// Skeleton region-mask benchmark: interior stencil + K boundary-condition kernels.
//
// Each half step of a Jacobi ping-pong is one interior 6-point stencil plus K small
// boundary kernels, all writing the same field. Boundary kernel k owns the k-th y-strip of
// the two x faces, so the K + 1 RegionMasks are pairwise disjoint and, with masks honoured,
// the K + 1 kernels can run side by side. Every kernel launches over the whole grid and
// filters itself (masks are declaration-only), which is the v1 cost model.
//
// Variants, each for K in {1, 4, 16} (or --k):
//   masks          : masks honoured, uncapped streams
//   ignored        : Options::ignore_region_masks (the pre-region-mask schedule)
//   masks_max2/4   : masks honoured, Options::max_streams = 2 / 4
//   ignored_graph  : masks ignored, single stream captured as a CUDA graph
//
// Prints one line per (variant, K):
//   us_per_iter=... host_us_per_iter=... variant=... k=... streams=... gpus=... n=...
// us_per_iter is wall time per full iteration (both half steps) including the final drain;
// host_us_per_iter is the time spent inside run() calls (enqueue cost: launches, events,
// fork/join), which grows with the stream count.

#include "bench_report.h"

#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/devices.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/memory_side.h>
#include <carbon/neighbour_value.h>
#include <carbon/region_mask.h>
#include <carbon/skeleton.h>
#include <carbon/stencil.h>
#include <carbon/system.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int64_t kDefaultN = 256;
constexpr int kDefaultIters = 200;
constexpr int kDefaultWarmup = 20;
constexpr int kDefaultRepeats = 3;

struct Args {
  int64_t n = kDefaultN;
  int iters = kDefaultIters;
  int warmup = kDefaultWarmup;
  int n_gpus = -1;
  int repeats = kDefaultRepeats;
  int k = -1;
  std::string report_path{};
};

[[nodiscard]] auto
parse_args(int argc,
           char** argv) -> Args {
  Args a{};
  a.n = static_cast<int64_t>(
      carbon::bench::parse_int_flag("--n", argc, argv, static_cast<int>(kDefaultN)));
  a.iters = carbon::bench::parse_int_flag("--iters", argc, argv, kDefaultIters);
  a.warmup = carbon::bench::parse_int_flag("--warmup", argc, argv, kDefaultWarmup);
  a.n_gpus = carbon::bench::parse_int_flag("--gpus", argc, argv, -1);
  a.repeats = carbon::bench::parse_int_flag("--repeats", argc, argv, kDefaultRepeats);
  a.k = carbon::bench::parse_int_flag("--k", argc, argv, -1);
  a.report_path = carbon::bench::parse_string_flag("--report", argc, argv);
  return a;
}

using DenseGrid = carbon::Dense;
using Span = DenseGrid::Span;
using Idx = DenseGrid::Idx;
using FieldT = DenseGrid::Field<float, 1>;
using carbon::RegionMask;

// Label 0 is the interior; label 1 + k is boundary strip k of the x faces.
struct Labeling {
  int64_t n;
  int k;

  [[nodiscard]] CARBON_HD auto
  label(const carbon::int64_3d& g) const -> int {
    if (g.x != 0 && g.x != n - 1) {
      return 0;
    }
    return 1 + static_cast<int>(g.y * k / n);
  }
};

[[nodiscard]] auto
build_interior(DenseGrid& grid,
               FieldT& fin,
               FieldT& fout,
               Labeling lab,
               std::string name) -> carbon::Kernel {
  return carbon::Kernel::compute(
      grid,
      [&fin, &fout, lab](carbon::KernelLoader& loader) {
        loader.region(RegionMask::bit(0));
        const auto in = loader.read_stencil(fin);
        const auto out = loader.write(fout);
        return [in, out, lab] CARBON_HD(Span s, Idx idx) -> void {
          if (lab.label(s.global_location(idx)) != 0) {
            return;
          }
          const carbon::NeighbourValue<float> xp =
              in.neighbour(s, idx, carbon::int8_3d{1, 0, 0}, 0);
          const carbon::NeighbourValue<float> xm =
              in.neighbour(s, idx, carbon::int8_3d{-1, 0, 0}, 0);
          const carbon::NeighbourValue<float> yp =
              in.neighbour(s, idx, carbon::int8_3d{0, 1, 0}, 0);
          const carbon::NeighbourValue<float> ym =
              in.neighbour(s, idx, carbon::int8_3d{0, -1, 0}, 0);
          const carbon::NeighbourValue<float> zp =
              in.neighbour(s, idx, carbon::int8_3d{0, 0, 1}, 0);
          const carbon::NeighbourValue<float> zm =
              in.neighbour(s, idx, carbon::int8_3d{0, 0, -1}, 0);
          if (!(xp.valid && xm.valid && yp.valid && ym.valid && zp.valid && zm.valid)) {
            out.write(s, idx, 0, 0.0F);
            return;
          }
          const float sum = xp.value + xm.value + yp.value + ym.value + zp.value + zm.value;
          out.write(s, idx, 0, sum * (1.0F / 6.0F));
        };
      },
      std::move(name));
}

[[nodiscard]] auto
build_boundary(DenseGrid& grid,
               FieldT& fin,
               FieldT& fout,
               Labeling lab,
               int strip,
               std::string name) -> carbon::Kernel {
  return carbon::Kernel::compute(
      grid,
      [&fin, &fout, lab, strip](carbon::KernelLoader& loader) {
        loader.region(RegionMask::bit(static_cast<unsigned>(1 + strip)));
        const auto in = loader.read(fin);
        const auto out = loader.write(fout);
        return [in, out, lab, strip] CARBON_HD(Span s, Idx idx) -> void {
          if (lab.label(s.global_location(idx)) != 1 + strip) {
            return;
          }
          out.write(s, idx, 0, 0.5F * in.read(s, idx, 0) + 0.01F * static_cast<float>(strip));
        };
      },
      std::move(name));
}

[[nodiscard]] auto
build_zero(DenseGrid& grid,
           FieldT& field,
           std::string name) -> carbon::Kernel {
  return carbon::Kernel::compute(
      grid,
      [&field](carbon::KernelLoader& loader) {
        const auto out = loader.write(field);
        return [out] CARBON_HD(Span s, Idx idx) -> void {
          out.write(s, idx, 0, 0.0F);
        };
      },
      std::move(name));
}

struct Variant {
  const char* name;
  carbon::Skeleton::Options options;
};

[[nodiscard]] auto
make_variants() -> std::vector<Variant> {
  carbon::Skeleton::Options ignored;
  ignored.ignore_region_masks = true;
  carbon::Skeleton::Options max2;
  max2.max_streams = 2U;
  carbon::Skeleton::Options max4;
  max4.max_streams = 4U;
  carbon::Skeleton::Options graph = ignored;
  graph.use_cuda_graph = true;
  return {{"masks", carbon::Skeleton::Options{}},
          {"ignored", ignored},
          {"masks_max2", max2},
          {"masks_max4", max4},
          {"ignored_graph", graph}};
}

struct Sample {
  double wall_seconds;
  double host_seconds;
  size_t streams;
};

auto
bench_once(DenseGrid& grid,
           carbon::System& sys,
           const Variant& variant,
           int k,
           int64_t n,
           int iters,
           int warmup) -> Sample {
  auto u = grid.newField<float, 1>(carbon::MemorySide::Device, "u");
  auto v = grid.newField<float, 1>(carbon::MemorySide::Device, "v");

  carbon::Skeleton init(sys);
  init.add(build_zero(grid, u, "zero_u"));
  init.add(build_zero(grid, v, "zero_v"));
  init.run();
  init.wait();

  const Labeling lab{n, k};
  carbon::Skeleton pipe(sys, variant.options);
  pipe.add(build_interior(grid, u, v, lab, "interior_uv"));
  for (int s = 0; s < k; ++s) {
    pipe.add(build_boundary(grid, u, v, lab, s, "bc_uv_" + std::to_string(s)));
  }
  pipe.add(build_interior(grid, v, u, lab, "interior_vu"));
  for (int s = 0; s < k; ++s) {
    pipe.add(build_boundary(grid, v, u, lab, s, "bc_vu_" + std::to_string(s)));
  }
  pipe.compile();

  for (int w = 0; w < warmup; ++w) {
    pipe.run();
  }
  pipe.wait();

  double host = 0.0;
  const auto t0 = std::chrono::steady_clock::now();
  for (int it = 0; it < iters; ++it) {
    const auto h0 = std::chrono::steady_clock::now();
    pipe.run();
    host += std::chrono::duration<double>(std::chrono::steady_clock::now() - h0).count();
  }
  pipe.wait();
  const auto t1 = std::chrono::steady_clock::now();
  return {std::chrono::duration<double>(t1 - t0).count(), host, pipe.num_streams()};
}

[[nodiscard]] auto
make_config_json(const Args& args,
                 const char* variant,
                 int k) -> std::string {
  std::ostringstream out;
  out << "{\n";
  out << "    \"n\": " << args.n << ",\n";
  out << "    \"iters\": " << args.iters << ",\n";
  out << "    \"warmup\": " << args.warmup << ",\n";
  out << "    \"repeats\": " << args.repeats << ",\n";
  out << "    \"k\": " << k << ",\n";
  out << "    \"variant\": \"" << variant << "\"\n";
  out << "  }";
  return out.str();
}

}  // namespace

int
main(int argc,
     char** argv) {
  carbon::init();
  const Args args = parse_args(argc, argv);

  std::vector<carbon::detail::Gpu> devices = carbon::get_all_devices();
  if (devices.empty()) {
    std::cerr << "skeleton_region_mask: no devices visible\n";
    return EXIT_FAILURE;
  }
  if (args.n_gpus > 0 && static_cast<size_t>(args.n_gpus) < devices.size()) {
    devices.erase(devices.begin() + args.n_gpus, devices.end());
  }
  if (args.k > 254 || args.k == 0 || args.n < 8) {
    std::cerr << "skeleton_region_mask: need 1 <= --k <= 254 and --n >= 8\n";
    return EXIT_FAILURE;
  }

  carbon::System sys(devices);
  const carbon::int64_3d global{args.n, args.n, args.n};
  const carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      true);

  try {
    DenseGrid grid(sys, global, stencil);
    const std::vector<int> ks = args.k > 0 ? std::vector<int>{args.k} : std::vector<int>{1, 4, 16};
    for (const int k : ks) {
      for (const Variant& variant : make_variants()) {
        std::vector<double> wall;
        double host = 0.0;
        size_t streams = 0;
        for (int r = 0; r < args.repeats; ++r) {
          const Sample s = bench_once(grid, sys, variant, k, args.n, args.iters, args.warmup);
          wall.push_back(s.wall_seconds);
          host += s.host_seconds;
          streams = s.streams;
        }
        const carbon::bench::TimingStats stats = carbon::bench::compute_timing_stats(wall);
        const double us_per_iter = stats.median_seconds * 1.0e6 / args.iters;
        const double host_us_per_iter = host * 1.0e6 / (args.iters * args.repeats);
        std::cout << "us_per_iter=" << us_per_iter << " host_us_per_iter=" << host_us_per_iter
                  << " variant=" << variant.name << " k=" << k << " streams=" << streams
                  << " gpus=" << sys.gpu_group().size() << " n=" << args.n
                  << " iters=" << args.iters << " repeats=" << args.repeats << '\n';

        if (!args.report_path.empty()) {
          const std::string path = args.report_path + "." + variant.name + ".k" + std::to_string(k);
          if (!carbon::bench::write_json_report(path, "skeleton_region_mask",
                                                make_config_json(args, variant.name, k), sys, stats,
                                                us_per_iter, "us_per_iter")) {
            return EXIT_FAILURE;
          }
        }
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "skeleton_region_mask: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
