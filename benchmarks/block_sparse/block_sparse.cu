// Block sparse (sphere-predicate) stencil throughput benchmark.
//
// Runs a Block Jacobi ping-pong pipeline on a sphere-inscribed cube and prints:
//   MLUPS=... elapsed_seconds=... halo_pair_count=... active_fraction=F grid=block gpus=N n=N

#include "bench_report.h"

#include <carbon/block.h>
#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/devices.h>
#include <carbon/halo_block.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/memory_side.h>
#include <carbon/neighbour_value.h>
#include <carbon/skeleton.h>
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

using SB = carbon::detail::block::DefaultStaticBlock;
using BlockGrid = carbon::Block<SB>;
using Span = BlockGrid::Span;
using Idx = BlockGrid::Idx;
using FieldT = BlockGrid::Field<float, 1>;

constexpr int64_t kDefaultN = 256;
constexpr int kDefaultIters = 200;
constexpr int kDefaultWarmup = 40;
constexpr int kDefaultRepeats = 3;

struct Args {
  int64_t n = kDefaultN;
  int iters = kDefaultIters;
  int warmup = kDefaultWarmup;
  int n_gpus = -1;
  int repeats = kDefaultRepeats;
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
  a.report_path = carbon::bench::parse_string_flag("--report", argc, argv);
  return a;
}

[[nodiscard]] auto
sphere_predicate(int64_t n,
                 const carbon::int64_3d& gc) -> bool {
  const double cx = (static_cast<double>(n) - 1.0) * 0.5;
  const double cy = (static_cast<double>(n) - 1.0) * 0.5;
  const double cz = (static_cast<double>(n) - 1.0) * 0.5;
  const double r = static_cast<double>(n) * 0.42;
  const double dx = static_cast<double>(gc.x) - cx;
  const double dy = static_cast<double>(gc.y) - cy;
  const double dz = static_cast<double>(gc.z) - cz;
  return (dx * dx + dy * dy + dz * dz) <= (r * r);
}

[[nodiscard]] auto
active_fraction(int64_t n) -> double {
  int64_t active = 0;
  for (int64_t z = 0; z < n; ++z) {
    for (int64_t y = 0; y < n; ++y) {
      for (int64_t x = 0; x < n; ++x) {
        if (sphere_predicate(n, carbon::int64_3d{x, y, z})) {
          ++active;
        }
      }
    }
  }
  return static_cast<double>(active) / static_cast<double>(n * n * n);
}

[[nodiscard]] auto
block_owned_active_cells(const BlockGrid& grid) -> int64_t {
  int64_t total = 0;
  for (size_t g = 0; g < grid.gpu_count(); ++g) {
    const auto& span = grid.host_spans()[g];
    const auto* bm = span.bitmask();
    const uint32_t n_owned = grid.block_count(static_cast<uint32_t>(g));
    for (uint32_t b = 0; b < n_owned; ++b) {
      for (int z = 0; z < SB::MemZ; ++z) {
        for (int y = 0; y < SB::MemY; ++y) {
          for (int x = 0; x < SB::MemX; ++x) {
            if (bm[b].is_active(x, y, z)) {
              ++total;
            }
          }
        }
      }
    }
  }
  return total;
}

[[nodiscard]] auto
build_jacobi_step(BlockGrid& grid,
                  FieldT& fin,
                  FieldT& fout,
                  std::string name) -> carbon::Kernel {
  return carbon::Kernel::compute(
      grid,
      [&fin, &fout](carbon::KernelLoader& loader) {
        const auto in = loader.read_stencil(fin);
        const auto out = loader.write(fout);
        return [in, out] CARBON_HD(Span s, Idx idx) -> void {
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
          if (!zm.valid) {
            out.write(s, idx, 0, +1.0F);
            return;
          }
          if (!zp.valid) {
            out.write(s, idx, 0, -1.0F);
            return;
          }
          if (!(xp.valid && xm.valid && yp.valid && ym.valid)) {
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
build_zero(BlockGrid& grid,
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

auto
bench_once(BlockGrid& grid,
           carbon::System& sys,
           int sweeps,
           int warmup) -> double {
  auto u = grid.newField<float, 1>(carbon::MemorySide::Device, "u");
  auto u_next = grid.newField<float, 1>(carbon::MemorySide::Device, "u_next");

  carbon::Skeleton init(sys);
  init.add(build_zero(grid, u, "zero_u"));
  init.add(build_zero(grid, u_next, "zero_u_next"));
  init.run();
  init.wait();

  carbon::Skeleton pipe(sys);
  pipe.add(build_jacobi_step(grid, u, u_next, "step_ab"));
  pipe.add(build_jacobi_step(grid, u_next, u, "step_ba"));
  pipe.compile();

  const int cycles = sweeps / 2;
  for (int w = 0; w < warmup / 2; ++w) {
    pipe.run();
  }
  pipe.wait();

  const auto t0 = std::chrono::steady_clock::now();
  for (int c = 0; c < cycles; ++c) {
    pipe.run();
  }
  pipe.wait();
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(t1 - t0).count();
}

[[nodiscard]] auto
bench_repeated(BlockGrid& grid,
               carbon::System& sys,
               int sweeps,
               int warmup,
               int repeats) -> carbon::bench::TimingStats {
  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(repeats));
  for (int r = 0; r < repeats; ++r) {
    samples.push_back(bench_once(grid, sys, sweeps, warmup));
  }
  return carbon::bench::compute_timing_stats(std::move(samples));
}

[[nodiscard]] auto
make_config_json(const Args& args,
                 double active_frac,
                 size_t halo_pairs) -> std::string {
  std::ostringstream out;
  out << "{\n";
  out << "    \"n\": " << args.n << ",\n";
  out << "    \"iters\": " << args.iters << ",\n";
  out << "    \"warmup\": " << args.warmup << ",\n";
  out << "    \"repeats\": " << args.repeats << ",\n";
  out << "    \"active_fraction\": " << active_frac << ",\n";
  out << "    \"halo_pair_count\": " << halo_pairs << "\n";
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
    std::cerr << "block_sparse: no devices visible\n";
    return EXIT_FAILURE;
  }
  if (args.n_gpus > 0 && static_cast<size_t>(args.n_gpus) < devices.size()) {
    devices.erase(devices.begin() + args.n_gpus, devices.end());
  }

  carbon::System sys(devices);
  const carbon::int64_3d global{args.n, args.n, args.n};
  const auto predicate = [n = args.n](const carbon::int64_3d& gc) -> bool {
    return sphere_predicate(n, gc);
  };

  try {
    BlockGrid grid(sys, global, predicate, "block_sparse");
    const double frac = active_fraction(args.n);
    const int64_t cells = block_owned_active_cells(grid);
    if (cells <= 0) {
      std::cerr << "block_sparse: no active cells\n";
      return EXIT_FAILURE;
    }

    auto probe = grid.newField<float, 1>(carbon::MemorySide::Device, "halo_probe");
    carbon::HaloBlock<float, 1> halo_probe(grid, probe);
    const size_t halo_pairs = halo_probe.pair_count();

    const int sweeps = (args.iters + 1) / 2 * 2;
    const carbon::bench::TimingStats stats =
        bench_repeated(grid, sys, sweeps, args.warmup, args.repeats);
    const double mlups =
        static_cast<double>(cells) * static_cast<double>(sweeps) / (1.0e6 * stats.median_seconds);

    std::cout << "MLUPS=" << mlups << " elapsed_seconds=" << stats.median_seconds
              << " halo_pair_count=" << halo_pairs << " active_fraction=" << frac
              << " grid=block gpus=" << sys.gpu_group().size() << " n=" << args.n
              << " iters=" << sweeps << " repeats=" << args.repeats << '\n';

    if (!args.report_path.empty()) {
      if (!carbon::bench::write_json_report(args.report_path, "block_sparse",
                                            make_config_json(args, frac, halo_pairs), sys, stats,
                                            mlups, "mlups")) {
        return EXIT_FAILURE;
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "block_sparse: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
