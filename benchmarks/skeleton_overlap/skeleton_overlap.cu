// Skeleton overlap-mode throughput benchmark.
//
// Runs a Dense Jacobi ping-pong pipeline under each OverlapMode and prints:
//   MLUPS=... elapsed_seconds=... overlap=none|standard|extended|two-way-extended gpus=N n=N

#include "bench_report.h"

#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/devices.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/memory_side.h>
#include <carbon/neighbour_value.h>
#include <carbon/skeleton.h>
#include <carbon/stencil.h>
#include <carbon/system.h>

#include <array>
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
constexpr int kDefaultWarmup = 40;
constexpr int kDefaultRepeats = 3;
constexpr int kDefaultMinElapsedMs = 200;

struct Args {
  int64_t n = kDefaultN;
  int iters = kDefaultIters;
  int warmup = kDefaultWarmup;
  int n_gpus = -1;
  int repeats = kDefaultRepeats;
  int min_elapsed_ms = kDefaultMinElapsedMs;
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
  a.min_elapsed_ms =
      carbon::bench::parse_int_flag("--min-elapsed-ms", argc, argv, kDefaultMinElapsedMs);
  a.report_path = carbon::bench::parse_string_flag("--report", argc, argv);
  return a;
}

[[nodiscard]] auto
overlap_name(carbon::OverlapMode m) -> const char* {
  switch (m) {
    case carbon::OverlapMode::None:
      return "none";
    case carbon::OverlapMode::Standard:
      return "standard";
    case carbon::OverlapMode::Extended:
      return "extended";
    case carbon::OverlapMode::TwoWayExtended:
      return "two-way-extended";
  }
  return "?";
}

using DenseGrid = carbon::Dense;
using Span = DenseGrid::Span;
using Idx = DenseGrid::Idx;
using FieldT = DenseGrid::Field<float, 1>;

[[nodiscard]] auto
build_jacobi_step(DenseGrid& grid,
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

auto
bench_mode_once(DenseGrid& grid,
                carbon::System& sys,
                carbon::OverlapMode mode,
                int sweeps,
                int warmup) -> double {
  auto u = grid.newField<float, 1>(carbon::MemorySide::Device, "u");
  auto u_next = grid.newField<float, 1>(carbon::MemorySide::Device, "u_next");

  carbon::Skeleton init(sys);
  init.add(build_zero(grid, u, "zero_u"));
  init.add(build_zero(grid, u_next, "zero_u_next"));
  init.run();
  init.wait();

  carbon::Skeleton pipe(sys, carbon::Skeleton::Options{mode});
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
bench_mode_repeated(DenseGrid& grid,
                    carbon::System& sys,
                    carbon::OverlapMode mode,
                    int sweeps,
                    int warmup,
                    int repeats) -> carbon::bench::TimingStats {
  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(repeats));
  for (int r = 0; r < repeats; ++r) {
    samples.push_back(bench_mode_once(grid, sys, mode, sweeps, warmup));
  }
  return carbon::bench::compute_timing_stats(std::move(samples));
}

[[nodiscard]] auto
make_config_json(const Args& args,
                 const char* overlap) -> std::string {
  std::ostringstream out;
  out << "{\n";
  out << "    \"n\": " << args.n << ",\n";
  out << "    \"iters\": " << args.iters << ",\n";
  out << "    \"warmup\": " << args.warmup << ",\n";
  out << "    \"repeats\": " << args.repeats << ",\n";
  out << "    \"overlap\": \"" << overlap << "\"\n";
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
    std::cerr << "skeleton_overlap: no devices visible\n";
    return EXIT_FAILURE;
  }
  if (args.n_gpus > 0 && static_cast<size_t>(args.n_gpus) < devices.size()) {
    devices.erase(devices.begin() + args.n_gpus, devices.end());
  }

  carbon::System sys(devices);
  const carbon::int64_3d global{args.n, args.n, args.n};
  const carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      true);

  try {
    DenseGrid grid(sys, global, stencil);
    const int sweeps = (args.iters + 1) / 2 * 2;
    const int64_t cells = global.volume();
    const double cell_updates = static_cast<double>(cells) * static_cast<double>(sweeps) / 1.0e6;

    const std::array<carbon::OverlapMode, 4> modes{
        carbon::OverlapMode::None, carbon::OverlapMode::Standard, carbon::OverlapMode::Extended,
        carbon::OverlapMode::TwoWayExtended};

    for (const auto mode : modes) {
      const carbon::bench::TimingStats stats =
          bench_mode_repeated(grid, sys, mode, sweeps, args.warmup, args.repeats);
      const double mlups = cell_updates / stats.median_seconds;
      const char* mode_name = overlap_name(mode);

      if (stats.median_seconds * 1000.0 < static_cast<double>(args.min_elapsed_ms)) {
        std::cerr << "WARNING launch_bound: skeleton_overlap overlap=" << mode_name
                  << " median_elapsed_ms=" << (stats.median_seconds * 1000.0)
                  << " < threshold=" << args.min_elapsed_ms << '\n';
      }

      std::cout << "MLUPS=" << mlups << " elapsed_seconds=" << stats.median_seconds
                << " overlap=" << mode_name << " gpus=" << sys.gpu_group().size() << " n=" << args.n
                << " iters=" << sweeps << " repeats=" << args.repeats << '\n';

      if (!args.report_path.empty()) {
        const std::string path = args.report_path + "." + std::string(mode_name);
        if (!carbon::bench::write_json_report(path, "skeleton_overlap",
                                              make_config_json(args, mode_name), sys, stats, mlups,
                                              "mlups")) {
          return EXIT_FAILURE;
        }
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "skeleton_overlap: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
