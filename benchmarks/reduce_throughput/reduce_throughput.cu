// Three-stage reduction throughput benchmark across grid families.
//
// Times repeated Sum reductions over a fixed domain and prints scrapeable
// throughput lines:
//   REDUCE_GBs=... ELEMENTS_PER_S=... elapsed_seconds=... grid=dense|disg|block gpus=N

#include <carbon/block.h>
#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/dense_disg.h>
#include <carbon/devices.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/memory_side.h>
#include <carbon/reduce.h>
#include <carbon/stencil.h>
#include <carbon/system.h>
#include "bench_report.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr int64_t kDefaultN = 256;
constexpr int kDefaultIters = 500;
constexpr int kDefaultWarmup = 50;
constexpr int kDefaultRepeats = 5;
constexpr int kDefaultBatchIters = 1;

struct Args {
  int64_t n = kDefaultN;
  int iters = kDefaultIters;
  int warmup = kDefaultWarmup;
  int n_gpus = -1;
  int repeats = kDefaultRepeats;
  int batch_iters = kDefaultBatchIters;
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
  a.batch_iters = carbon::bench::parse_int_flag("--batch-iters", argc, argv, kDefaultBatchIters);
  a.report_path = carbon::bench::parse_string_flag("--report", argc, argv);
  return a;
}

template <typename Grid>
[[nodiscard]] auto
dense_like_owned_cells(const Grid& grid) -> int64_t {
  int64_t total = 0;
  for (size_t g = 0; g < grid.gpu_count(); ++g) {
    total += grid.spans()[g].iter_extent(carbon::DataView::Standard).volume();
  }
  return total;
}

template <typename SBlock>
[[nodiscard]] auto
block_owned_active_cells(const carbon::Block<SBlock>& grid) -> int64_t {
  int64_t total = 0;
  for (size_t g = 0; g < grid.gpu_count(); ++g) {
    const auto& span = grid.host_spans()[g];
    const auto* bm = span.bitmask();
    const uint32_t n_owned = grid.block_count(static_cast<uint32_t>(g));
    for (uint32_t b = 0; b < n_owned; ++b) {
      for (int z = 0; z < SBlock::MemZ; ++z) {
        for (int y = 0; y < SBlock::MemY; ++y) {
          for (int x = 0; x < SBlock::MemX; ++x) {
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

template <typename Grid>
auto
bench_reduce_once(Grid& grid,
                  carbon::System& sys,
                  int iters,
                  int warmup,
                  int batch_iters) -> double {
  using Span = typename Grid::Span;
  using Idx = typename Grid::Idx;

  auto field = grid.template newField<float, 1>(carbon::MemorySide::Device, "f");
  carbon::ReductionResult<float> result(sys);
  carbon::Kernel reducer = carbon::Kernel::reduce<float>(
      grid,
      [&field](carbon::KernelLoader& loader) {
        const auto in = loader.read(field);
        return [in] CARBON_HD(Span s, Idx idx) -> float {
          return in.read(s, idx, 0);
        };
      },
      result, carbon::reduce::Sum<float>{}, "sum", 0U);

  carbon::Stream stream = sys.stream();
  for (int i = 0; i < warmup; ++i) {
    for (int b = 0; b < batch_iters; ++b) {
      reducer.launch(stream);
      result.sync(stream);
    }
    stream.wait();
  }

  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    for (int b = 0; b < batch_iters; ++b) {
      reducer.launch(stream);
      result.sync(stream);
    }
    stream.wait();
  }
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(t1 - t0).count();
}

template <typename Grid>
[[nodiscard]] auto
bench_reduce_repeated(Grid& grid,
                      carbon::System& sys,
                      const Args& args) -> carbon::bench::TimingStats {
  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(args.repeats));
  for (int r = 0; r < args.repeats; ++r) {
    samples.push_back(bench_reduce_once(grid, sys, args.iters, args.warmup, args.batch_iters));
  }
  return carbon::bench::compute_timing_stats(std::move(samples));
}

[[nodiscard]] auto
make_config_json(const Args& args,
                 std::string_view grid_tag) -> std::string {
  std::ostringstream out;
  out << "{\n";
  out << "    \"n\": " << args.n << ",\n";
  out << "    \"iters\": " << args.iters << ",\n";
  out << "    \"warmup\": " << args.warmup << ",\n";
  out << "    \"repeats\": " << args.repeats << ",\n";
  out << "    \"batch_iters\": " << args.batch_iters << ",\n";
  out << "    \"grid\": \"" << grid_tag << "\"\n";
  out << "  }";
  return out.str();
}

template <typename Grid>
auto
report_dense_family(std::string_view tag,
                    Grid& grid,
                    carbon::System& sys,
                    const Args& args) -> int {
  const int64_t cells = dense_like_owned_cells(grid);
  if (cells <= 0) {
    std::cerr << "reduce_throughput: empty domain for grid=" << tag << '\n';
    return EXIT_FAILURE;
  }

  const carbon::bench::TimingStats stats = bench_reduce_repeated(grid, sys, args);
  const double total_reductions = static_cast<double>(cells) * static_cast<double>(args.iters) *
                                  static_cast<double>(args.batch_iters);
  const double eps = total_reductions / stats.median_seconds;
  const double gbs = eps * static_cast<double>(sizeof(float)) / 1.0e9;

  std::cout << "REDUCE_GBs=" << gbs << " ELEMENTS_PER_S=" << eps
            << " elapsed_seconds=" << stats.median_seconds << " grid=" << tag
            << " gpus=" << sys.gpu_group().size() << " n=" << args.n << " iters=" << args.iters
            << " batch_iters=" << args.batch_iters << " repeats=" << args.repeats << '\n';

  if (!args.report_path.empty()) {
    const std::string path = args.report_path + "." + std::string(tag);
    if (!carbon::bench::write_json_report(path, "reduce_throughput", make_config_json(args, tag),
                                          sys, stats, gbs, "reduce_gbs")) {
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}

template <typename SBlock>
auto
report_block(std::string_view tag,
             carbon::Block<SBlock>& grid,
             carbon::System& sys,
             const Args& args) -> int {
  const int64_t cells = block_owned_active_cells(grid);
  if (cells <= 0) {
    std::cerr << "reduce_throughput: empty domain for grid=" << tag << '\n';
    return EXIT_FAILURE;
  }

  const carbon::bench::TimingStats stats = bench_reduce_repeated(grid, sys, args);
  const double total_reductions = static_cast<double>(cells) * static_cast<double>(args.iters) *
                                  static_cast<double>(args.batch_iters);
  const double eps = total_reductions / stats.median_seconds;
  const double gbs = eps * static_cast<double>(sizeof(float)) / 1.0e9;

  std::cout << "REDUCE_GBs=" << gbs << " ELEMENTS_PER_S=" << eps
            << " elapsed_seconds=" << stats.median_seconds << " grid=" << tag
            << " gpus=" << sys.gpu_group().size() << " n=" << args.n << " iters=" << args.iters
            << " batch_iters=" << args.batch_iters << " repeats=" << args.repeats << '\n';

  if (!args.report_path.empty()) {
    const std::string path = args.report_path + "." + std::string(tag);
    if (!carbon::bench::write_json_report(path, "reduce_throughput", make_config_json(args, tag),
                                          sys, stats, gbs, "reduce_gbs")) {
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
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

}  // namespace

int
main(int argc,
     char** argv) {
  carbon::init();
  const Args args = parse_args(argc, argv);

  std::vector<carbon::detail::Gpu> devices = carbon::get_all_devices();
  if (devices.empty()) {
    std::cerr << "reduce_throughput: no devices visible\n";
    return EXIT_FAILURE;
  }
  if (args.n_gpus > 0 && static_cast<size_t>(args.n_gpus) < devices.size()) {
    devices.erase(devices.begin() + args.n_gpus, devices.end());
  }

  carbon::System sys(devices);
  const carbon::int64_3d global{args.n, args.n, args.n};

  try {
    const carbon::Stencil stencil(
        std::vector<carbon::int32_3d>{
            {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
        true);
    carbon::Dense dense(sys, global, stencil);
    if (report_dense_family("dense", dense, sys, args) != EXIT_SUCCESS) {
      return EXIT_FAILURE;
    }

    carbon::DenseDisg disg(sys, global, 1);
    if (report_dense_family("disg", disg, sys, args) != EXIT_SUCCESS) {
      return EXIT_FAILURE;
    }

    using SB = carbon::detail::block::DefaultStaticBlock;
    using BlockGrid = carbon::Block<SB>;
    const auto predicate = [n = args.n](const carbon::int64_3d& gc) -> bool {
      return sphere_predicate(n, gc);
    };
    BlockGrid block(sys, global, predicate, "reduce_block");
    if (report_block("block", block, sys, args) != EXIT_SUCCESS) {
      return EXIT_FAILURE;
    }
  } catch (const std::exception& e) {
    std::cerr << "reduce_throughput: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
