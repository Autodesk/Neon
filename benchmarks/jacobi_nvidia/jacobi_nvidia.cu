// NVIDIA multi-gpu-programming-models Jacobi Laplace port on carbon::Dense.
//
// Maps reference grid ny x nx (y-decomposed, periodic y) to carbon extent
// {nx, 1, ny - 2} with BoundaryMode::PeriodicZ and a four-neighbour stencil in x/z.

#include "bench_report.h"

#include <carbon/boundary_mode.h>
#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/detail/gpu.h>
#include <carbon/devices.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/memory_side.h>
#include <carbon/reduce.h>
#include <carbon/skeleton.h>
#include <carbon/stencil.h>
#include <carbon/system.h>
#include <carbon/transfer.h>

#ifdef CARBON_HAVE_MPI_NCCL
#include <carbon/distributed.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr float kTol = 1.0e-8F;
constexpr int kDefaultNx = 16384;
constexpr int kDefaultNy = 16384;
constexpr int kDefaultNiter = 1000;
constexpr int kDefaultNccheck = 1;
constexpr int kWarmupIters = 10;
constexpr int kDefaultRepeats = 1;

struct Args {
  int nx = kDefaultNx;
  int ny = kDefaultNy;
  int niter = kDefaultNiter;
  int nccheck = kDefaultNccheck;
  int n_gpus = -1;
  int repeats = kDefaultRepeats;
  int warmup = kWarmupIters;
  bool csv = false;
  bool skip_validation = false;
  bool fixed_iters = false;
  std::string overlap = "none";
  std::string block = "64x1x8";
  std::string reduce_finalize = "device";
  std::string reduce_block{};
  std::string norm_sync = "blocking";
  std::string dump_path{};
  std::string report_path{};
};

using DenseGrid = carbon::Dense;
using Span = DenseGrid::Span;
using Idx = DenseGrid::Idx;
using FieldT = DenseGrid::Field<float, 1>;

[[nodiscard]] auto
pi() -> double {
  return 2.0 * std::asin(1.0);
}

[[nodiscard]] auto
get_argval_int(char** begin,
               char** end,
               std::string_view flag,
               int fallback) -> int {
  for (char** it = begin; it != end; ++it) {
    if (std::string_view{*it} == flag && it + 1 != end) {
      return std::stoi(*(it + 1));
    }
  }
  return fallback;
}

[[nodiscard]] auto
get_argval_string(char** begin,
                  char** end,
                  std::string_view flag) -> std::string {
  for (char** it = begin; it != end; ++it) {
    if (std::string_view{*it} == flag && it + 1 != end) {
      return std::string{*(it + 1)};
    }
  }
  return {};
}

[[nodiscard]] auto
has_flag(char** begin,
         char** end,
         std::string_view flag) -> bool {
  for (char** it = begin; it != end; ++it) {
    if (std::string_view{*it} == flag) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] auto
parse_args(int argc,
           char** argv) -> Args {
  Args a{};
  a.nx = get_argval_int(argv, argv + argc, "-nx", kDefaultNx);
  a.ny = get_argval_int(argv, argv + argc, "-ny", kDefaultNy);
  a.niter = get_argval_int(argv, argv + argc, "-niter", kDefaultNiter);
  a.nccheck = get_argval_int(argv, argv + argc, "-nccheck", kDefaultNccheck);
  a.n_gpus = get_argval_int(argv, argv + argc, "-gpus", -1);
  a.repeats = get_argval_int(argv, argv + argc, "-repeats", kDefaultRepeats);
  a.warmup = get_argval_int(argv, argv + argc, "--warmup", kWarmupIters);
  a.csv = has_flag(argv, argv + argc, "-csv");
  a.skip_validation =
      has_flag(argv, argv + argc, "--skip-validation") ||
      has_flag(argv, argv + argc, "-skip-validation");
  a.fixed_iters = has_flag(argv, argv + argc, "--fixed-iters");
  const std::string overlap = get_argval_string(argv, argv + argc, "--overlap");
  if (!overlap.empty()) {
    a.overlap = overlap;
  }
  const std::string block = get_argval_string(argv, argv + argc, "--block");
  if (!block.empty()) {
    a.block = block;
  }
  a.dump_path = get_argval_string(argv, argv + argc, "--dump");
  a.report_path = get_argval_string(argv, argv + argc, "--report");
  const std::string reduce_finalize =
      get_argval_string(argv, argv + argc, "--reduce-finalize");
  if (!reduce_finalize.empty()) {
    a.reduce_finalize = reduce_finalize;
  }
  {
    const std::string rb = get_argval_string(argv, argv + argc, "--reduce-block");
    if (!rb.empty()) {
      a.reduce_block = rb;
    }
  }
  const std::string norm_sync = get_argval_string(argv, argv + argc, "--norm-sync");
  if (!norm_sync.empty()) {
    a.norm_sync = norm_sync;
  }
  return a;
}

enum class NormSyncMode { Blocking, Pipelined };

[[nodiscard]] auto
parse_norm_sync(std::string_view s) -> std::optional<NormSyncMode> {
  if (s == "blocking" || s.empty()) {
    return NormSyncMode::Blocking;
  }
  if (s == "pipelined") {
    return NormSyncMode::Pipelined;
  }
  return std::nullopt;
}

[[nodiscard]] auto
apply_reduce_block_spec(carbon::reduce::ReduceOptions& opts,
                        std::string_view spec) -> bool {
  if (spec.empty()) {
    return true;
  }
  std::string s{spec};
  for (char& c : s) {
    if (c == 'x' || c == 'X') {
      c = ' ';
    }
  }
  unsigned a = 0U;
  unsigned b = 0U;
  std::istringstream in{s};
  in >> a;
  if (in >> b) {
    try {
      const auto [bx, by] = carbon::reduce::normalize_dense_stage1_block_2d(a, b);
      opts.stage1_threads = static_cast<unsigned>(bx);
      opts.stage1_threads_y = static_cast<unsigned>(by);
    } catch (const std::invalid_argument&) {
      return false;
    }
    return true;
  }
  if (a == 0U) {
    return true;
  }
  try {
    opts.stage1_threads = static_cast<unsigned>(carbon::reduce::normalize_dense_stage1_threads(a));
  } catch (const std::invalid_argument&) {
    return false;
  }
  return true;
}

[[nodiscard]] auto
make_reduce_options(std::string_view finalize,
                    std::string_view reduce_block) -> std::optional<carbon::reduce::ReduceOptions> {
  carbon::reduce::ReduceOptions opts{};
  if (finalize == "device" || finalize.empty()) {
    opts.finalize = carbon::reduce::ReduceFinalize::DeviceReduce;
  } else if (finalize == "atomic") {
    opts.finalize = carbon::reduce::ReduceFinalize::BlockAtomic;
  } else {
    return std::nullopt;
  }
  if (!apply_reduce_block_spec(opts, reduce_block)) {
    return std::nullopt;
  }
  return opts;
}

[[nodiscard]] auto
parse_block_shape(std::string_view spec) -> carbon::LaunchShape {
  unsigned bx = 64U;
  unsigned by = 1U;
  unsigned bz = 8U;
  std::string s{spec};
  for (char& c : s) {
    if (c == 'x' || c == 'X') {
      c = ' ';
    }
  }
  std::istringstream in{s};
  in >> bx >> by >> bz;
  return carbon::LaunchShape{bx, by, bz};
}

[[nodiscard]] auto
parse_overlap(std::string_view s) -> std::optional<carbon::OverlapMode> {
  if (s == "none" || s.empty()) {
    return carbon::OverlapMode::None;
  }
  if (s == "standard") {
    return carbon::OverlapMode::Standard;
  }
  if (s == "extended") {
    return carbon::OverlapMode::Extended;
  }
  if (s == "two-way-extended" || s == "twoway") {
    return carbon::OverlapMode::TwoWayExtended;
  }
  return std::nullopt;
}

[[nodiscard]] auto
make_stencil() -> carbon::Stencil {
  return carbon::Stencil(
      std::vector<carbon::int32_3d>{{1, 0, 0}, {-1, 0, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);
}

[[nodiscard]] auto
build_fused_jacobi_step(DenseGrid& grid,
                        FieldT& fin,
                        FieldT& fout,
                        carbon::ReductionResult<float>& sumsq,
                        int64_t nx_global,
                        bool mirror_periodic_z,
                        carbon::reduce::ReduceOptions reduce_opts,
                        std::string name) -> carbon::Kernel {
  return carbon::Kernel::reduce<float>(
      grid,
      [&fin, &fout, nx_global, mirror_periodic_z](carbon::KernelLoader& loader) {
        const auto in = loader.read_stencil(fin);
        const auto old_p = loader.read(fin);
        const auto out = loader.write(fout);
        return [in, old_p, out, nx_global,
                mirror_periodic_z] CARBON_HD(Span s, Idx idx) -> float {
          const auto g = s.global_location(idx);
          if (g.x <= 0 || g.x >= nx_global - 1) {
            return 0.0F;
          }
          const float xp = in.neighbour(s, idx, carbon::int8_3d{1, 0, 0}, 0).value;
          const float xm = in.neighbour(s, idx, carbon::int8_3d{-1, 0, 0}, 0).value;
          const float zp = in.neighbour(s, idx, carbon::int8_3d{0, 0, 1}, 0).value;
          const float zm = in.neighbour(s, idx, carbon::int8_3d{0, 0, -1}, 0).value;
          const float old = old_p.read(s, idx, 0);
          const float next = 0.25F * (xp + xm + zp + zm);
          out.write(s, idx, 0, next);
          if (mirror_periodic_z) {
            const int64_t gz_max = s.global_extent().z - 1;
            const int ix = static_cast<int>(g.x);
            if (g.z == 0) {
              const Idx top{ix, 0, static_cast<int>(s.haloed_extent().z - 1)};
              out.write(s, top, 0, next);
            }
            if (g.z == gz_max) {
              const Idx bot{ix, 0, 0};
              out.write(s, bot, 0, next);
            }
          }
          const float d = next - old;
          return d * d;
        };
      },
      sumsq, carbon::reduce::Sum<float>{}, std::move(name), /*shared_mem=*/0U, reduce_opts);
}

auto
initialize_fields(DenseGrid& grid,
                  carbon::System& sys,
                  FieldT& u,
                  FieldT& u_next,
                  int nx,
                  int ny) -> void {
  const double p = pi();
  const int64_t nx64 = nx;
  const int64_t ny_m1 = static_cast<int64_t>(ny - 1);
  const auto set_host = [&](FieldT& field) {
    field.fill_owned_host([=](carbon::int64_3d g, int) -> float {
      if (g.x == 0 || g.x == nx64 - 1) {
        const double iy = static_cast<double>(g.z + 1);
        return static_cast<float>(std::sin(2.0 * p * iy / static_cast<double>(ny_m1)));
      }
      return 0.0F;
    });
  };
  set_host(u);
  set_host(u_next);
  carbon::Stream st = sys.stream();
  carbon::Transfer<float>(sys, u.cpu_buffers(), u.gpu_buffers())(st);
  carbon::Transfer<float>(sys, u_next.cpu_buffers(), u_next.gpu_buffers())(st);
  st.wait();
}

struct SolveResult {
  double seconds = 0.0;
  int iterations = 0;
  float final_l2 = 0.0F;
};

struct SolveRun {
  SolveResult stats;
  FieldT u;
  FieldT u_next;
  int iterations = 0;
};

[[nodiscard]] auto
final_solution_field(SolveRun& run) -> FieldT& {
  return (run.iterations % 2 == 1) ? run.u_next : run.u;
}

auto
run_solver(DenseGrid& grid,
           carbon::System& sys,
           carbon::OverlapMode overlap_mode,
           int niter,
           int nccheck,
           bool csv,
           bool enable_timing,
           int warmup_iters,
           bool fixed_iterations,
           carbon::reduce::ReduceOptions reduce_opts,
           NormSyncMode norm_sync) -> SolveRun {
  SolveRun run{};
  run.u = grid.newField<float, 1>(carbon::MemorySide::HostDevice, "u");
  run.u_next = grid.newField<float, 1>(carbon::MemorySide::HostDevice, "u_next");
  const int nx = static_cast<int>(grid.global_extent().x);
  const int ny = static_cast<int>(grid.global_extent().z) + 2;
  initialize_fields(grid, sys, run.u, run.u_next, nx, ny);

  {
    carbon::Kernel prime_u = run.u.make_halo_kernel("prime_u");
    carbon::Kernel prime_un = run.u_next.make_halo_kernel("prime_u_next");
    prime_u.launch(sys.stream());
    prime_un.launch(sys.stream());
    sys.stream().wait();
  }

  carbon::ReductionResult<float> sumsq(
      sys, norm_sync == NormSyncMode::Pipelined
               ? carbon::reduce::ReductionResultOptions{.slots = 2U}
               : carbon::reduce::ReductionResultOptions{});
  if (norm_sync == NormSyncMode::Pipelined) {
    reduce_opts.host_copy = carbon::reduce::ReduceHostCopy::Deferred;
  }
  const int64_t nx64 = grid.global_extent().x;
  const bool mirror_periodic = (grid.global_slab_count() == 1U);
  auto step_ab = build_fused_jacobi_step(grid, run.u, run.u_next, sumsq, nx64, mirror_periodic,
                                         reduce_opts, "step_ab");
  auto step_ba = build_fused_jacobi_step(grid, run.u_next, run.u, sumsq, nx64, mirror_periodic,
                                         reduce_opts, "step_ba");

  carbon::Skeleton sk_ab(sys, carbon::Skeleton::Options{overlap_mode});
  sk_ab.add(std::move(step_ab));
  sk_ab.compile();

  carbon::Skeleton sk_ba(sys, carbon::Skeleton::Options{overlap_mode});
  sk_ba.add(std::move(step_ba));
  sk_ba.compile();

  auto one_iter = [&](int iter) -> float {
    if ((iter % 2) == 0) {
      sk_ab.run();
    } else {
      sk_ba.run();
    }
    const bool need_norm =
        (iter % nccheck) == 0 || csv || (iter % 100) == 0;
    if (!need_norm) {
      return 0.0F;
    }
    if (norm_sync == NormSyncMode::Pipelined) {
      if (iter == 0) {
        return 0.0F;
      }
      return std::sqrt(sumsq.wait_value(carbon::DataView::Standard, /*lag=*/1U));
    }
    return std::sqrt(sumsq.wait_value());
  };

  for (int w = 0; w < warmup_iters; ++w) {
    (void)one_iter(w);
  }

  float l2 = 1.0F;
  const auto t0 = std::chrono::steady_clock::now();
  for (int iter = 0; iter < niter; ++iter) {
    if (!fixed_iterations && l2 <= kTol) {
      break;
    }
    l2 = one_iter(warmup_iters + iter);
    if (!csv && (iter % 100) == 0 && l2 > 0.0F) {
      std::printf("%5d, %0.6f\n", iter, static_cast<double>(l2));
    }
    run.iterations = iter + 1;
    run.stats.iterations = iter + 1;
    run.stats.final_l2 = l2;
  }
  const auto t1 = std::chrono::steady_clock::now();
  if (enable_timing) {
    run.stats.seconds = std::chrono::duration<double>(t1 - t0).count();
  }
  if (norm_sync == NormSyncMode::Pipelined) {
    run.stats.final_l2 = std::sqrt(sumsq.wait_value());
  }
  return run;
}

auto
download_reference_layout(const DenseGrid& grid,
                          carbon::System& sys,
                          const FieldT& u,
                          int nx,
                          int ny,
                          std::vector<float>& out) -> void {
  out.assign(static_cast<size_t>(nx) * static_cast<size_t>(ny), 0.0F);
  FieldT& mut_u = const_cast<FieldT&>(u);
  carbon::Stream st = sys.stream();
  carbon::Transfer<float>(sys, mut_u.gpu_buffers(), mut_u.cpu_buffers())(st);
  st.wait();

  const double p = pi();
  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx; ++ix) {
      if (ix == 0 || ix == nx - 1) {
        out[static_cast<size_t>(iy) * static_cast<size_t>(nx) + static_cast<size_t>(ix)] =
            static_cast<float>(std::sin(2.0 * p * static_cast<double>(iy) /
                                        static_cast<double>(ny - 1)));
      }
    }
  }

  for (size_t g = 0; g < grid.gpu_count(); ++g) {
    const Span& span = grid.spans()[g];
    const float* host = mut_u.cpu_buffers()[g].data();
    const carbon::int64_3d owned = span.logic_extent();
    for (int64_t iz = 0; iz < owned.z; ++iz) {
      for (int64_t ix = 0; ix < owned.x; ++ix) {
        const auto vi = span.set_and_validate(carbon::DataView::Standard, ix, 0, iz);
        if (!vi.valid) {
          continue;
        }
        const auto loc = span.global_location(vi.idx);
        const int iy = static_cast<int>(loc.z) + 1;
        const int64_t off = span.linear_offset(vi.idx, 0);
        out[static_cast<size_t>(iy) * static_cast<size_t>(nx) + static_cast<size_t>(loc.x)] =
            host[off];
      }
    }
    // Reference rows iy=0 and iy=ny-1 live in z halos (buffer rows 0 and haloed.z-1).
    for (int ix = 1; ix < nx - 1; ++ix) {
      const Idx bot{ix, 0, 0};
      const Idx top{ix, 0, static_cast<int>(span.haloed_extent().z - 1)};
      out[static_cast<size_t>(ix)] = host[span.linear_offset(bot, 0)];
      out[static_cast<size_t>(ny - 1) * static_cast<size_t>(nx) + static_cast<size_t>(ix)] =
          host[span.linear_offset(top, 0)];
    }
  }
}

auto
write_dump(const std::string& path,
           const std::vector<float>& data) -> bool {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    return false;
  }
  out.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(float)));
  return out.good();
}

auto
validate_interior(const std::vector<float>& got,
                  const std::vector<float>& ref,
                  int nx,
                  int ny) -> bool {
  bool ok = true;
  for (int iy = 1; ok && iy < ny - 1; ++iy) {
    for (int ix = 1; ix < nx - 1; ++ix) {
      const size_t off = static_cast<size_t>(iy) * static_cast<size_t>(nx) + static_cast<size_t>(ix);
      if (std::fabs(got[off] - ref[off]) > kTol) {
        std::fprintf(stderr,
                     "ERROR: mismatch at iy=%d ix=%d got=%f ref=%f\n",
                     iy, ix, static_cast<double>(got[off]), static_cast<double>(ref[off]));
        ok = false;
        break;
      }
    }
  }
  return ok;
}

auto
run_reference_single_gpu(int nx,
                         int ny,
                         int niter,
                         int nccheck,
                         int warmup_iters,
                         bool fixed_iterations,
                         std::vector<float>& ref_out) -> double {
  std::vector<carbon::detail::Gpu> one = carbon::get_all_devices();
  if (one.empty()) {
    return 0.0;
  }
  if (one.size() > 1U) {
    one.erase(one.begin() + 1, one.end());
  }
  carbon::System ref_sys(one);
  const carbon::int64_3d global{nx, 1, ny - 2};
  const carbon::LaunchShape block{64U, 1U, 8U};
  DenseGrid ref_grid(ref_sys, global, make_stencil(), carbon::BoundaryMode::PeriodicZ,
                     carbon::float32_3d{}, 1.0F, block);
  SolveRun ref_run =
      run_solver(ref_grid, ref_sys, carbon::OverlapMode::None, niter, nccheck, true, true,
                 warmup_iters, fixed_iterations, carbon::reduce::ReduceOptions{},
                 NormSyncMode::Blocking);
  download_reference_layout(ref_grid, ref_sys, final_solution_field(ref_run), nx, ny, ref_out);
  return ref_run.stats.seconds;
}

struct BenchContext {
  DenseGrid grid;
  carbon::System sys;
  int rank = 0;
  int world_size = 1;
};

[[nodiscard]] auto
make_context(const Args& args) -> BenchContext {
  BenchContext ctx{};
#ifdef CARBON_HAVE_MPI_NCCL
  auto distributed = std::make_shared<const carbon::Distributed>();
  ctx.world_size = distributed->world_size();
  ctx.rank = distributed->world_rank();
  if (ctx.world_size > 1) {
    ctx.sys = carbon::System(distributed);
  } else {
#endif
    std::vector<carbon::detail::Gpu> devices = carbon::get_all_devices();
    if (devices.empty()) {
      throw std::runtime_error("jacobi_nvidia: no CUDA devices visible");
    }
    if (args.n_gpus > 0 && static_cast<size_t>(args.n_gpus) < devices.size()) {
      devices.erase(devices.begin() + static_cast<std::ptrdiff_t>(args.n_gpus), devices.end());
    }
    ctx.sys = carbon::System(devices);
#ifdef CARBON_HAVE_MPI_NCCL
  }
#endif

  if (ctx.sys.gpu_group().size() == 0U) {
    throw std::runtime_error("jacobi_nvidia: System has zero GPUs");
  }

  const carbon::int64_3d global{args.nx, 1, args.ny - 2};
  const auto overlap = parse_overlap(args.overlap);
  if (!overlap) {
    throw std::runtime_error("jacobi_nvidia: unknown --overlap value");
  }
  const carbon::LaunchShape block = parse_block_shape(args.block);
  ctx.grid = DenseGrid(ctx.sys, global, make_stencil(), carbon::BoundaryMode::PeriodicZ,
                       carbon::float32_3d{}, 1.0F, block);
  (void)overlap;
  return ctx;
}

}  // namespace

int
main(int argc,
     char** argv) {
  carbon::init();
  const Args args = parse_args(argc, argv);

  if (args.nccheck != 1) {
    std::fprintf(stderr, "jacobi_nvidia: only -nccheck 1 is supported\n");
    return EXIT_FAILURE;
  }
  if (args.ny < 4) {
    std::fprintf(stderr, "jacobi_nvidia: -ny must be >= 4\n");
    return EXIT_FAILURE;
  }

  try {
    const auto overlap = parse_overlap(args.overlap);
    if (!overlap) {
      std::fprintf(stderr, "jacobi_nvidia: unknown --overlap '%s'\n", args.overlap.c_str());
      return EXIT_FAILURE;
    }
    const auto reduce_opts = make_reduce_options(args.reduce_finalize, args.reduce_block);
    if (!reduce_opts) {
      std::fprintf(stderr,
                   "jacobi_nvidia: invalid --reduce-finalize '%s' (device|atomic) and/or "
                   "--reduce-block (256, 32x32, …)\n",
                   args.reduce_finalize.c_str());
      return EXIT_FAILURE;
    }
    const auto norm_sync = parse_norm_sync(args.norm_sync);
    if (!norm_sync) {
      std::fprintf(stderr,
                   "jacobi_nvidia: unknown --norm-sync '%s' (use blocking or pipelined)\n",
                   args.norm_sync.c_str());
      return EXIT_FAILURE;
    }

    BenchContext ctx = make_context(args);
    const size_t num_devices_local = ctx.sys.gpu_group().size();
    const size_t num_devices = (ctx.world_size > 1)
                                   ? static_cast<size_t>(ctx.world_size)
                                   : num_devices_local;

    if (ctx.rank == 0 && !args.csv) {
      std::printf(
          "Jacobi (carbon): %d iterations on %d x %d mesh, norm check every %d, gpus=%zu "
          "overlap=%s block=%s reduce-finalize=%s reduce-block=%s norm-sync=%s\n",
          args.niter, args.ny, args.nx, args.nccheck, num_devices, args.overlap.c_str(),
          args.block.c_str(), args.reduce_finalize.c_str(),
          args.reduce_block.empty() ? "256" : args.reduce_block.c_str(), args.norm_sync.c_str());
    }

    double runtime_serial = 0.0;
    std::vector<float> ref;
    bool result_correct = true;
    if (!args.skip_validation && ctx.rank == 0 && ctx.world_size == 1) {
      runtime_serial = run_reference_single_gpu(args.nx, args.ny, args.niter, args.nccheck,
                                                args.warmup, args.fixed_iters, ref);
    }

#ifdef CARBON_HAVE_MPI_NCCL
    if (const auto* dist = ctx.sys.distributed()) {
      dist->barrier();
    }
#endif

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(args.repeats));
    SolveRun last_run{};
    for (int r = 0; r < args.repeats; ++r) {
      last_run = run_solver(ctx.grid, ctx.sys, *overlap, args.niter, args.nccheck, args.csv, true,
                            args.warmup, args.fixed_iters, *reduce_opts, *norm_sync);
      samples.push_back(last_run.stats.seconds);
    }
    const carbon::bench::TimingStats stats = carbon::bench::compute_timing_stats(std::move(samples));

    if (ctx.rank == 0 && ctx.world_size == 1) {
      if (!args.dump_path.empty()) {
        std::vector<float> got;
        download_reference_layout(ctx.grid, ctx.sys, final_solution_field(last_run), args.nx,
                                  args.ny, got);
        if (!write_dump(args.dump_path, got)) {
          std::fprintf(stderr, "jacobi_nvidia: failed to write dump %s\n", args.dump_path.c_str());
          return EXIT_FAILURE;
        }
      }
      if (!args.skip_validation) {
        std::vector<float> got;
        download_reference_layout(ctx.grid, ctx.sys, final_solution_field(last_run), args.nx,
                                  args.ny, got);
        result_correct = validate_interior(got, ref, args.nx, args.ny);
        if (!result_correct) {
          return EXIT_FAILURE;
        }
      }
    }

    if (ctx.rank != 0) {
      return EXIT_SUCCESS;
    }

    if (!result_correct) {
      return EXIT_FAILURE;
    }

    if (args.csv) {
      if (num_devices <= 1U) {
        std::printf("carbon_jacobi, %d, %d, %d, %d, %f\n", args.nx, args.ny, args.niter,
                    args.nccheck, stats.median_seconds);
      } else {
        std::printf("carbon_jacobi, %d, %d, %d, %d, %zu, 1, %f, %f\n", args.nx, args.ny, args.niter,
                    args.nccheck, num_devices, stats.median_seconds, runtime_serial);
      }
    } else {
      if (num_devices <= 1U) {
        std::printf("%dx%d: 1 GPU (carbon): %8.4f s\n", args.ny, args.nx, stats.median_seconds);
      } else {
        const double speedup = runtime_serial / stats.median_seconds;
        const double efficiency = speedup / static_cast<double>(num_devices) * 100.0;
        std::printf("Num GPUs: %zu.\n", num_devices);
        std::printf(
            "%dx%d: 1 GPU: %8.4f s, %zu GPUs (carbon): %8.4f s, speedup: %8.2f, efficiency: "
            "%8.2f\n",
            args.ny, args.nx, runtime_serial, num_devices, stats.median_seconds, speedup,
            efficiency);
      }
    }

    if (!args.report_path.empty()) {
      std::ostringstream cfg;
      cfg << "{\n"
          << "    \"nx\": " << args.nx << ",\n"
          << "    \"ny\": " << args.ny << ",\n"
          << "    \"niter\": " << args.niter << ",\n"
          << "    \"overlap\": \"" << args.overlap << "\",\n"
          << "    \"block\": \"" << args.block << "\"\n"
          << "  }";
      if (!carbon::bench::write_json_report(args.report_path, "jacobi_nvidia", cfg.str(), ctx.sys,
                                            stats, 0.0, "seconds")) {
        return EXIT_FAILURE;
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "jacobi_nvidia: %s\n", e.what());
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
