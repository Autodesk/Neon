// Unified Jacobi smoothing example across every carbon grid + every transport.
//
// One source file, one Skeleton-only main loop, and (most importantly) one
// per-cell device lambda that compiles and runs on:
//
//   - Grids:     carbon::Dense   (dense Cartesian box)
//                carbon::DenseDisg (disaggregated dense slabs, region-partitioned)
//                carbon::Block<DefaultStaticBlock> (sparse 8^3 brick grid)
//   - Backends:  CUDA  (default, nvcc)
//                SYCL  (AdaptiveCpp -DCARBON_BACKEND=Sycl)
//   - Transports: single-GPU, multi-GPU same node, NCCL/RCCL/oneCCL cross-rank
//                 (the latter is enabled by -DCARBON_ENABLE_MPI=ON + mpiexec)
//
// What is unified
// ---------------
// The per-cell update lambda - the only part the user actually writes - lives
// inside @c build_jacobi_step . It reads the six face neighbours through
// @c Partition::neighbour and branches on the @c NeighbourValue::valid flag:
//
//   - @c !zm.valid               -> u_out = +1.0  (hot south face)
//   - @c !zp.valid               -> u_out = -1.0  (cold north face)
//   - any other invalid face     -> u_out =  0.0  (insulated sides)
//   - all six neighbours valid   -> 1/6 sum (interior Jacobi step)
//
// All three grids report @c valid=false in exactly the right places:
//   - Dense / DenseDisg under @c BoundaryMode::Open : neighbours off the global
//     Cartesian box. The hot / cold / insulated faces fall out automatically
//     because the Dirichlet face cells are the ones whose @c z- / @c z+ /
//     @c x?/@c y? neighbour is out-of-domain.
//   - Block under the default predicate (a sphere inscribed in a 64-cube):
//     neighbours into an inactive voxel (out-of-domain, dropped brick, or
//     bitmask-clear). The sphere's poles become the hot / cold faces and the
//     shell becomes the insulated boundary.
//
// What Skeleton buys you
// ----------------------
// The iteration loop reduces to @c sk.run() inside a for-loop. We never write
// a halo call by hand: the loader records @c read_stencil(fin) tokens at
// @c make_kernel time, and @c Skeleton::compile auto-inserts the matching
// halo kernel (@c HaloKernelImpl for Dense, @c HaloDisgKernelImpl for
// DenseDisg, @c HaloBlockKernelImpl for Block) wherever the dependency
// analysis says "stencil read of a field that was just written without
// re-exchanging halos". The same pipeline definition therefore exchanges
//
//   - nothing on a single-GPU run (halo is a no-op),
//   - cross-GPU intra-process via @c Transfer<T> on multi-GPU same-node,
//   - cross-rank via NCCL / RCCL / oneCCL when @c sys.distributed() is set.
//
// Overlap mode
// ------------
// The @c --overlap CLI flag picks the @c Skeleton::OverlapMode applied to the
// main ping-pong pipeline:
//
//   - @c none            : no OCC rewrite. Halos run serially before each
//                          stencil step. Bit-for-bit equivalent to the
//                          single-stream reference. (Default.)
//   - @c standard        : carbon::OverlappedStencil-shaped split. For each
//                          stencil node with a preceding halo, the kernel is
//                          split into Internal (concurrent with the halo on
//                          a separate stream) + Boundary (fenced behind the
//                          halo). Recovers halo latency on multi-GPU /
//                          cross-rank runs.
//   - @c extended        : Standard plus one-step propagation: the kernel
//                          immediately downstream of a Boundary launch is
//                          also split, so the overlap chain extends one
//                          stencil deeper.
//   - @c two-way-extended: Extended plus propagation in the other direction
//                          (the kernel immediately upstream of a Halo's
//                          stencil consumer is also split).
//
// All three OCC modes produce results that match @c none to within the
// documented multi-stream tolerance ( ~5e-3 absolute, 1e-4 RMS over 20 Jacobi
// iters; see @c tests/test_overlapped_stencil.cu ); single-stream Jacobi
// stays bit-for-bit identical because every step has the previous step as a
// direct predecessor.
//
// Build + run
// -----------
//   ./carbon_example_jacobi_unified                          # default: dense, 200 iters, overlap=none
//   ./carbon_example_jacobi_unified --grid disg --overlap standard
//   ./carbon_example_jacobi_unified --grid block --iters 400
//   ./carbon_example_jacobi_unified --grid dense --iters 400 --size 48 --overlap two-way-extended
//   mpiexec -n 2 ./carbon_example_jacobi_unified --grid block --overlap standard
//
// Knobs: @c --grid dense|disg|block , @c --iters N , @c --size N (cube side),
// @c --overlap none|standard|extended|two-way-extended ,
// @c --dot (dump the inferred Skeleton DAG as GraphViz DOT on stdout).
//
// Block + MPI cross-rank stencil parity (sphere predicate, six-direction
// neighbour reads, Jacobi sum|u|^2) is covered by
// tests/test_block_dist_stencil_parity.cu under mpiexec -n 2.

#include <carbon/block.h>
#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/dense_disg.h>
#include <carbon/detail/backend/select.h>
#include <carbon/devices.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/reduce.h>
#include <carbon/skeleton.h>
#include <carbon/stencil.h>
#include <carbon/system.h>

#ifdef CARBON_HAVE_MPI_NCCL
#include <carbon/distributed.h>
#endif

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr int kDefaultIters = 200;
constexpr int kDefaultExtent = 64;

using SB = carbon::detail::block::DefaultStaticBlock;

// Sphere inscribed in the cube, radius = 0.42 * N. Used as the @c Block
// predicate so the sparse layout retains a recognisable subset of bricks.
[[nodiscard]] auto
sphere_predicate(int64_t domain_n,
                 const carbon::int64_3d& gc) -> bool {
  const double cx = (static_cast<double>(domain_n) - 1.0) * 0.5;
  const double cy = (static_cast<double>(domain_n) - 1.0) * 0.5;
  const double cz = (static_cast<double>(domain_n) - 1.0) * 0.5;
  const double r = static_cast<double>(domain_n) * 0.42;
  const double dx = static_cast<double>(gc.x) - cx;
  const double dy = static_cast<double>(gc.y) - cy;
  const double dz = static_cast<double>(gc.z) - cz;
  return (dx * dx + dy * dy + dz * dz) <= (r * r);
}

// The one per-cell Jacobi body, parameterised on `Grid`. All three grid
// families share the same `Span` / `Idx` / `Partition::neighbour` /
// `Partition::read` / `Partition::write` surface, so this lambda compiles
// unchanged for Dense, DenseDisg, and Block (and for both CUDA and SYCL,
// since `CARBON_HOST_DEVICE` is `__host__ __device__` under CUDA and a no-op
// under SYCL where `parallel_for` already treats the lambda as device code).
template <typename Grid>
[[nodiscard]] auto
build_jacobi_step(const Grid& grid,
                  typename Grid::template Field<float,
                                                1>& fin,
                  typename Grid::template Field<float,
                                                1>& fout,
                  std::string name) -> carbon::Kernel {
  using Span = typename Grid::Span;
  using Idx = typename Grid::Idx;
  return carbon::Kernel::compute(
      grid,
      [&fin, &fout](carbon::KernelLoader& loader) {
        const auto in = loader.read_stencil(fin);
        const auto out = loader.write(fout);
        return [in, out] CARBON_HOST_DEVICE(Span s, Idx idx) -> void {
          const auto xp = in.neighbour(s, idx, carbon::int8_3d{1, 0, 0}, 0);
          const auto xm = in.neighbour(s, idx, carbon::int8_3d{-1, 0, 0}, 0);
          const auto yp = in.neighbour(s, idx, carbon::int8_3d{0, 1, 0}, 0);
          const auto ym = in.neighbour(s, idx, carbon::int8_3d{0, -1, 0}, 0);
          const auto zp = in.neighbour(s, idx, carbon::int8_3d{0, 0, 1}, 0);
          const auto zm = in.neighbour(s, idx, carbon::int8_3d{0, 0, -1}, 0);
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

// Zero-initialise every owned cell of `field`. We could memset the device
// buffer in a backend-specific way, but doing it via a one-liner kernel keeps
// the example backend-neutral and pulls the init under the same Skeleton
// machinery as the main loop. Block grids only invoke the lambda for active
// voxels (the stage-1 launch's set_and_validate filters inactive bricks);
// dense grids invoke it for every owned cell.
template <typename Grid>
[[nodiscard]] auto
build_zero_init(const Grid& grid,
                typename Grid::template Field<float,
                                              1>& field,
                std::string name) -> carbon::Kernel {
  using Span = typename Grid::Span;
  using Idx = typename Grid::Idx;
  return carbon::Kernel::compute(
      grid,
      [&field](carbon::KernelLoader& loader) {
        const auto out = loader.write(field);
        return [out] CARBON_HOST_DEVICE(Span s, Idx idx) -> void {
          out.write(s, idx, 0, 0.0F);
        };
      },
      std::move(name));
}

// Sum-of-squares reduction. `Kernel::reduce` already dispatches on
// `Grid::BlockPolicy` internally, so this same call routes through
// `ReduceKernelImpl` for dense grids and `BlockReduceKernelImpl` for block
// grids - and through stage-3 NCCL/RCCL allreduce when `sys.distributed()`
// is non-null. Result lands in `dst.host_value()` after a `dst.sync(stream)`.
template <typename Grid>
[[nodiscard]] auto
build_sumsq(const Grid& grid,
            typename Grid::template Field<float,
                                          1>& field,
            carbon::ReductionResult<float>& dst,
            std::string name) -> carbon::Kernel {
  using Span = typename Grid::Span;
  using Idx = typename Grid::Idx;
  return carbon::Kernel::reduce<float>(
      grid,
      [&field](carbon::KernelLoader& loader) {
        const auto in = loader.read(field);
        return [in] CARBON_HOST_DEVICE(Span s, Idx idx) -> float {
          const float v = in.read(s, idx, 0);
          return v * v;
        };
      },
      dst, carbon::reduce::Sum<float>{}, std::move(name), /*shared_mem=*/0U);
}

[[nodiscard]] auto
overlap_mode_name(carbon::OverlapMode m) -> const char* {
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

template <typename Grid>
auto
run_jacobi(Grid& grid,
           carbon::System& sys,
           int n_iters,
           std::string_view grid_tag,
           int rank,
           bool dump_dot,
           carbon::OverlapMode overlap_mode) -> int {
  // Round up to even so the ping-pong always lands on `u` (step_ab then step_ba).
  const int sweeps = (n_iters + 1) / 2 * 2;
  const int cycles = sweeps / 2;

  auto u = grid.template newField<float, 1>(carbon::MemorySide::Device, "u");
  auto u_next = grid.template newField<float, 1>(carbon::MemorySide::Device, "u_next");

  // One-shot init skeleton: zero both ping-pong slots before the loop. The two
  // zero kernels write disjoint fields, so the dependency analyser puts them
  // on parallel streams when more than one is available.
  carbon::Skeleton init(sys);
  init.add(build_zero_init(grid, u, "zero_u"));
  init.add(build_zero_init(grid, u_next, "zero_u_next"));
  init.run();
  init.wait();

  // Main pipeline: two ping-pong stencil steps. Halos are auto-inserted by
  // Skeleton::compile via `loader.read_stencil(fin)` - no manual halo plumbing.
  // When overlap_mode != None the same compile pass also rewrites each
  // stencil node into the Internal + Boundary split that recovers halo
  // latency on multi-GPU / cross-rank runs.
  carbon::Skeleton main_pipe(sys, carbon::Skeleton::Options{overlap_mode});
  main_pipe.add(build_jacobi_step(grid, u, u_next, "step_ab"));
  main_pipe.add(build_jacobi_step(grid, u_next, u, "step_ba"));
  main_pipe.compile();
  if (rank == 0) {
    CARBON_LOG_INFO(
        "jacobi_unified[{}]: compiled - user_kernels={} effective_kernels={} overlap_mode={} "
        "(auto-halos included)",
        grid_tag, main_pipe.size(), main_pipe.effective_kernels().size(),
        overlap_mode_name(overlap_mode));
    if (dump_dot) {
      // Pipe to `dot -Tsvg -o jacobi_unified.svg` to render. The DOT export is
      // what populates docs/skeleton/jacobi_skeleton.{dot,svg,png}.
      CARBON_LOG_INFO("jacobi_unified[{}]: --- DAG (GraphViz DOT) ---\n{}", grid_tag,
                      main_pipe.export_dot("jacobi_unified"));
    }
  }

  for (int c = 0; c < cycles; ++c) {
    main_pipe.run();
  }
  main_pipe.wait();

  // L2 norm via a one-shot reduce skeleton over the final `u`. Cross-rank
  // allreduce kicks in automatically when sys.distributed() != nullptr.
  carbon::ReductionResult<float> sumsq_result(sys);
  carbon::Skeleton norm_pipe(sys);
  norm_pipe.add(build_sumsq(grid, u, sumsq_result, "l2_sumsq"));
  norm_pipe.run();
  norm_pipe.wait();
  sumsq_result.sync(sys.stream());
  sys.stream().wait();

  if (rank == 0) {
    const double sumsq = static_cast<double>(sumsq_result.host_value());
    const double l2 = std::sqrt(sumsq);
    CARBON_LOG_INFO("jacobi_unified[{}]: done after {} sweeps - L2(u) = {:.6e} (sum|u|^2 = {:.6e})",
                    grid_tag, sweeps, l2, sumsq);
  }
  return EXIT_SUCCESS;
}

// Tiny argv parser - no shell, no exec, no user-controlled paths or commands.
[[nodiscard]] auto
arg_value(int argc,
          char** argv,
          std::string_view flag) -> std::string {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string_view{argv[i]} == flag) {
      return std::string{argv[i + 1]};
    }
  }
  return {};
}

[[nodiscard]] auto
arg_int(int argc,
        char** argv,
        std::string_view flag,
        int fallback) -> int {
  const std::string v = arg_value(argc, argv, flag);
  if (v.empty()) {
    return fallback;
  }
  try {
    return std::stoi(v);
  } catch (...) {
    return fallback;
  }
}

// Maps the @c --overlap CLI value to @c carbon::OverlapMode . Returns
// @c std::nullopt when the user passed a string we do not recognise, so the
// caller can emit a clear diagnostic instead of silently defaulting.
[[nodiscard]] auto
parse_overlap_mode(std::string_view s) -> std::optional<carbon::OverlapMode> {
  if (s == "none" || s.empty()) {
    return carbon::OverlapMode::None;
  }
  if (s == "standard") {
    return carbon::OverlapMode::Standard;
  }
  if (s == "extended") {
    return carbon::OverlapMode::Extended;
  }
  if (s == "two-way-extended" || s == "two_way_extended" || s == "twoway") {
    return carbon::OverlapMode::TwoWayExtended;
  }
  return std::nullopt;
}

}  // namespace

auto
main(int argc,
     char** argv) -> int {
  carbon::init();

  const std::string grid_kind_raw = arg_value(argc, argv, "--grid");
  const std::string grid_kind = grid_kind_raw.empty() ? std::string{"dense"} : grid_kind_raw;
  const int n_iters = arg_int(argc, argv, "--iters", kDefaultIters);
  const int extent = arg_int(argc, argv, "--size", kDefaultExtent);
  const std::string overlap_raw = arg_value(argc, argv, "--overlap");
  const std::optional<carbon::OverlapMode> overlap_opt = parse_overlap_mode(overlap_raw);
  if (!overlap_opt) {
    CARBON_LOG_ERROR(
        "jacobi_unified: unknown --overlap '{}' (expected none|standard|extended|two-way-extended)",
        overlap_raw);
    return EXIT_FAILURE;
  }
  const carbon::OverlapMode overlap_mode = *overlap_opt;
  bool dump_dot = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string_view{argv[i]} == "--dot") {
      dump_dot = true;
    }
  }

#ifdef CARBON_HAVE_MPI_NCCL
  auto distributed = std::make_shared<const carbon::Distributed>();
  const int world_size = distributed->world_size();
  const int world_rank = distributed->world_rank();
  carbon::System sys =
      (world_size > 1) ? carbon::System(distributed) : carbon::System(carbon::get_all_devices());
  const bool is_distributed = (world_size > 1);
#else
  carbon::System sys(carbon::get_all_devices());
  const int world_size = 1;
  const int world_rank = 0;
  const bool is_distributed = false;
#endif

  if (sys.gpu_group().size() == 0U) {
    CARBON_LOG_ERROR("jacobi_unified: no devices visible");
    return EXIT_FAILURE;
  }

  if (world_rank == 0) {
    CARBON_LOG_INFO(
        "jacobi_unified: grid={} size={} iters={} overlap={} backend={} gpus_local={} "
        "world_size={} distributed={}",
        grid_kind, extent, n_iters, overlap_mode_name(overlap_mode),
#if CARBON_BACKEND_IS_SYCL
        "SYCL",
#else
        "CUDA",
#endif
        sys.gpu_group().size(), world_size, is_distributed);
  }

  const carbon::int64_3d global{extent, extent, extent};

  try {
    if (grid_kind == "dense") {
      // 7-point face stencil -> halo_z=1.
      carbon::Stencil stencil(
          std::vector<carbon::int32_3d>{
              {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
          /*remove_center=*/true);
      carbon::Dense dense(sys, global, stencil);
      return run_jacobi(dense, sys, n_iters, "dense", world_rank, dump_dot, overlap_mode);
    }
    if (grid_kind == "disg") {
      carbon::DenseDisg disg(sys, global, /*halo_z=*/1);
      return run_jacobi(disg, sys, n_iters, "disg", world_rank, dump_dot, overlap_mode);
    }
    if (grid_kind == "block") {
      const auto predicate = [extent](const carbon::int64_3d& gc) -> bool {
        return sphere_predicate(extent, gc);
      };
      using BlockGrid = carbon::Block<SB>;
      BlockGrid block(sys, global, predicate, "jacobi_unified_block");
      return run_jacobi(block, sys, n_iters, "block", world_rank, dump_dot, overlap_mode);
    }
    CARBON_LOG_ERROR("jacobi_unified: unknown grid '{}' (expected dense|disg|block)", grid_kind);
    return EXIT_FAILURE;
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("jacobi_unified: failed: {}", e.what());
    return EXIT_FAILURE;
  }
}
