#pragma once

/**
 * @file detail/launch_traits.h
 *
 * Per-grid-family launch trait — hoists the variation between
 * @c detail::dense::launch_dense_grid , @c detail::dense_disg::launch_disg_grid ,
 * and @c detail::block::launch_block_grid behind a single uniform interface so
 * the @c KernelImpl tree can converge on one device-launch shape. Resolves
 * P1.α from @c docs/audit_2026_05_followup.md : Block + Dense + DenseDisg
 * compute-path impls share @c DeviceKernelImpl ; the trait absorbs the per-
 * family launch-helper signature differences.
 *
 * Surface (per specialisation of @c DeviceLaunchTraits<Grid> ):
 *
 * - @c preflight(grid, view) — family-specific sanity checks against the
 *   per-GPU launch grid (Dense / DenseDisg expose @c gpu_grid(view) /
 *   @c gpu_block(view) and must match @c gpu_count() ; Block derives its
 *   dims internally from @c SBlock::Mem{X,Y,Z} + @c span.iter_block_count(view)
 *   and has nothing to size-check). Returns @c gpu_count() either way so the
 *   kernel impl gets it in one call.
 * - @c cuda_launch(grid, i, shared_mem, cuda_stream, view, fn) — runs the
 *   family's launch helper for one GPU. Returns @c cudaError_t .
 * - @c sycl_launch(grid, i, shared_mem, queue, view, fn) — same on SYCL.
 *   Returns @c void ; the helper repackages any @c sycl::exception into
 *   @c std::runtime_error before returning.
 *
 * Concepts: the primary template covers any grid that exposes the dense /
 * disg launch contract ( @c gpu_grid / @c gpu_block accessors). Block
 * specialises because @c block::launch_block_grid has a different parameter
 * shape (no per-GPU grid / block dims passed by caller). The trait dispatch
 * is purely compile-time; no virtual or runtime branch.
 */

#include <carbon/data_view.h>
#include <carbon/detail/backend/select.h>
#include <carbon/detail/block/idx.h>
#include <carbon/detail/block/launch.h>
#include <carbon/detail/block/span.h>
#include <carbon/detail/dense/launch.h>
#include <carbon/detail/dense/reduce_kernel.h>
#include <carbon/detail/dense_disg/launch.h>
#include <carbon/detail/dense_disg/reduce_kernel.h>
#include <carbon/detail/gpu.h>
#include <carbon/integer3d.h>
#include <carbon/logging.h>

#if CARBON_BACKEND_IS_CUDALIKE
#include <carbon/detail/backend/cudalike/block_reduce_kernel.h>
#if CARBON_CUDALIKE_DEVICE_TU
#include <carbon/detail/backend/cudalike/dense_reduce_launch.h>
#endif
#elif CARBON_BACKEND_IS_SYCL
#include <carbon/detail/backend/sycl/block_reduce_kernel.h>
#endif

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

// Unconditional include: the existing `*_host_kernel_impl.h` files pulled
// `<omp.h>` in directly too. nvcc-compiled .cu TUs do not always define
// `_OPENMP` at the trait header parse time, but `<omp.h>` is still available
// (and its `#pragma omp` directives are no-ops if OpenMP support is absent at
// the active compiler stage).
#include <omp.h>

// Provides the cuda-like runtime types (gpu_stream_t / gpu_error_t / ...) and, under a
// cuda-like device TU, the <cuda_runtime.h> / <hip/hip_runtime.h> header that defines
// dim3. Expands to nothing under the SYCL backend.
#include <carbon/detail/backend/cuda_like.h>

namespace carbon {

template <typename SBlock>
class Block;

namespace detail {

/**
 * Primary template: covers Dense + DenseDisg. Both expose
 * @c gpu_grid(view) / @c gpu_block(view) and the only difference is which
 * launch helper runs under the hood — picked by the @c DisgPolicy probe.
 */
template <typename Grid>
struct DeviceLaunchTraits {
  /**
   * @return @c grid.gpu_count() . Throws on per-GPU launch-grid size drift.
   */
  static auto
  preflight(const Grid& grid,
            ::carbon::DataView view) -> size_t {
    const size_t n = grid.gpu_count();
    const auto& gpu_grid = grid.gpu_grid(view);
    const auto& gpu_block = grid.gpu_block(view);
    if (gpu_grid.size() != n || gpu_block.size() != n) {
      log::throw_with_log<std::logic_error>(
          "Kernel::launch: gpu_grid size {} and gpu_block size {} must equal gpu_count() {}",
          gpu_grid.size(), gpu_block.size(), n);
    }
    return n;
  }

#if CARBON_BACKEND_IS_CUDALIKE && CARBON_CUDALIKE_DEVICE_TU
  template <typename F>
  static auto
  cuda_launch(const Grid& grid,
              size_t i,
              size_t shared_mem,
              gpu_stream_t stream,
              ::carbon::DataView view,
              F& fn) -> gpu_error_t {
    const auto& gpu_grid = grid.gpu_grid(view);
    const auto& gpu_block = grid.gpu_block(view);
    if constexpr (requires { typename Grid::DisgPolicy; }) {
      return ::carbon::detail::dense_disg::launch_disg_grid(grid, gpu_grid[i], gpu_block[i], i,
                                                            shared_mem, stream, view, fn);
    } else {
      return ::carbon::detail::dense::launch_dense_grid(grid, gpu_grid[i], gpu_block[i], i,
                                                        shared_mem, stream, view, fn);
    }
  }
#endif

#if CARBON_BACKEND_IS_SYCL
  template <typename Queue,
            typename F>
  static auto
  sycl_launch(const Grid& grid,
              size_t i,
              size_t shared_mem,
              Queue& q,
              ::carbon::DataView view,
              F& fn) -> void {
    const auto& gpu_grid = grid.gpu_grid(view);
    const auto& gpu_block = grid.gpu_block(view);
    if constexpr (requires { typename Grid::DisgPolicy; }) {
      ::carbon::detail::dense_disg::launch_disg_grid(grid, gpu_grid[i], gpu_block[i], i, shared_mem,
                                                     q, view, fn);
    } else {
      ::carbon::detail::dense::launch_dense_grid(grid, gpu_grid[i], gpu_block[i], i, shared_mem, q,
                                                 view, fn);
    }
  }
#endif
};

/**
 * Specialisation for @c Block . Brick iteration: the launch helper takes no
 * per-GPU grid / block dims (it derives them from @c SBlock::Mem{X,Y,Z}
 * and @c span.iter_block_count(view) ) and there is no caller-side launch-
 * grid array to size-check. The @c BlockPolicy template argument is the
 * @c SBlock policy itself, threaded through the helper as a template arg.
 */
template <typename SBlock>
struct DeviceLaunchTraits<::carbon::Block<SBlock>> {
  using Grid = ::carbon::Block<SBlock>;

  static auto
  preflight(const Grid& grid,
            ::carbon::DataView /*view*/) -> size_t {
    return grid.gpu_count();
  }

#if CARBON_BACKEND_IS_CUDALIKE && CARBON_CUDALIKE_DEVICE_TU
  template <typename F>
  static auto
  cuda_launch(const Grid& grid,
              size_t i,
              size_t shared_mem,
              gpu_stream_t stream,
              ::carbon::DataView view,
              F& fn) -> gpu_error_t {
    return ::carbon::detail::block::launch_block_grid<SBlock>(grid, i, shared_mem, stream, view,
                                                              fn);
  }
#endif

#if CARBON_BACKEND_IS_SYCL
  template <typename Queue,
            typename F>
  static auto
  sycl_launch(const Grid& grid,
              size_t i,
              size_t shared_mem,
              Queue& q,
              ::carbon::DataView view,
              F& fn) -> void {
    ::carbon::detail::block::launch_block_grid<SBlock>(grid, i, shared_mem, q, view, fn);
  }
#endif
};

/**
 * Compile-time probe: does @c Grid use the dense-style cell-iteration model
 * (Dense / DenseDisg) where @c Span exposes @c iter_extent / @c logic_extent
 * / @c xy_slab_elements , or the brick-iteration model (Block) where the
 * Span talks @c iter_block_count(view) instead? The compute-path concept
 * anchors in @c DeviceKernelImpl 's ctor are gated on this probe:
 * @c if constexpr (grid_has_dense_iteration_v<Grid>) selects
 * @c carbon::detail::DenseLikeSpanConcept , the @c else branch selects
 * @c carbon::detail::BlockLikeSpanConcept ; the shared
 * @c carbon::detail::PartitionConcept fires on both branches with
 * @c Grid::template Partition<int, 1> as the representative
 * @c (T, Components) . That way a fourth grid family that ships a mis-shaped
 * @c Span @b or @c Partition lands on a clean concept diagnostic at the
 * kernel-impl ctor instead of a deep launch-path instantiation stack — see
 * @c include/carbon/detail/grid_concepts.h for the full contract narrative.
 */
template <typename Grid>
inline constexpr bool grid_has_dense_iteration_v = !requires {
  typename Grid::BlockPolicy;
};

/**
 * Per-grid-family @b host (OpenMP) launch trait. Twin of @c DeviceLaunchTraits
 * for the @c launch_on_host path: surfaces the same family-specific variation
 * (which span vector to walk, what the per-slab iteration shape looks like)
 * behind a uniform interface so the host @c KernelImpl tree can converge on
 * one impl per kernel kind.
 *
 * Surface (per specialisation):
 *
 * - @c spans_for_host(grid) — Dense / DenseDisg use @c grid.spans() (device-
 *   pointer spans dereference fine on host because the dense @c Span is pure
 *   value data — no device pointers); Block uses @c grid.host_spans() because
 *   the device @c Span's @c bitmask() / @c connectivity() pointers are not
 *   reachable from a host thread.
 * - @c iterate_slab(span, view, host_fn) — runs the per-slab OpenMP loop.
 *   Dense / DenseDisg : 3-deep @c collapse(3) over @c iter_extent(view) with
 *   the @c ValidatedIdx-returning @c set_and_validate . Block : outer
 *   sequential loop over @c iter_block_count(view) bricks plus inner
 *   @c collapse(3) over @c MemX*MemY*MemZ voxels with the @c bool-returning
 *   @c set_and_validate(b, ix, iy, iz, idx) .
 *
 * Stream-side machinery (@c stream.wait() , token discovery, etc.) stays in
 * the kernel impl — the trait only owns the iteration shape itself.
 */
template <typename Grid>
struct HostLaunchTraits {
  static auto
  spans_for_host(const Grid& grid) -> const auto& {
    return grid.spans();
  }

  template <typename Span,
            typename HostFn>
  static auto
  iterate_slab(const Span& span,
               ::carbon::DataView view,
               HostFn& host_fn) -> void {
    const ::carbon::int64_3d ext = span.iter_extent(view);
    if (ext.x <= 0 || ext.y <= 0 || ext.z <= 0) {
      return;
    }
#pragma omp parallel for collapse(3) default(shared)
    for (int64_t iz = 0; iz < ext.z; ++iz) {
      for (int64_t iy = 0; iy < ext.y; ++iy) {
        for (int64_t ix = 0; ix < ext.x; ++ix) {
          const auto vi = span.set_and_validate(view, ix, iy, iz);
          if (vi.valid) {
            host_fn(span, vi.idx);
          }
        }
      }
    }
  }

  /**
   * Per-slab host reduction. Returns the slab-local accumulator obtained by
   * combining per-thread partials via @p op . Empty domain returns
   * @c op.identity() . Twin of @c iterate_slab but for the reduce host path:
   * the loading lambda is expected to return a value the operator folds.
   */
  template <typename Span,
            typename HostFn,
            typename T,
            typename Op>
  static auto
  iterate_slab_reduce(const Span& span,
                      ::carbon::DataView view,
                      HostFn& host_fn,
                      Op& op) -> T {
    const ::carbon::int64_3d ext = span.iter_extent(view);
    if (ext.volume() <= 0) {
      return op.identity();
    }
    const int max_threads = ::omp_get_max_threads();
    std::vector<T> partials(static_cast<size_t>(max_threads), op.identity());
#pragma omp parallel default(shared)
    {
      const int tid = ::omp_get_thread_num();
      T local = op.identity();
#pragma omp for collapse(3) nowait
      for (int64_t iz = 0; iz < ext.z; ++iz) {
        for (int64_t iy = 0; iy < ext.y; ++iy) {
          for (int64_t ix = 0; ix < ext.x; ++ix) {
            const auto vi = span.set_and_validate(view, ix, iy, iz);
            if (vi.valid) {
              local = op(local, host_fn(span, vi.idx));
            }
          }
        }
      }
      partials[static_cast<size_t>(tid)] = local;
    }
    T slab_acc = op.identity();
    for (const T& p : partials) {
      slab_acc = op(slab_acc, p);
    }
    return slab_acc;
  }
};

/**
 * Specialisation for @c Block . Iterates per brick, with the brick walk
 * sequential (so per-thread bookkeeping stays consistent across brick
 * boundaries — same contract @c BlockHostKernelImpl had pre-collapse) and
 * the inner @c MemX*MemY*MemZ voxel walk parallel under @c collapse(3) .
 */
template <typename SBlock>
struct HostLaunchTraits<::carbon::Block<SBlock>> {
  using Grid = ::carbon::Block<SBlock>;

  static auto
  spans_for_host(const Grid& grid) -> const auto& {
    return grid.host_spans();
  }

  template <typename Span,
            typename HostFn>
  static auto
  iterate_slab(const Span& span,
               ::carbon::DataView view,
               HostFn& host_fn) -> void {
    constexpr int MemX = SBlock::MemX;
    constexpr int MemY = SBlock::MemY;
    constexpr int MemZ = SBlock::MemZ;
    const uint32_t iter_b = span.iter_block_count(view);
    if (iter_b == 0U) {
      return;
    }
    for (uint32_t iter_idx = 0; iter_idx < iter_b; ++iter_idx) {
      const uint32_t b = span.view_block_idx(view, iter_idx);
#pragma omp parallel for collapse(3) default(shared)
      for (int iz = 0; iz < MemZ; ++iz) {
        for (int iy = 0; iy < MemY; ++iy) {
          for (int ix = 0; ix < MemX; ++ix) {
            ::carbon::detail::block::Idx idx{};
            if (span.set_and_validate(b, ix, iy, iz, idx)) {
              host_fn(span, idx);
            }
          }
        }
      }
    }
  }

  /**
   * Per-slab host reduction over the brick range. One @c #pragma omp parallel
   * region drives a @c collapse(4) over @c (iter_idx, iz, iy, ix) with per-
   * thread partials combined sequentially at slab end. Matches the dense host
   * reducer's semantics on the per-thread + combine structure.
   */
  template <typename Span,
            typename HostFn,
            typename T,
            typename Op>
  static auto
  iterate_slab_reduce(const Span& span,
                      ::carbon::DataView view,
                      HostFn& host_fn,
                      Op& op) -> T {
    constexpr int MemX = SBlock::MemX;
    constexpr int MemY = SBlock::MemY;
    constexpr int MemZ = SBlock::MemZ;
    const uint32_t iter_b = span.iter_block_count(view);
    if (iter_b == 0U) {
      return op.identity();
    }
    const int max_threads = ::omp_get_max_threads();
    std::vector<T> partials(static_cast<size_t>(max_threads), op.identity());
#pragma omp parallel default(shared)
    {
      const int tid = ::omp_get_thread_num();
      T local = op.identity();
#pragma omp for collapse(4) nowait
      for (uint32_t iter_idx = 0; iter_idx < iter_b; ++iter_idx) {
        for (int iz = 0; iz < MemZ; ++iz) {
          for (int iy = 0; iy < MemY; ++iy) {
            for (int ix = 0; ix < MemX; ++ix) {
              const uint32_t b = span.view_block_idx(view, iter_idx);
              ::carbon::detail::block::Idx idx{};
              if (span.set_and_validate(b, ix, iy, iz, idx)) {
                local = op(local, host_fn(span, idx));
              }
            }
          }
        }
      }
      partials[static_cast<size_t>(tid)] = local;
    }
    T slab_acc = op.identity();
    for (const T& p : partials) {
      slab_acc = op(slab_acc, p);
    }
    return slab_acc;
  }
};

/**
 * Per-grid-family @b reduce launch trait. Sibling of @c DeviceLaunchTraits /
 * @c HostLaunchTraits ; covers the stage-1 fused "map + block-reduce" kernel
 * the reduction pipeline issues per local GPU. Stages 2 / 3 (intra-process
 * combine + cross-rank @c ncclAllReduce ) sit in @c detail/reduce_stages.h
 * and are family-agnostic — they only touch the per-GPU @c ReductionResult
 * scalars and carbon system primitives.
 *
 * Surface (per specialisation):
 *
 * - @c stage1_block_threads() — constexpr CUDA thread count per CUB
 *   @c BlockReduce instance (Dense: @c kReduceBlockThreads ; DenseDisg:
 *   @c kDisgReduceBlockThreads ; Block: @c SBlock::VoxelCount ).
 * - @c stage1_unit_count(grid, g_idx, view) — number of CUDA blocks the
 *   stage-1 launch dispatches for @p view on local GPU @p g_idx . Dense /
 *   DenseDisg : @c ceil_div(iter_extent.volume(), stage1_block_threads()) .
 *   Block : @c iter_block_count(view) (one CUDA block per retained brick).
 * - @c is_empty_domain(grid, g_idx, view) — true iff the stage-1 launch
 *   would have zero work. Dense / DenseDisg : @c iter_cells == 0 . Block :
 *   @c iter_block_count(view) == 0 .
 * - @c iter_cell_count(grid, g_idx, view) — total iteration cell count
 *   passed as the last arg to the Dense / DenseDisg stage-1 kernel; unused
 *   by Block (whose stage-1 kernel signature drops it).
 * - @c cuda_stage1(...) — runs the stage-1 launch for one GPU; returns
 *   @c cudaError_t .
 * - @c sycl_singlepass(...) — runs the single-pass @c sycl::reduction
 *   launcher on SYCL (Dense / Disg: takes @c total_cells ; Block: derives
 *   it from @c iter_block_count internally).
 * - @c assert_reduce_fn<F, T, Op>() — compile-time concept check on the
 *   per-cell device callable.
 */
template <typename Grid>
struct ReduceLaunchTraits {
  static constexpr auto
  stage1_block_threads() noexcept -> int {
    if constexpr (requires { typename Grid::DisgPolicy; }) {
      return ::carbon::detail::dense_disg::kDisgReduceBlockThreads;
    } else {
      return ::carbon::detail::dense::kReduceBlockThreads;
    }
  }

  static auto
  stage1_unit_count(const Grid& grid,
                    size_t g_idx,
                    ::carbon::DataView view,
                    int dense_stage1_threads = 0,
                    int dense_stage1_threads_y = 0) -> int64_t {
    const auto span = grid.spans()[g_idx];
    const ::carbon::int64_3d ext = span.iter_extent(view);
    const int64_t cells = static_cast<int64_t>(ext.volume());
    if constexpr (
        !requires { typename Grid::DisgPolicy; } && !requires { typename Grid::SBlock; }) {
      if (dense_stage1_threads_y > 0) {
        const int bx = dense_stage1_threads > 0 ? dense_stage1_threads : 32;
        const int by = dense_stage1_threads_y;
        if (ext.x <= 0 || ext.z <= 0) {
          return 0;
        }
        const int64_t gx = (ext.x + bx - 1) / bx;
        const int64_t gz = (ext.z + by - 1) / by;
        return gx * gz;
      }
    }
    int bt = stage1_block_threads();
    if constexpr (
        !requires { typename Grid::DisgPolicy; } && !requires { typename Grid::SBlock; }) {
      if (dense_stage1_threads > 0) {
        bt = dense_stage1_threads;
      }
    }
    return cells == 0 ? 0 : static_cast<int64_t>((cells + bt - 1) / bt);
  }

  static auto
  is_empty_domain(const Grid& grid,
                  size_t g_idx,
                  ::carbon::DataView view) -> bool {
    return grid.spans()[g_idx].iter_extent(view).volume() == 0;
  }

  static auto
  iter_cell_count(const Grid& grid,
                  size_t g_idx,
                  ::carbon::DataView view) -> int64_t {
    return static_cast<int64_t>(grid.spans()[g_idx].iter_extent(view).volume());
  }

#if CARBON_BACKEND_IS_CUDALIKE && CARBON_CUDALIKE_DEVICE_TU
  template <typename F,
            typename T,
            typename Op>
  static auto
  assert_reduce_fn() noexcept -> void {
    if constexpr (requires { typename Grid::DisgPolicy; }) {
      static_assert(::carbon::detail::dense_disg::disg_reduce_fn<F, T>,
                    "Kernel::reduce::launch (DenseDisg): the device callable returned by the "
                    "loading function must be invocable as fn(DisgSpan, DisgIdx) -> T.");
    } else {
      static_assert(::carbon::detail::dense::dense_reduce_fn<F, T>,
                    "Kernel::reduce::launch (Dense): the device callable returned by the loading "
                    "function must be invocable as fn(Span, Idx) -> T and trivially copyable.");
    }
  }

  template <typename F,
            typename T,
            typename Op>
  static auto
  cuda_stage1(const ::carbon::detail::Gpu& gpu,
              int64_t blocks,
              size_t shared_mem,
              gpu_stream_t stream,
              const typename Grid::Span& span,
              ::carbon::DataView view,
              F& fn,
              Op& op,
              T identity,
              T* partials,
              int64_t iter_cells,
              int dense_stage1_threads = 0,
              int dense_stage1_threads_y = 0) -> gpu_error_t {
    if constexpr (requires { typename Grid::DisgPolicy; }) {
      return gpu
          .template typed_launch<&::carbon::detail::dense_disg::disg_grid_reduce_kernel<F, T, Op>>(
              dim3(static_cast<unsigned>(blocks), 1U, 1U),
              dim3(static_cast<unsigned>(stage1_block_threads()), 1U, 1U), shared_mem, stream, span,
              view, fn, op, identity, partials, iter_cells);
    } else {
      if (dense_stage1_threads_y > 0) {
        const int bx = dense_stage1_threads > 0 ? dense_stage1_threads : 32;
        const int by = dense_stage1_threads_y;
        return ::carbon::detail::dense::launch_dense_grid_reduce_kernel_2d<F, T, Op>(
            bx, by, gpu, blocks, shared_mem, stream, span, view, fn, op, identity, partials,
            iter_cells);
      }
      const int bt = dense_stage1_threads > 0 ? dense_stage1_threads
                                              : ::carbon::detail::dense::kReduceBlockThreads;
      return ::carbon::detail::dense::launch_dense_grid_reduce_kernel<F, T, Op>(
          bt, gpu, blocks, shared_mem, stream, span, view, fn, op, identity, partials, iter_cells);
    }
  }

  template <typename F,
            typename T,
            typename Op>
  static auto
  cuda_stage1_block_atomic(const ::carbon::detail::Gpu& gpu,
                           int64_t blocks,
                           size_t shared_mem,
                           gpu_stream_t stream,
                           const typename Grid::Span& span,
                           ::carbon::DataView view,
                           F& fn,
                           Op& op,
                           T identity,
                           T* global_out,
                           int64_t iter_cells,
                           int dense_stage1_threads = 0,
                           int dense_stage1_threads_y = 0) -> gpu_error_t {
    if constexpr (requires { typename Grid::DisgPolicy; }) {
      return gpu.template typed_launch<
          &::carbon::detail::dense_disg::disg_grid_reduce_block_atomic_kernel<F, T, Op>>(
          dim3(static_cast<unsigned>(blocks), 1U, 1U),
          dim3(static_cast<unsigned>(stage1_block_threads()), 1U, 1U), shared_mem, stream, span,
          view, fn, op, identity, global_out, iter_cells);
    } else {
      if (dense_stage1_threads_y > 0) {
        const int bx = dense_stage1_threads > 0 ? dense_stage1_threads : 32;
        const int by = dense_stage1_threads_y;
        return ::carbon::detail::dense::launch_dense_grid_reduce_block_atomic_kernel_2d<F, T, Op>(
            bx, by, gpu, blocks, shared_mem, stream, span, view, fn, op, identity, global_out,
            iter_cells);
      }
      const int bt = dense_stage1_threads > 0 ? dense_stage1_threads
                                              : ::carbon::detail::dense::kReduceBlockThreads;
      return ::carbon::detail::dense::launch_dense_grid_reduce_block_atomic_kernel<F, T, Op>(
          bt, gpu, blocks, shared_mem, stream, span, view, fn, op, identity, global_out,
          iter_cells);
    }
  }
#endif  // CARBON_BACKEND_IS_CUDALIKE && CARBON_CUDALIKE_DEVICE_TU

#if CARBON_BACKEND_IS_SYCL
  template <typename F,
            typename T,
            typename Op>
  static auto
  sycl_singlepass(sycl::queue& q,
                  const typename Grid::Span& span,
                  ::carbon::DataView view,
                  F& fn,
                  Op& op,
                  T identity,
                  T* out,
                  int64_t iter_cells) -> void {
    if constexpr (requires { typename Grid::DisgPolicy; }) {
      ::carbon::detail::dense_disg::launch_disg_reduce<F, T, Op>(q, span, view, fn, op, identity,
                                                                 out, iter_cells);
    } else {
      ::carbon::detail::dense::launch_dense_reduce<F, T, Op>(q, span, view, fn, op, identity, out,
                                                             iter_cells);
    }
  }
#endif
};

template <typename SBlock>
struct ReduceLaunchTraits<::carbon::Block<SBlock>> {
  using Grid = ::carbon::Block<SBlock>;

  static constexpr auto
  stage1_block_threads() noexcept -> int {
    return SBlock::VoxelCount;
  }

  static auto
  stage1_unit_count(const Grid& grid,
                    size_t g_idx,
                    ::carbon::DataView view,
                    int /*dense_stage1_threads*/ = 0,
                    int /*dense_stage1_threads_y*/ = 0) -> int64_t {
    return static_cast<int64_t>(grid.spans()[g_idx].iter_block_count(view));
  }

  static auto
  is_empty_domain(const Grid& grid,
                  size_t g_idx,
                  ::carbon::DataView view) -> bool {
    return grid.spans()[g_idx].iter_block_count(view) == 0U;
  }

  static auto
  iter_cell_count(const Grid& grid,
                  size_t g_idx,
                  ::carbon::DataView view) -> int64_t {
    return stage1_unit_count(grid, g_idx, view);
  }

#if CARBON_BACKEND_IS_CUDALIKE && CARBON_CUDALIKE_DEVICE_TU
  template <typename F,
            typename T,
            typename Op>
  static auto
  assert_reduce_fn() noexcept -> void {
    static_assert(::carbon::detail::block::block_reduce_fn<F, SBlock, T>,
                  "Kernel::reduce::launch (Block): the device callable returned by the loading "
                  "function must be invocable as fn(Span<SBlock>, Idx) -> T.");
  }

  template <typename F,
            typename T,
            typename Op>
  static auto
  cuda_stage1(const ::carbon::detail::Gpu& gpu,
              int64_t blocks,
              size_t shared_mem,
              gpu_stream_t stream,
              const typename Grid::Span& span,
              ::carbon::DataView view,
              F& fn,
              Op& op,
              T identity,
              T* partials,
              int64_t /*iter_cells*/,
              int /*dense_stage1_threads*/ = 0,
              int /*dense_stage1_threads_y*/ = 0) -> gpu_error_t {
    return gpu.template typed_launch<
        &::carbon::detail::block::block_grid_reduce_kernel<SBlock, F, T, Op>>(
        dim3(static_cast<unsigned>(blocks), 1U, 1U),
        dim3(static_cast<unsigned>(stage1_block_threads()), 1U, 1U), shared_mem, stream, span, view,
        fn, op, identity, partials);
  }

  template <typename F,
            typename T,
            typename Op>
  static auto
  cuda_stage1_block_atomic(const ::carbon::detail::Gpu& gpu,
                           int64_t blocks,
                           size_t shared_mem,
                           gpu_stream_t stream,
                           const typename Grid::Span& span,
                           ::carbon::DataView view,
                           F& fn,
                           Op& op,
                           T identity,
                           T* global_out,
                           int64_t /*iter_cells*/,
                           int /*dense_stage1_threads*/ = 0,
                           int /*dense_stage1_threads_y*/ = 0) -> gpu_error_t {
    return gpu.template typed_launch<
        &::carbon::detail::block::block_grid_reduce_block_atomic_kernel<SBlock, F, T, Op>>(
        dim3(static_cast<unsigned>(blocks), 1U, 1U),
        dim3(static_cast<unsigned>(stage1_block_threads()), 1U, 1U), shared_mem, stream, span, view,
        fn, op, identity, global_out);
  }
#endif

#if CARBON_BACKEND_IS_SYCL
  template <typename F,
            typename T,
            typename Op>
  static auto
  sycl_singlepass(sycl::queue& q,
                  const typename Grid::Span& span,
                  ::carbon::DataView view,
                  F& fn,
                  Op& op,
                  T identity,
                  T* out,
                  int64_t /*iter_cells*/) -> void {
    ::carbon::detail::block::launch_block_reduce<SBlock, F, T, Op>(q, span, view, fn, op, identity,
                                                                   out);
  }
#endif
};

}  // namespace detail
}  // namespace carbon
