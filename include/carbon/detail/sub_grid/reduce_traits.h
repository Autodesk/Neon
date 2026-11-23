#pragma once

/**
 * @c ReduceLaunchTraits / @c HostLaunchTraits for @c carbon::SubGrid , so the grid-agnostic
 * reduction impls ( @c ReduceKernelImpl , @c ReduceHostKernelImpl ,
 * @c UniversalReduceKernelImpl ) run SubGrid reductions unchanged: multi-GPU combine,
 * cross-rank stage, @c ReduceFinalize::BlockAtomic , deferred host copy, and the OCC
 * Partial / Merge split. Stage 1 iterates the lanes of the requested @ref DataView ;
 * internal-first lane order makes Internal and Boundary two contiguous ranges.
 */

#include <carbon/data_view.h>
#include <carbon/detail/backend/select.h>
#include <carbon/detail/launch_traits.h>
#include <carbon/detail/sub_grid/fwd.h>

#if CARBON_BACKEND_IS_CUDALIKE
#include <carbon/detail/backend/cudalike/sub_grid_reduce.h>
#elif CARBON_BACKEND_IS_SYCL
#include <carbon/detail/backend/sycl/sub_grid_reduce.h>
#endif

#include <omp.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace carbon::detail {

template <typename G>
struct ReduceLaunchTraits<::carbon::SubGrid<G>> {
  using Grid = ::carbon::SubGrid<G>;

  static constexpr auto
  stage1_block_threads() noexcept -> int {
    return 256;
  }

  static auto
  stage1_unit_count(const Grid& grid,
                    size_t g_idx,
                    ::carbon::DataView view,
                    int /*dense_stage1_threads*/ = 0,
                    int /*dense_stage1_threads_y*/ = 0) -> int64_t {
    const int64_t lanes = iter_cell_count(grid, g_idx, view);
    return (lanes + stage1_block_threads() - 1) / stage1_block_threads();
  }

  static auto
  is_empty_domain(const Grid& grid,
                  size_t g_idx,
                  ::carbon::DataView view) -> bool {
    return iter_cell_count(grid, g_idx, view) == 0;
  }

  static auto
  iter_cell_count(const Grid& grid,
                  size_t g_idx,
                  ::carbon::DataView view) -> int64_t {
    return grid.spans()[g_idx].sub().lane_count(view);
  }

#if CARBON_BACKEND_IS_CUDALIKE && CARBON_CUDALIKE_DEVICE_TU
  template <typename F,
            typename T,
            typename Op>
  static auto
  assert_reduce_fn() noexcept -> void {
    static_assert(std::is_invocable_r_v<T, F, typename Grid::Span, typename Grid::Idx>,
                  "Kernel::reduce (SubGrid): the device callable returned by the loading "
                  "function must be invocable as fn(SubGrid::Span, SubGrid::Idx) -> T.");
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
    return sub_grid::launch_lane_reduce<typename Grid::Idx, /*BlockAtomic=*/false>(
        gpu, blocks, shared_mem, stream, span, view, fn, op, identity, partials);
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
    return sub_grid::launch_lane_reduce<typename Grid::Idx, /*BlockAtomic=*/true>(
        gpu, blocks, shared_mem, stream, span, view, fn, op, identity, global_out);
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
    sub_grid::launch_lane_reduce<typename Grid::Idx>(q, span, view, fn, op, identity, out);
  }

  template <typename F,
            typename T>
  static auto
  sycl_block_atomic(sycl::queue& q,
                    const typename Grid::Span& span,
                    ::carbon::DataView view,
                    F& fn,
                    T* out,
                    int64_t /*iter_cells*/) -> void {
    sub_grid::launch_lane_block_atomic_sum<typename Grid::Idx>(q, span, view, fn, out);
  }
#endif
};

template <typename G>
struct HostLaunchTraits<::carbon::SubGrid<G>> {
  using Grid = ::carbon::SubGrid<G>;

  /** Host lane tables: the device spans' tables are not reachable from host threads. */
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
    using Idx = typename Grid::Idx;
    const int64_t begin = span.sub().lane_offset(view);
    const int64_t count = span.sub().lane_count(view);
#pragma omp parallel for default(shared)
    for (int64_t k = 0; k < count; ++k) {
      const int64_t lane = begin + k;
      host_fn(span, Idx(span.sub().parent_idx(lane), lane));
    }
  }

  template <typename Span,
            typename HostFn,
            typename T,
            typename Op>
  static auto
  iterate_slab_reduce(const Span& span,
                      ::carbon::DataView view,
                      HostFn& host_fn,
                      Op& op) -> T {
    using Idx = typename Grid::Idx;
    const int64_t begin = span.sub().lane_offset(view);
    const int64_t count = span.sub().lane_count(view);
    if (count <= 0) {
      return op.identity();
    }
    std::vector<T> partials(static_cast<size_t>(::omp_get_max_threads()), op.identity());
#pragma omp parallel default(shared)
    {
      T local = op.identity();
#pragma omp for nowait
      for (int64_t k = 0; k < count; ++k) {
        const int64_t lane = begin + k;
        local = op(local, host_fn(span, Idx(span.sub().parent_idx(lane), lane)));
      }
      partials[static_cast<size_t>(::omp_get_thread_num())] = local;
    }
    T acc = op.identity();
    for (const T& p : partials) {
      acc = op(acc, p);
    }
    return acc;
  }
};

}  // namespace carbon::detail
