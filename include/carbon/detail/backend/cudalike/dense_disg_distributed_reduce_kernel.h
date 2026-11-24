#pragma once

/**
 * Device-side machinery for grid-wide reductions on @c DenseDisgDistributed slabs. Twin of
 * @c detail/backend/cudalike/dense_disg_reduce_kernel.h parameterised on
 * @c dense_disg_distributed::Span / @c Idx . Include only from @c .cu translation units (or any
 * TU compiled with nvcc).
 */

#include <carbon/data_view.h>
#include <carbon/detail/backend/cudalike/dense_disg_reduce_kernel.h>
#include <carbon/detail/dense_disg_distributed/span.h>
#include <carbon/integer3d.h>

#include <cstdint>
#include <type_traits>

#if CARBON_CUDALIKE_DEVICE_TU
#include <carbon/detail/backend/cuda_like.h>
#include <carbon/detail/backend/cudalike/reduce_atomic.h>
#endif

namespace carbon::detail::dense_disg_distributed {

/** Concept for a reduce kernel's per-cell device lambda: @c (Span, Idx) -> T . */
template <typename F, typename T>
concept disg_reduce_fn = std::is_invocable_r_v<T, F, Span, Idx>;

/** Block-flat thread count for the reduce kernel; matches @c dense_disg . */
constexpr int kDisgReduceBlockThreads = 256;

#if CARBON_CUDALIKE_DEVICE_TU

/** Fused "map + block-reduce" kernel; same shape as @c dense_disg::disg_grid_reduce_kernel . */
template <typename F,
          typename T,
          typename Op>
__global__ void
disg_grid_reduce_kernel(Span span,
                        ::carbon::DataView view,
                        F fn,
                        Op op,
                        T init,
                        T* partials,
                        int64_t total_cells) {
  using BlockReduceT = gpucub::BlockReduce<T, kDisgReduceBlockThreads>;
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  const int64_t flat = static_cast<int64_t>(blockIdx.x) * kDisgReduceBlockThreads +
                       static_cast<int64_t>(threadIdx.x);

  T thread_val = init;
  if (flat < total_cells) {
    const Span local = span;
    const ::carbon::int64_3d ext = local.iter_extent(view);
    const int64_t ix = flat % ext.x;
    const int64_t rest = flat / ext.x;
    const int64_t iy = rest % ext.y;
    const int64_t iz = rest / ext.y;
    const ValidatedIdx vi = local.set_and_validate(view, ix, iy, iz);
    if (vi.valid) {
      thread_val = fn(local, vi.idx);
    }
  }

  const T block_val = BlockReduceT(temp_storage).Reduce(thread_val, op);
  if (threadIdx.x == 0U) {
    partials[blockIdx.x] = block_val;
  }
}

/** Block-atomic twin of @ref disg_grid_reduce_kernel ; accumulates straight into @p global_out . */
template <typename F,
          typename T,
          typename Op>
__global__ void
disg_grid_reduce_block_atomic_kernel(Span span,
                                     ::carbon::DataView view,
                                     F fn,
                                     Op op,
                                     T init,
                                     T* global_out,
                                     int64_t total_cells) {
  using BlockReduceT = gpucub::BlockReduce<T, kDisgReduceBlockThreads>;
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  const int64_t flat = static_cast<int64_t>(blockIdx.x) * kDisgReduceBlockThreads +
                       static_cast<int64_t>(threadIdx.x);

  T thread_val = init;
  if (flat < total_cells) {
    const Span local = span;
    const ::carbon::int64_3d ext = local.iter_extent(view);
    const int64_t ix = flat % ext.x;
    const int64_t rest = flat / ext.x;
    const int64_t iy = rest % ext.y;
    const int64_t iz = rest / ext.y;
    const ValidatedIdx vi = local.set_and_validate(view, ix, iy, iz);
    if (vi.valid) {
      thread_val = fn(local, vi.idx);
    }
  }

  const T block_val = BlockReduceT(temp_storage).Reduce(thread_val, op);
  if (threadIdx.x == 0U) {
    ::carbon::detail::reduce_atomic_accumulate(global_out, block_val);
  }
}

#endif  // CARBON_CUDALIKE_DEVICE_TU

}  // namespace carbon::detail::dense_disg_distributed
