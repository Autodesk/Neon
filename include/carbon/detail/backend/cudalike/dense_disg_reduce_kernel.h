#pragma once

/**
 * Device-side machinery for grid-wide reductions on @c DenseDisg slabs. Twin of
 * @c detail/backend/cudalike/dense_reduce_kernel.h but parameterised on @c dense_disg::Span
 * / @c dense_disg::Idx . Include only from @c .cu translation units (or any TU compiled
 * with nvcc).
 */

#include <carbon/data_view.h>
#include <carbon/detail/dense_disg/span.h>
#include <carbon/integer3d.h>

#include <concepts>
#include <cstdint>
#include <type_traits>

#if CARBON_CUDALIKE_DEVICE_TU
#include <carbon/detail/backend/cuda_like.h>
#endif

namespace carbon::detail::dense_disg {

/** Concept for a disg reduce kernel's per-cell device lambda: @c (Span, Idx) -> T . */
template <typename F, typename T>
concept disg_reduce_fn = std::is_invocable_r_v<T, F, Span, Idx>;

/** Block-flat thread count for the disg reduce kernel; matches the aggregated path. */
constexpr int kDisgReduceBlockThreads = 256;

#if CARBON_CUDALIKE_DEVICE_TU

/**
 * Fused "map + block-reduce" kernel for a @c DenseDisg slab under any @ref DataView .
 * Same shape as @c dense_grid_reduce_kernel ; only the Span / Idx types differ.
 */
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
    const ::carbon::int64_3d ext = span.iter_extent(view);
    const int64_t ix = flat % ext.x;
    const int64_t rest = flat / ext.x;
    const int64_t iy = rest % ext.y;
    const int64_t iz = rest / ext.y;
    const ValidatedIdx vi = span.set_and_validate(view, ix, iy, iz);
    if (vi.valid) {
      thread_val = fn(span, vi.idx);
    }
  }

  const T block_val = BlockReduceT(temp_storage).Reduce(thread_val, op);
  if (threadIdx.x == 0U) {
    partials[blockIdx.x] = block_val;
  }
}

template <typename T>
CARBON_HD inline auto
disg_reduce_atomic_accumulate(T* dst,
                              T value) -> void {
  if constexpr (std::is_same_v<T, float>) {
    atomicAdd(dst, value);
  } else if constexpr (std::is_same_v<T, double>) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
    atomicAdd(dst, value);
#else
    (void)dst;
    (void)value;
#endif
  } else if constexpr (std::is_same_v<T, int32_t>) {
    atomicAdd(dst, value);
  } else if constexpr (std::is_same_v<T, int64_t>) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
    atomicAdd(reinterpret_cast<unsigned long long*>(dst),
              static_cast<unsigned long long>(value));
#else
    (void)dst;
    (void)value;
#endif
  } else {
    (void)dst;
    (void)value;
  }
}

/**
 * Disg twin of @ref carbon::detail::dense::dense_grid_reduce_block_atomic_kernel .
 */
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
    const ::carbon::int64_3d ext = span.iter_extent(view);
    const int64_t ix = flat % ext.x;
    const int64_t rest = flat / ext.x;
    const int64_t iy = rest % ext.y;
    const int64_t iz = rest / ext.y;
    const ValidatedIdx vi = span.set_and_validate(view, ix, iy, iz);
    if (vi.valid) {
      thread_val = fn(span, vi.idx);
    }
  }

  const T block_val = BlockReduceT(temp_storage).Reduce(thread_val, op);
  if (threadIdx.x == 0U) {
    disg_reduce_atomic_accumulate(global_out, block_val);
  }
}

#endif  // CARBON_CUDALIKE_DEVICE_TU

}  // namespace carbon::detail::dense_disg
