#pragma once

/**
 * Device-side machinery for grid-wide reductions on @c Dense slabs. The fused
 * "map + block-reduce" CUDA kernel that runs the user's per-cell device lambda and
 * collapses each block to a single partial via @c gpucub::BlockReduce. The grid-level
 * collapse (the second pass over the per-block partials) is done by
 * @c gpucub::DeviceReduce::Reduce on the host side; see
 * @c carbon/detail/reduce_kernel_impl.h for the orchestration (the impl behind
 * @c carbon::Kernel::reduce in @c carbon/kernel.h).
 *
 * Include only from @c .cu translation units (or any TU compiled with nvcc): pulls in
 * @c <cub/...> and a @c __global__ template.
 */

#include <carbon/data_view.h>
#include <carbon/detail/dense/span.h>
#include <carbon/integer3d.h>

#include <concepts>
#include <cstdint>
#include <type_traits>

#if CARBON_CUDALIKE_DEVICE_TU
#include <carbon/detail/backend/cuda_like.h>
#endif

namespace carbon::detail::dense {

/**
 * Concept for a reduce kernel's per-cell device lambda: callable as @c (Span, Idx) -> T,
 * with @c T returned by value. Same shape as @ref dense_kernel_fn but the return type is
 * the contribution to the reduction instead of @c void.
 *
 * No @c std::is_trivially_copyable_v check: nvcc wraps extended host-device lambdas in
 * @c __nv_hdl_wrapper_t which fails the standard trait even though the closure is safe to
 * pass by value into a @c __global__ kernel (CUDA's own contract is more permissive than
 * the C++ trait). The existing @ref dense_kernel_fn for the void-returning compute case
 * doesn't enforce trivial copyability either, and we follow the same convention here.
 */
template <typename F, typename T>
concept dense_reduce_fn = std::is_invocable_r_v<T, F, Span, Idx>;

/**
 * Default block-flat thread count when @ref carbon::reduce::ReduceOptions::stage1_threads
 * is zero.
 */
constexpr int kReduceBlockThreads = 256;

#if CARBON_CUDALIKE_DEVICE_TU

template <typename T>
CARBON_HD inline auto
reduce_atomic_accumulate(T* dst,
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
    atomicAdd(reinterpret_cast<unsigned long long*>(dst), static_cast<unsigned long long>(value));
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
 * Fused "map + block-reduce" kernel for a @c Dense slab under any @ref DataView .
 * @p BlockThreads must match the launch configuration (see @c dense_reduce_launch.h ).
 */
template <int BlockThreads,
          typename F,
          typename T,
          typename Op>
__global__ void
dense_grid_reduce_kernel_t(Span span,
                           ::carbon::DataView view,
                           F fn,
                           Op op,
                           T init,
                           T* partials,
                           int64_t total_cells) {
  using BlockReduceT = gpucub::BlockReduce<T, BlockThreads>;
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  const int64_t flat =
      static_cast<int64_t>(blockIdx.x) * BlockThreads + static_cast<int64_t>(threadIdx.x);

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

/**
 * Same as @ref dense_grid_reduce_kernel_t but block sums are merged with
 * @ref reduce_atomic_accumulate into @p global_out instead of a partials vector.
 */
template <int BlockThreads,
          typename F,
          typename T,
          typename Op>
__global__ void
dense_grid_reduce_block_atomic_kernel_t(Span span,
                                        ::carbon::DataView view,
                                        F fn,
                                        Op op,
                                        T init,
                                        T* global_out,
                                        int64_t total_cells) {
  using BlockReduceT = gpucub::BlockReduce<T, BlockThreads>;
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  const int64_t flat =
      static_cast<int64_t>(blockIdx.x) * BlockThreads + static_cast<int64_t>(threadIdx.x);

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
    reduce_atomic_accumulate(global_out, block_val);
  }
}

/**
 * 2D CUDA block over @c (ix, iz) — matches NVIDIA @c jacobi_kernel launch shape on Carbon
 * @c Dense slabs where @c y == 1 and @c z is the second mesh axis.
 */
template <int BlockDimX,
          int BlockDimY,
          typename F,
          typename T,
          typename Op>
__global__ void
dense_grid_reduce_kernel_2d_t(Span span,
                              ::carbon::DataView view,
                              F fn,
                              Op op,
                              T init,
                              T* partials,
                              int64_t /*total_cells*/) {
  using BlockReduceT =
      gpucub::BlockReduce<T, BlockDimX, gpucub::BLOCK_REDUCE_WARP_REDUCTIONS, BlockDimY>;
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  const ::carbon::int64_3d ext = span.iter_extent(view);
  const int64_t ix =
      static_cast<int64_t>(blockIdx.x) * BlockDimX + static_cast<int64_t>(threadIdx.x);
  const int64_t iz =
      static_cast<int64_t>(blockIdx.y) * BlockDimY + static_cast<int64_t>(threadIdx.y);

  T thread_val = init;
  if (ix < ext.x && iz < ext.z) {
    const ValidatedIdx vi = span.set_and_validate(view, ix, 0, iz);
    if (vi.valid) {
      thread_val = fn(span, vi.idx);
    }
  }

  const T block_val = BlockReduceT(temp_storage).Reduce(thread_val, op);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    const size_t block_linear = static_cast<size_t>(blockIdx.y) * static_cast<size_t>(gridDim.x) +
                                static_cast<size_t>(blockIdx.x);
    partials[block_linear] = block_val;
  }
}

template <int BlockDimX,
          int BlockDimY,
          typename F,
          typename T,
          typename Op>
__global__ void
dense_grid_reduce_block_atomic_kernel_2d_t(Span span,
                                           ::carbon::DataView view,
                                           F fn,
                                           Op op,
                                           T init,
                                           T* global_out,
                                           int64_t /*total_cells*/) {
  using BlockReduceT =
      gpucub::BlockReduce<T, BlockDimX, gpucub::BLOCK_REDUCE_WARP_REDUCTIONS, BlockDimY>;
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  const ::carbon::int64_3d ext = span.iter_extent(view);
  const int64_t ix =
      static_cast<int64_t>(blockIdx.x) * BlockDimX + static_cast<int64_t>(threadIdx.x);
  const int64_t iz =
      static_cast<int64_t>(blockIdx.y) * BlockDimY + static_cast<int64_t>(threadIdx.y);

  T thread_val = init;
  if (ix < ext.x && iz < ext.z) {
    const ValidatedIdx vi = span.set_and_validate(view, ix, 0, iz);
    if (vi.valid) {
      thread_val = fn(span, vi.idx);
    }
  }

  const T block_val = BlockReduceT(temp_storage).Reduce(thread_val, op);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    reduce_atomic_accumulate(global_out, block_val);
  }
}

#endif  // CARBON_CUDALIKE_DEVICE_TU

}  // namespace carbon::detail::dense
