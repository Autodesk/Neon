#pragma once

/**
 * Device-side machinery for grid-wide reductions over a @c carbon::Block grid.
 * Fused "map + block-reduce" CUDA kernel: each CUDA block processes one
 * @ref carbon::Block storage brick, each thread processes one voxel, and a
 * single @c gpucub::BlockReduce::Reduce collapses the brick to one partial.
 * The grid-level collapse (second pass over the per-brick partials) goes
 * through @c gpucub::DeviceReduce::Reduce on the host side; the orchestration
 * lives in @c carbon/detail/block_reduce_kernel_impl.h .
 *
 * Include only from @c .cu translation units (or any TU compiled with nvcc):
 * pulls in @c <cub/...> and a @c __global__ template.
 */

#include <carbon/data_view.h>
#include <carbon/detail/block/idx.h>
#include <carbon/detail/block/span.h>
#include <carbon/detail/block/static_block.h>

#include <concepts>
#include <cstdint>
#include <type_traits>

#if CARBON_CUDALIKE_DEVICE_TU
#include <carbon/detail/backend/cuda_like.h>
#endif

namespace carbon::detail::block {

/**
 * Concept for a Block-reduce kernel's per-voxel device lambda: callable as
 * @c (Span<SBlock>, Idx) -> T , with @c T returned by value. The lambda is
 * invoked only for voxels whose bitmask bit is set; inactive voxels in the
 * brick @b never call the lambda and contribute the reduction operator's
 * identity element. Same convention as the dense reducer: no
 * @c is_trivially_copyable_v static-assert because nvcc-wrapped extended
 * lambdas fail that trait even though they're safe to pass by value.
 */
template <typename F, typename SBlock, typename T>
concept block_reduce_fn = std::is_invocable_r_v<T, F, Span<SBlock>, Idx>;

#if CARBON_CUDALIKE_DEVICE_TU

/**
 * Fused "map + block-reduce" kernel for one storage brick of a
 * @c carbon::Block under any @ref DataView .
 *
 * Grid layout: 1D, @c gridDim.x == @c span.iter_block_count(view) ; threads
 * are 1D with @c blockDim.x == @c SBlock::VoxelCount (e.g. 512 for the
 * default 8×8×8 brick). Each CUDA block handles exactly one storage brick
 * (the one returned by @c span.view_block_idx(view, blockIdx.x) ) and each
 * thread handles one voxel of that brick — decoded row-major
 * (@c x fastest, @c z slowest) from @c threadIdx.x .
 *
 * Per thread:
 * - Decode @c tid in @c [0, VoxelCount) to @c (ix, iy, iz) .
 * - If the brick's bitmask reports the voxel as inactive (e.g. carved-out
 *   voxel inside an active brick), contribute @p init .
 * - Otherwise invoke @p fn(span, idx) and feed its return into the per-thread
 *   contribution.
 *
 * Block stage: @c gpucub::BlockReduce<T, BLOCK_THREADS_X> with
 * @c BLOCK_THREADS_X == @c SBlock::VoxelCount collapses the brick to one
 * @c T . Thread 0 writes the brick partial into @p partials at index
 * @c blockIdx.x (i.e. the iteration index, @b not the storage block index —
 * the impl's stage-2 gpucub::DeviceReduce::Reduce over @c partials runs over
 * exactly @c iter_block_count entries).
 *
 * @tparam SBlock @c StaticBlock policy of the underlying @ref carbon::Block .
 * @tparam F      User per-voxel device callable, must satisfy
 *                @c block_reduce_fn<F, SBlock, T> .
 * @tparam T      Reduction value type (CUB-supported; arithmetic types or
 *                trivially-copyable aggregates with a custom Op).
 * @tparam Op     Binary reduction functor invoked as @c op(a, b) and forwarded
 *                to @c gpucub::BlockReduce::Reduce . Must be trivially copyable.
 */
template <typename SBlock,
          typename F,
          typename T,
          typename Op>
__global__ void
block_grid_reduce_kernel(Span<SBlock> span,
                         ::carbon::DataView view,
                         F fn,
                         Op op,
                         T init,
                         T* partials) {
  constexpr int BLOCK_THREADS = SBlock::VoxelCount;
  using BlockReduceT = gpucub::BlockReduce<T, BLOCK_THREADS>;
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  const uint32_t iter_b = blockIdx.x;
  // Translate the iteration index to a storage brick index. For Standard /
  // Internal this is @c iter_b ; for Boundary it is @c n_internal + iter_b .
  const uint32_t b = span.view_block_idx(view, iter_b);

  const int tid = static_cast<int>(threadIdx.x);
  const int ix = tid % SBlock::MemX;
  const int iy = (tid / SBlock::MemX) % SBlock::MemY;
  const int iz = tid / (SBlock::MemX * SBlock::MemY);

  T thread_val = init;
  const auto& bm = span.bitmask()[b];
  if (bm.is_active(ix, iy, iz)) {
    const Idx idx{b, static_cast<int8_t>(ix), static_cast<int8_t>(iy), static_cast<int8_t>(iz)};
    thread_val = fn(span, idx);
  }

  // BlockReduce::Reduce needs every thread to participate (warp shuffles +
  // shared memory writes). Inactive-voxel threads above already hold @c init
  // so they contribute the identity element.
  const T brick_val = BlockReduceT(temp_storage).Reduce(thread_val, op);
  if (threadIdx.x == 0U) {
    partials[iter_b] = brick_val;
  }
}

template <typename T>
CARBON_HD inline auto
block_reduce_atomic_accumulate(T* dst,
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
 * Block-grid reduce with block partials merged via atomics into @p global_out
 * ( @ref ReduceFinalize::BlockAtomic ), analogous to @ref dense_grid_reduce_block_atomic_kernel .
 */
template <typename SBlock,
          typename F,
          typename T,
          typename Op>
__global__ void
block_grid_reduce_block_atomic_kernel(Span<SBlock> span,
                                      ::carbon::DataView view,
                                      F fn,
                                      Op op,
                                      T init,
                                      T* global_out) {
  constexpr int BLOCK_THREADS = SBlock::VoxelCount;
  using BlockReduceT = gpucub::BlockReduce<T, BLOCK_THREADS>;
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  const uint32_t iter_b = blockIdx.x;
  const uint32_t b = span.view_block_idx(view, iter_b);

  const int tid = static_cast<int>(threadIdx.x);
  const int ix = tid % SBlock::MemX;
  const int iy = (tid / SBlock::MemX) % SBlock::MemY;
  const int iz = tid / (SBlock::MemX * SBlock::MemY);

  T thread_val = init;
  const auto& bm = span.bitmask()[b];
  if (bm.is_active(ix, iy, iz)) {
    const Idx idx{b, static_cast<int8_t>(ix), static_cast<int8_t>(iy), static_cast<int8_t>(iz)};
    thread_val = fn(span, idx);
  }

  const T brick_val = BlockReduceT(temp_storage).Reduce(thread_val, op);
  if (threadIdx.x == 0U) {
    block_reduce_atomic_accumulate(global_out, brick_val);
  }
}

#endif  // CARBON_CUDALIKE_DEVICE_TU

}  // namespace carbon::detail::block
