#pragma once

/**
 * CUDA block-grid kernel. Counterpart of @c detail/backend/cudalike/dense_kernel.h for the
 * @c carbon::Block (block-sparse) family.
 *
 * Launch shape:
 *
 *   - grid  = @c (block_count, 1, 1)               — one CUDA block per retained brick.
 *   - block = @c (MemX, MemY, MemZ)                — one thread per voxel inside the
 *                                                    brick.
 *
 * Each thread reads its @c (block_idx, ix, iy, iz) from @c blockIdx.x and @c threadIdx ,
 * runs @c span.set_and_validate to test the per-block bitmask, and (when the bit is
 * set) invokes the user's per-cell callable. Threads on tail bricks whose
 * @c (ix, iy, iz) falls on an inactive voxel (clear bit) early-out without invoking
 * @p fn .
 *
 * Mirrors Neon's @c bGrid @c d1b3 launch (one block per data block, brick-sized
 * thread block). Include only from @c .cu translation units (or any TU compiled with
 * nvcc).
 */

#include <carbon/data_view.h>
#include <carbon/detail/block/idx.h>
#include <carbon/detail/block/span.h>
#include <carbon/detail/block/static_block.h>

#include <concepts>
#include <utility>

#if CARBON_CUDALIKE_DEVICE_TU
#include <carbon/detail/backend/cuda_like.h>
#endif

namespace carbon::detail::block {

/**
 * Callable type for @ref block_grid_kernel : @c void fn(span, idx) where @c span is a
 * @c Span<SBlock> and @c idx is a @c block::Idx . Same shape as @c dense_kernel_fn ,
 * just over the block-sparse types.
 */
template <typename F, typename SBlock>
concept block_kernel_fn = requires(F fn, Span<SBlock> span, Idx idx) {
  {fn(span, idx)}->std::same_as<void>;
};

#if CARBON_CUDALIKE_DEVICE_TU

/**
 * Per-voxel CUDA entry point for @c carbon::Block kernels. Skips inactive voxels via
 * @c Span::set_and_validate and invokes @p fn only for voxels whose bitmask bit is
 * set.
 *
 * @note No @c requires clause on the @c __global__ template — nvcc's stub generator
 *       has historically choked on constrained device function templates that take a
 *       device-lambda type and a separate static-type parameter; the constraint is
 *       enforced at the launch helper instead (see @c launch_block_grid below).
 */
template <typename SBlock,
          typename F>
__global__ void
block_grid_kernel(Span<SBlock> span,
                  ::carbon::DataView view,
                  F fn) {
  const uint32_t iter_b = blockIdx.x;
  const uint32_t b = span.view_block_idx(view, iter_b);
  const int ix = static_cast<int>(threadIdx.x);
  const int iy = static_cast<int>(threadIdx.y);
  const int iz = static_cast<int>(threadIdx.z);
  Idx idx{};
  if (!span.set_and_validate(b, ix, iy, iz, idx)) {
    return;
  }
  fn(span, idx);
}

#endif

}  // namespace carbon::detail::block
