#pragma once

/**
 * SYCL twin of @c detail/backend/cudalike/block_reduce_kernel.h: fused per-voxel
 * map+reduce over a @c carbon::detail::block::Span<SBlock> using
 * @c sycl::reduction over @c parallel_for . Replaces the CUDA backend's two-stage
 * @c cub::BlockReduce + per-brick partials + @c cub::DeviceReduce::Reduce pipeline
 * with the single SYCL primitive -- same trade-off the dense SYCL reducer makes.
 *
 * @par Concept
 * Same @c block_reduce_fn concept as the CUDA backend so user lambdas authored
 * against @c Block::make_reduce_kernel satisfy both.
 *
 * @par Iteration shape
 *
 *   total_voxels = iter_block_count(view) * MemX * MemY * MemZ
 *
 * A single @c sycl::range<1>(total_voxels) traversal walks every (brick, ix, iy, iz)
 * tuple; the decoder inside the kernel splits the flat index into
 * @c (brick_iter_idx, iz, iy, ix) with brick slowest and x fastest. The same brick
 * ordering the CUDA path uses ( @c blockIdx.x = brick_iter_idx ) so any non-
 * associative-op corner case lands on the same accumulation tree shape modulo the
 * SYCL runtime's own internal reduction tree (associative by contract).
 *
 * @par Inactive voxels
 * @c span.set_and_validate(b, ix, iy, iz, idx) returns @c false for inactive voxels;
 * the kernel skips @c red.combine for those cases so the operator's identity is the
 * only contribution they make. Mirrors the CUDA backend's block_grid_reduce_kernel
 * inactive-voxel branch.
 */

#include <carbon/data_view.h>
#include <carbon/detail/block/idx.h>
#include <carbon/detail/block/span.h>
#include <carbon/detail/block/static_block.h>
#include <carbon/logging.h>

#include <sycl/sycl.hpp>

#include <concepts>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace carbon::detail::block {

/**
 * Callable type for the SYCL block reducer: @c (Span<SBlock>, Idx) -> T returned
 * by value. Same shape as the CUDA-backend concept.
 */
template <typename F, typename SBlock, typename T>
concept block_reduce_fn = std::is_invocable_r_v<T, F, Span<SBlock>, Idx>;

/**
 * @brief Launch a fused map+reduce over every retained brick in @p span 's
 *        @p view iteration range and write the final scalar to @p out .
 *
 * @tparam SBlock  Brick policy (provides @c MemX / @c MemY / @c MemZ ).
 * @tparam F       Per-cell device callable; must satisfy
 *                 @c block_reduce_fn<F, SBlock, T> .
 * @tparam T       Reduction value type.
 * @tparam Op      Binary reduction functor (commutative-associative).
 *
 * @param queue        SYCL queue (already pinned to the right device).
 * @param span         Per-GPU block::Span<SBlock>.
 * @param view         Iteration view (Standard / Internal / Boundary).
 * @param fn           Per-voxel device callable.
 * @param op           Reduction operator.
 * @param init         Identity element for @p op .
 * @param out          Device pointer to a single @c T receiving the final result.
 */
template <typename SBlock,
          typename F,
          typename T,
          typename Op>
requires block_reduce_fn<F,
                         SBlock,
                         T> inline auto
launch_block_reduce(sycl::queue& queue,
                    Span<SBlock> span,
                    ::carbon::DataView view,
                    F fn,
                    Op op,
                    T init,
                    T* out) -> void {
  const uint32_t iter_blocks = span.iter_block_count(view);
  if (iter_blocks == 0U) {
    try {
      queue.fill(out, init, 1U);
    } catch (const sycl::exception& ex) {
      ::carbon::log::throw_with_log<std::runtime_error>(
          "launch_block_reduce: empty-domain queue.fill failed: {}", ex.what());
    }
    return;
  }

  constexpr int MemX = SBlock::MemX;
  constexpr int MemY = SBlock::MemY;
  constexpr int MemZ = SBlock::MemZ;
  constexpr int VoxelCount = SBlock::VoxelCount;

  const int64_t total_voxels = static_cast<int64_t>(iter_blocks) * static_cast<int64_t>(VoxelCount);

  // AdaptiveCpp 25.02 was observed to ignore the initialize_to_identity property
  // on USM pointers in some shared-context multi-GPU configurations: the runtime
  // would atomically combine into @c *out without first resetting it, so any
  // stale device-memory value would be folded into the reduction. Explicit
  // queue.fill before the parallel_for forces the destination to start from
  // @p init , matching the documented semantics regardless of runtime quirks.
  try {
    queue.fill(out, init, 1U);
  } catch (const sycl::exception& ex) {
    ::carbon::log::throw_with_log<std::runtime_error>(
        "launch_block_reduce: queue.fill(out=init) failed: {}", ex.what());
  }

  try {
    queue.submit([&](sycl::handler& cgh) {
      auto reducer =
          sycl::reduction(out, init, op, sycl::property::reduction::initialize_to_identity{});
      cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_voxels)), reducer,
                       [=](sycl::id<1> i, auto& red) {
                         const int64_t flat = static_cast<int64_t>(i[0]);
                         // brick slowest, then z, y, x (x fastest) -- same ordering as
                         // the CUDA block_grid_reduce_kernel.
                         const int64_t ix = flat % static_cast<int64_t>(MemX);
                         int64_t rest = flat / static_cast<int64_t>(MemX);
                         const int64_t iy = rest % static_cast<int64_t>(MemY);
                         rest = rest / static_cast<int64_t>(MemY);
                         const int64_t iz = rest % static_cast<int64_t>(MemZ);
                         const int64_t iter_b = rest / static_cast<int64_t>(MemZ);
                         const uint32_t b =
                             span.view_block_idx(view, static_cast<uint32_t>(iter_b));
                         Idx idx{};
                         // Every work-item calls red.combine exactly once -- inactive
                         // voxels contribute @p init (the operator's identity) so the
                         // reduction is unaffected. This avoids skip-based reductions
                         // some SYCL runtimes implement non-conformantly (AdaptiveCpp
                         // 25.02's hipSYCL backend was observed to over-count under
                         // skip-then-combine; calling combine unconditionally is the
                         // documented safe pattern).
                         T contribution = init;
                         if (span.set_and_validate(b, static_cast<int>(ix), static_cast<int>(iy),
                                                   static_cast<int>(iz), idx)) {
                           contribution = fn(span, idx);
                         }
                         red.combine(contribution);
                       });
    });
  } catch (const sycl::exception& ex) {
    ::carbon::log::throw_with_log<std::runtime_error>(
        "launch_block_reduce: SYCL parallel_for submission failed: {}", ex.what());
  }
}

}  // namespace carbon::detail::block
