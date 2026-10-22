#pragma once

/**
 * Single-GPU launch helper for @c carbon::detail::block::block_grid_kernel . Lives in
 * @c detail/ so that the public @c carbon/block.h does not have to mention any
 * backend stream / error type. Symmetric with @c detail/backend/cudalike/dense_launch.h .
 *
 * Internal API: only @c carbon::detail::BlockKernelImpl reaches in here. User code
 * composes a @c carbon::Kernel via @c Block::make_kernel / @c Kernel::compute and
 * calls @c Kernel::launch — never @c launch_block_grid directly.
 */

#include <carbon/block.h>
#include <carbon/data_view.h>
#include <carbon/detail/backend/cudalike/block_kernel.h>
#include <carbon/detail/gpu.h>
#include <carbon/logging.h>

#include <carbon/detail/backend/cuda_like.h>

#include <cstddef>
#include <stdexcept>

namespace carbon::detail::block {

template <typename>
inline constexpr bool block_launch_requires_cudacc_ = false;

#if CARBON_CUDALIKE_DEVICE_TU

/**
 * @brief Launch @ref block_grid_kernel on @p relative_gpu_id .
 *
 * @param grid             The @c Block layout this launch is bound to.
 * @param relative_gpu_id  Index into @c grid.system().gpu_group().devices() . The
 *                         caller is expected to have already pushed the matching
 *                         CUDA context (typically via @c detail::Gpu::push_context )
 *                         and to have selected the per-GPU stream out of the public
 *                         @c Stream group.
 * @param shared_mem       Dynamic shared memory in bytes per launch.
 * @param stream           CUDA stream to enqueue on; the caller has already picked the
 *                         per-GPU handle out of the public @c Stream group.
 * @param fn               Per-cell device callable; must satisfy
 *                         @c block_kernel_fn<F, SBlock> .
 *
 * @return @c gpu_error_t result of @c gpu_runtime_launch_kernel — non-success values are
 *         surfaced to the user through @c BlockKernelImpl 's error path.
 */
template <typename SBlock,
          typename F>
requires block_kernel_fn<F,
                         SBlock> [[nodiscard]] inline auto
launch_block_grid(const ::carbon::Block<SBlock>& grid,
                  size_t relative_gpu_id,
                  size_t shared_mem,
                  gpu_stream_t stream,
                  ::carbon::DataView view,
                  F fn) -> gpu_error_t {
  if (relative_gpu_id >= grid.system().gpu_group().size()) {
    ::carbon::log::throw_with_log<std::out_of_range>(
        "launch_block_grid: relative_gpu_id out of range (relative_gpu_id={}, gpu_count={})",
        relative_gpu_id, grid.system().gpu_group().size());
  }
  const auto span = grid.span(relative_gpu_id);
  const uint32_t iter_block_count = span.iter_block_count(view);
  if (iter_block_count == 0U) {
    return gpu_success;
  }
  const ::carbon::detail::Gpu& gpu = grid.system().gpu_group().devices()[relative_gpu_id];
  const dim3 grid_dim(iter_block_count, 1U, 1U);
  const dim3 block_dim(static_cast<unsigned int>(SBlock::MemX),
                       static_cast<unsigned int>(SBlock::MemY),
                       static_cast<unsigned int>(SBlock::MemZ));
  return gpu.template typed_launch<&block_grid_kernel<SBlock, F>>(grid_dim, block_dim, shared_mem,
                                                                  stream, span, view, fn);
}

#else

template <typename SBlock,
          typename F>
requires block_kernel_fn<F,
                         SBlock> [[nodiscard]] inline auto
launch_block_grid(const ::carbon::Block<SBlock>&,
                  size_t,
                  size_t,
                  gpu_stream_t,
                  ::carbon::DataView,
                  F) -> gpu_error_t {
  static_assert(
      block_launch_requires_cudacc_<F>,
      "carbon::detail::block::launch_block_grid is only supported in translation "
      "units compiled with nvcc (CARBON_CUDALIKE_DEVICE_TU). Move the call site to a .cu file (or "
      "build the TU with nvcc) and include <carbon/kernel.h> from there.");
  return gpu_error_not_supported;
}

#endif

}  // namespace carbon::detail::block
