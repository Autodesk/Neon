#pragma once

/**
 * SYCL twin of @c detail/backend/cudalike/block_launch.h . Maps the
 * @c (block_count, 1, 1) × @c (MemX, MemY, MemZ) CUDA launch shape onto a
 * @c sycl::nd_range<3> :
 *
 *   - SYCL dim 0 (slowest) = block_idx
 *   - SYCL dim 1           = z (intra-block)
 *   - SYCL dim 2 (fastest) = x (intra-block)... wait
 *
 * Actually we mirror the CUDA mapping: thread @c (ix, iy, iz) is fastest -> slowest in
 * carbon (x fastest), so SYCL dim 2 = ix (fastest), dim 1 = iy, dim 0 = iz; the CUDA
 * grid axis @c blockIdx.x (block_idx) is folded onto dim 0 by multiplying its extent
 * into the slowest local dim. Concretely:
 *
 *   global = (block_count * MemZ, MemY, MemX)
 *   local  = (MemZ, MemY, MemX)
 *   block_idx = global_id(0) / MemZ
 *   iz        = global_id(0) % MemZ
 *   iy        = global_id(1)
 *   ix        = global_id(2)
 *
 * Same trick the SYCL @c launch_dense_grid uses to fold a 3D CUDA grid into a 3D
 * @c nd_range : SYCL has the same dim count as CUDA so we can keep the shape parallel.
 */

#include <carbon/block.h>
#include <carbon/data_view.h>
#include <carbon/detail/backend/sycl/block_kernel.h>
#include <carbon/logging.h>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace carbon::detail::block {

template <typename SBlock,
          typename F>
requires block_kernel_fn<F,
                         SBlock> inline auto
launch_block_grid(const ::carbon::Block<SBlock>& grid,
                  size_t relative_gpu_id,
                  size_t shared_mem,
                  sycl::queue& queue,
                  ::carbon::DataView view,
                  F fn) -> void {
  if (relative_gpu_id >= grid.system().gpu_group().size()) {
    ::carbon::log::throw_with_log<std::out_of_range>(
        "launch_block_grid: relative_gpu_id out of range (relative_gpu_id={}, gpu_count={})",
        relative_gpu_id, grid.system().gpu_group().size());
  }
  if (shared_mem != 0U) {
    ::carbon::log::throw_with_log<std::runtime_error>(
        "launch_block_grid: dynamic shared memory ({} bytes) is not yet supported under the "
        "SYCL backend.",
        shared_mem);
  }
  const auto span = grid.span(relative_gpu_id);
  const uint32_t iter_block_count = span.iter_block_count(view);
  if (iter_block_count == 0U) {
    return;
  }
  const sycl::range<3> local{static_cast<size_t>(SBlock::MemZ), static_cast<size_t>(SBlock::MemY),
                             static_cast<size_t>(SBlock::MemX)};
  const sycl::range<3> global{
      static_cast<size_t>(iter_block_count) * static_cast<size_t>(SBlock::MemZ),
      static_cast<size_t>(SBlock::MemY), static_cast<size_t>(SBlock::MemX)};
  try {
    queue.parallel_for(sycl::nd_range<3>{global, local}, [=](sycl::nd_item<3> it) {
      const auto g0 = it.get_global_id(0);
      const uint32_t iter_b = static_cast<uint32_t>(g0 / static_cast<size_t>(SBlock::MemZ));
      const uint32_t b = span.view_block_idx(view, iter_b);
      const int iz = static_cast<int>(g0 % static_cast<size_t>(SBlock::MemZ));
      const int iy = static_cast<int>(it.get_global_id(1));
      const int ix = static_cast<int>(it.get_global_id(2));
      Idx idx{};
      if (!span.set_and_validate(b, ix, iy, iz, idx)) {
        return;
      }
      fn(span, idx);
    });
  } catch (const sycl::exception& ex) {
    ::carbon::log::throw_with_log<std::runtime_error>(
        "launch_block_grid: SYCL parallel_for submission failed (relative_gpu_id={}): {}",
        relative_gpu_id, ex.what());
  }
}

}  // namespace carbon::detail::block
