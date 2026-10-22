#pragma once

/**
 * SYCL twin of @c detail/backend/cudalike/block_kernel.h. Provides the same
 * @c block_kernel_fn concept; the actual launch is wired in
 * @c detail/backend/sycl/block_launch.h via @c sycl::queue::parallel_for over an
 * @c nd_range that mirrors the CUDA @c (block_count, 1, 1) × @c (MemX, MemY, MemZ)
 * launch shape.
 */

#include <carbon/detail/block/idx.h>
#include <carbon/detail/block/span.h>
#include <carbon/detail/block/static_block.h>

#include <concepts>

namespace carbon::detail::block {

template <typename F, typename SBlock>
concept block_kernel_fn = requires(F fn, Span<SBlock> span, Idx idx) {
  {fn(span, idx)}->std::same_as<void>;
};

}  // namespace carbon::detail::block
