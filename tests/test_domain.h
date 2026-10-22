#pragma once

/**
 * Domain-sizing helpers for tests that build a grid across *every* visible GPU.
 *
 * Every grid family refuses a domain it cannot partition: @c Dense wants at least
 * @c 2 * halo_z owned z planes per slab (otherwise the Internal and Boundary bands
 * overlap), and @c Block / @c MultiResBlock want at least one brick row per slab. A test
 * that hard-codes its extent therefore silently caps the GPU count it can run on, and
 * fails with a precondition error above it — which is what the whole @c mgrid / @c block /
 * @c skeleton group did on an 8-GPU host while passing on 1 and 2.
 *
 * These helpers only ever grow the extent, so the shape a test asks for is preserved on the
 * small GPU counts it was written against.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace carbon::test {

/**
 * @brief Smallest z extent >= @p desired that leaves every slab enough owned planes for the
 *        Internal / Boundary split.
 *
 * @c Dense hands each of @p n_gpus slabs @c floor(z / n_gpus) owned planes and rejects any
 * slab with fewer than @c 2 * halo_z of them.
 */
[[nodiscard]] inline auto
dense_slab_z(int64_t desired,
             size_t n_gpus,
             int64_t halo_z = 1) -> int64_t {
  const int64_t needed = static_cast<int64_t>(n_gpus) * 2 * halo_z;
  return std::max(desired, needed);
}

/**
 * @brief Smallest z extent >= @p desired that leaves every slab at least one brick row.
 *
 * @p brick_z is the z span of one brick in cells: @c SBlock::MemZ for @c Block , and for
 * @c MultiResBlock the *coarsest* level's span, @c MemZ * ref_factor^(depth - 1) — that is
 * the level whose brick count is checked against the slab count. The result is a multiple of
 * @p brick_z whenever @p desired is, so callers keep any alignment their grid requires.
 */
[[nodiscard]] inline auto
block_slab_z(int64_t desired,
             size_t n_gpus,
             int64_t brick_z) -> int64_t {
  const int64_t needed = brick_z * static_cast<int64_t>(n_gpus);
  return std::max(desired, needed);
}

}  // namespace carbon::test
