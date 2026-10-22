#pragma once

/**
 * Brick-index volume for the block-sparse grid families ( @c Block and, one per level,
 * @c MultiResBlock ).
 *
 * A @c Block knows, for a voxel it is already visiting, everything about that voxel's
 * neighbourhood: the per-block bitmask answers "is this voxel active" and the 27-way
 * connectivity table answers "which brick sits one step away". What it cannot answer is
 * the inverse question — @e which cell contains an arbitrary point — because the only
 * topology it carries is a flat @c Span::origins() list, so the query degrades to a linear
 * scan over every retained brick.
 *
 * @ref Locator closes that gap with a dense @c uint32_t index over the brick lattice:
 * one entry per brick holding the storage @c block_idx , or @ref Locator::k_no_block .
 * At the default @c 8×8×8 brick that is 4 bytes per 512 voxels — 0.2% of a single-component
 * @c float field — and the same table doubles as the empty-space-skipping structure a
 * ray-march DDA walks, so it is both the locator and the acceleration structure.
 *
 * @par Ownership and multi-GPU
 * The table is @b slab-local and covers this GPU's owned bricks @b plus its ghost ring, and
 * those two roles must not be confused. @ref owned_z_lo / @ref owned_z_hi carry the owned
 * brick-row band in world units, and that band is what a caller integrating along a ray must
 * clip to: the owned bands are disjoint across GPUs and ranks and tile the lattice, so
 * per-slab results composite without double counting. The ghost entries exist for the other
 * role — a trilinear gather anchored at an owned cell needs corners one voxel past the seam,
 * and without them every slab boundary would fall back to a lower-order value and leave a
 * visible seam in the render. This is exactly the split @c Dense draws between its owned band
 * and its haloed storage band. Ghost cells hold stale data until a @c HaloBlock update runs,
 * same as a dense halo.
 *
 * The consequence to respect: @ref locate succeeds for a point in the ghost ring on @b two
 * GPUs (its owner and its ghoster). Anything accumulating across GPUs must therefore gate on
 * the owned band rather than on @c locate 's validity alone.
 *
 * @par World transform
 * @c Block has no intrinsic world frame — @c Span::global_location returns integer voxel
 * coordinates, and @c Span::spacing() means something different here than on @c Dense (an
 * integer refinement factor, not a world size). The world frame therefore lives on the
 * locator: @ref world_origin is the world position of the level's voxel @c (0, 0, 0) corner
 * and @ref world_spacing the world size of one voxel at this level ( @c R^level in an mGrid
 * tower). The table itself is transform-independent, so one built table serves any frame.
 */

#include <carbon/cuda_annotation.h>
#include <carbon/detail/block/idx.h>
#include <carbon/detail/block/span.h>
#include <carbon/floating3d.h>
#include <carbon/integer3d.h>

#include <stdint.h>
#include <cmath>

namespace carbon::detail::block {

/**
 * Host-side geometry of one GPU's brick-index table. Kept separate from @ref Locator so
 * @c Block can cache the lattice shape once and stamp out device views with different
 * world frames without rebuilding the table.
 */
struct LocatorGeometry {
  /** Table extent in bricks. Zero on a GPU whose slab retained no brick. */
  int64_3d lattice{};
  /**
   * Brick coordinates of table entry @c (0, 0, 0) : the retained-brick bounding box corner
   * grown by one brick on every side so the ghost ring has somewhere to live.
   */
  int64_3d lattice_origin{};
  /** Voxel extent of the whole level, used to reject out-of-domain queries. */
  int64_3d voxel_extent{};
  /** Owned brick-row band @c [begin, end) in the global brick lattice. */
  int64_t owned_bz_begin{0};
  int64_t owned_bz_end{0};

  /** Number of @c uint32_t entries this GPU's table occupies. */
  [[nodiscard]] auto
  entry_count() const noexcept -> int64_t {
    return lattice.x * lattice.y * lattice.z;
  }
};

/**
 * Device-visible view over one GPU's brick-index table plus the world frame it is read in.
 *
 * Aggregate on purpose (same shape as @c carbon::Camera ): kernel loading lambdas build one
 * by designated initialisation and capture it by value.
 */
template <typename SBlock>
struct Locator {
  using SpanT = Span<SBlock>;

  static constexpr int MemX = SBlock::MemX;
  static constexpr int MemY = SBlock::MemY;
  static constexpr int MemZ = SBlock::MemZ;

  /** Table entry for a brick that is not retained on this GPU. */
  static constexpr uint32_t k_no_block = kInvalidBlockIdx;

  const uint32_t* table{nullptr};
  int64_3d lattice{};
  int64_3d lattice_origin{};
  int64_3d voxel_extent{};
  float32_3d world_origin{};
  float world_spacing{1.0F};
  float owned_z_lo{0.0F};
  float owned_z_hi{0.0F};

  /** @c false on a GPU whose slab retained no brick (every query then misses). */
  [[nodiscard]] CARBON_HD constexpr auto
  valid() const noexcept -> bool {
    return table != nullptr && lattice.x > 0 && lattice.y > 0 && lattice.z > 0;
  }

  /**
   * Continuous voxel-space coordinate of @p world . Voxel @c i spans @c [i, i + 1) , so its
   * centre sits at @c i + 0.5 — the same convention @c Dense uses.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  to_voxel_space(float32_3d world) const noexcept -> float32_3d {
    const float inv = 1.0F / world_spacing;
    return {
        (world.x - world_origin.x) * inv,
        (world.y - world_origin.y) * inv,
        (world.z - world_origin.z) * inv,
    };
  }

  /** World position of the centre of voxel @p voxel . */
  [[nodiscard]] CARBON_HD constexpr auto
  voxel_centre(int64_3d voxel) const noexcept -> float32_3d {
    return {
        world_origin.x + (static_cast<float>(voxel.x) + 0.5F) * world_spacing,
        world_origin.y + (static_cast<float>(voxel.y) + 0.5F) * world_spacing,
        world_origin.z + (static_cast<float>(voxel.z) + 0.5F) * world_spacing,
    };
  }

  /** World-space axis-aligned box of the whole level. */
  [[nodiscard]] CARBON_HD constexpr auto
  world_bounds_min() const noexcept -> float32_3d {
    return world_origin;
  }

  [[nodiscard]] CARBON_HD constexpr auto
  world_bounds_max() const noexcept -> float32_3d {
    return {
        world_origin.x + static_cast<float>(voxel_extent.x) * world_spacing,
        world_origin.y + static_cast<float>(voxel_extent.y) * world_spacing,
        world_origin.z + static_cast<float>(voxel_extent.z) * world_spacing,
    };
  }

  /**
   * Storage @c block_idx of the brick at brick coordinates @p bx , @p by , @p bz , or
   * @ref k_no_block when that brick is outside this GPU's table or was not retained.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  brick_at(int64_t bx,
           int64_t by,
           int64_t bz) const noexcept -> uint32_t {
    if (table == nullptr) {
      return k_no_block;
    }
    const int64_t lx = bx - lattice_origin.x;
    const int64_t ly = by - lattice_origin.y;
    const int64_t lz = bz - lattice_origin.z;
    if (lx < 0 || ly < 0 || lz < 0 || lx >= lattice.x || ly >= lattice.y || lz >= lattice.z) {
      return k_no_block;
    }
    return table[(lz * lattice.y + ly) * lattice.x + lx];
  }

  /** Brick coordinates containing voxel @p voxel (assumes non-negative components). */
  [[nodiscard]] CARBON_HD constexpr auto
  voxel_to_brick(int64_3d voxel) const noexcept -> int64_3d {
    return {voxel.x / MemX, voxel.y / MemY, voxel.z / MemZ};
  }

  /** @c true when @p voxel lies inside the level's voxel domain. */
  [[nodiscard]] CARBON_HD constexpr auto
  in_domain(int64_3d voxel) const noexcept -> bool {
    return voxel.x >= 0 && voxel.y >= 0 && voxel.z >= 0 && voxel.x < voxel_extent.x &&
           voxel.y < voxel_extent.y && voxel.z < voxel_extent.z;
  }

  /**
   * Resolve the cell containing voxel @p voxel into an @c Idx on @p span .
   *
   * The returned @c ValidatedIdx is invalid when the voxel is outside the domain, when its
   * brick was not retained on this GPU, or when the voxel itself is bit-clear — the same
   * three-way outcome @c Span::set_and_validate produces for a cell the kernel is visiting.
   */
  /**
   * Resolve a global voxel to a cell on this GPU. Invalid when the voxel is outside the
   * domain, its brick is neither owned nor ghosted here, or its bit is clear.
   *
   * Validates through @c Span::bitmask directly rather than through
   * @c Span::set_and_validate , which rejects any slot at or beyond the @b owned block
   * count and so would reject every ghost. This mirrors what @c Partition::neighbour
   * already does to walk across a slab edge — the device bitmask array covers ghost slots
   * precisely so that read is legal.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  locate(const SpanT& span,
         int64_3d voxel) const noexcept -> ValidatedIdx {
    ValidatedIdx out{};
    if (!in_domain(voxel)) {
      return out;
    }
    const int64_3d brick = voxel_to_brick(voxel);
    const uint32_t block_idx = brick_at(brick.x, brick.y, brick.z);
    if (block_idx == k_no_block) {
      return out;
    }
    const int ix = static_cast<int>(voxel.x - brick.x * MemX);
    const int iy = static_cast<int>(voxel.y - brick.y * MemY);
    const int iz = static_cast<int>(voxel.z - brick.z * MemZ);
    const typename SBlock::BitMask* mask = span.bitmask();
    if (mask == nullptr || !mask[block_idx].is_active(ix, iy, iz)) {
      return out;
    }
    out.idx =
        Idx{block_idx, static_cast<int8_t>(ix), static_cast<int8_t>(iy), static_cast<int8_t>(iz)};
    out.valid = true;
    return out;
  }

  /** Convenience overload taking a world position. */
  [[nodiscard]] CARBON_HD auto
  locate_world(const SpanT& span,
               float32_3d world) const noexcept -> ValidatedIdx {
    const float32_3d v = to_voxel_space(world);
    return locate(
        span, int64_3d{static_cast<int64_t>(std::floor(v.x)), static_cast<int64_t>(std::floor(v.y)),
                       static_cast<int64_t>(std::floor(v.z))});
  }
};

}  // namespace carbon::detail::block
