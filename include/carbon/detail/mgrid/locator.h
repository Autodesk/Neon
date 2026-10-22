#pragma once

/**
 * Multi-resolution brick-index tower: one @c detail::block::Locator per refinement level,
 * bundled with that level's @c Span so a device function can answer "which cell of which
 * level contains this point" without any traversal structure of its own.
 *
 * The tower is the mGrid analogue of the single-level locator described in
 * @c detail/block/locator.h , and inherits its ownership contract unchanged: each level's
 * table is slab-local and indexes owned bricks only, so the owned world-z band still
 * bounds what a ray on this GPU may integrate. Because mGrid keeps a parent brick and all
 * its descendants co-resident on one GPU (the rule that makes cross-level access
 * transfer-free), a point resident on this slab at one level is resident at every level,
 * and @ref MultiResLocator::find_finest never has to leave the device to descend.
 *
 * Levels share one world frame: @c world_origin is common and level @c L 's voxel is
 * @c base_spacing · R^L wide, which is exactly what @c MultiResDescriptor::voxel_spacing
 * reports in finest-level units.
 */

#include <carbon/cuda_annotation.h>
#include <carbon/detail/block/locator.h>
#include <carbon/detail/block/span.h>
#include <carbon/floating3d.h>
#include <carbon/integer3d.h>

#include <stdint.h>

namespace carbon::detail::mgrid {

/**
 * Device-visible per-level locator + span bundle.
 *
 * @tparam SBlock    Brick policy, shared by every level of the tower.
 * @tparam MaxLevels Compile-time cap on tower depth. Sized for the depths mGrid towers use
 *                   in practice (2-4); the struct is captured by value into kernels, so the
 *                   cap keeps it out of the parameter-space limit. @c MultiResBlock throws
 *                   when a deeper tower asks for a locator.
 */
template <typename SBlock, int MaxLevels = 6>
struct MultiResLocator {
  using LocatorT = ::carbon::detail::block::Locator<SBlock>;
  using SpanT = ::carbon::detail::block::Span<SBlock>;
  using ValidatedIdx = ::carbon::detail::block::ValidatedIdx;

  static constexpr int k_max_levels = MaxLevels;

  LocatorT level_locator[MaxLevels]{};
  SpanT level_span[MaxLevels]{};
  int depth{0};

  [[nodiscard]] CARBON_HD constexpr auto
  valid() const noexcept -> bool {
    return depth > 0;
  }

  /**
   * Finest level holding an active cell at @p world .
   *
   * Probes finest-first and stops at the first hit, so a point inside a refined region
   * resolves to the finest data available there and a point in the coarse far field falls
   * through to the level that actually stores it. Returns @c false when no level has an
   * active cell at @p world (outside the domain, or in a hole).
   *
   * @param world     World-space query position.
   * @param out_level Receives the resolved level on success; untouched otherwise.
   * @param out_idx   Receives the resolved cell on success; untouched otherwise.
   */
  [[nodiscard]] CARBON_HD auto
  find_finest(float32_3d world,
              int& out_level,
              ::carbon::detail::block::Idx& out_idx) const noexcept -> bool {
    for (int l = 0; l < depth; ++l) {
      const ValidatedIdx hit = level_locator[l].locate_world(level_span[l], world);
      if (hit.valid) {
        out_level = l;
        out_idx = hit.idx;
        return true;
      }
    }
    return false;
  }

  /**
   * Coarsest-first variant: useful when a caller wants the level whose cell size best
   * matches a screen-space footprint rather than the finest data available.
   */
  [[nodiscard]] CARBON_HD auto
  find_coarsest(float32_3d world,
                int& out_level,
                ::carbon::detail::block::Idx& out_idx) const noexcept -> bool {
    for (int l = depth - 1; l >= 0; --l) {
      const ValidatedIdx hit = level_locator[l].locate_world(level_span[l], world);
      if (hit.valid) {
        out_level = l;
        out_idx = hit.idx;
        return true;
      }
    }
    return false;
  }

  /** Owned world-z band of the tower, taken from the finest level (all levels agree). */
  [[nodiscard]] CARBON_HD constexpr auto
  owned_z_lo() const noexcept -> float {
    return depth > 0 ? level_locator[0].owned_z_lo : 0.0F;
  }

  [[nodiscard]] CARBON_HD constexpr auto
  owned_z_hi() const noexcept -> float {
    return depth > 0 ? level_locator[0].owned_z_hi : 0.0F;
  }
};

}  // namespace carbon::detail::mgrid
