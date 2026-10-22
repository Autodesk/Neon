#pragma once

#include <carbon/cuda_annotation.h>
#include <carbon/integer3d.h>

#include <stdint.h>

namespace carbon::detail::mgrid {

/**
 * Level layout descriptor for the multi-resolution (mGrid) grid family.
 *
 * Trimmed port of Neon's @c mGridDescriptor
 * (@c Neon/libNeonDomain/include/Neon/domain/details/mGrid/mGridDescriptor.h ): this pass
 * supports a @b uniform refinement factor across every level (the common octree case),
 * with a runtime @c depth . Per-level refinement factors are a documented follow-up — the
 * @c ref_factor(level) accessor already takes a level so generalising later does not change
 * call sites.
 *
 * @par Conventions (level 0 is the finest)
 *
 * - @c ref_factor : the per-axis branching factor @c R between adjacent levels. One parent
 *   voxel at level @c L+1 has @c R·R·R child voxels at level @c L . For an octree @c R == 2 .
 *   Neon couples this to the brick policy by requiring @c R == @c SBlock::UserX (the user
 *   block size, @b not @c BlockRatio ); @c MultiResBlock enforces the same.
 * - @c voxel_spacing(level) = @c R^level : the size of a level- @c level voxel measured in
 *   finest-level (level-0) voxels. @c voxel_spacing(0) == 1 . Matches @c Span::spacing() from
 *   the pre-mGrid PR3 pass.
 * - @c spacing(level) = @c R^(level+1) : Neon's @c getSpacing — used by the cross-level
 *   @c get_parent arithmetic in finest units. Provided for parity / completeness.
 * - @c level_extent(finest, level) = @c ceil_div(finest, R^level) per axis: the voxel extent
 *   of level @c level over a domain whose finest-level extent is @c finest . The ceil-div
 *   identity @c ceil(ceil(n/a)/b) == ceil(n/(a·b)) makes these consistent across levels, so a
 *   block origin mapped from one level to an adjacent one always lands in-bounds.
 */
class MultiResDescriptor {
  int depth_{1};
  int ref_factor_{2};

public:
  CARBON_HD constexpr MultiResDescriptor() noexcept = default;

  CARBON_HD constexpr MultiResDescriptor(int depth,
                                         int ref_factor = 2) noexcept
      : depth_(depth),
        ref_factor_(ref_factor) {}

  /** Number of levels in the tower (level 0 finest, @c depth-1 coarsest). */
  [[nodiscard]] CARBON_HD constexpr auto
  depth() const noexcept -> int {
    return depth_;
  }

  /**
   * Per-axis branching factor between level @p level and its parent @c level+1 . Uniform in
   * this pass, so independent of @p level ; the parameter is kept for forward compatibility
   * with per-level factors.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  ref_factor([[maybe_unused]] int level = 0) const noexcept -> int {
    return ref_factor_;
  }

  /** Size of a level- @p level voxel in finest-level (level-0) voxels: @c R^level . */
  [[nodiscard]] CARBON_HD constexpr auto
  voxel_spacing(int level) const noexcept -> int64_t {
    int64_t s = 1;
    for (int l = 0; l < level; ++l) {
      s *= static_cast<int64_t>(ref_factor_);
    }
    return s;
  }

  /** Neon's @c getSpacing : @c R^(level+1) (finest-level units). */
  [[nodiscard]] CARBON_HD constexpr auto
  spacing(int level) const noexcept -> int64_t {
    return voxel_spacing(level) * static_cast<int64_t>(ref_factor_);
  }

  /** Voxel extent of @p level given a finest-level extent @p finest : @c ceil_div per axis. */
  [[nodiscard]] CARBON_HD constexpr auto
  level_extent(const int64_3d& finest,
               int level) const noexcept -> int64_3d {
    const int64_t s = voxel_spacing(level);
    return int64_3d{ceil_div_(finest.x, s), ceil_div_(finest.y, s), ceil_div_(finest.z, s)};
  }

private:
  [[nodiscard]] CARBON_HD constexpr static auto
  ceil_div_(int64_t a,
            int64_t b) noexcept -> int64_t {
    return (a + b - 1) / b;
  }
};

}  // namespace carbon::detail::mgrid
