#pragma once

#include <carbon/cuda_annotation.h>
#include <carbon/integer3d.h>

#include <stdint.h>

namespace carbon::detail::dense_disg {

/**
 * Local logical index for a CUDA thread / cell on a @b disaggregated dense grid. Mirrors
 * Neon's @c dIndex from @c Neon::domain::details::disaggregated::dGrid (under
 * @c Neon-disg/libNeonDomain/include/Neon/domain/details/dGridDisg/dIndex.h).
 *
 * The disaggregated layout splits each per-GPU buffer into five physically disjoint
 * z-regions — lower ghost, down boundary, internal, up boundary, upper ghost — and lays
 * each region out per-cardinality (per-region SoA). Resolving a cell in that layout needs
 * three precomputed numbers in addition to the local @c (x, y, z):
 *
 * - @c offset_local_no_card : in-region xy/inner-stack offset (in elements, ignoring the
 *   cardinality dimension). For the boundary / ghost regions this is just
 *   @c x + y * dim.x ; for the internal region it additionally folds the in-region z slab
 *   index, @c x + y * dim.x + (z_in_region) * (dim.x * dim.y) .
 * - @c region_first_z : index of the first z-slab of the cell's region, expressed in the
 *   @b owned-after-shift coordinate system (lower ghost = 0, down boundary = 1, internal
 *   = 2, up boundary = z, upper ghost = z+1, where @c z is the owned z-thickness). The
 *   pitch formula multiplies this by @c xy_pitch * components to step past every region
 *   that precedes the current one.
 * - @c region_z_dim : number of z-slabs in the cell's region (1 for boundary / ghost, the
 *   inner z-thickness for the internal region). Multiplied by @c xy_pitch in the pitch
 *   formula to step from one cardinality plane of this region to the next.
 *
 * @ref Span @c set_and_validate populates all three fields once at thread-launch time;
 * @ref Partition::read uses them to compute the final byte offset without any branches.
 *
 * @par Why a class wrapping these fields?
 *
 * Symmetric reason to @c carbon::detail::dense::Idx : the kernel-facing @c Idx is the
 * extension point each grid family customises. The aggregated dense grid carries a 3D
 * cell only; the disaggregated dense grid additionally carries the three region fields
 * above. Sparse / block / multi-resolution grids will carry their own augmentations.
 * Storing them on @c Idx (instead of recomputing them from @c (x, y, z) on every neighbour
 * read) matches Neon's design and keeps the per-cell device math branch-free.
 */
class Idx {
  int64_3d cell_{};
  int64_t offset_local_no_card_{};
  int32_t region_first_z_{};
  int32_t region_z_dim_{};

public:
  CARBON_HD constexpr Idx() noexcept = default;

  /**
   * @param ix Local logical x.
   * @param iy Local logical y.
   * @param iz Buffer-relative z (already shifted into the haloed buffer's z coordinate
   *           system by @ref Span::set_and_validate).
   * @param offset_local_no_card In-region xy/inner-stack offset; see class docs.
   * @param region_first_z First z-slab of the cell's region; see class docs.
   * @param region_z_dim Number of z-slabs in the cell's region; see class docs.
   */
  CARBON_HD constexpr Idx(int64_t ix,
                          int64_t iy,
                          int64_t iz,
                          int64_t offset_local_no_card,
                          int32_t region_first_z,
                          int32_t region_z_dim) noexcept
      : cell_{ix,
              iy,
              iz},
        offset_local_no_card_(offset_local_no_card),
        region_first_z_(region_first_z),
        region_z_dim_(region_z_dim) {}

  /** Local extent holding @c (ix, iy, iz). */
  [[nodiscard]] CARBON_HD constexpr auto
  cell() const noexcept -> int64_3d {
    return cell_;
  }

  /** In-region xy/inner-stack offset (in elements, ignoring the cardinality dimension). */
  [[nodiscard]] CARBON_HD constexpr auto
  offset_local_no_card() const noexcept -> int64_t {
    return offset_local_no_card_;
  }

  /** Index of the first z-slab of the cell's region, in owned-after-shift coordinates. */
  [[nodiscard]] CARBON_HD constexpr auto
  region_first_z() const noexcept -> int32_t {
    return region_first_z_;
  }

  /** Number of z-slabs in the cell's region. */
  [[nodiscard]] CARBON_HD constexpr auto
  region_z_dim() const noexcept -> int32_t {
    return region_z_dim_;
  }
};

}  // namespace carbon::detail::dense_disg
