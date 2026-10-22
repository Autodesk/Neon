#pragma once

#include <carbon/boundary_mode.h>
#include <carbon/data_view.h>
#include <carbon/detail/dense_disg/idx.h>
#include <carbon/detail/validated_idx.h>
#include <carbon/floating3d.h>

namespace carbon::detail::dense_disg {

/**
 * Result of @ref Span::set_and_validate : a buffer-relative @ref Idx (with the disg region
 * fields already populated) and a flag indicating whether the originating launch-grid
 * thread fell inside @c iter_extent(view) . Aliased to the cross-family
 * @c carbon::detail::Validated template so every grid family's value-returning
 * @c set_and_validate shares the same return carrier shape (see
 * @c detail/validated_idx.h ); the local @c ValidatedIdx spelling is preserved
 * as a source-compat alias.
 */
using ValidatedIdx = ::carbon::detail::Validated<Idx>;

/**
 * Layout of one GPU's @b disaggregated dense subvolume. Mirrors Neon's @c dSpan (under
 * @c Neon-disg/libNeonDomain/include/Neon/domain/details/dGridDisg/dSpan.h ) and pairs with
 * @ref Idx + @ref Partition to form the disg port of carbon's dense primitives.
 *
 * @par Disaggregated z layout
 *
 * Each slab's storage is split into five physically disjoint z-regions, indexed below in
 * the @b owned-after-shift coordinate system (the haloed-buffer z system, where buffer-z=0
 * is the lower ghost):
 *
 * @code
 *   region            firstZ  zDim                 storage span (in xy planes)
 *   ---------------   ------  -------------------  ----------------------------------
 *   Lower ghost       0       1                    [0           ,   C            )
 *   Down boundary     1       1                    [   C        ,  2C            )
 *   Internal          2       Lz - 2               [  2C        ,  2C + C(Lz-2)  )
 *   Up boundary       Lz      1                    [Lz·C        ,  Lz·C + C      )
 *   Upper ghost       Lz+1    1                    [(Lz+1)·C    , (Lz+1)·C + C   )
 * @endcode
 *
 * where @c Lz is the @b owned z-thickness ( @c haloed_extent.z - 2·halo_z , equivalently
 * Neon's @c mMaxZInDomain ) and @c C is the per-cell component count. The total storage
 * is @c (Lz + 2)·C·xy_pitch elements per slab — same byte count as the aggregated dense
 * grid, just regrouped: each region's data is laid out per-cardinality, so the internal
 * block becomes a card-major SoA while the thin boundary / ghost slabs stay AoS-like.
 *
 * The pitch formula consumed by @ref Partition is:
 * @code
 *   addr = idx.offset_local_no_card()
 *        + xy_pitch * (idx.region_first_z() * components + card * idx.region_z_dim())
 * @endcode
 * mirroring Neon's @c dPartition::getPitch with its default pitch.
 *
 * @par Hard invariants
 *
 * The disg layout pins both the ghost radius and the boundary radius to 1 (the block
 * decomposition above does not generalise to thicker boundaries without changing the
 * region count). @c set_and_validate and @ref linear_offset assume @c halo_z == 1 ;
 * carbon's owning grid wrapper enforces this at construction.
 */
class Span {
  int64_3d haloed_extent_{};
  int64_3d global_extent_{};
  int64_t xy_slab_elements_{};
  int64_t z_offset_{};
  int64_t halo_z_{};
  BoundaryMode boundary_mode_{BoundaryMode::Open};
  float32_3d origin_{};
  float spacing_{1.0F};

public:
  /** Both ghost radius and boundary radius are pinned to 1 for the disaggregated layout. */
  static constexpr int64_t ghost_radius = 1;
  static constexpr int64_t boundary_radius = 1;

  CARBON_HD constexpr Span() noexcept = default;

  /**
   * @param haloed_extent     Storage extent: @c (owned.x, owned.y, owned.z + 2 * halo_z) .
   *                          @c halo_z must equal @ref ghost_radius (the only supported
   *                          radius in the disaggregated layout).
   * @param global_extent     Full Cartesian grid extent shared by every slab.
   * @param xy_slab_elements  Elements per XY plane (pitch in elements). Equals
   *                          @c haloed_extent.x * haloed_extent.y for the canonical
   *                          single-buffer layout.
   * @param z_offset          Global z-index of buffer-local @c z=0 (start of the lower
   *                          ghost slab).
   * @param halo_z            Halo thickness on each z side. Must equal @ref ghost_radius .
   * @param boundary_mode     Boundary semantics propagated from the owning grid; consumed
   *                          by the @c HaloDisg adjacency builder (PeriodicZ pairs the
   *                          top slab's upper face with the bottom slab's lower face) and
   *                          reserved for a future @c Partition::neighbour overload —
   *                          today @c dense_disg::Partition is read/write-only.
   * @param origin            Physical-space origin of global cell @c (0,0,0)'s corner.
   * @param spacing           Isotropic cell spacing (finite and > 0).
   */
  CARBON_HD constexpr Span(int64_3d haloed_extent,
                           int64_3d global_extent,
                           int64_t xy_slab_elements,
                           int64_t z_offset,
                           int64_t halo_z = ghost_radius,
                           BoundaryMode boundary_mode = BoundaryMode::Open,
                           float32_3d origin = {},
                           float spacing = 1.0F) noexcept
      : haloed_extent_(haloed_extent),
        global_extent_(global_extent),
        xy_slab_elements_(xy_slab_elements),
        z_offset_(z_offset),
        halo_z_(halo_z),
        boundary_mode_(boundary_mode),
        origin_(origin),
        spacing_(spacing) {}

  [[nodiscard]] CARBON_HD constexpr auto
  haloed_extent() const noexcept -> int64_3d {
    return haloed_extent_;
  }

  [[nodiscard]] CARBON_HD constexpr auto
  global_extent() const noexcept -> int64_3d {
    return global_extent_;
  }

  /** Owned-cell extent (Lz on z): @c (haloed.x, haloed.y, haloed.z - 2 * halo_z) . */
  [[nodiscard]] CARBON_HD constexpr auto
  logic_extent() const noexcept -> int64_3d {
    return int64_3d(haloed_extent_.x, haloed_extent_.y, haloed_extent_.z - 2 * halo_z_);
  }

  /**
   * Iteration extent for @p view , i.e. the shape of the launch grid the user should
   * iterate over. Mirrors Neon's @c dGrid_imp.h population of @c span.mSpanDim per
   * @c DataView : Standard covers every owned slab, Internal covers the inner band that
   * is independent of neighbour data ( @c Lz - 2·boundary_radius ), and Boundary covers
   * exactly the two boundary slabs ( @c 2·boundary_radius ).
   *
   * @c Internal + @c Boundary together cover every owned cell exactly once, same as a
   * single @c Standard launch.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  iter_extent(::carbon::DataView view) const noexcept -> int64_3d {
    const int64_t lz = haloed_extent_.z - 2 * halo_z_;
    switch (view) {
      case ::carbon::DataView::Standard:
        return int64_3d(haloed_extent_.x, haloed_extent_.y, lz);
      case ::carbon::DataView::Internal: {
        const int64_t inner = lz - 2 * boundary_radius;
        return int64_3d(haloed_extent_.x, haloed_extent_.y, inner > 0 ? inner : 0);
      }
      case ::carbon::DataView::Boundary:
        return int64_3d(haloed_extent_.x, haloed_extent_.y, 2 * boundary_radius);
    }
    return int64_3d(haloed_extent_.x, haloed_extent_.y, lz);
  }

  [[nodiscard]] CARBON_HD constexpr auto
  xy_slab_elements() const noexcept -> int64_t {
    return xy_slab_elements_;
  }

  [[nodiscard]] CARBON_HD constexpr auto
  z_offset() const noexcept -> int64_t {
    return z_offset_;
  }

  [[nodiscard]] CARBON_HD constexpr auto
  halo_z() const noexcept -> int64_t {
    return halo_z_;
  }

  [[nodiscard]] CARBON_HD constexpr auto
  boundary_mode() const noexcept -> BoundaryMode {
    return boundary_mode_;
  }

  [[nodiscard]] CARBON_HD constexpr auto
  origin() const noexcept -> float32_3d {
    return origin_;
  }

  [[nodiscard]] CARBON_HD constexpr auto
  spacing() const noexcept -> float {
    return spacing_;
  }

  /**
   * Total storage elements per slab: @c (Lz + 2) * components * xy_slab_elements() ,
   * matching the five-region map in the class docs. Multiply by @c sizeof(T) for bytes.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  linear_storage_elements(int components) const noexcept -> int64_t {
    return per_card_elements() * static_cast<int64_t>(components);
  }

  /**
   * Canonical (cross-family) spelling for the per-component buffer pitch:
   * @c haloed_extent_.z * xy_slab_elements_ , i.e. one cardinality plane's
   * worth of storage. Matches @c block::Span::per_card_elements and
   * @c dense::Span::per_card_elements ; new cross-grid code should prefer
   * this spelling over computing @c linear_storage_elements(1) .
   */
  [[nodiscard]] CARBON_HD constexpr auto
  per_card_elements() const noexcept -> int64_t {
    return haloed_extent_.z * xy_slab_elements_;
  }

  /**
   * Global location of @p idx within the grid: @c (cell.x, cell.y, cell.z + z_offset()) .
   * Note that the cell's @c z is in the haloed-buffer coordinate system, so adding
   * @c z_offset() lands directly in global coordinates.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  global_location(const Idx& idx) const noexcept -> int64_3d;

  /**
   * Element offset (in units of @c T ) from the start of a partition buffer to the cell
   * at @p idx for component @p component_id , using the disaggregated pitch:
   * @code
   *   addr = idx.offset_local_no_card()
   *        + xy_slab_elements() * (idx.region_first_z() * components
   *                                + component_id * idx.region_z_dim());
   * @endcode
   * Mirrors Neon's @c dPartition::getPitch . No bounds checking.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  linear_offset(const Idx& idx,
                int components,
                int component_id = 0) const noexcept -> int64_t;

  /**
   * Translate a launch-grid thread @p (ix, iy, iz) into a buffer-relative @ref Idx for
   * @p view and report whether the thread falls inside @c iter_extent(view) . Mirrors
   * Neon's @c dSpan::setAndValidate verbatim, including the per-region precomputation
   * of @c offset_local_no_card / @c region_first_z / @c region_z_dim that
   * @ref Partition::read consumes. Boundary semantics:
   *
   * - @c Standard: @c iz ∈ [0, Lz) ; the cell is shifted to buffer-z @c iz + 1 . If the
   *   resulting buffer-z lies strictly inside the internal band ( @c bz > 1 && bz < Lz )
   *   the cell maps into the internal SoA region; otherwise it maps into the down /
   *   up boundary slab.
   * - @c Internal: @c iz ∈ [0, Lz - 2) ; the cell is shifted to buffer-z @c iz + 2 and
   *   always maps into the internal SoA region.
   * - @c Boundary: @c iz ∈ [0, 2) ; @c iz == 0 maps to the down boundary ( @c bz = 1 ),
   *   @c iz == 1 maps to the up boundary ( @c bz = Lz ).
   *
   * The returned @ref Idx is in the same buffer-relative coordinate system as
   * @ref linear_offset / @ref global_location .
   */
  [[nodiscard]] CARBON_HD constexpr auto
  set_and_validate(::carbon::DataView view,
                   int64_t ix,
                   int64_t iy,
                   int64_t iz) const noexcept -> ValidatedIdx;
};

[[nodiscard]] CARBON_HD constexpr auto
Span::global_location(const Idx& idx) const noexcept -> int64_3d {
  const int64_3d cell = idx.cell();
  return {cell.x, cell.y, cell.z + z_offset_};
}

[[nodiscard]] CARBON_HD constexpr auto
Span::linear_offset(const Idx& idx,
                    int components,
                    int component_id) const noexcept -> int64_t {
  return idx.offset_local_no_card() +
         xy_slab_elements_ *
             (static_cast<int64_t>(idx.region_first_z()) * components +
              static_cast<int64_t>(component_id) * static_cast<int64_t>(idx.region_z_dim()));
}

/**
 * Result of resolving a neighbour offset against a disg @c Span : the neighbour's
 * disaggregated @ref Idx (with the per-region fields populated) and a @c valid flag
 * indicating whether the neighbour is reachable (in-buffer @b and inside the global
 * Cartesian grid). Internal helper struct; not part of the user-facing API.
 */
struct NeighbourIdx {
  Idx idx{};
  bool valid{false};
};

/**
 * Internal helper: resolve @p (idx + offset) against @p span and report whether the
 * result is reachable, populating the disaggregated-pitch metadata on the resulting
 * @ref Idx so @ref Partition::read works without an extra recomputation. Twin of
 * @c carbon::detail::dense::neighbour_idx ; the difference is the per-region pitch
 * setup at the end (the dense version stops at a buffer-z; the disg version classifies
 * the buffer-z into a region and fills @c offset_local_no_card / @c region_first_z /
 * @c region_z_dim ).
 *
 * Two checks combine, matching the aggregated dense helper:
 * - @b Buffer bounds: @c (cell + offset) must be inside @c span.haloed_extent() .
 * - @b Global-domain bounds on @b z: under @c BoundaryMode::Open the halo cell that
 *   maps outside the global Cartesian grid is rejected; under @c BoundaryMode::PeriodicZ
 *   it is accepted (the halo exchange has wrapped the data into the same buffer slot).
 */
[[nodiscard]] CARBON_HD constexpr auto
neighbour_idx(const Span& span,
              const Idx& idx,
              ::carbon::int8_3d offset) noexcept -> NeighbourIdx {
  const int64_3d cell = idx.cell();
  const int64_3d haloed = span.haloed_extent();
  const int64_t bx = cell.x + static_cast<int64_t>(offset.x);
  const int64_t by = cell.y + static_cast<int64_t>(offset.y);
  const int64_t bz = cell.z + static_cast<int64_t>(offset.z);
  if (bx < 0 || bx >= haloed.x || by < 0 || by >= haloed.y || bz < 0 || bz >= haloed.z) {
    return {};
  }
  const int64_t gz = bz + span.z_offset();
  const int64_3d global_extent = span.global_extent();
  if (gz < 0 || gz >= global_extent.z) {
    if (span.boundary_mode() != BoundaryMode::PeriodicZ) {
      return {};
    }
  }
  // Classify bz into the five-region map (see Span class docs). lz here is the owned
  // z-thickness ( @c haloed.z - 2 * halo_z ) — the buffer-z range is @c [0, lz + 2) .
  const int64_t lz = haloed.z - 2 * span.halo_z();
  const int64_t xy = haloed.x;
  int64_t offset_local_no_card = bx + by * xy;
  int32_t region_first_z = static_cast<int32_t>(bz);
  int32_t region_z_dim = 1;
  if (bz > 1 && bz < lz) {
    region_first_z = 2;
    region_z_dim = static_cast<int32_t>(lz - 2);
    offset_local_no_card += (bz - 2) * span.xy_slab_elements();
  }
  return NeighbourIdx{Idx(bx, by, bz, offset_local_no_card, region_first_z, region_z_dim), true};
}

/**
 * Internal helper: build the disaggregated @ref Idx for @p (idx + offset) with @b no
 * validity check. Twin of @ref neighbour_idx minus the buffer-bounds and global-domain
 * tests — the caller guarantees the neighbour is in-buffer (e.g. via a precomputed wall
 * mask). Backs @c Partition::read_pull so a pull-stream kernel can address the shifted
 * cell branch-free while still going through the correct five-region pitch classification
 * (the classification itself is inherent to the disg layout, not extra per-neighbour work).
 */
[[nodiscard]] CARBON_HD constexpr auto
pull_idx(const Span& span,
         const Idx& idx,
         ::carbon::int8_3d offset) noexcept -> Idx {
  const int64_3d cell = idx.cell();
  const int64_t bx = cell.x + static_cast<int64_t>(offset.x);
  const int64_t by = cell.y + static_cast<int64_t>(offset.y);
  const int64_t bz = cell.z + static_cast<int64_t>(offset.z);
  const int64_t lz = span.haloed_extent().z - 2 * span.halo_z();
  const int64_t xy = span.haloed_extent().x;
  int64_t offset_local_no_card = bx + by * xy;
  int32_t region_first_z = static_cast<int32_t>(bz);
  int32_t region_z_dim = 1;
  if (bz > 1 && bz < lz) {
    region_first_z = 2;
    region_z_dim = static_cast<int32_t>(lz - 2);
    offset_local_no_card += (bz - 2) * span.xy_slab_elements();
  }
  return Idx(bx, by, bz, offset_local_no_card, region_first_z, region_z_dim);
}

[[nodiscard]] CARBON_HD constexpr auto
Span::set_and_validate(::carbon::DataView view,
                       int64_t ix,
                       int64_t iy,
                       int64_t iz) const noexcept -> ValidatedIdx {
  const int64_3d ext = iter_extent(view);
  const bool ok = (ix >= 0 && iy >= 0 && iz >= 0 && ix < ext.x && iy < ext.y && iz < ext.z);
  const int64_t lz = haloed_extent_.z - 2 * halo_z_;
  const int64_t xy = haloed_extent_.x;

  int64_t bz = 0;
  int64_t offset_local_no_card = 0;
  int32_t region_first_z = 0;
  int32_t region_z_dim = 1;

  switch (view) {
    case ::carbon::DataView::Standard: {
      bz = iz + ghost_radius;
      offset_local_no_card = ix + iy * xy;
      region_first_z = static_cast<int32_t>(bz);
      region_z_dim = 1;
      if (bz > 1 && bz < lz) {
        region_first_z = 2;
        region_z_dim = static_cast<int32_t>(lz - 2);
        offset_local_no_card += (bz - 2) * xy_slab_elements_;
      }
      break;
    }
    case ::carbon::DataView::Internal: {
      bz = iz + 2;
      region_first_z = 2;
      region_z_dim = static_cast<int32_t>(lz - 2);
      offset_local_no_card = ix + iy * xy + iz * xy_slab_elements_;
      break;
    }
    case ::carbon::DataView::Boundary: {
      bz = (iz == 0) ? 1 : lz;
      region_first_z = static_cast<int32_t>(bz);
      region_z_dim = 1;
      offset_local_no_card = ix + iy * xy;
      break;
    }
  }

  return ValidatedIdx{Idx(ix, iy, bz, offset_local_no_card, region_first_z, region_z_dim), ok};
}

}  // namespace carbon::detail::dense_disg
