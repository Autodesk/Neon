#pragma once

#include <carbon/boundary_mode.h>
#include <carbon/data_view.h>
#include <carbon/detail/dense/idx.h>
#include <carbon/detail/validated_idx.h>
#include <carbon/floating3d.h>

namespace carbon::detail::dense {

/**
 * Result of @ref Span::set_and_validate: the buffer-relative @ref Idx and a flag indicating
 * whether the originating launch-grid thread fell inside @c iter_extent(view). Aliased to the
 * cross-family @c carbon::detail::Validated template (see
 * @c include/carbon/detail/validated_idx.h ) so every grid family's
 * value-returning @c set_and_validate shares the same return carrier shape;
 * the local @c ValidatedIdx spelling is preserved as a source-compat alias.
 */
using ValidatedIdx = ::carbon::detail::Validated<Idx>;

/**
 * Layout of one GPU's dense subvolume expressed in @b storage (haloed) terms: 3D haloed extent,
 * elements per XY plane (pitch in elements), global z-index where the slab's storage starts, and
 * the halo thickness on each z side.
 *
 * Convention:
 * - @c haloed_extent() is the full storage extent, equal to
 *   @c (owned.x, owned.y, owned.z + 2 * halo_z()) . When the grid has no stencil, @c halo_z() is
 *   0 and @c haloed_extent() equals @ref logic_extent().
 * - @c z_offset() is the @b buffer z-origin: buffer-local @c z=0 maps to global
 *   @c z = z_offset(); the lower halo occupies @c [z_offset(), z_offset() + halo_z()),
 *   owned cells @c [z_offset() + halo_z(), z_offset() + halo_z() + logic_extent().z), and the
 *   upper halo the remaining @c halo_z() layers.
 * - @c xy_slab_elements() is the number of T-sized elements between two adjacent z planes in
 *   the haloed buffer (may include pitch padding). Because carbon's current halo is z-only, the
 *   minimum value is @c haloed_extent().x * haloed_extent().y . The haloed extent is the only
 *   reference frame here — there is no separate "logical" pitch, since halos do not pad XY.
 * - @c global_extent() is the full Cartesian grid extent shared by every slab. It is needed to
 *   distinguish, in @c Partition::neighbour, an in-buffer halo cell that maps to a real global
 *   cell (interior slab boundary) from one that lies outside the global domain (top/bottom slab
 *   in a non-toroidal grid).
 *
 * Kernels that iterate owned cells should drive their loop bounds from @ref logic_extent().
 * Pointer math (@ref linear_offset, @ref linear_storage_elements) uses the haloed view.
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
  CARBON_HD constexpr Span() noexcept = default;

  /**
   * Construct a span over its @b storage (haloed) extent.
   *
   * @param haloed_extent     Storage extent: @c (owned.x, owned.y, owned.z + 2 * halo_z). When
   *                          @p halo_z is 0, this also equals @ref logic_extent().
   * @param global_extent     Full Cartesian grid extent shared by every slab. For a single-slab
   *                          grid this equals @c (haloed_extent.x, haloed_extent.y,
   *                          haloed_extent.z - 2 * halo_z). Used by @c Partition::neighbour to
   *                          distinguish in-domain reads from out-of-domain halo cells at the
   *                          top/bottom of a non-toroidal grid.
   * @param xy_slab_elements  Elements per XY plane (pitch, in elements).
   * @param z_offset          Global z-index of buffer-local @c z=0 (the start of the lower halo).
   * @param halo_z            Halo thickness on each z side; must satisfy
   *                          @c haloed_extent.z >= 2 * halo_z. Default 0 (no halo).
   * @param boundary_mode     Boundary semantics propagated from the owning @c Dense; consumed by
   *                          @c Partition::neighbour and the halo exchange. Defaults to
   *                          @c BoundaryMode::Open (today's behaviour).
   * @param origin            Physical-space origin of global cell @c (0,0,0)'s corner (same
   *                          convention as @c global_location on integer indices).
   * @param spacing           Isotropic cell spacing along every axis (must be finite and > 0).
   */
  CARBON_HD constexpr Span(int64_3d haloed_extent,
                           int64_3d global_extent,
                           int64_t xy_slab_elements,
                           int64_t z_offset,
                           int64_t halo_z = 0,
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

  /** Storage extent (includes halo layers on z when @ref halo_z is non-zero). */
  [[nodiscard]] CARBON_HD constexpr auto
  haloed_extent() const noexcept -> int64_3d {
    return haloed_extent_;
  }

  /** Full Cartesian grid extent shared by every slab. */
  [[nodiscard]] CARBON_HD constexpr auto
  global_extent() const noexcept -> int64_3d {
    return global_extent_;
  }

  /**
   * Owned-cell extent: @c (haloed.x, haloed.y, haloed.z - 2 * halo_z). Independent of
   * @ref DataView — this is the storage-side owned region and is the same for every view.
   * Use @ref iter_extent if you need the iteration shape for a specific @ref DataView.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  logic_extent() const noexcept -> int64_3d {
    return int64_3d(haloed_extent_.x, haloed_extent_.y, haloed_extent_.z - 2 * halo_z_);
  }

  /**
   * Iteration extent for @p view, i.e. the shape of the launch grid the user should iterate
   * over (paired with @ref set_and_validate to translate a thread (ix, iy, iz) into a
   * buffer-relative @ref Idx). @c x and @c y are always @c haloed_extent().x / .y; only @c z
   * differs by view, and the split is purely geometric — there is no special-casing on the
   * GPU/rank count, so @c "Internal + Boundary == Standard" holds in single- and multi-GPU
   * runs alike.
   *
   * Let @c B = halo_z() and @c Lz = logic_extent().z (owned z-thickness):
   *
   * - @c Standard: iteration extent z = @c Lz. Covers every owned cell exactly once.
   * - @c Internal: iteration extent z = @c max(Lz - 2 * B, 0). Covers the inner band of
   *   owned cells that does not depend on data outside the slab. With @c B == 0 this
   *   collapses to @c Standard.
   * - @c Boundary: iteration extent z = @c (B == 0) ? 0 : 2 * B. Covers the lower and upper
   *   bands of owned cells (one halo-thick slab on each side) that depend on neighbouring
   *   slab data through halo exchange. The two bands are folded into a single 1D
   *   thread-z domain @c [0, 2*B): threads with @c z < B map to the lower band, threads
   *   with @c z >= B map to the upper band. With @c B == 0 the band is empty.
   *
   * The ranges are non-overlapping, so dispatching Internal and Boundary back-to-back
   * (in any order) writes every owned cell exactly once — same coverage as a single
   * Standard launch. The owning @ref Dense rejects geometries with @c B > 0 && Lz < 2*B
   * at construction so the upper / lower bands cannot collide here.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  iter_extent(::carbon::DataView view) const noexcept -> int64_3d {
    const int64_t lz = haloed_extent_.z - 2 * halo_z_;
    switch (view) {
      case ::carbon::DataView::Standard:
        return int64_3d(haloed_extent_.x, haloed_extent_.y, lz);
      case ::carbon::DataView::Internal: {
        const int64_t inner = lz - 2 * halo_z_;
        return int64_3d(haloed_extent_.x, haloed_extent_.y, inner > 0 ? inner : 0);
      }
      case ::carbon::DataView::Boundary:
        return int64_3d(haloed_extent_.x, haloed_extent_.y, halo_z_ > 0 ? 2 * halo_z_ : 0);
    }
    return int64_3d(haloed_extent_.x, haloed_extent_.y, lz);
  }

  /** Elements per XY plane (pitch in elements), same for halo and owned layers. */
  [[nodiscard]] CARBON_HD constexpr auto
  xy_slab_elements() const noexcept -> int64_t {
    return xy_slab_elements_;
  }

  /** Global z-index of buffer-local @c z=0 (start of lower halo when @ref halo_z is non-zero). */
  [[nodiscard]] CARBON_HD constexpr auto
  z_offset() const noexcept -> int64_t {
    return z_offset_;
  }

  /** Halo thickness on each z side (0 when the grid carries no stencil). */
  [[nodiscard]] CARBON_HD constexpr auto
  halo_z() const noexcept -> int64_t {
    return halo_z_;
  }

  /** Boundary semantics this span was built with (propagated from the owning @c Dense). */
  [[nodiscard]] CARBON_HD constexpr auto
  boundary_mode() const noexcept -> BoundaryMode {
    return boundary_mode_;
  }

  /** Physical-space origin at global integer cell @c (0,0,0) (cell-corner convention). */
  [[nodiscard]] CARBON_HD constexpr auto
  origin() const noexcept -> float32_3d {
    return origin_;
  }

  /** Isotropic spacing between adjacent cell corners along @c x, @c y, and @c z. */
  [[nodiscard]] CARBON_HD constexpr auto
  spacing() const noexcept -> float {
    return spacing_;
  }

  /**
   * Contiguous element count for this slab's storage: @c haloed_extent().z layers times
   * @c xy_slab_elements() per layer. Matches the buffer sizing used by
   * @c Dense::newField (lower halo + owned + upper halo).
   */
  [[nodiscard]] CARBON_HD constexpr auto
  linear_storage_elements() const noexcept -> int64_t {
    return haloed_extent_.z * xy_slab_elements_;
  }

  /**
   * Canonical (cross-family) spelling for the per-component buffer pitch:
   * same value as the no-arg @ref linear_storage_elements above, named to
   * match @c block::Span::per_card_elements (carbon's "per cardinality
   * plane" terminology) and @c dense_disg::Span::per_card_elements . New
   * cross-grid code should prefer this spelling; the historical no-arg
   * @c linear_storage_elements stays as a source-compat alias.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  per_card_elements() const noexcept -> int64_t {
    return linear_storage_elements();
  }

  /**
   * Total per-slab buffer elements for @p components cardinality planes —
   * @c per_card_elements() * components . Mirrors the @c (int) -arity overload
   * on @c block::Span and @c dense_disg::Span so a generic kernel impl that
   * wants the full per-slab allocation size can spell it the same way across
   * grid families.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  linear_storage_elements(int components) const noexcept -> int64_t {
    return per_card_elements() * static_cast<int64_t>(components);
  }

  /**
   * Build an @ref Idx for local coordinates @p (ix, iy, iz). Typical use: pass indices derived
   * from @c blockIdx / @c threadIdx. The slab's z-origin is not carried on @ref Idx; use
   * @ref global_location on this span when global coordinates are needed.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  make_idx(int64_t ix,
           int64_t iy,
           int64_t iz) const noexcept -> Idx;

  /** Same as @ref make_idx(cell.x, cell.y, cell.z). */
  [[nodiscard]] CARBON_HD constexpr auto
  make_idx(int64_3d cell) const noexcept -> Idx;

  /**
   * Global location of @p idx within the grid: @c (cell.x, cell.y, cell.z + z_offset()).
   * Use @c .z on the result if only the global z is needed.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  global_location(const Idx& idx) const noexcept -> int64_3d;

  /**
   * Element offset (in units of @c T) from the start of a partition buffer to the cell at @p idx
   * for component @p component_id, using this span's SoA dense layout:
   * @c component_id * linear_storage_elements() + iz * xy_slab_elements() + iy * haloed_extent().x + ix.
   *
   * Matches Neon's @c dPartition::getPitch with its default pitch. No bounds checking.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  linear_offset(const Idx& idx,
                int component_id = 0) const noexcept -> int64_t;

  /**
   * Translate a launch-grid thread @p (ix, iy, iz) into a buffer-relative @ref Idx for @p view
   * and report whether the thread falls inside @c iter_extent(view). Mirrors Neon's
   * @c dSpan::setAndValidate.
   *
   * The returned @c Idx is in the same buffer-relative coordinate system as
   * @ref linear_offset / @ref global_location: x and y are passed through verbatim, while z
   * has been shifted into the owned / halo buffer layout for @p view:
   *
   * - @c Standard: @c bz = halo_z + iz.
   * - @c Internal: @c bz = 2 * halo_z + iz (skip past the halo plus the lower boundary band).
   * - @c Boundary: lower band (@c iz < halo_z) → @c bz = halo_z + iz; upper band
   *   (@c iz >= halo_z) → @c bz = (Lz - halo_z) + iz, where @c Lz is @c logic_extent().z.
   *
   * The @c valid flag in the returned @ref ValidatedIdx is @c true iff every component of
   * @c (ix, iy, iz) is within @c iter_extent(view); kernels should early-out on a @c false
   * return. With @c halo_z == 0 the Boundary path produces no valid threads (its iteration
   * extent is zero on z).
   */
  [[nodiscard]] CARBON_HD constexpr auto
  set_and_validate(::carbon::DataView view,
                   int64_t ix,
                   int64_t iy,
                   int64_t iz) const noexcept -> ValidatedIdx;
};

[[nodiscard]] CARBON_HD constexpr auto
Span::make_idx(int64_t ix,
               int64_t iy,
               int64_t iz) const noexcept -> Idx {
  return Idx(ix, iy, iz);
}

[[nodiscard]] CARBON_HD constexpr auto
Span::make_idx(int64_3d cell) const noexcept -> Idx {
  return Idx(cell);
}

[[nodiscard]] CARBON_HD constexpr auto
Span::global_location(const Idx& idx) const noexcept -> int64_3d {
  const int64_3d cell = idx.cell();
  return {cell.x, cell.y, cell.z + z_offset_};
}

[[nodiscard]] CARBON_HD constexpr auto
Span::linear_offset(const Idx& idx,
                    int component_id) const noexcept -> int64_t {
  const int64_3d cell = idx.cell();
  return static_cast<int64_t>(component_id) * linear_storage_elements() +
         cell.z * xy_slab_elements_ + cell.y * haloed_extent_.x + cell.x;
}

[[nodiscard]] CARBON_HD constexpr auto
Span::set_and_validate(::carbon::DataView view,
                       int64_t ix,
                       int64_t iy,
                       int64_t iz) const noexcept -> ValidatedIdx {
  const int64_3d ext = iter_extent(view);
  const bool ok = (ix >= 0 && iy >= 0 && iz >= 0 && ix < ext.x && iy < ext.y && iz < ext.z);
  int64_t bz = 0;
  switch (view) {
    case ::carbon::DataView::Standard:
      bz = halo_z_ + iz;
      break;
    case ::carbon::DataView::Internal:
      bz = 2 * halo_z_ + iz;
      break;
    case ::carbon::DataView::Boundary: {
      const int64_t lz = haloed_extent_.z - 2 * halo_z_;
      bz = (iz < halo_z_) ? (halo_z_ + iz) : (lz - halo_z_ + iz);
      break;
    }
  }
  return ValidatedIdx{Idx(ix, iy, bz), ok};
}

/**
 * Result of resolving a neighbour offset against a @c Span: the neighbour's @c Idx and a
 * @c valid flag indicating whether the neighbour is reachable (in-buffer @b and inside the
 * global Cartesian grid). Internal helper struct for @ref neighbour_idx; not part of the
 * user-facing API.
 */
struct NeighbourIdx {
  Idx idx{};
  bool valid{false};
};

/**
 * Internal helper: resolve @p (idx + offset) against @p span and report whether the result is
 * reachable. The shape-and-bounds half of @c Partition::neighbour, factored out so the
 * partition's @c neighbour() member is purely the data-access half (pointer dereference +
 * SoA stride math via @c Span::linear_offset).
 *
 * Two checks combine, mirroring @c Partition::neighbour 's previous body:
 * - @b Buffer (storage) bounds: @c (cell + offset) must be inside @c span.haloed_extent(). On
 *   @b x and @b y this is also the global-domain check (no x/y halos); on @b z it ensures the
 *   offset stays within @c [-halo_z, +halo_z].
 * - @b Global-domain bounds on @b z: @c gz = bz + span.z_offset() must lie in
 *   @c [0, span.global_extent().z) — rejecting halo cells that map outside the global grid on
 *   the top/bottom slab in a non-toroidal mode.
 *
 * Lives at namespace scope (not on @c Span) so the Span class's public surface stays minimal;
 * grid families that need different resolution semantics (block grids, sparse grids) provide
 * their own @c neighbour_idx overload taking their own @c Span and @c Idx types and ADL picks
 * the right one inside their family's @c Partition.
 *
 * Boundary modes:
 * - @c BoundaryMode::Open: out-of-domain reads return @c valid = false (no neighbour).
 * - @c BoundaryMode::PeriodicZ: the z-axis wraps. A halo cell whose global z lies outside
 *   @c [0, global_extent.z) is still reachable because the halo exchange has filled it with
 *   the wrapped data; we accept the buffer-local @c bz and return @c valid = true. (X and Y
 *   stay Open: carbon's storage is z-haloed only, so x/y out-of-domain reads remain invalid.)
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
    // The halo cell maps outside the global Cartesian grid. In Open mode that's a non-existent
    // cell; in PeriodicZ the halo exchange has already wrapped the data into this same buffer
    // location, so we just accept the local bz and let the caller dereference normally.
    if (span.boundary_mode() == BoundaryMode::PeriodicZ) {
      return NeighbourIdx{Idx(bx, by, bz), true};
    }
    return {};
  }
  return NeighbourIdx{Idx(bx, by, bz), true};
}

}  // namespace carbon::detail::dense
