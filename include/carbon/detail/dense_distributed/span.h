#pragma once

#include <carbon/boundary_mode_distributed.h>
#include <carbon/data_view.h>
#include <carbon/detail/dense/idx.h>
#include <carbon/detail/validated_idx.h>
#include <carbon/floating3d.h>

#include <carbon/process_grid.h>

namespace carbon::detail::dense_distributed {

/**
 * Cell index shared with the Z-slab @c Dense family: @c Idx carries only the buffer-relative cell
 * coordinates, so it has no per-family state and a second copy of the type would add nothing.
 */
using Idx = ::carbon::detail::dense::Idx;

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
 * Layout of one GPU's dense subvolume expressed in @b storage (haloed) terms: per-axis haloed
 * extent, elements per XY plane (pitch in elements), global coordinate where the chunk's storage
 * starts, and the halo thickness on each axis.
 *
 * Convention:
 * - @c haloed_extent() is the full storage extent, equal to
 *   @c owned + 2 * halo() component-wise. When the grid has no stencil, @c halo() is zero and
 *   @c haloed_extent() equals @ref logic_extent().
 * - @c offset() is the @b buffer origin: buffer-local @c (0,0,0) maps to global
 *   @c offset(); the lower halo on each axis occupies @c [offset(), offset() + halo()), owned
 *   cells follow, and the upper halo the remaining @c halo() layers.
 * - @c xy_slab_elements() is the number of T-sized elements between two adjacent z planes in
 *   the haloed buffer. It is at least @c haloed_extent().x * haloed_extent().y ; halos are
 *   included in the extent rather than encoded as pitch padding.
 * - @c global_extent() is the full Cartesian grid extent shared by every chunk. It is needed to
 *   distinguish, in @c Partition::neighbour, an in-buffer halo cell that maps to a real global
 *   cell (interior chunk boundary) from one that lies outside the global domain (an open face).
 *
 * Kernels that iterate owned cells should drive their loop bounds from @ref logic_extent().
 * Pointer math (@ref linear_offset, @ref linear_storage_elements) uses the haloed view.
 */
class Span {
  int64_3d haloed_extent_{};
  int64_3d global_extent_{};
  int64_t xy_slab_elements_{};
  int64_3d offset_{};
  int64_3d halo_{};
  BoundaryModeDistributed boundary_mode_{BoundaryModeDistributed::Open};
  bool legacy_slab_{false};
  float32_3d origin_{};
  float spacing_{1.0F};

public:
  CARBON_HD constexpr Span() noexcept = default;

  CARBON_HD constexpr Span(int64_3d haloed_extent,
                           int64_3d global_extent,
                           int64_t xy_slab_elements,
                           int64_3d offset,
                           int64_3d halo = {},
                           BoundaryModeDistributed boundary_mode = BoundaryModeDistributed::Open,
                           bool legacy_slab = false,
                           float32_3d origin = {},
                           float spacing = 1.0F) noexcept
      : haloed_extent_(haloed_extent),
        global_extent_(global_extent),
        xy_slab_elements_(xy_slab_elements),
        offset_(offset),
        halo_(halo),
        boundary_mode_(boundary_mode),
        legacy_slab_(legacy_slab),
        origin_(origin),
        spacing_(spacing) {}

  /**
   * Construct a span over its @b storage (haloed) extent.
   *
   * @param haloed_extent     Storage extent: @c owned + 2 * halo(), component-wise. When
   *                          @p halo is zero this equals @ref logic_extent().
   * @param global_extent     Full Cartesian grid extent shared by every chunk. Used by
   *                          @c Partition::neighbour to distinguish in-domain reads from
   *                          out-of-domain halo cells on an open face.
   * @param xy_slab_elements  Elements per XY plane (pitch, in elements).
   * @param offset            Global coordinate of buffer-local @c (0,0,0).
   * @param halo              Halo thickness on each axis; must satisfy
   *                          @c logic_extent() >= 0. Default 0 (no halo).
   * @param boundary_mode     Per-axis boundary semantics propagated from the owning @c DenseDistributed;
   *                          consumed by @c Partition::neighbour and the halo exchange.
   *                          Defaults to @c BoundaryModeDistributed::Open.
   * @param legacy_slab       Selects the historical two-band z-only Boundary mapping. This is valid
   *                          only when @c halo.x == halo.y == 0 and @c logic_extent().z >=
   *                          2 * halo.z; @c DenseDistributed passes true only for legacy slabs.
   *                          Defaults to false, so z-only non-legacy chunks use the compact shell.
   * @param origin            Physical-space origin of global cell @c (0,0,0)'s corner (same
   *                          convention as @c global_location on integer indices).
   * @param spacing           Isotropic cell spacing along every axis (must be finite and > 0).
   */
  /** Z-only convenience overload: equivalent to full @c offset={0,0,z_offset} and
   *  @c halo={0,0,halo_z}. @p legacy_slab defaults to false; pass true for a legacy z-slab. */
  CARBON_HD constexpr Span(int64_3d haloed_extent,
                           int64_3d global_extent,
                           int64_t xy_slab_elements,
                           int64_t z_offset,
                           int64_t halo_z = 0,
                           BoundaryModeDistributed boundary_mode = BoundaryModeDistributed::Open,
                           bool legacy_slab = false,
                           float32_3d origin = {},
                           float spacing = 1.0F) noexcept
      : Span(haloed_extent,
             global_extent,
             xy_slab_elements,
             int64_3d{0,
                      0,
                      z_offset},
             int64_3d{0,
                      0,
                      halo_z},
             boundary_mode,
             legacy_slab,
             origin,
             spacing) {}

  /** Storage extent (includes halo layers on every axis where @ref halo() is non-zero). */
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
   * Owned-cell extent: @c haloed_extent() - 2 * halo(), component-wise. Independent of
   * @ref DataView — this is the storage-side owned region and is the same for every view.
   * Use @ref iter_extent if you need the iteration shape for a specific @ref DataView.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  logic_extent() const noexcept -> int64_3d {
    return int64_3d{haloed_extent_.x - 2 * halo_.x, haloed_extent_.y - 2 * halo_.y,
                    haloed_extent_.z - 2 * halo_.z};
  }

  /**
   * Iteration extent for @p view, i.e. the shape of the launch grid the user should iterate
   * over (paired with @ref set_and_validate to translate a thread (ix, iy, iz) into a
   * buffer-relative @ref Idx). The split is purely geometric — there is no special-casing on
   * the GPU/rank count, so @c "Internal + Boundary == Standard" holds in single- and multi-GPU
   * runs alike.
   *
   * - @c Standard: @c logic_extent(). Covers every owned cell exactly once.
   * - @c Internal: @c max(owned - 2 * halo(), 0) component-wise. Covers the inner region that
   *   does not depend on data outside the chunk. Zero halo on an axis leaves that axis whole.
   * - @c Boundary: the owned cells that are not in @c Internal (the shell of thickness
   *   @c halo() ). Legacy z-only chunks ( @c legacy_slab == true and @c halo.x == halo.y == 0 )
   *   use the compact @c 2 * halo.z z-band domain. Non-legacy z-only chunks (for example a
   *   pencil/brick or periodic-X/Y slab with a z-only stencil) and multi-axis chunks use a
   *   @b compact shell domain: the shell is split into six non-overlapping slabs (Z faces over
   *   the full XY extent, then Y faces over the middle Z range, then X faces over the middle Y
   *   and Z ranges), and @c iter_extent is a 2-D @c (W, ceil(S / W), 1) launch box over the @c S
   *   shell cells. @ref set_and_validate maps the linear thread index to its cell and returns
   *   @c valid=false only for the padded tail ( @c < W trailing threads), so a Boundary launch
   *   runs ~ @c S threads instead of one per owned cell. Because of that mapping, consecutive
   *   Boundary threads are @b not spatial neighbours: a kernel must not assume that thread
   *   @c (ix, iy, iz) addresses cell @c (ix, iy, iz) on this view (Standard and Internal keep
   *   that property).
   *
   * The ranges are non-overlapping, so dispatching Internal and Boundary back-to-back
   * (in any order) writes every owned cell exactly once — same coverage as a single
   * Standard launch.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  iter_extent(::carbon::DataView view) const noexcept -> int64_3d {
    const int64_3d owned = logic_extent();
    switch (view) {
      case ::carbon::DataView::Standard:
        return owned;

      case ::carbon::DataView::Internal: {
        const int64_t ix = owned.x - 2 * halo_.x;
        const int64_t iy = owned.y - 2 * halo_.y;
        const int64_t iz = owned.z - 2 * halo_.z;
        return int64_3d{ix > 0 ? ix : 0, iy > 0 ? iy : 0, iz > 0 ? iz : 0};
      }

      case ::carbon::DataView::Boundary: {
        if (legacy_slab_ && halo_.x == 0 && halo_.y == 0) {
          return int64_3d{owned.x, owned.y, halo_.z > 0 ? 2 * halo_.z : 0};
        }
        const int64_t shell = shell_bands_().total;
        if (shell == 0) {
          return int64_3d{0, 0, 0};
        }
        const int64_t w = shell >= kShellRowWidth ? kShellRowWidth : shell;
        return int64_3d{w, (shell + w - 1) / w, 1};
      }
    }
    return owned;
  }

  /** Elements per XY plane (pitch in elements), same for halo and owned layers. */
  [[nodiscard]] CARBON_HD constexpr auto
  xy_slab_elements() const noexcept -> int64_t {
    return xy_slab_elements_;
  }

  [[nodiscard]] CARBON_HD constexpr auto
  offset() const noexcept -> int64_3d {
    return offset_;
  }

  [[nodiscard]] CARBON_HD constexpr auto
  x_offset() const noexcept -> int64_t {
    return offset_.x;
  }

  [[nodiscard]] CARBON_HD constexpr auto
  y_offset() const noexcept -> int64_t {
    return offset_.y;
  }

  [[nodiscard]] CARBON_HD constexpr auto
  z_offset() const noexcept -> int64_t {
    return offset_.z;
  }

  [[nodiscard]] CARBON_HD constexpr auto
  halo() const noexcept -> int64_3d {
    return halo_;
  }

  [[nodiscard]] CARBON_HD constexpr auto
  halo_x() const noexcept -> int64_t {
    return halo_.x;
  }

  [[nodiscard]] CARBON_HD constexpr auto
  halo_y() const noexcept -> int64_t {
    return halo_.y;
  }

  [[nodiscard]] CARBON_HD constexpr auto
  halo_z() const noexcept -> int64_t {
    return halo_.z;
  }

  /** Boundary semantics this span was built with (propagated from the owning @c DenseDistributed). */
  [[nodiscard]] CARBON_HD constexpr auto
  boundary_mode() const noexcept -> BoundaryModeDistributed {
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
   * @c DenseDistributed::newField (lower halo + owned + upper halo).
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
   * Global location of @p idx within the grid: @c cell + offset(), component-wise.
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
   * @ref linear_offset / @ref global_location: every axis is shifted into the owned / halo
   * buffer layout for @p view.
   *
   * - @c Standard: @c b = halo() + i.
   * - @c Internal: @c b = 2 * halo() + i (skip past the halo plus the boundary band).
   * - @c Boundary: legacy z-only chunks ( @c legacy_slab == true ) use the compact lower/upper z
   *   bands; non-legacy chunks decode the linear shell index @c t = iy * iter_extent.x + ix into
   *   an owned cell (see @ref iter_extent ) and return @c valid=false for the padded tail.
   *
   * The @c valid flag in the returned @ref ValidatedIdx is @c true iff every component of
   * @c (ix, iy, iz) is within @c iter_extent(view); kernels should early-out on a @c false
   * return.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  set_and_validate(::carbon::DataView view,
                   int64_t ix,
                   int64_t iy,
                   int64_t iz) const noexcept -> ValidatedIdx;

private:
  /** Row width of the 2-D Boundary launch box (two full 64-wide blocks per row). */
  static constexpr int64_t kShellRowWidth = 128;

  /**
   * The Boundary shell as six disjoint boxes over the owned extent. Per axis the low / high band
   * thickness is clamped so the two never overlap (a chunk thinner than @c 2 * halo is shell
   * entirely and @c Internal is empty). Box order: Z-low, Z-high (full XY); Y-low, Y-high
   * (full X, middle Z); X-low, X-high (middle Y, middle Z).
   */
  struct ShellBands {
    int64_3d lo{};
    int64_3d hi{};
    int64_t vol[6]{};
    int64_t total{0};
  };

  [[nodiscard]] CARBON_HD constexpr auto
  shell_bands_() const noexcept -> ShellBands {
    const int64_3d n = logic_extent();
    ShellBands b;
    b.lo = int64_3d{halo_.x < n.x ? halo_.x : n.x, halo_.y < n.y ? halo_.y : n.y,
                    halo_.z < n.z ? halo_.z : n.z};
    const int64_t rx = n.x - b.lo.x;
    const int64_t ry = n.y - b.lo.y;
    const int64_t rz = n.z - b.lo.z;
    b.hi = int64_3d{halo_.x < rx ? halo_.x : rx, halo_.y < ry ? halo_.y : ry,
                    halo_.z < rz ? halo_.z : rz};
    const int64_t mz = n.z - b.lo.z - b.hi.z;  // middle extents (>= 0 by the clamps)
    const int64_t my = n.y - b.lo.y - b.hi.y;
    b.vol[0] = n.x * n.y * b.lo.z;
    b.vol[1] = n.x * n.y * b.hi.z;
    b.vol[2] = n.x * b.lo.y * mz;
    b.vol[3] = n.x * b.hi.y * mz;
    b.vol[4] = b.lo.x * my * mz;
    b.vol[5] = b.hi.x * my * mz;
    b.total = b.vol[0] + b.vol[1] + b.vol[2] + b.vol[3] + b.vol[4] + b.vol[5];
    return b;
  }

  /** Owned-local cell of linear shell index @p t in [0, bands.total); x varies fastest. */
  [[nodiscard]] CARBON_HD constexpr auto
  shell_cell_(const ShellBands& b,
              int64_t t) const noexcept -> int64_3d {
    const int64_3d n = logic_extent();
    int box = 0;
    while (box < 5 && t >= b.vol[box]) {
      t -= b.vol[box];
      ++box;
    }
    int64_t bx = 0;  // box origin
    int64_t by = 0;
    int64_t bz = 0;
    int64_t w = n.x;  // box XY footprint
    int64_t h = n.y;
    switch (box) {
      case 0:
        break;
      case 1:
        bz = n.z - b.hi.z;
        break;
      case 2:
        bz = b.lo.z;
        h = b.lo.y;
        break;
      case 3:
        by = n.y - b.hi.y;
        bz = b.lo.z;
        h = b.hi.y;
        break;
      case 4:
        by = b.lo.y;
        bz = b.lo.z;
        w = b.lo.x;
        h = n.y - b.lo.y - b.hi.y;
        break;
      default:
        bx = n.x - b.hi.x;
        by = b.lo.y;
        bz = b.lo.z;
        w = b.hi.x;
        h = n.y - b.lo.y - b.hi.y;
        break;
    }
    const int64_t x = t % w;
    const int64_t r = t / w;
    return int64_3d{bx + x, by + r % h, bz + r / h};
  }
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
  return {cell.x + offset_.x, cell.y + offset_.y, cell.z + offset_.z};
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
  if (ix < 0 || iy < 0 || iz < 0 || ix >= ext.x || iy >= ext.y || iz >= ext.z) {
    return ValidatedIdx{Idx{}, false};
  }

  switch (view) {
    case ::carbon::DataView::Standard:
      return ValidatedIdx{Idx(halo_.x + ix, halo_.y + iy, halo_.z + iz), true};

    case ::carbon::DataView::Internal:
      return ValidatedIdx{Idx(2 * halo_.x + ix, 2 * halo_.y + iy, 2 * halo_.z + iz), true};

    case ::carbon::DataView::Boundary: {
      if (legacy_slab_ && halo_.x == 0 && halo_.y == 0) {
        const int64_t owned_z = logic_extent().z;
        const int64_t bz = (iz < halo_.z) ? (halo_.z + iz) : (owned_z - halo_.z + iz);
        return ValidatedIdx{Idx(ix, iy, bz), true};
      }

      // Compact shell: thread (ix, iy) is the linear shell index t (iz is 0 by the range check
      // above); the padded tail of the last row is invalid.
      const ShellBands bands = shell_bands_();
      const int64_t t = iy * ext.x + ix;
      if (t >= bands.total) {
        return ValidatedIdx{Idx{}, false};
      }
      const int64_3d c = shell_cell_(bands, t);
      return ValidatedIdx{Idx(halo_.x + c.x, halo_.y + c.y, halo_.z + c.z), true};
    }
  }
  return ValidatedIdx{Idx{}, false};
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
 * Two checks combine:
 * - @b Buffer (storage) bounds: @c (cell + offset) must be inside @c span.haloed_extent() on
 *   every axis. This also bounds the offset to @c [-halo(), +halo()].
 * - @b Global-domain bounds: @c cell + offset + span.offset() must lie in
 *   @c [0, span.global_extent()) on every axis — rejecting halo cells that map outside the
 *   global grid on an open face.
 *
 * Lives at namespace scope (not on @c Span) so the Span class's public surface stays minimal;
 * grid families that need different resolution semantics (block grids, sparse grids) provide
 * their own @c neighbour_idx overload taking their own @c Span and @c Idx types and ADL picks
 * the right one inside their family's @c Partition.
 *
 * Boundary modes:
 * - @c BoundaryModeDistributed::Open: out-of-domain reads return @c valid = false (no neighbour).
 * - @c BoundaryModeDistributed::PeriodicX / @c PeriodicY / @c PeriodicZ: the corresponding axis wraps. A
 *   halo cell whose global coordinate lies outside @c [0, global_extent) is still reachable
 *   because the halo exchange has filled it with the wrapped data; the buffer-local index is
 *   accepted and @c valid = true is returned.
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
  const int64_3d global_extent = span.global_extent();
  const int64_3d off = span.offset();
  const BoundaryModeDistributed mode = span.boundary_mode();

  const int64_t gx = bx + off.x;
  if ((gx < 0 || gx >= global_extent.x) && !is_periodic(mode, Axis::X)) {
    return {};
  }

  const int64_t gy = by + off.y;
  if ((gy < 0 || gy >= global_extent.y) && !is_periodic(mode, Axis::Y)) {
    return {};
  }

  const int64_t gz = bz + off.z;
  if ((gz < 0 || gz >= global_extent.z) && !is_periodic(mode, Axis::Z)) {
    return {};
  }

  return NeighbourIdx{Idx(bx, by, bz), true};
}

}  // namespace carbon::detail::dense_distributed
