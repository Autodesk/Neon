#pragma once

#include <carbon/boundary_mode_distributed.h>
#include <carbon/data_view.h>
#include <carbon/detail/dense_disg_distributed/idx.h>
#include <carbon/detail/validated_idx.h>
#include <carbon/floating3d.h>
#include <carbon/integer3d.h>
#include <carbon/process_grid.h>

#include <stdint.h>

namespace carbon::detail::dense_disg_distributed {

/**
 * Result of @ref Span::set_and_validate : a buffer-relative @ref Idx (with the disg region
 * fields already populated) and a flag indicating whether the originating launch-grid thread
 * fell inside @c iter_extent(view) . Aliased to the cross-family @c carbon::detail::Validated
 * template (see @c detail/validated_idx.h ).
 */
using ValidatedIdx = ::carbon::detail::Validated<Idx>;

/** Compile-time storage map selector for device-kernel specialisation. */
enum class StorageKind : uint8_t { Slab = 0, Pencil = 1, Brick = 2 };

/**
 * The physically disjoint storage groups of a disaggregated chunk buffer. The first
 * @ref kPencilRegionCount enumerators are the slab / pencil groups, in their storage order: each
 * is a box in @c (y, z) spanning the full @c x extent, stored card-major
 * ( @c [component][z][y][x] ). On slab storage ( @c ghost_y() == 0 ) only the five z-regions
 * @c LowerGhost / @c DownBoundary / @c Internal / @c UpBoundary / @c UpperGhost are non-empty;
 * the rest have zero size.
 *
 * The remaining enumerators exist only on brick storage ( @c ghost_x() == 1 , zero size
 * otherwise): the x faces, the ghost edges / corners touching an x ghost, and the send strips of
 * the x-crossing edges and the corners. Brick storage reuses every pencil enumerator with the
 * same geometric meaning but restricted to the owned @c x range, and lays the groups out in its
 * own nested order (see @ref Span ). Corner groups are numbered
 * @c base + xhi + 2·yhi + 4·zhi .
 */
enum class Region : uint8_t {
  LowerGhostYLo = 0,
  LowerGhost = 1,
  LowerGhostYHi = 2,
  DownBoundary = 3,
  YLowerGhost = 4,
  YDownBoundary = 5,
  Internal = 6,
  YUpBoundary = 7,
  YUpperGhost = 8,
  UpBoundary = 9,
  UpperGhostYLo = 10,
  UpperGhost = 11,
  UpperGhostYHi = 12,
  EdgeSendYLoZLo = 13,
  EdgeSendYHiZLo = 14,
  EdgeSendYLoZHi = 15,
  EdgeSendYHiZHi = 16,
  XLowerGhost = 17,
  XDownBoundary = 18,
  XUpBoundary = 19,
  XUpperGhost = 20,
  LowerGhostXLo = 21,
  LowerGhostXHi = 22,
  UpperGhostXLo = 23,
  UpperGhostXHi = 24,
  YLowerGhostXLo = 25,
  YLowerGhostXHi = 26,
  YUpperGhostXLo = 27,
  YUpperGhostXHi = 28,
  CornerGhostXLoYLoZLo = 29,
  CornerGhostXHiYLoZLo = 30,
  CornerGhostXLoYHiZLo = 31,
  CornerGhostXHiYHiZLo = 32,
  CornerGhostXLoYLoZHi = 33,
  CornerGhostXHiYLoZHi = 34,
  CornerGhostXLoYHiZHi = 35,
  CornerGhostXHiYHiZHi = 36,
  EdgeSendXLoZLo = 37,
  EdgeSendXHiZLo = 38,
  EdgeSendXLoZHi = 39,
  EdgeSendXHiZHi = 40,
  EdgeSendXLoYLo = 41,
  EdgeSendXHiYLo = 42,
  EdgeSendXLoYHi = 43,
  EdgeSendXHiYHi = 44,
  CornerSendXLoYLoZLo = 45,
  CornerSendXHiYLoZLo = 46,
  CornerSendXLoYHiZLo = 47,
  CornerSendXHiYHiZLo = 48,
  CornerSendXLoYLoZHi = 49,
  CornerSendXHiYLoZHi = 50,
  CornerSendXLoYHiZHi = 51,
  CornerSendXHiYHiZHi = 52,
};

/** Number of @ref Region enumerators. */
inline constexpr int kRegionCount = 53;

/** Number of slab / pencil @ref Region enumerators ( @c LowerGhostYLo .. @c EdgeSendYHiZHi ). */
inline constexpr int kPencilRegionCount = 17;

/** Brick storage: the internal box's x pitch is a multiple of this many elements (128 B of fp32). */
inline constexpr int64_t kInternalPitchAlign = 32;

/** Brick storage: row width of the flattened 1-D @c DataView::Boundary launch. */
inline constexpr int64_t kBoundaryRowWidth = 64;

/** Corner group @c base + xhi + 2·yhi + 4·zhi (see @ref Region ). */
[[nodiscard]] CARBON_HD constexpr auto
corner_region(Region base,
              bool xhi,
              bool yhi,
              bool zhi) noexcept -> Region {
  return static_cast<Region>(static_cast<int>(base) + (xhi ? 1 : 0) + (yhi ? 2 : 0) +
                             (zhi ? 4 : 0));
}

/**
 * Layout of one GPU's @b disaggregated subvolume on a @ref DenseDisgDistributed grid. Combines
 * the per-axis geometry surface of @c dense_distributed::Span ( @c offset() / @c halo() and their
 * per-axis spellings) with the disaggregated storage map of @c dense_disg::Span .
 *
 * Buffer coordinates: @c bx in @c [0, X) , @c by in @c [0, Hy) , @c bz in @c [0, Lz + 2) , where
 * @c Hy = Ly + 2·ghost_y() . Owned cells are @c by in @c [ghost_y, ghost_y + Ly) and
 * @c bz in @c [1, Lz] .
 *
 * @par Storage unit and pitch formula
 *
 * Every region size / prefix is counted in @c storage_unit() elements: one xy plane
 * ( @c X·Hy ) on slab storage, one x row ( @c X ) on pencil storage. The @ref Idx region fields
 * carry those counts, and the address of component @c c of a cell is
 * @code
 *   addr = idx.offset_local_no_card() + unit * (idx.region_first_z() * C + c * idx.region_z_dim())
 * @endcode
 * so a region occupies @c [first·C·unit, (first + size)·C·unit) with its components back to
 * back.
 *
 * @par Slab storage ( @c ghost_y() == 0 )
 *
 * Five z-regions in storage order (unit = xy plane): lower ghost ( @c bz = 0 ), down boundary
 * ( @c 1 ), internal ( @c 2 .. Lz-1 ), up boundary ( @c Lz ), upper ghost ( @c Lz+1 ). Map, pitch
 * and every accessor value are identical to @c dense_disg::Span (bit-exact storage).
 *
 * @par Pencil storage ( @c ghost_y() == 1 ) — nested, z outermost
 *
 * Sizes in x rows; @c mid = (Ly-2)·(Lz-2) .
 * @code
 *   region           cells (by, bz)                     rows   in-region row
 *   LowerGhostYLo    (0,      0)                        1      0
 *   LowerGhost       (1..Ly,  0)                        Ly     by-1
 *   LowerGhostYHi    (Ly+1,   0)                        1      0
 *   DownBoundary     (1..Ly,  1)                        Ly     by-1
 *   YLowerGhost      (0,      1..Lz)                    Lz     bz-1
 *   YDownBoundary    (1,      1..Lz)  [*]               Lz     bz-1
 *   Internal         (2..Ly-1, 2..Lz-1)                 mid    (bz-2)(Ly-2) + by-2
 *   YUpBoundary      (Ly,     1..Lz)  [*]               Lz     bz-1
 *   YUpperGhost      (Ly+1,   1..Lz)                    Lz     bz-1
 *   UpBoundary       (1..Ly,  Lz)                       Ly     by-1
 *   UpperGhostYLo    (0,      Lz+1)                     1      0
 *   UpperGhost       (1..Ly,  Lz+1)                     Ly     by-1
 *   UpperGhostYHi    (Ly+1,   Lz+1)                     1      0
 *   EdgeSend{YLo,YHi}{ZLo,ZHi}   send strip of corner (by in {1,Ly}, bz in {1,Lz})   1 each
 * @endcode
 * [*] Rows @c bz = 1 and @c bz = Lz of the two y-boundary groups are @b send @b shadows of the
 * edge cells, whose canonical storage is in @c DownBoundary / @c UpBoundary ; only rows
 * @c 2 .. Lz-1 are canonical. Shadows and edge strips are refreshed by the halo just before it
 * sends, so every face / edge message is one contiguous range of the buffer:
 *
 * - z face: @c DownBoundary / @c UpBoundary block → peer's @c UpperGhost / @c LowerGhost .
 * - y face: @c YDownBoundary / @c YUpBoundary block (shadows included) → peer's
 *   @c YUpperGhost / @c YLowerGhost .
 * - yz edge: @c EdgeSend* strip → peer's corner ghost group.
 *
 * Total storage is @c ((Ly+2)(Lz+2) + 8)·C·X : the extended box plus four shadow and four strip
 * rows per component.
 *
 * @par Brick storage ( @c ghost_x() == 1 ) — nested, z outermost, then y, then x
 *
 * Buffer coordinates gain an x ghost: @c bx in @c [0, X + 2) , owned @c bx in @c [1, X] (and
 * @c ghost_y() == 1 ). The storage unit is one element, so the region fields of an @ref Idx are
 * element counts and the in-region offset already folds every axis. Every group covers only the
 * owned range of the axes it does not pin. Storage order and in-region layout (per component;
 * @c Ix = X-2 , @c Iy = Y-2 , @c Iz = Z-2 , @c P = Ix rounded up to @ref kInternalPitchAlign ):
 * @code
 *   z ghost plane bz = 0 (then bz = Z+1, same order with Upper* names), [by][bx] x fastest:
 *     CornerGhostXLoYLoZLo, LowerGhostYLo [bx], CornerGhostXHiYLoZLo,
 *     LowerGhostXLo [by], LowerGhost [by][bx], LowerGhostXHi [by],
 *     CornerGhostXLoYHiZLo, LowerGhostYHi [bx], CornerGhostXHiYHiZLo
 *   DownBoundary              bz = 1           [by][bx]
 *   YLowerGhostXLo [bz], YLowerGhost [bz][bx], YLowerGhostXHi [bz]          by = 0
 *   YDownBoundary             by = 1           [bz][bx]   rows bz = 1, Z are shadows
 *   XLowerGhost               bx = 0           [bz][by]   y fastest
 *   XDownBoundary             bx = 1           [bz][by]   y fastest; rows bz = 1, Z and
 *                                                         columns by = 1, Y are shadows
 *   (pad to kInternalPitchAlign)
 *   Internal                  2..X-1, 2..Y-1, 2..Z-1      [bz][by] rows of pitch P
 *   XUpBoundary, XUpperGhost  bx = X, X+1      [bz][by]
 *   YUpBoundary               by = Y           [bz][bx]
 *   YUpperGhostXLo [bz], YUpperGhost [bz][bx], YUpperGhostXHi [bz]          by = Y+1
 *   UpBoundary                bz = Z           [by][bx]
 *   (z ghost plane bz = Z+1)
 *   EdgeSendY{Lo,Hi}Z{Lo,Hi} [bx], EdgeSendX{Lo,Hi}Z{Lo,Hi} [by], EdgeSendX{Lo,Hi}Y{Lo,Hi} [bz],
 *   CornerSend* (one cell each)
 * @endcode
 * Every owned cell has one canonical home: the z-boundary planes own their whole @c (bx, by)
 * plane, the y-boundary groups rows @c bz = 2..Z-1 , the x-boundary groups the
 * @c (by, bz) = (2..Y-1, 2..Z-1) block. The other rows / columns of the y and x face groups and the
 * edge / corner send strips are @b send @b shadows the halo refreshes just before it sends, so
 * each of the six face, twelve edge and eight corner messages is one contiguous range landing in
 * the matching ghost group of the peer. The internal box starts on a @ref kInternalPitchAlign
 * element boundary and its rows use pitch @c P , so every internal row of every component starts
 * 128-byte aligned for 4-byte elements.
 *
 * @par Storage vs. halo
 *
 * The storage extent always reserves the ghost planes / rows, so @c offset() is the global
 * coordinate of buffer @c (0,0,0) . @c halo() is the @b exchanged thickness: with a zero halo the
 * ghosts exist but are never filled, and @ref neighbour_idx rejects reads into them.
 *
 * @par Hard invariants
 *
 * Ghost and boundary radius are pinned to 1, @c halo().x == 0 outside brick storage,
 * @c logic_extent().z >= 2 and, on pencil / brick storage, @c logic_extent().y >= 2 (and
 * @c logic_extent().x >= 2 on brick storage). The owning grid enforces them at construction.
 */
class Span {
  int64_3d haloed_extent_{};
  int64_3d global_extent_{};
  int64_t xy_slab_elements_{};
  int64_3d offset_{};
  int64_3d halo_{};
  BoundaryModeDistributed boundary_mode_{BoundaryModeDistributed::Open};
  float32_3d origin_{};
  float spacing_{1.0F};
  int64_t ghost_y_{};
  int64_t ghost_x_{};
  /** Cached @ref storage_unit so @ref linear_offset stays branch-free on the hot path. */
  int64_t storage_unit_{};

public:
  /** Both ghost radius and boundary radius are pinned to 1 for the disaggregated layout. */
  static constexpr int64_t ghost_radius = 1;
  static constexpr int64_t boundary_radius = 1;

  CARBON_HD constexpr Span() noexcept = default;

  /**
   * @param haloed_extent     Storage extent:
   *                          @c (owned.x + 2·ghost_x, owned.y + 2·ghost_y, owned.z + 2) .
   * @param global_extent     Full Cartesian grid extent shared by every chunk.
   * @param xy_slab_elements  Elements per XY plane of the haloed box ( @c Hx·Hy ).
   * @param offset            Global coordinate of buffer-local @c (0,0,0) .
   * @param halo              Exchanged halo thickness per side ( @c {0|1, 0|1, 0|1} ).
   * @param boundary_mode     Per-axis boundary semantics propagated from the owning grid.
   * @param origin            Physical-space origin of global cell @c (0,0,0)'s corner.
   * @param spacing           Isotropic cell spacing (finite and > 0).
   * @param ghost_y           Ghost rows reserved per @c y side: @c 0 (slab) or @c 1 (pencil,
   *                          brick).
   * @param ghost_x           Ghost columns reserved per @c x side: @c 1 selects brick storage
   *                          (requires @p ghost_y == 1 ).
   */
  CARBON_HD constexpr Span(int64_3d haloed_extent,
                           int64_3d global_extent,
                           int64_t xy_slab_elements,
                           int64_3d offset,
                           int64_3d halo = {},
                           BoundaryModeDistributed boundary_mode = BoundaryModeDistributed::Open,
                           float32_3d origin = {},
                           float spacing = 1.0F,
                           int64_t ghost_y = 0,
                           int64_t ghost_x = 0) noexcept
      : haloed_extent_(haloed_extent),
        global_extent_(global_extent),
        xy_slab_elements_(xy_slab_elements),
        offset_(offset),
        halo_(halo),
        boundary_mode_(boundary_mode),
        origin_(origin),
        spacing_(spacing),
        ghost_y_(ghost_y),
        ghost_x_(ghost_x),
        storage_unit_(ghost_x != 0 ? int64_t{1}
                                   : (ghost_y != 0 ? haloed_extent.x : xy_slab_elements)) {}

  /** Storage extent (includes the ghost planes on @c z and ghost rows on @c y ). */
  [[nodiscard]] CARBON_HD constexpr auto
  haloed_extent() const noexcept -> int64_3d {
    return haloed_extent_;
  }

  [[nodiscard]] CARBON_HD constexpr auto
  global_extent() const noexcept -> int64_3d {
    return global_extent_;
  }

  /** Ghost rows reserved on each @c y side: @c 0 on slab storage, @c 1 on pencil storage. */
  [[nodiscard]] CARBON_HD constexpr auto
  ghost_y() const noexcept -> int64_t {
    return ghost_y_;
  }

  /** Ghost columns reserved on each @c x side: @c 1 on brick storage, @c 0 otherwise. */
  [[nodiscard]] CARBON_HD constexpr auto
  ghost_x() const noexcept -> int64_t {
    return ghost_x_;
  }

  /**
   * Elements per region-size unit: @c xy_slab_elements() (slab), @c haloed_extent().x (pencil)
   * or @c 1 (brick).
   */
  [[nodiscard]] CARBON_HD constexpr auto
  storage_unit() const noexcept -> int64_t {
    return storage_unit_;
  }

  /** True when this span uses the five-region DenseDisg-identical slab map. */
  [[nodiscard]] CARBON_HD constexpr auto
  is_slab() const noexcept -> bool {
    return ghost_x_ == 0 && ghost_y_ == 0;
  }

  /** Runtime storage-map kind (slab / pencil / brick). */
  [[nodiscard]] CARBON_HD constexpr auto
  storage_kind() const noexcept -> StorageKind {
    if (ghost_x_ != 0) {
      return StorageKind::Brick;
    }
    return ghost_y_ != 0 ? StorageKind::Pencil : StorageKind::Slab;
  }

  /** Owned-cell extent: @c (Hx - 2·ghost_x, Hy - 2·ghost_y, Hz - 2·ghost_radius) . */
  [[nodiscard]] CARBON_HD constexpr auto
  logic_extent() const noexcept -> int64_3d {
    return int64_3d(haloed_extent_.x - 2 * ghost_x_, haloed_extent_.y - 2 * ghost_y_,
                    haloed_extent_.z - 2 * ghost_radius);
  }

  /**
   * Brick storage: row pitch of the @c Internal box, @c X - 2 rounded up to
   * @ref kInternalPitchAlign elements ( @c 0 when the box is empty). @c 0 outside brick storage.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  internal_pitch() const noexcept -> int64_t {
    if (ghost_x_ == 0) {
      return 0;
    }
    const int64_t ix = haloed_extent_.x - 4;
    if (ix <= 0) {
      return 0;
    }
    return (ix + kInternalPitchAlign - 1) / kInternalPitchAlign * kInternalPitchAlign;
  }

  /** Brick storage: owned cells outside the @c Internal box (the 1-D @c Boundary launch size). */
  [[nodiscard]] CARBON_HD constexpr auto
  boundary_cell_count() const noexcept -> int64_t {
    const int64_3d n = logic_extent();
    const int64_t ix = n.x - 2 > 0 ? n.x - 2 : 0;
    const int64_t iy = n.y - 2 > 0 ? n.y - 2 : 0;
    const int64_t iz = n.z - 2 > 0 ? n.z - 2 : 0;
    return n.x * n.y * n.z - ix * iy * iz;
  }

  /**
   * Iteration extent for @p view . @c Internal + @c Boundary covers every owned cell once.
   *
   * - Slab: Standard covers every owned slab, Internal the inner band ( @c Lz - 2 ), Boundary
   *   exactly the two boundary slabs (same split as @c dense_disg::Span ).
   * - Pencil: Standard is @c (X, Ly, Lz) , Internal the @c Internal region
   *   @c (X, Ly-2, Lz-2) , Boundary a 1-D run over the boundary rows
   *   @c (X, 2·Ly + 2·(Lz-2), 1) : @c DownBoundary , @c UpBoundary , then the canonical rows of
   *   @c YDownBoundary and @c YUpBoundary .
   * - Brick: Standard is @c (X, Y, Z) , Internal the @c Internal box @c (X-2, Y-2, Z-2) ,
   *   Boundary a flattened 1-D run @c (kBoundaryRowWidth, ceil(nb / kBoundaryRowWidth), 1) over
   *   the @c nb = @ref boundary_cell_count canonical boundary cells in storage order
   *   ( @c DownBoundary , @c YDownBoundary , @c XDownBoundary , @c XUpBoundary ,
   *   @c YUpBoundary , @c UpBoundary ); the tail threads past @c nb are invalid.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  iter_extent(::carbon::DataView view) const noexcept -> int64_3d {
    const int64_3d n = logic_extent();
    const int64_t inner_z = n.z - 2 * boundary_radius > 0 ? n.z - 2 * boundary_radius : 0;
    if (ghost_x_ != 0) {
      const int64_t inner_x = n.x - 2 > 0 ? n.x - 2 : 0;
      const int64_t inner_y = n.y - 2 > 0 ? n.y - 2 : 0;
      switch (view) {
        case ::carbon::DataView::Standard:
          return n;
        case ::carbon::DataView::Internal:
          return int64_3d(inner_x, inner_y, inner_z);
        case ::carbon::DataView::Boundary: {
          const int64_t nb = boundary_cell_count();
          return int64_3d(kBoundaryRowWidth, (nb + kBoundaryRowWidth - 1) / kBoundaryRowWidth, 1);
        }
      }
      return n;
    }
    if (ghost_y_ == 0) {
      switch (view) {
        case ::carbon::DataView::Standard:
          return n;
        case ::carbon::DataView::Internal:
          return int64_3d(n.x, n.y, inner_z);
        case ::carbon::DataView::Boundary:
          return int64_3d(n.x, n.y, 2 * boundary_radius);
      }
      return n;
    }
    const int64_t inner_y = n.y - 2 * boundary_radius > 0 ? n.y - 2 * boundary_radius : 0;
    switch (view) {
      case ::carbon::DataView::Standard:
        return n;
      case ::carbon::DataView::Internal:
        return int64_3d(n.x, inner_y, inner_z);
      case ::carbon::DataView::Boundary:
        return int64_3d(n.x, 2 * n.y + 2 * inner_z, 1);
    }
    return n;
  }

  [[nodiscard]] CARBON_HD constexpr auto
  xy_slab_elements() const noexcept -> int64_t {
    return xy_slab_elements_;
  }

  /** Global coordinate of buffer-local @c (0,0,0) . */
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

  /** Exchanged halo thickness per side. */
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

  [[nodiscard]] CARBON_HD constexpr auto
  boundary_mode() const noexcept -> BoundaryModeDistributed {
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

  /** Per-component storage pitch: total region units times @c storage_unit() . */
  [[nodiscard]] CARBON_HD constexpr auto
  per_card_elements() const noexcept -> int64_t {
    const Region last = ghost_x_ != 0 ? Region::CornerSendXHiYHiZHi : Region::EdgeSendYHiZHi;
    return (region_first_z(last) + region_z_dim(last)) * storage_unit();
  }

  /** Total storage elements per chunk: @c per_card_elements() * components . */
  [[nodiscard]] CARBON_HD constexpr auto
  linear_storage_elements(int components) const noexcept -> int64_t {
    return per_card_elements() * static_cast<int64_t>(components);
  }

  /** Size of region @p r in @c storage_unit() elements per component. */
  [[nodiscard]] CARBON_HD constexpr auto
  region_z_dim(Region r) const noexcept -> int64_t {
    if (ghost_x_ != 0) {
      return brick_region_dim_(r);
    }
    if (static_cast<int>(r) >= kPencilRegionCount) {
      return 0;
    }
    const int64_3d n = logic_extent();
    const int64_t inner_z = n.z - 2 > 0 ? n.z - 2 : 0;
    const int64_t g = ghost_y_;
    const int64_t own = g != 0 ? n.y : 1;
    const int64_t ycol = g != 0 ? n.z : 0;
    const int64_t mid = g != 0 ? (n.y - 2 > 0 ? n.y - 2 : 0) * inner_z : inner_z;
    switch (r) {
      case Region::LowerGhost:
      case Region::DownBoundary:
      case Region::UpBoundary:
      case Region::UpperGhost:
        return own;
      case Region::YLowerGhost:
      case Region::YDownBoundary:
      case Region::YUpBoundary:
      case Region::YUpperGhost:
        return ycol;
      case Region::Internal:
        return mid;
      default:
        return g;
    }
  }

  /** First storage unit of region @p r ( @c region_first_z on the @ref Idx of its cells). */
  [[nodiscard]] CARBON_HD constexpr auto
  region_first_z(Region r) const noexcept -> int64_t {
    if (ghost_x_ != 0) {
      return brick_region_first_(r);
    }
    const int64_t corner = region_z_dim(Region::LowerGhostYLo);
    const int64_t own = region_z_dim(Region::LowerGhost);
    const int64_t ycol = region_z_dim(Region::YLowerGhost);
    const int64_t mid = region_z_dim(Region::Internal);
    const int64_t y_lower_ghost = 2 * corner + 2 * own;
    const int64_t internal = y_lower_ghost + 2 * ycol;
    const int64_t up_boundary = internal + mid + 2 * ycol;
    const int64_t edges = up_boundary + 2 * own + 2 * corner;
    switch (r) {
      case Region::LowerGhostYLo:
        return 0;
      case Region::LowerGhost:
        return corner;
      case Region::LowerGhostYHi:
        return corner + own;
      case Region::DownBoundary:
        return 2 * corner + own;
      case Region::YLowerGhost:
        return y_lower_ghost;
      case Region::YDownBoundary:
        return y_lower_ghost + ycol;
      case Region::Internal:
        return internal;
      case Region::YUpBoundary:
        return internal + mid;
      case Region::YUpperGhost:
        return internal + mid + ycol;
      case Region::UpBoundary:
        return up_boundary;
      case Region::UpperGhostYLo:
        return up_boundary + own;
      case Region::UpperGhost:
        return up_boundary + own + corner;
      case Region::UpperGhostYHi:
        return up_boundary + 2 * own + corner;
      case Region::EdgeSendYLoZLo:
        return edges;
      case Region::EdgeSendYHiZLo:
        return edges + corner;
      case Region::EdgeSendYLoZHi:
        return edges + 2 * corner;
      case Region::EdgeSendYHiZHi:
        return edges + 3 * corner;
      default:
        return 0;
    }
  }

  /**
   * Element offset of region @p r 's block in a @p components -cardinality buffer:
   * @c region_first_z(r) * components * storage_unit() . All cardinalities of a region are
   * stored back to back, so @c [offset, offset + region_storage_elements(r, components)) is the
   * whole region and component @c c starts at
   * @c offset + c * region_storage_elements(r, 1) .
   */
  [[nodiscard]] CARBON_HD constexpr auto
  region_storage_offset(Region r,
                        int components) const noexcept -> int64_t {
    return region_first_z(r) * static_cast<int64_t>(components) * storage_unit();
  }

  /** Element count of region @p r 's block: @c region_z_dim(r) * components * storage_unit() . */
  [[nodiscard]] CARBON_HD constexpr auto
  region_storage_elements(Region r,
                          int components) const noexcept -> int64_t {
    return region_z_dim(r) * static_cast<int64_t>(components) * storage_unit();
  }

  /**
   * Build the disaggregated @ref Idx of buffer-relative cell @p (bx, by, bz) : classifies the
   * cell into its region ( @c y first, then @c z ; brick: ghosts by axis @c z , @c y , @c x ,
   * then owned cells by @c z , @c y , @c x ) and fills @c offset_local_no_card /
   * @c region_first_z / @c region_z_dim . Edge cells resolve to their canonical storage, never
   * to a send shadow. No bounds checking.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  make_idx(int64_t bx,
           int64_t by,
           int64_t bz) const noexcept -> Idx;

  /** Same as @ref make_idx(cell.x, cell.y, cell.z) . */
  [[nodiscard]] CARBON_HD constexpr auto
  make_idx(int64_3d cell) const noexcept -> Idx;

  /** Global location of @p idx : @c idx.cell() + offset() , component-wise. */
  [[nodiscard]] CARBON_HD constexpr auto
  global_location(const Idx& idx) const noexcept -> int64_3d;

  /**
   * Element offset (in units of @c T ) from the start of a partition buffer to the cell at
   * @p idx for component @p component_id , using the disaggregated pitch. On slab storage this
   * is identical to @c dense_disg::Span::linear_offset . No bounds checking.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  linear_offset(const Idx& idx,
                int components,
                int component_id = 0) const noexcept -> int64_t;

  /**
   * Translate a launch-grid thread @p (ix, iy, iz) into a buffer-relative @ref Idx for @p view
   * and report whether the thread falls inside @c iter_extent(view) .
   *
   * - Slab: same mapping as @c dense_disg::Span::set_and_validate .
   * - Pencil Standard: buffer cell @c (ix, iy + 1, iz + 1) , classified by @ref make_idx .
   * - Pencil Internal: buffer cell @c (ix, iy + 2, iz + 2) , always @c Internal .
   * - Pencil Boundary: @c iy indexes the boundary rows in storage order ( @c DownBoundary ,
   *   @c UpBoundary , canonical @c YDownBoundary rows, canonical @c YUpBoundary rows); each
   *   range maps to its region directly, no division.
   * - Brick Standard: buffer cell @c (ix + 1, iy + 1, iz + 1) , classified by @ref make_idx .
   * - Brick Internal: buffer cell @c (ix + 2, iy + 2, iz + 2) , always @c Internal .
   * - Brick Boundary: @c t = ix + kBoundaryRowWidth·iy walks the canonical boundary cells group
   *   by group in storage order, so consecutive threads write consecutive addresses (x fastest in
   *   the z / y groups, y fastest in the x groups); one 32-bit division per thread.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  set_and_validate(::carbon::DataView view,
                   int64_t ix,
                   int64_t iy,
                   int64_t iz) const noexcept -> ValidatedIdx;

private:
  [[nodiscard]] CARBON_HD constexpr auto
  region_idx_(Region r,
              int64_t bx,
              int64_t by,
              int64_t bz,
              int64_t row) const noexcept -> Idx {
    return Idx(bx, by, bz, bx + row * haloed_extent_.x, static_cast<int32_t>(region_first_z(r)),
               static_cast<int32_t>(region_z_dim(r)));
  }

  template <Region R>
  [[nodiscard]] CARBON_HD constexpr auto
  brick_idx_(int64_t bx,
             int64_t by,
             int64_t bz,
             int64_t local) const noexcept -> Idx {
    return Idx(bx, by, bz, local, static_cast<int32_t>(brick_region_first_(R)),
               static_cast<int32_t>(brick_region_dim_(R)));
  }

  template <Region Lo,
            Region Hi>
  [[nodiscard]] CARBON_HD constexpr auto
  brick_pick_(bool hi,
              int64_t bx,
              int64_t by,
              int64_t bz,
              int64_t local) const noexcept -> Idx {
    return hi ? brick_idx_<Hi>(bx, by, bz, local) : brick_idx_<Lo>(bx, by, bz, local);
  }

  [[nodiscard]] CARBON_HD constexpr auto
  brick_region_dim_(Region r) const noexcept -> int64_t {
    const int64_3d n = logic_extent();
    switch (r) {
      case Region::LowerGhostYLo:
      case Region::LowerGhostYHi:
      case Region::UpperGhostYLo:
      case Region::UpperGhostYHi:
      case Region::EdgeSendYLoZLo:
      case Region::EdgeSendYHiZLo:
      case Region::EdgeSendYLoZHi:
      case Region::EdgeSendYHiZHi:
        return n.x;
      case Region::LowerGhostXLo:
      case Region::LowerGhostXHi:
      case Region::UpperGhostXLo:
      case Region::UpperGhostXHi:
      case Region::EdgeSendXLoZLo:
      case Region::EdgeSendXHiZLo:
      case Region::EdgeSendXLoZHi:
      case Region::EdgeSendXHiZHi:
        return n.y;
      case Region::YLowerGhostXLo:
      case Region::YLowerGhostXHi:
      case Region::YUpperGhostXLo:
      case Region::YUpperGhostXHi:
      case Region::EdgeSendXLoYLo:
      case Region::EdgeSendXHiYLo:
      case Region::EdgeSendXLoYHi:
      case Region::EdgeSendXHiYHi:
        return n.z;
      case Region::LowerGhost:
      case Region::DownBoundary:
      case Region::UpBoundary:
      case Region::UpperGhost:
        return n.x * n.y;
      case Region::YLowerGhost:
      case Region::YDownBoundary:
      case Region::YUpBoundary:
      case Region::YUpperGhost:
        return n.x * n.z;
      case Region::XLowerGhost:
      case Region::XDownBoundary:
      case Region::XUpBoundary:
      case Region::XUpperGhost:
        return n.y * n.z;
      case Region::Internal: {
        const int64_t iy = n.y - 2 > 0 ? n.y - 2 : 0;
        const int64_t iz = n.z - 2 > 0 ? n.z - 2 : 0;
        return internal_pitch() * iy * iz;
      }
      default:
        return 1;
    }
  }

  [[nodiscard]] CARBON_HD constexpr auto
  brick_region_first_(Region r) const noexcept -> int64_t {
    const int64_3d n = logic_extent();
    const int64_t x = n.x;
    const int64_t y = n.y;
    const int64_t z = n.z;
    const int64_t xy = x * y;
    const int64_t xz = x * z;
    const int64_t yz = y * z;
    const int64_t plane = (x + 2) * (y + 2);
    const int64_t c0 = plane + xy;
    const int64_t before_internal = c0 + 2 * z + 2 * xz + 2 * yz;
    const int64_t i0 =
        (before_internal + kInternalPitchAlign - 1) / kInternalPitchAlign * kInternalPitchAlign;
    const int64_t u0 = i0 + brick_region_dim_(Region::Internal);
    const int64_t d0 = u0 + 2 * yz + 2 * xz + 2 * z;
    const int64_t e0 = d0 + xy;
    const int64_t f0 = e0 + plane;
    const int64_t f1 = f0 + 4 * x;
    const int64_t f2 = f1 + 4 * y;
    const int64_t f3 = f2 + 4 * z;
    switch (r) {
      case Region::CornerGhostXLoYLoZLo:
        return 0;
      case Region::LowerGhostYLo:
        return 1;
      case Region::CornerGhostXHiYLoZLo:
        return 1 + x;
      case Region::LowerGhostXLo:
        return 2 + x;
      case Region::LowerGhost:
        return 2 + x + y;
      case Region::LowerGhostXHi:
        return 2 + x + y + xy;
      case Region::CornerGhostXLoYHiZLo:
        return 2 + x + 2 * y + xy;
      case Region::LowerGhostYHi:
        return 3 + x + 2 * y + xy;
      case Region::CornerGhostXHiYHiZLo:
        return 3 + 2 * x + 2 * y + xy;
      case Region::DownBoundary:
        return plane;
      case Region::YLowerGhostXLo:
        return c0;
      case Region::YLowerGhost:
        return c0 + z;
      case Region::YLowerGhostXHi:
        return c0 + z + xz;
      case Region::YDownBoundary:
        return c0 + 2 * z + xz;
      case Region::XLowerGhost:
        return c0 + 2 * z + 2 * xz;
      case Region::XDownBoundary:
        return c0 + 2 * z + 2 * xz + yz;
      case Region::Internal:
        return i0;
      case Region::XUpBoundary:
        return u0;
      case Region::XUpperGhost:
        return u0 + yz;
      case Region::YUpBoundary:
        return u0 + 2 * yz;
      case Region::YUpperGhostXLo:
        return u0 + 2 * yz + xz;
      case Region::YUpperGhost:
        return u0 + 2 * yz + xz + z;
      case Region::YUpperGhostXHi:
        return u0 + 2 * yz + 2 * xz + z;
      case Region::UpBoundary:
        return d0;
      case Region::CornerGhostXLoYLoZHi:
        return e0;
      case Region::UpperGhostYLo:
        return e0 + 1;
      case Region::CornerGhostXHiYLoZHi:
        return e0 + 1 + x;
      case Region::UpperGhostXLo:
        return e0 + 2 + x;
      case Region::UpperGhost:
        return e0 + 2 + x + y;
      case Region::UpperGhostXHi:
        return e0 + 2 + x + y + xy;
      case Region::CornerGhostXLoYHiZHi:
        return e0 + 2 + x + 2 * y + xy;
      case Region::UpperGhostYHi:
        return e0 + 3 + x + 2 * y + xy;
      case Region::CornerGhostXHiYHiZHi:
        return e0 + 3 + 2 * x + 2 * y + xy;
      case Region::EdgeSendYLoZLo:
        return f0;
      case Region::EdgeSendYHiZLo:
        return f0 + x;
      case Region::EdgeSendYLoZHi:
        return f0 + 2 * x;
      case Region::EdgeSendYHiZHi:
        return f0 + 3 * x;
      case Region::EdgeSendXLoZLo:
        return f1;
      case Region::EdgeSendXHiZLo:
        return f1 + y;
      case Region::EdgeSendXLoZHi:
        return f1 + 2 * y;
      case Region::EdgeSendXHiZHi:
        return f1 + 3 * y;
      case Region::EdgeSendXLoYLo:
        return f2;
      case Region::EdgeSendXHiYLo:
        return f2 + z;
      case Region::EdgeSendXLoYHi:
        return f2 + 2 * z;
      case Region::EdgeSendXHiYHi:
        return f2 + 3 * z;
      case Region::CornerSendXLoYLoZLo:
      case Region::CornerSendXHiYLoZLo:
      case Region::CornerSendXLoYHiZLo:
      case Region::CornerSendXHiYHiZLo:
      case Region::CornerSendXLoYLoZHi:
      case Region::CornerSendXHiYLoZHi:
      case Region::CornerSendXLoYHiZHi:
      case Region::CornerSendXHiYHiZHi:
        return f3 + static_cast<int64_t>(static_cast<int>(r) -
                                         static_cast<int>(Region::CornerSendXLoYLoZLo));
    }
    return 0;
  }

  [[nodiscard]] CARBON_HD constexpr auto
  make_brick_idx_(int64_t bx,
                  int64_t by,
                  int64_t bz) const noexcept -> Idx;

  [[nodiscard]] CARBON_HD constexpr auto
  brick_set_and_validate_(::carbon::DataView view,
                          int64_t ix,
                          int64_t iy,
                          int64_t iz,
                          bool ok) const noexcept -> ValidatedIdx;
};

[[nodiscard]] CARBON_HD constexpr auto
Span::make_idx(int64_t bx,
               int64_t by,
               int64_t bz) const noexcept -> Idx {
  if (ghost_x_ != 0) {
    return make_brick_idx_(bx, by, bz);
  }
  const int64_t lz = haloed_extent_.z - 2 * ghost_radius;
  if (ghost_y_ == 0) {
    int64_t offset_local_no_card = bx + by * haloed_extent_.x;
    int32_t region_first_z = static_cast<int32_t>(bz);
    int32_t region_z_dim = 1;
    if (bz > 1 && bz < lz) {
      region_first_z = 2;
      region_z_dim = static_cast<int32_t>(lz - 2);
      offset_local_no_card += (bz - 2) * xy_slab_elements_;
    }
    return Idx(bx, by, bz, offset_local_no_card, region_first_z, region_z_dim);
  }
  const int64_t ly = haloed_extent_.y - 2;
  if (by == 0) {
    if (bz == 0) {
      return region_idx_(Region::LowerGhostYLo, bx, by, bz, 0);
    }
    if (bz == lz + 1) {
      return region_idx_(Region::UpperGhostYLo, bx, by, bz, 0);
    }
    return region_idx_(Region::YLowerGhost, bx, by, bz, bz - 1);
  }
  if (by == ly + 1) {
    if (bz == 0) {
      return region_idx_(Region::LowerGhostYHi, bx, by, bz, 0);
    }
    if (bz == lz + 1) {
      return region_idx_(Region::UpperGhostYHi, bx, by, bz, 0);
    }
    return region_idx_(Region::YUpperGhost, bx, by, bz, bz - 1);
  }
  if (bz == 0) {
    return region_idx_(Region::LowerGhost, bx, by, bz, by - 1);
  }
  if (bz == 1) {
    return region_idx_(Region::DownBoundary, bx, by, bz, by - 1);
  }
  if (bz == lz) {
    return region_idx_(Region::UpBoundary, bx, by, bz, by - 1);
  }
  if (bz == lz + 1) {
    return region_idx_(Region::UpperGhost, bx, by, bz, by - 1);
  }
  if (by == 1) {
    return region_idx_(Region::YDownBoundary, bx, by, bz, bz - 1);
  }
  if (by == ly) {
    return region_idx_(Region::YUpBoundary, bx, by, bz, bz - 1);
  }
  return region_idx_(Region::Internal, bx, by, bz, (bz - 2) * (ly - 2) + (by - 2));
}

[[nodiscard]] CARBON_HD constexpr auto
Span::make_brick_idx_(int64_t bx,
                      int64_t by,
                      int64_t bz) const noexcept -> Idx {
  const int64_3d n = logic_extent();
  if (bz == 0 || bz == n.z + 1) {
    const bool zhi = bz != 0;
    const bool ylo = by == 0;
    const bool yhi = by == n.y + 1;
    const bool xlo = bx == 0;
    const bool xhi = bx == n.x + 1;
    if (ylo || yhi) {
      if (xlo || xhi) {
        if (zhi) {
          return yhi ? brick_pick_<Region::CornerGhostXLoYHiZHi, Region::CornerGhostXHiYHiZHi>(
                           xhi, bx, by, bz, 0)
                     : brick_pick_<Region::CornerGhostXLoYLoZHi, Region::CornerGhostXHiYLoZHi>(
                           xhi, bx, by, bz, 0);
        }
        return yhi ? brick_pick_<Region::CornerGhostXLoYHiZLo, Region::CornerGhostXHiYHiZLo>(
                         xhi, bx, by, bz, 0)
                   : brick_pick_<Region::CornerGhostXLoYLoZLo, Region::CornerGhostXHiYLoZLo>(
                         xhi, bx, by, bz, 0);
      }
      if (zhi) {
        return brick_pick_<Region::UpperGhostYLo, Region::UpperGhostYHi>(yhi, bx, by, bz, bx - 1);
      }
      return brick_pick_<Region::LowerGhostYLo, Region::LowerGhostYHi>(yhi, bx, by, bz, bx - 1);
    }
    if (xlo || xhi) {
      if (zhi) {
        return brick_pick_<Region::UpperGhostXLo, Region::UpperGhostXHi>(xhi, bx, by, bz, by - 1);
      }
      return brick_pick_<Region::LowerGhostXLo, Region::LowerGhostXHi>(xhi, bx, by, bz, by - 1);
    }
    return brick_pick_<Region::LowerGhost, Region::UpperGhost>(zhi, bx, by, bz,
                                                               (by - 1) * n.x + bx - 1);
  }
  if (by == 0 || by == n.y + 1) {
    const bool yhi = by != 0;
    if (bx == 0) {
      return brick_pick_<Region::YLowerGhostXLo, Region::YUpperGhostXLo>(yhi, bx, by, bz, bz - 1);
    }
    if (bx == n.x + 1) {
      return brick_pick_<Region::YLowerGhostXHi, Region::YUpperGhostXHi>(yhi, bx, by, bz, bz - 1);
    }
    return brick_pick_<Region::YLowerGhost, Region::YUpperGhost>(yhi, bx, by, bz,
                                                                 (bz - 1) * n.x + bx - 1);
  }
  if (bx == 0) {
    return brick_idx_<Region::XLowerGhost>(bx, by, bz, (bz - 1) * n.y + by - 1);
  }
  if (bx == n.x + 1) {
    return brick_idx_<Region::XUpperGhost>(bx, by, bz, (bz - 1) * n.y + by - 1);
  }
  if (bz == 1) {
    return brick_idx_<Region::DownBoundary>(bx, by, bz, (by - 1) * n.x + bx - 1);
  }
  if (bz == n.z) {
    return brick_idx_<Region::UpBoundary>(bx, by, bz, (by - 1) * n.x + bx - 1);
  }
  if (by == 1) {
    return brick_idx_<Region::YDownBoundary>(bx, by, bz, (bz - 1) * n.x + bx - 1);
  }
  if (by == n.y) {
    return brick_idx_<Region::YUpBoundary>(bx, by, bz, (bz - 1) * n.x + bx - 1);
  }
  if (bx == 1) {
    return brick_idx_<Region::XDownBoundary>(bx, by, bz, (bz - 1) * n.y + by - 1);
  }
  if (bx == n.x) {
    return brick_idx_<Region::XUpBoundary>(bx, by, bz, (bz - 1) * n.y + by - 1);
  }
  return brick_idx_<Region::Internal>(bx, by, bz,
                                      ((bz - 2) * (n.y - 2) + by - 2) * internal_pitch() + bx - 2);
}

[[nodiscard]] CARBON_HD constexpr auto
Span::make_idx(int64_3d cell) const noexcept -> Idx {
  return make_idx(cell.x, cell.y, cell.z);
}

[[nodiscard]] CARBON_HD constexpr auto
Span::global_location(const Idx& idx) const noexcept -> int64_3d {
  const int64_3d cell = idx.cell();
  return {cell.x + offset_.x, cell.y + offset_.y, cell.z + offset_.z};
}

[[nodiscard]] CARBON_HD constexpr auto
Span::linear_offset(const Idx& idx,
                    int components,
                    int component_id) const noexcept -> int64_t {
  return idx.offset_local_no_card() +
         storage_unit() *
             (static_cast<int64_t>(idx.region_first_z()) * components +
              static_cast<int64_t>(component_id) * static_cast<int64_t>(idx.region_z_dim()));
}

[[nodiscard]] CARBON_HD constexpr auto
Span::set_and_validate(::carbon::DataView view,
                       int64_t ix,
                       int64_t iy,
                       int64_t iz) const noexcept -> ValidatedIdx {
  const int64_3d ext = iter_extent(view);
  const bool ok = (ix >= 0 && iy >= 0 && iz >= 0 && ix < ext.x && iy < ext.y && iz < ext.z);
  if (ghost_x_ != 0) {
    return brick_set_and_validate_(view, ix, iy, iz, ok);
  }
  const int64_t lz = haloed_extent_.z - 2 * ghost_radius;
  const int64_t xy = haloed_extent_.x;

  if (ghost_y_ == 0) {
    // Bit-identical to dense_disg::Span::set_and_validate — keep the Standard path
    // fully inlined (no make_idx) so the slab hot kernel does not pull in pencil/brick
    // classification.
    switch (view) {
      case ::carbon::DataView::Standard: {
        const int64_t bz = iz + ghost_radius;
        int64_t offset_local_no_card = ix + iy * xy;
        int32_t region_first_z = static_cast<int32_t>(bz);
        int32_t region_z_dim = 1;
        if (bz > 1 && bz < lz) {
          region_first_z = 2;
          region_z_dim = static_cast<int32_t>(lz - 2);
          offset_local_no_card += (bz - 2) * xy_slab_elements_;
        }
        return ValidatedIdx{Idx(ix, iy, bz, offset_local_no_card, region_first_z, region_z_dim),
                            ok};
      }
      case ::carbon::DataView::Internal:
        return ValidatedIdx{Idx(ix, iy, iz + 2, ix + iy * xy + iz * xy_slab_elements_, 2,
                                static_cast<int32_t>(lz - 2)),
                            ok};
      case ::carbon::DataView::Boundary: {
        const int64_t bz = (iz == 0) ? 1 : lz;
        return ValidatedIdx{Idx(ix, iy, bz, ix + iy * xy, static_cast<int32_t>(bz), 1), ok};
      }
    }
    return ValidatedIdx{Idx{}, false};
  }

  const int64_t ly = haloed_extent_.y - 2;
  switch (view) {
    case ::carbon::DataView::Standard:
      return ValidatedIdx{make_idx(ix, iy + 1, iz + 1), ok};
    case ::carbon::DataView::Internal:
      return ValidatedIdx{region_idx_(Region::Internal, ix, iy + 2, iz + 2, iz * (ly - 2) + iy),
                          ok};
    case ::carbon::DataView::Boundary: {
      if (iy < ly) {
        return ValidatedIdx{region_idx_(Region::DownBoundary, ix, iy + 1, 1, iy), ok};
      }
      if (iy < 2 * ly) {
        return ValidatedIdx{region_idx_(Region::UpBoundary, ix, iy - ly + 1, lz, iy - ly), ok};
      }
      const int64_t inner_z = lz - 2;
      const int64_t r = iy - 2 * ly;
      if (r < inner_z) {
        return ValidatedIdx{region_idx_(Region::YDownBoundary, ix, 1, r + 2, r + 1), ok};
      }
      return ValidatedIdx{
          region_idx_(Region::YUpBoundary, ix, ly, r - inner_z + 2, r - inner_z + 1), ok};
    }
  }
  return ValidatedIdx{Idx{}, false};
}

/**
 * Quotient of two non-negative values below @c 2^32 in 32-bit arithmetic (the owning grid caps a
 * chunk's per-component storage below @c 2^31 elements).
 */
[[nodiscard]] CARBON_HD constexpr auto
udiv32(int64_t a,
       int64_t b) noexcept -> int64_t {
  return static_cast<int64_t>(static_cast<uint32_t>(a) / static_cast<uint32_t>(b));
}

[[nodiscard]] CARBON_HD constexpr auto
Span::brick_set_and_validate_(::carbon::DataView view,
                              int64_t ix,
                              int64_t iy,
                              int64_t iz,
                              bool ok) const noexcept -> ValidatedIdx {
  const int64_3d n = logic_extent();
  const int64_t inner_y = n.y - 2;
  const int64_t inner_z = n.z - 2;
  switch (view) {
    case ::carbon::DataView::Standard:
      return ValidatedIdx{make_brick_idx_(ix + 1, iy + 1, iz + 1), ok};
    case ::carbon::DataView::Internal:
      return ValidatedIdx{brick_idx_<Region::Internal>(ix + 2, iy + 2, iz + 2,
                                                       (iz * inner_y + iy) * internal_pitch() + ix),
                          ok};
    case ::carbon::DataView::Boundary: {
      int64_t u = ix + kBoundaryRowWidth * iy;
      const bool valid = ok && u < boundary_cell_count();
      const int64_t plane = n.x * n.y;
      const int64_t yface = n.x * inner_z;
      const int64_t xface = inner_y * inner_z;
      if (u < plane) {
        const int64_t q = udiv32(u, n.x);
        return ValidatedIdx{brick_idx_<Region::DownBoundary>(u - q * n.x + 1, q + 1, 1, u), valid};
      }
      u -= plane;
      if (u < yface) {
        const int64_t q = udiv32(u, n.x);
        return ValidatedIdx{brick_idx_<Region::YDownBoundary>(u - q * n.x + 1, 1, q + 2, u + n.x),
                            valid};
      }
      u -= yface;
      if (u < 2 * xface) {
        const bool up = u >= xface;
        const int64_t v = up ? u - xface : u;
        const int64_t q = udiv32(v, inner_y);
        const int64_t r = v - q * inner_y;
        return ValidatedIdx{brick_pick_<Region::XDownBoundary, Region::XUpBoundary>(
                                up, up ? n.x : 1, r + 2, q + 2, (q + 1) * n.y + r + 1),
                            valid};
      }
      u -= 2 * xface;
      if (u < yface) {
        const int64_t q = udiv32(u, n.x);
        return ValidatedIdx{brick_idx_<Region::YUpBoundary>(u - q * n.x + 1, n.y, q + 2, u + n.x),
                            valid};
      }
      u -= yface;
      const int64_t q = udiv32(u, n.x);
      return ValidatedIdx{brick_idx_<Region::UpBoundary>(u - q * n.x + 1, q + 1, n.z, u), valid};
    }
  }
  return ValidatedIdx{Idx{}, false};
}

/**
 * Result of resolving a neighbour offset against a @ref Span : the neighbour's disaggregated
 * @ref Idx and a @c valid flag. Internal helper struct; not part of the user-facing API.
 */
struct NeighbourIdx {
  Idx idx{};
  bool valid{false};
};

/**
 * Internal helper: resolve @p (idx + offset) against @p span and report whether the result is
 * reachable, populating the region fields on the returned @ref Idx .
 *
 * - @b Storage bounds: @c x must stay inside the buffer; @c y / @c z may step into a ghost row /
 *   plane only when the span exchanges a halo on that axis, since an unexchanged ghost is never
 *   filled.
 * - @b Global-domain bounds: per axis, a cell mapping outside @c [0, global_extent) is rejected
 *   unless @c boundary_mode() wraps that axis.
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
  const int64_t x_lo = span.ghost_x() - span.halo_x();
  const int64_t x_hi = haloed.x - span.ghost_x() + span.halo_x();
  const int64_t y_lo = span.ghost_y() - span.halo_y();
  const int64_t y_hi = haloed.y - span.ghost_y() + span.halo_y();
  const int64_t z_lo = Span::ghost_radius - span.halo_z();
  const int64_t z_hi = haloed.z - Span::ghost_radius + span.halo_z();
  if (bx < x_lo || bx >= x_hi || by < y_lo || by >= y_hi || bz < z_lo || bz >= z_hi) {
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
  return NeighbourIdx{span.make_idx(bx, by, bz), true};
}

/**
 * Slab-only trusted pull classification — bit-identical to @c dense_disg::pull_idx .
 * Kept separate so the slab @c read_pull seam path never instantiates the nested
 * pencil/brick @c make_idx .
 */
[[nodiscard]] CARBON_HD constexpr auto
slab_pull_idx(const Span& span,
              const Idx& idx,
              ::carbon::int8_3d offset) noexcept -> Idx {
  const int64_3d cell = idx.cell();
  const int64_t bx = cell.x + static_cast<int64_t>(offset.x);
  const int64_t by = cell.y + static_cast<int64_t>(offset.y);
  const int64_t bz = cell.z + static_cast<int64_t>(offset.z);
  const int64_t lz = span.haloed_extent().z - 2 * Span::ghost_radius;
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

/**
 * Internal helper: disaggregated @ref Idx for @p (idx + offset) with @b no validity check (the
 * caller guarantees the neighbour is in-buffer). Backs @c Partition::read_pull 's seam path.
 */
[[nodiscard]] CARBON_HD constexpr auto
pull_idx(const Span& span,
         const Idx& idx,
         ::carbon::int8_3d offset) noexcept -> Idx {
  if (span.is_slab()) {
    return slab_pull_idx(span, idx, offset);
  }
  const int64_3d cell = idx.cell();
  return span.make_idx(cell.x + static_cast<int64_t>(offset.x),
                       cell.y + static_cast<int64_t>(offset.y),
                       cell.z + static_cast<int64_t>(offset.z));
}

/**
 * Device-launch wrapper that stamps @p Kind onto a @ref Span so @c Partition hot paths can
 * @c if constexpr -specialize (slab codegen matches @c dense_disg ). Constructed from the
 * runtime @ref Span at kernel entry; user lambdas should take the span as @c auto .
 */
template <StorageKind Kind>
struct TypedSpan : Span {
  static constexpr StorageKind kind = Kind;

  CARBON_HD constexpr TypedSpan() noexcept = default;
  CARBON_HD constexpr explicit TypedSpan(Span span) noexcept : Span(span) {}
};

}  // namespace carbon::detail::dense_disg_distributed
