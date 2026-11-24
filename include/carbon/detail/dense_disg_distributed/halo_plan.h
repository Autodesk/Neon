#pragma once

/**
 * @file halo_plan.h
 * @brief Host-side planning for the @c DenseDisgDistributed pencil and brick halos.
 *
 * A pencil chunk exchanges with at most eight peers in one NCCL group: four face peers
 * ( @c ±y , @c ±z ) and four yz-edge peers. Each (direction, component run) is one message
 * between two contiguous ranges of the chunk buffers (see @c dense_disg_distributed::Span ):
 *
 * @code
 *   direction (dy, dz)   sender region       receiver region     runs
 *   ( 0, +1)             UpBoundary          LowerGhost          maximal c_z runs (as slab)
 *   ( 0, -1)             DownBoundary        UpperGhost          maximal c_z runs (as slab)
 *   (+1,  0)             YUpBoundary         YLowerGhost         one covering run
 *   (-1,  0)             YDownBoundary       YUpperGhost         one covering run
 *   (+1, +1)             EdgeSendYHiZHi      LowerGhostYLo       one covering run
 *   (+1, -1)             EdgeSendYHiZLo      UpperGhostYLo       one covering run
 *   (-1, +1)             EdgeSendYLoZHi      LowerGhostYHi       one covering run
 *   (-1, -1)             EdgeSendYLoZLo      UpperGhostYHi       one covering run
 * @endcode
 *
 * Component selection per direction: @c Standard moves every component; @c Lattice moves the
 * populations whose @c c_q matches the direction's sign on every axis it crosses;
 * @c LatticeDistributed moves the populations with a nonzero @c c_q on every axis it crosses.
 * y-face and edge messages are widened to the single run covering every selected population, so
 * each of those peers gets exactly one message (the extra populations land in ghost storage
 * nothing else writes during the exchange).
 *
 * Before a y face or edge is sent its shadow rows / strip are refreshed from the canonical
 * z-boundary rows by @ref append_send_refresh (two shadow rows per y face, one strip per edge).
 *
 * Brick chunks use the 26-direction table @ref kBrickDirs (faces, edges, corners) with
 * @ref send_region / @ref recv_region overloads on @ref BrickDir : z faces ship the boundary
 * planes, y / x faces their face groups, edges and corners their send strips, each into the
 * matching ghost group of the peer. Every direction moves exactly one widened run
 * ( @ref brick_run ), and @ref append_brick_send_refresh refreshes the shadows / strips first.
 */

#include <carbon/boundary_mode_distributed.h>
#include <carbon/detail/dense_disg_distributed/row_copy.h>
#include <carbon/detail/dense_disg_distributed/span.h>
#include <carbon/halo_disg.h>
#include <carbon/integer3d.h>
#include <carbon/process_grid.h>
#include <carbon/stencil.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace carbon::detail::dense_disg_distributed {

/** Exchange direction from sender to receiver in process-grid @c (y, z) coordinates. */
struct PencilDir {
  int32_t dy{};
  int32_t dz{};
};

/**
 * Canonical direction order. Every rank posts, per direction, its Send to @c coords + d and then
 * its Recv from @c coords - d , so the per-peer NCCL FIFOs pair Send ↔ Recv by call order.
 */
inline constexpr PencilDir kPencilDirs[8] = {
    {0, 1}, {0, -1}, {1, 0}, {-1, 0}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1},
};

/** Region the sender ships for direction @p d . */
[[nodiscard]] constexpr auto
send_region(PencilDir d) noexcept -> Region {
  if (d.dy == 0) {
    return d.dz > 0 ? Region::UpBoundary : Region::DownBoundary;
  }
  if (d.dz == 0) {
    return d.dy > 0 ? Region::YUpBoundary : Region::YDownBoundary;
  }
  if (d.dy > 0) {
    return d.dz > 0 ? Region::EdgeSendYHiZHi : Region::EdgeSendYHiZLo;
  }
  return d.dz > 0 ? Region::EdgeSendYLoZHi : Region::EdgeSendYLoZLo;
}

/** Ghost region the receiver fills for direction @p d (on its @c -d side). */
[[nodiscard]] constexpr auto
recv_region(PencilDir d) noexcept -> Region {
  if (d.dy == 0) {
    return d.dz > 0 ? Region::LowerGhost : Region::UpperGhost;
  }
  if (d.dz == 0) {
    return d.dy > 0 ? Region::YLowerGhost : Region::YUpperGhost;
  }
  if (d.dy > 0) {
    return d.dz > 0 ? Region::LowerGhostYLo : Region::UpperGhostYLo;
  }
  return d.dz > 0 ? Region::LowerGhostYHi : Region::UpperGhostYHi;
}

/** Component runs moved for direction @p d ; empty when no component crosses it. */
template <int Components>
[[nodiscard]] inline auto
pencil_runs(StencilSemantic semantic,
            const Stencil& stencil,
            PencilDir d) -> std::vector<halo_disg::CardinalityRun> {
  const bool bidirectional = semantic == StencilSemantic::LatticeDistributed;
  const auto crosses = [bidirectional](int32_t sign, int32_t c) noexcept -> bool {
    if (sign == 0) {
      return true;
    }
    if (bidirectional) {
      return c != 0;
    }
    return sign > 0 ? c > 0 : c < 0;
  };
  std::vector<halo_disg::CardinalityRun> runs;
  int j_start = -1;
  int j_len = 0;
  for (int j = 0; j < Components; ++j) {
    bool active = true;
    if (semantic != StencilSemantic::Standard) {
      const int8_3d p = stencil.points()[static_cast<size_t>(j)];
      active = crosses(d.dy, p.y) && crosses(d.dz, p.z);
    }
    if (active) {
      if (j_len == 0) {
        j_start = j;
      }
      ++j_len;
    } else if (j_len > 0) {
      runs.push_back({j_start, j_len});
      j_len = 0;
    }
  }
  if (j_len > 0) {
    runs.push_back({j_start, j_len});
  }
  if (d.dy != 0 && runs.size() > 1U) {
    const int first = runs.front().j_start;
    const int last = runs.back().j_start + runs.back().j_len;
    runs.assign(1U, halo_disg::CardinalityRun{first, last - first});
  }
  return runs;
}

/**
 * Process coordinates of the rank at @p coords + @p d , wrapping @c y / @c z when @p mode makes
 * them periodic; @c std::nullopt past an open edge of the process grid.
 */
[[nodiscard]] inline auto
pencil_peer(const ProcessGrid& pg,
            int32_3d coords,
            PencilDir d,
            BoundaryModeDistributed mode) -> std::optional<int32_3d> {
  const int32_3d dims = pg.dims();
  const auto step = [](int32_t c, int32_t delta, int32_t n,
                       bool periodic) -> std::optional<int32_t> {
    const int32_t v = c + delta;
    if (v >= 0 && v < n) {
      return v;
    }
    if (!periodic) {
      return std::nullopt;
    }
    return ((v % n) + n) % n;
  };
  const auto y = step(coords.y, d.dy, dims.y, is_periodic(mode, Axis::Y));
  const auto z = step(coords.z, d.dz, dims.z, is_periodic(mode, Axis::Z));
  if (!y.has_value() || !z.has_value()) {
    return std::nullopt;
  }
  return int32_3d{coords.x, *y, *z};
}

/** Contiguous element range of one message inside a chunk buffer. */
struct PencilMessage {
  size_t first{};
  size_t count{};
};

/** Range of components @p run of region @p r in a @p components -cardinality buffer of @p s . */
[[nodiscard]] inline auto
region_run(const Span& s,
           Region r,
           int components,
           halo_disg::CardinalityRun run) -> PencilMessage {
  const int64_t per_component = s.region_storage_elements(r, 1);
  return PencilMessage{
      static_cast<size_t>(s.region_storage_offset(r, components) + run.j_start * per_component),
      static_cast<size_t>(run.j_len * per_component)};
}

/**
 * Append the row copies that refresh the send block of direction @p d for components @p run :
 * the two edge-row shadows of a y face, or the strip of a yz edge. A pure z face needs none.
 */
inline auto
append_send_refresh(const Span& s,
                    int components,
                    PencilDir d,
                    halo_disg::CardinalityRun run,
                    RowCopyArgs& args) -> void {
  if (d.dy == 0 || run.j_len <= 0) {
    return;
  }
  const int64_t nx = s.haloed_extent().x;
  const int64_t ly = s.logic_extent().y;
  const int64_t plane_stride = s.region_storage_elements(Region::DownBoundary, 1);
  const int64_t row = d.dy > 0 ? ly - 1 : 0;
  const auto source = [&](Region plane) -> int64_t {
    return s.region_storage_offset(plane, components) +
           static_cast<int64_t>(run.j_start) * plane_stride + row * nx;
  };
  const Region dst = send_region(d);
  if (d.dz == 0) {
    const int64_t column_stride = s.region_storage_elements(dst, 1);
    const int64_t base = s.region_storage_offset(dst, components) +
                         static_cast<int64_t>(run.j_start) * column_stride;
    const int64_t lz = s.logic_extent().z;
    args.push(
        RowCopyJob{source(Region::DownBoundary), plane_stride, base, column_stride, run.j_len, nx});
    args.push(RowCopyJob{source(Region::UpBoundary), plane_stride, base + (lz - 1) * nx,
                         column_stride, run.j_len, nx});
    return;
  }
  const Region plane = d.dz > 0 ? Region::UpBoundary : Region::DownBoundary;
  args.push(
      RowCopyJob{source(plane), plane_stride,
                 s.region_storage_offset(dst, components) + static_cast<int64_t>(run.j_start) * nx,
                 nx, run.j_len, nx});
}

/** Exchange direction from sender to receiver in process-grid @c (x, y, z) coordinates. */
struct BrickDir {
  int32_t dx{};
  int32_t dy{};
  int32_t dz{};
};

/**
 * Canonical brick direction order: six faces, twelve edges, eight corners. Same pairing contract
 * as @ref kPencilDirs : per direction, Send to @c coords + d then Recv from @c coords - d .
 */
inline constexpr BrickDir kBrickDirs[26] = {
    {0, 0, 1},   {0, 0, -1}, {0, 1, 0},   {0, -1, 0},  {1, 0, 0},    {-1, 0, 0}, {0, 1, 1},
    {0, 1, -1},  {0, -1, 1}, {0, -1, -1}, {1, 0, 1},   {1, 0, -1},   {-1, 0, 1}, {-1, 0, -1},
    {1, 1, 0},   {1, -1, 0}, {-1, 1, 0},  {-1, -1, 0}, {1, 1, 1},    {1, 1, -1}, {1, -1, 1},
    {1, -1, -1}, {-1, 1, 1}, {-1, 1, -1}, {-1, -1, 1}, {-1, -1, -1},
};

/** Number of nonzero axes of @p d : 1 face, 2 edge, 3 corner. */
[[nodiscard]] constexpr auto
brick_dir_rank(BrickDir d) noexcept -> int {
  return (d.dx != 0 ? 1 : 0) + (d.dy != 0 ? 1 : 0) + (d.dz != 0 ? 1 : 0);
}

/** Region the sender ships for brick direction @p d (its owned layer on the @c +d side). */
[[nodiscard]] constexpr auto
send_region(BrickDir d) noexcept -> Region {
  const bool xh = d.dx > 0;
  const bool yh = d.dy > 0;
  const bool zh = d.dz > 0;
  switch (brick_dir_rank(d)) {
    case 1:
      if (d.dx != 0) {
        return xh ? Region::XUpBoundary : Region::XDownBoundary;
      }
      if (d.dy != 0) {
        return yh ? Region::YUpBoundary : Region::YDownBoundary;
      }
      return zh ? Region::UpBoundary : Region::DownBoundary;
    case 2:
      if (d.dx == 0) {
        return zh ? (yh ? Region::EdgeSendYHiZHi : Region::EdgeSendYLoZHi)
                  : (yh ? Region::EdgeSendYHiZLo : Region::EdgeSendYLoZLo);
      }
      if (d.dy == 0) {
        return zh ? (xh ? Region::EdgeSendXHiZHi : Region::EdgeSendXLoZHi)
                  : (xh ? Region::EdgeSendXHiZLo : Region::EdgeSendXLoZLo);
      }
      return yh ? (xh ? Region::EdgeSendXHiYHi : Region::EdgeSendXLoYHi)
                : (xh ? Region::EdgeSendXHiYLo : Region::EdgeSendXLoYLo);
    default:
      return corner_region(Region::CornerSendXLoYLoZLo, xh, yh, zh);
  }
}

/**
 * Ghost region the receiver fills for brick direction @p d : its ghost layer on the @c -d side
 * (low ghost on every axis where @c d is positive).
 */
[[nodiscard]] constexpr auto
recv_region(BrickDir d) noexcept -> Region {
  const bool xh = d.dx < 0;
  const bool yh = d.dy < 0;
  const bool zh = d.dz < 0;
  switch (brick_dir_rank(d)) {
    case 1:
      if (d.dx != 0) {
        return xh ? Region::XUpperGhost : Region::XLowerGhost;
      }
      if (d.dy != 0) {
        return yh ? Region::YUpperGhost : Region::YLowerGhost;
      }
      return zh ? Region::UpperGhost : Region::LowerGhost;
    case 2:
      if (d.dx == 0) {
        return zh ? (yh ? Region::UpperGhostYHi : Region::UpperGhostYLo)
                  : (yh ? Region::LowerGhostYHi : Region::LowerGhostYLo);
      }
      if (d.dy == 0) {
        return zh ? (xh ? Region::UpperGhostXHi : Region::UpperGhostXLo)
                  : (xh ? Region::LowerGhostXHi : Region::LowerGhostXLo);
      }
      return yh ? (xh ? Region::YUpperGhostXHi : Region::YUpperGhostXLo)
                : (xh ? Region::YLowerGhostXHi : Region::YLowerGhostXLo);
    default:
      return corner_region(Region::CornerGhostXLoYLoZLo, xh, yh, zh);
  }
}

/**
 * The single component run moved for brick direction @p d , or @c std::nullopt when no component
 * crosses it. Selection is the same as @ref pencil_runs on all three axes; the selected
 * components are then widened to the one run covering them, so every face, edge and corner peer
 * gets exactly one message (the extra populations land in ghost storage nothing else writes
 * during the exchange).
 */
template <int Components>
[[nodiscard]] inline auto
brick_run(StencilSemantic semantic,
          const Stencil& stencil,
          BrickDir d) -> std::optional<halo_disg::CardinalityRun> {
  const bool bidirectional = semantic == StencilSemantic::LatticeDistributed;
  const auto crosses = [bidirectional](int32_t sign, int32_t c) noexcept -> bool {
    if (sign == 0) {
      return true;
    }
    if (bidirectional) {
      return c != 0;
    }
    return sign > 0 ? c > 0 : c < 0;
  };
  int first = -1;
  int last = -1;
  for (int j = 0; j < Components; ++j) {
    bool active = true;
    if (semantic != StencilSemantic::Standard) {
      const int8_3d p = stencil.points()[static_cast<size_t>(j)];
      active = crosses(d.dx, p.x) && crosses(d.dy, p.y) && crosses(d.dz, p.z);
    }
    if (active) {
      first = first < 0 ? j : first;
      last = j;
    }
  }
  if (first < 0) {
    return std::nullopt;
  }
  return halo_disg::CardinalityRun{first, last - first + 1};
}

/**
 * Process coordinates of the rank at @p coords + @p d , wrapping @c y / @c z when @p mode makes
 * them periodic; @c std::nullopt past an open edge of the process grid ( @c x never wraps).
 */
[[nodiscard]] inline auto
brick_peer(const ProcessGrid& pg,
           int32_3d coords,
           BrickDir d,
           BoundaryModeDistributed mode) -> std::optional<int32_3d> {
  const int32_t x = coords.x + d.dx;
  if (x < 0 || x >= pg.dims().x) {
    return std::nullopt;
  }
  const auto yz = pencil_peer(pg, coords, PencilDir{d.dy, d.dz}, mode);
  if (!yz.has_value()) {
    return std::nullopt;
  }
  return int32_3d{x, yz->y, yz->z};
}

/**
 * Append the copies that refresh the send block of brick direction @p d for components @p run
 * (sources are canonical storage only, see @c Span ):
 *
 * - z face: none (the z-boundary plane is the send block).
 * - y face: rows @c bz = 1, Z of the y-boundary group, from the z planes (contiguous).
 * - x face: rows @c bz = 1, Z of the x-boundary group, gathered from the z planes (step @c X ),
 *   and columns @c by = 1, Y for @c bz = 2..Z-1 , gathered from the y-boundary groups.
 * - edge along x / y: one row / column of a z plane.
 * - edge along z: its two end cells from the z planes and its middle from a y-boundary group.
 * - corner: one cell of a z plane.
 */
inline auto
append_brick_send_refresh(const Span& s,
                          int components,
                          BrickDir d,
                          halo_disg::CardinalityRun run,
                          RowCopyArgs& args) -> void {
  const int rank = brick_dir_rank(d);
  if (run.j_len <= 0 || (rank == 1 && d.dz != 0)) {
    return;
  }
  const int64_3d n = s.logic_extent();
  const int64_t xr = d.dx > 0 ? n.x - 1 : 0;
  const int64_t yr = d.dy > 0 ? n.y - 1 : 0;
  const Region dst = send_region(d);
  const auto copy = [&](Region from, int64_t from_local, int64_t from_step, int64_t to_local,
                        int64_t to_step, int64_t len) {
    const int64_t from_dim = s.region_storage_elements(from, 1);
    const int64_t to_dim = s.region_storage_elements(dst, 1);
    args.push(RowCopyJob{
        s.region_storage_offset(from, components) + run.j_start * from_dim + from_local, from_dim,
        s.region_storage_offset(dst, components) + run.j_start * to_dim + to_local, to_dim,
        run.j_len, len, from_step, to_step});
  };
  if (rank == 1 && d.dy != 0) {
    copy(Region::DownBoundary, yr * n.x, 1, 0, 1, n.x);
    copy(Region::UpBoundary, yr * n.x, 1, (n.z - 1) * n.x, 1, n.x);
    return;
  }
  if (rank == 1) {
    copy(Region::DownBoundary, xr, n.x, 0, 1, n.y);
    copy(Region::UpBoundary, xr, n.x, (n.z - 1) * n.y, 1, n.y);
    copy(Region::YDownBoundary, n.x + xr, n.x, n.y, n.y, n.z - 2);
    copy(Region::YUpBoundary, n.x + xr, n.x, 2 * n.y - 1, n.y, n.z - 2);
    return;
  }
  const Region plane = d.dz > 0 ? Region::UpBoundary : Region::DownBoundary;
  if (rank == 3) {
    copy(plane, yr * n.x + xr, 1, 0, 1, 1);
    return;
  }
  if (d.dx == 0) {
    copy(plane, yr * n.x, 1, 0, 1, n.x);
    return;
  }
  if (d.dy == 0) {
    copy(plane, xr, n.x, 0, 1, n.y);
    return;
  }
  copy(Region::DownBoundary, yr * n.x + xr, 1, 0, 1, 1);
  copy(d.dy > 0 ? Region::YUpBoundary : Region::YDownBoundary, n.x + xr, n.x, 1, 1, n.z - 2);
  copy(Region::UpBoundary, yr * n.x + xr, 1, n.z - 1, 1, 1);
}

}  // namespace carbon::detail::dense_disg_distributed
