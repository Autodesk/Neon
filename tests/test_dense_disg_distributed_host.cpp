// Host-side verification of DenseDisgDistributed (Slab, Pencil and Brick).
//
// Part A  layout helpers (detail/dense_disg_distributed/layout.h): slab, pencil and brick chunks
//         against hand expectations (brick ownership also against dense_distributed::rank_chunk),
//         ghost-axis selection, boundary-mode and stencil-radius validation.
// Part B  Span parity with dense_disg::Span on emulated multi-slab worlds (no GPU): every view's
//         Idx and address, region offsets, neighbour resolution and read_pull addresses must match
//         the DenseDisg span built for the same slab.
// Part C  grid checks on the visible devices. Slab: a DenseDisgDistributed and a DenseDisg with the
//         same System / extent / halo have the same logic extent and storage size per span, and
//         fill_owned_host produces bit-identical host buffers. Pencil: owned-cell values after
//         fill_owned_host match a DenseDistributed on the same ProcessGrid, and the host and device
//         halos (Standard and D3Q19 lattice) fill every ghost with the cell it mirrors. Brick
//         (multi-GPU, x split): owned cells sit where dense_distributed::rank_chunk puts them and
//         hold their fill_owned_host values, and the host / device halos fill every ghost
//         (faces, edges, corners). Skipped when no device is visible.
// Part D  pencil storage oracle on emulated worlds (no GPU): region sizes / prefixes against an
//         independent table, the full storage map (cell -> address) against the nested layout
//         formula, coverage (only the eight shadow / strip rows per component unmapped), the
//         Standard / Internal / Boundary views, neighbour_idx validity and read_pull addresses.
// Part E  pencil halo plan on emulated worlds (no GPU): shadow / strip refresh, then every face
//         and edge message copied as one contiguous range between host buffers. Each send block
//         must hold exactly its face / edge cells in storage order, and every ghost must end up
//         holding the cell it mirrors (lattice: the populations crossing into it).
// Part F  brick storage oracle on emulated worlds (no GPU): region sizes / prefixes against the
//         spec order (128-byte aligned Internal pitch), the full storage map, coverage (only
//         padding, shadows and send strips unmapped), x faces stored y-fastest and contiguous,
//         the Standard / Internal / flattened Boundary views, neighbour_idx and read_pull.
// Part G  brick halo plan on emulated worlds (no GPU): refresh, then one contiguous message per
//         face / edge / corner peer (none for D3Q19 corners); every ghost must end up holding the
//         cell it mirrors.

#include <carbon/boundary_mode.h>
#include <carbon/boundary_mode_distributed.h>
#include <carbon/carbon.h>
#include <carbon/data_view.h>
#include <carbon/dense_disg.h>
#include <carbon/dense_disg_distributed.h>
#include <carbon/dense_distributed.h>
#include <carbon/detail/dense_disg/span.h>
#include <carbon/detail/dense_disg_distributed/halo_plan.h>
#include <carbon/detail/dense_disg_distributed/layout.h>
#include <carbon/detail/dense_disg_distributed/partition.h>
#include <carbon/detail/dense_disg_distributed/row_copy.h>
#include <carbon/detail/dense_disg_distributed/span.h>
#include <carbon/halo_disg_distributed.h>
#include <carbon/integer3d.h>
#include <carbon/memory_side.h>
#include <carbon/process_grid.h>
#include <carbon/stencil.h>
#include <carbon/transfer.h>

#include "expect_throw.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

using carbon::Axis;
using carbon::BoundaryMode;
using carbon::BoundaryModeDistributed;
using carbon::DataView;
using carbon::DecompositionMode;
using carbon::int32_3d;
using carbon::int64_3d;
using carbon::int8_3d;
using carbon::ProcessGrid;
namespace ddd = carbon::detail::dense_disg_distributed;
namespace dsg = carbon::detail::dense_disg;

auto
eq3(const int64_3d& a,
    const int64_3d& b) -> bool {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

auto
radius1_stencil() -> carbon::Stencil {
  return carbon::Stencil(
      std::vector<int32_3d>{{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);
}

auto
d3q19_stencil() -> carbon::Stencil {
  std::vector<int32_3d> pts{{0, 0, 0}};
  for (int z = -1; z <= 1; ++z) {
    for (int y = -1; y <= 1; ++y) {
      for (int x = -1; x <= 1; ++x) {
        const int nz = (x != 0) + (y != 0) + (z != 0);
        if (nz >= 1 && nz <= 2) {
          pts.push_back({x, y, z});
        }
      }
    }
  }
  return carbon::Stencil(pts);
}

auto
same_idx(const ddd::Idx& a,
         const dsg::Idx& b) -> bool {
  return eq3(a.cell(), b.cell()) && a.offset_local_no_card() == b.offset_local_no_card() &&
         a.region_first_z() == b.region_first_z() && a.region_z_dim() == b.region_z_dim();
}

auto
wrap_axis(int64_t v,
          int64_t n,
          bool periodic,
          int64_t& out) noexcept -> bool {
  if (v >= 0 && v < n) {
    out = v;
    return true;
  }
  if (!periodic) {
    return false;
  }
  out = ((v % n) + n) % n;
  return true;
}

auto
wrap_global(const int64_3d& g,
            const int64_3d& global,
            BoundaryModeDistributed mode,
            int64_3d& out) noexcept -> bool {
  return wrap_axis(g.x, global.x, carbon::is_periodic(mode, Axis::X), out.x) &&
         wrap_axis(g.y, global.y, carbon::is_periodic(mode, Axis::Y), out.y) &&
         wrap_axis(g.z, global.z, carbon::is_periodic(mode, Axis::Z), out.z);
}

auto
encode(const int64_3d& g,
       int c) noexcept -> int64_t {
  return g.x + 10 * g.y + 1000 * g.z + 100000 * static_cast<int64_t>(c);
}

/** Every chunk of a slab / pencil / brick world, built the way DenseDisgDistributed builds them. */
auto
make_world(const int64_3d& global,
           const ProcessGrid& pg,
           BoundaryModeDistributed mode,
           int64_t halo) -> std::vector<ddd::Span> {
  std::vector<ddd::Span> spans;
  const int64_t ghost_y = ddd::ghost_y_for(pg, mode);
  const int64_t ghost_x = ddd::ghost_x_for(pg);
  for (int32_t r = 0; r < pg.world_size(); ++r) {
    const auto chunk = ddd::disg_chunk(global, pg, pg.coords_of(r), halo, mode);
    spans.emplace_back(chunk.haloed, global, chunk.xy_pitch, chunk.offset, chunk.halo, mode,
                       carbon::float32_3d{}, 1.0F, ghost_y, ghost_x);
  }
  return spans;
}

auto
part_a_layout() -> void {
  const int64_3d global{5, 4, 10};
  const ProcessGrid pg = ProcessGrid::custom(3, 1, 1, 3);
  const int64_t want_owned[] = {4, 3, 3};
  const int64_t want_origin[] = {0, 4, 7};
  for (const int64_t halo_z : {int64_t{0}, int64_t{1}}) {
    for (int32_t r = 0; r < 3; ++r) {
      const auto chunk = ddd::slab_chunk(global, pg, pg.coords_of(r), halo_z);
      assert(eq3(chunk.owned, {5, 4, want_owned[r]}));
      assert(eq3(chunk.haloed, {5, 4, want_owned[r] + 2 * ddd::kGhostRadius}));
      assert(eq3(chunk.offset, {0, 0, want_origin[r] - ddd::kGhostRadius}));
      assert(eq3(chunk.halo, {0, 0, halo_z}));
      assert(chunk.xy_pitch == 20);
    }
  }

  CARBON_EXPECT_THROW(ddd::slab_chunk(global, ProcessGrid::custom(2, 2, 1, 1), {0, 0, 0}, 1),
                      std::invalid_argument);
  CARBON_EXPECT_THROW(ddd::slab_chunk(global, ProcessGrid::custom(2, 1, 2, 1), {0, 0, 0}, 1),
                      std::invalid_argument);

  assert(ddd::ghost_x_for(ProcessGrid::make(8, DecompositionMode::Brick)) == 1);
  assert(ddd::ghost_x_for(ProcessGrid::make(4, DecompositionMode::Brick)) ==
         (ProcessGrid::make(4, DecompositionMode::Brick).dims().x > 1 ? 1 : 0));
  assert(ddd::ghost_x_for(ProcessGrid::make(1, DecompositionMode::Brick)) == 0);
  assert(ddd::ghost_x_for(ProcessGrid::custom(2, 2, 1, 1)) == 1);
  assert(ddd::ghost_x_for(ProcessGrid::custom(4, 1, 2, 2)) == 0);
  assert(ddd::ghost_x_for(ProcessGrid::custom(4, 1, 1, 4)) == 0);
  assert(ddd::ghost_y_for(ProcessGrid::custom(2, 2, 1, 1), BoundaryModeDistributed::Open) == 1);

  const ProcessGrid brick8 = ProcessGrid::custom(8, 2, 2, 2);
  const int64_3d bglobal{7, 5, 6};
  const int64_t want_bx_owned[] = {4, 3};
  const int64_t want_by_owned[] = {3, 2};
  for (const int64_t halo : {int64_t{0}, int64_t{1}}) {
    for (int32_t r = 0; r < 8; ++r) {
      const int32_3d co = brick8.coords_of(r);
      const auto chunk = ddd::disg_chunk(bglobal, brick8, co, halo, BoundaryModeDistributed::Open);
      const int64_3d owned{want_bx_owned[co.x], want_by_owned[co.y], 3};
      assert(eq3(chunk.owned, owned));
      assert(eq3(chunk.haloed, {owned.x + 2, owned.y + 2, owned.z + 2}));
      assert(eq3(chunk.offset, {4 * co.x - 1, 3 * co.y - 1, 3 * co.z - 1}));
      assert(eq3(chunk.halo, {halo, halo, halo}));
      assert(chunk.xy_pitch == (owned.x + 2) * (owned.y + 2));
      const auto dense =
          carbon::detail::dense_distributed::rank_chunk(bglobal, brick8, co, int64_3d{1, 1, 1});
      assert(eq3(chunk.owned, dense.owned));
      assert(eq3(chunk.offset, dense.offset));
    }
  }

  assert(ddd::to_boundary_mode(BoundaryModeDistributed::Open) == BoundaryMode::Open);
  assert(ddd::to_boundary_mode(BoundaryModeDistributed::PeriodicZ) == BoundaryMode::PeriodicZ);
  assert(ddd::to_boundary_mode(BoundaryModeDistributed::PeriodicY) == BoundaryMode::Open);
  assert(ddd::to_boundary_mode(BoundaryModeDistributed::PeriodicY |
                               BoundaryModeDistributed::PeriodicZ) == BoundaryMode::PeriodicZ);
  CARBON_EXPECT_THROW(ddd::to_boundary_mode(BoundaryModeDistributed::PeriodicX),
                      std::invalid_argument);
  CARBON_EXPECT_THROW(ddd::to_boundary_mode(BoundaryModeDistributed::PeriodicX |
                                            BoundaryModeDistributed::PeriodicZ),
                      std::invalid_argument);

  const ProcessGrid slab4 = ProcessGrid::custom(4, 1, 1, 4);
  const ProcessGrid pencil4 = ProcessGrid::custom(4, 1, 2, 2);
  assert(ddd::ghost_y_for(slab4, BoundaryModeDistributed::Open) == 0);
  assert(ddd::ghost_y_for(slab4, BoundaryModeDistributed::PeriodicZ) == 0);
  assert(ddd::ghost_y_for(slab4, BoundaryModeDistributed::PeriodicY) == 1);
  assert(ddd::ghost_y_for(pencil4, BoundaryModeDistributed::Open) == 1);

  const int64_3d pglobal{5, 7, 10};
  const int64_t want_y_owned[] = {4, 3};
  const int64_t want_y_origin[] = {0, 4};
  for (const int64_t halo : {int64_t{0}, int64_t{1}}) {
    for (int32_t r = 0; r < 4; ++r) {
      const int32_3d co = pencil4.coords_of(r);
      const auto chunk = ddd::disg_chunk(pglobal, pencil4, co, halo, BoundaryModeDistributed::Open);
      const int64_t oy = want_y_owned[co.y];
      assert(eq3(chunk.owned, {5, oy, 5}));
      assert(eq3(chunk.haloed, {5, oy + 2, 7}));
      assert(eq3(chunk.offset, {0, want_y_origin[co.y] - 1, 5 * co.z - 1}));
      assert(eq3(chunk.halo, {0, halo, halo}));
      assert(chunk.xy_pitch == 5 * (oy + 2));
    }
  }
  const auto periodic_y_slab =
      ddd::disg_chunk(pglobal, slab4, {0, 0, 1}, 1, BoundaryModeDistributed::PeriodicY);
  assert(eq3(periodic_y_slab.owned, {5, 7, 3}));
  assert(eq3(periodic_y_slab.haloed, {5, 9, 5}));
  assert(eq3(periodic_y_slab.offset, {0, -1, 2}));
  assert(eq3(periodic_y_slab.halo, {0, 1, 1}));
  const auto open_slab =
      ddd::disg_chunk(pglobal, slab4, {0, 0, 1}, 1, BoundaryModeDistributed::Open);
  assert(eq3(open_slab.haloed, {5, 7, 5}));
  assert(eq3(open_slab.halo, {0, 0, 1}));

  assert(ddd::halo_z_from_stencil(carbon::Stencil{}) == 0);
  assert(ddd::halo_z_from_stencil(radius1_stencil()) == 1);
  CARBON_EXPECT_THROW(
      ddd::halo_z_from_stencil(carbon::Stencil(std::vector<int32_3d>{{0, 0, 2}, {0, 0, -2}})),
      std::invalid_argument);
}

auto
check_slab_parity(const ddd::Span& s,
                  const dsg::Span& d,
                  int components) -> void {
  assert(eq3(s.logic_extent(), d.logic_extent()));
  assert(eq3(s.haloed_extent(), d.haloed_extent()));
  assert(s.z_offset() == d.z_offset());
  assert(s.halo_z() == d.halo_z());
  assert(s.xy_slab_elements() == d.xy_slab_elements());
  assert(s.linear_storage_elements(components) == d.linear_storage_elements(components));

  const int64_t lz = s.logic_extent().z;
  const int64_t blk = static_cast<int64_t>(components) * s.xy_slab_elements();
  assert(s.region_storage_offset(ddd::Region::LowerGhost, components) == 0);
  assert(s.region_storage_offset(ddd::Region::DownBoundary, components) == blk);
  assert(s.region_storage_offset(ddd::Region::Internal, components) == 2 * blk);
  assert(s.region_storage_offset(ddd::Region::UpBoundary, components) == lz * blk);
  assert(s.region_storage_offset(ddd::Region::UpperGhost, components) == (lz + 1) * blk);
  assert(s.region_storage_elements(ddd::Region::Internal, components) == (lz - 2) * blk);
  assert(s.region_storage_elements(ddd::Region::UpBoundary, components) == blk);

  for (const DataView view : {DataView::Standard, DataView::Internal, DataView::Boundary}) {
    const int64_3d e = s.iter_extent(view);
    assert(eq3(e, d.iter_extent(view)));
    for (int64_t iz = 0; iz <= e.z; ++iz) {
      for (int64_t iy = 0; iy < e.y; ++iy) {
        for (int64_t ix = 0; ix < e.x; ++ix) {
          const auto a = s.set_and_validate(view, ix, iy, iz);
          const auto b = d.set_and_validate(view, ix, iy, iz);
          assert(a.valid == b.valid);
          if (!a.valid) {
            continue;
          }
          assert(same_idx(a.idx, b.idx));
          assert(eq3(s.global_location(a.idx), d.global_location(b.idx)));
          for (int c = 0; c < components; ++c) {
            assert(s.linear_offset(a.idx, components, c) == d.linear_offset(b.idx, components, c));
          }
          for (int8_t dz = -1; dz <= 1; ++dz) {
            for (int8_t dy = -1; dy <= 1; ++dy) {
              for (int8_t dx = -1; dx <= 1; ++dx) {
                const int8_3d off{dx, dy, dz};
                const auto na = ddd::neighbour_idx(s, a.idx, off);
                const auto nb = dsg::neighbour_idx(d, b.idx, off);
                assert(na.valid == nb.valid);
                if (na.valid) {
                  assert(same_idx(na.idx, nb.idx));
                  for (int c = 0; c < components; ++c) {
                    assert(s.linear_offset(na.idx, components, c) ==
                           d.linear_offset(nb.idx, components, c));
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  std::vector<int64_t> buf(static_cast<size_t>(s.linear_storage_elements(components)));
  for (size_t k = 0; k < buf.size(); ++k) {
    buf[k] = static_cast<int64_t>(k);
  }
  const ddd::Partition<int64_t, 1> part(buf.data());
  if (components == 1) {
    const int64_3d e = s.iter_extent(DataView::Standard);
    for (int64_t iz = 0; iz < e.z; ++iz) {
      for (int64_t iy = 1; iy + 1 < e.y; ++iy) {
        for (int64_t ix = 1; ix + 1 < e.x; ++ix) {
          const auto a = s.set_and_validate(DataView::Standard, ix, iy, iz);
          for (int8_t dz = -1; dz <= 1; ++dz) {
            const int8_3d off{1, -1, dz};
            const ddd::Idx p = ddd::pull_idx(s, a.idx, off);
            assert(part.read_pull(s, a.idx, off, 0) ==
                   buf[static_cast<size_t>(s.linear_offset(p, 1, 0))]);
          }
        }
      }
    }
  }
}

auto
part_b_span_parity() -> size_t {
  struct Case {
    int pz;
    BoundaryModeDistributed bm;
    int64_t gx, gy, gz;
    int components;
  };
  const Case cases[] = {
      {1, BoundaryModeDistributed::Open, 3, 4, 6, 1},
      {1, BoundaryModeDistributed::PeriodicZ, 3, 4, 6, 2},
      {3, BoundaryModeDistributed::Open, 4, 3, 11, 1},
      {3, BoundaryModeDistributed::PeriodicZ, 4, 3, 11, 3},
      {4, BoundaryModeDistributed::Open, 2, 5, 9, 19},
  };
  size_t slabs = 0;
  for (const Case& c : cases) {
    const ProcessGrid pg = ProcessGrid::custom(c.pz, 1, 1, c.pz);
    const int64_3d global{c.gx, c.gy, c.gz};
    const BoundaryMode bm = ddd::to_boundary_mode(c.bm);
    int64_t running_z = 0;
    for (int32_t r = 0; r < c.pz; ++r) {
      const auto chunk = ddd::slab_chunk(global, pg, pg.coords_of(r), /*halo_z=*/1);
      const ddd::Span s(chunk.haloed, global, chunk.xy_pitch, chunk.offset, chunk.halo, c.bm);

      const int64_t owned_z = chunk.owned.z;
      const int64_3d haloed{global.x, global.y, owned_z + 2};
      const dsg::Span d(haloed, global, global.x * global.y, running_z - 1, 1, bm);
      running_z += owned_z;

      check_slab_parity(s, d, c.components);
      ++slabs;
    }
    assert(running_z == c.gz);
  }
  return slabs;
}

struct PencilCase {
  const char* label;
  int32_t py;
  int32_t pz;
  BoundaryModeDistributed mode;
  int64_3d global;
};

auto
pencil_cases() -> std::vector<PencilCase> {
  constexpr auto kO = BoundaryModeDistributed::Open;
  constexpr auto kY = BoundaryModeDistributed::PeriodicY;
  constexpr auto kZ = BoundaryModeDistributed::PeriodicZ;
  return {
      {"1x2x2-open", 2, 2, kO, {3, 5, 6}},      {"1x2x2-periodicYZ", 2, 2, kY | kZ, {3, 5, 6}},
      {"1x1x1-periodicY", 1, 1, kY, {3, 4, 4}}, {"1x1x3-periodicY", 1, 3, kY, {2, 3, 7}},
      {"1x2x3-open", 2, 3, kO, {2, 7, 9}},      {"1x3x1-open", 3, 1, kO, {4, 6, 2}},
      {"1x3x2-periodicY", 3, 2, kY, {3, 8, 5}},
  };
}

auto
pencil_grid(const PencilCase& pc) -> ProcessGrid {
  return ProcessGrid::custom(pc.py * pc.pz, 1, pc.py, pc.pz);
}

auto
expected_rows(ddd::Region r,
              int64_t ly,
              int64_t lz) -> int64_t {
  switch (r) {
    case ddd::Region::LowerGhost:
    case ddd::Region::DownBoundary:
    case ddd::Region::UpBoundary:
    case ddd::Region::UpperGhost:
      return ly;
    case ddd::Region::YLowerGhost:
    case ddd::Region::YDownBoundary:
    case ddd::Region::YUpBoundary:
    case ddd::Region::YUpperGhost:
      return lz;
    case ddd::Region::Internal:
      return (ly - 2) * (lz - 2);
    default:
      return 1;
  }
}

struct Placement {
  ddd::Region region;
  int64_t row;
};

auto
expected_placement(int64_t by,
                   int64_t bz,
                   int64_t ly,
                   int64_t lz) -> Placement {
  const bool ylo = by == 0;
  const bool yhi = by == ly + 1;
  if (bz == 0 || bz == lz + 1) {
    const bool lower = bz == 0;
    if (ylo) {
      return {lower ? ddd::Region::LowerGhostYLo : ddd::Region::UpperGhostYLo, 0};
    }
    if (yhi) {
      return {lower ? ddd::Region::LowerGhostYHi : ddd::Region::UpperGhostYHi, 0};
    }
    return {lower ? ddd::Region::LowerGhost : ddd::Region::UpperGhost, by - 1};
  }
  if (ylo) {
    return {ddd::Region::YLowerGhost, bz - 1};
  }
  if (yhi) {
    return {ddd::Region::YUpperGhost, bz - 1};
  }
  if (bz == 1) {
    return {ddd::Region::DownBoundary, by - 1};
  }
  if (bz == lz) {
    return {ddd::Region::UpBoundary, by - 1};
  }
  if (by == 1) {
    return {ddd::Region::YDownBoundary, bz - 1};
  }
  if (by == ly) {
    return {ddd::Region::YUpBoundary, bz - 1};
  }
  return {ddd::Region::Internal, (bz - 2) * (ly - 2) + (by - 2)};
}

auto
check_pencil_storage(const ddd::Span& s,
                     int components) -> void {
  const int64_3d n = s.logic_extent();
  const int64_3d h = s.haloed_extent();
  const int64_t nx = h.x;
  const int64_t ly = n.y;
  const int64_t lz = n.z;
  const int64_t cc = components;
  assert(s.ghost_y() == 1);
  assert(h.y == ly + 2 && h.z == lz + 2);
  assert(s.storage_unit() == nx);

  std::array<int64_t, ddd::kPencilRegionCount> first{};
  int64_t prefix = 0;
  for (int ri = 0; ri < ddd::kPencilRegionCount; ++ri) {
    const auto r = static_cast<ddd::Region>(ri);
    const int64_t rows = expected_rows(r, ly, lz);
    assert(s.region_first_z(r) == prefix);
    assert(s.region_z_dim(r) == rows);
    assert(s.region_storage_offset(r, components) == prefix * cc * nx);
    assert(s.region_storage_elements(r, components) == rows * cc * nx);
    first[static_cast<size_t>(ri)] = prefix;
    prefix += rows;
  }
  assert(prefix == (ly + 2) * (lz + 2) + 8);
  const int64_t total = s.linear_storage_elements(components);
  assert(total == prefix * cc * nx);

  std::vector<int> hits(static_cast<size_t>(total), 0);
  for (int64_t bz = 0; bz < h.z; ++bz) {
    for (int64_t by = 0; by < h.y; ++by) {
      for (int64_t bx = 0; bx < nx; ++bx) {
        const Placement p = expected_placement(by, bz, ly, lz);
        const int64_t rows = expected_rows(p.region, ly, lz);
        const int64_t base = first[static_cast<size_t>(p.region)];
        const ddd::Idx idx = s.make_idx(bx, by, bz);
        assert(eq3(idx.cell(), {bx, by, bz}));
        for (int c = 0; c < components; ++c) {
          const int64_t want = (base * cc + c * rows) * nx + p.row * nx + bx;
          const int64_t got = s.linear_offset(idx, components, c);
          assert(got == want);
          ++hits[static_cast<size_t>(got)];
        }
      }
    }
  }

  std::vector<char> spare(static_cast<size_t>(total), 0);
  const auto mark_row = [&](ddd::Region r, int64_t row) {
    for (int c = 0; c < components; ++c) {
      const int64_t start =
          s.region_storage_offset(r, components) + c * s.region_storage_elements(r, 1) + row * nx;
      for (int64_t x = 0; x < nx; ++x) {
        spare[static_cast<size_t>(start + x)] = 1;
      }
    }
  };
  for (const ddd::Region r : {ddd::Region::YDownBoundary, ddd::Region::YUpBoundary}) {
    mark_row(r, 0);
    mark_row(r, lz - 1);
  }
  for (const ddd::Region r : {ddd::Region::EdgeSendYLoZLo, ddd::Region::EdgeSendYHiZLo,
                              ddd::Region::EdgeSendYLoZHi, ddd::Region::EdgeSendYHiZHi}) {
    mark_row(r, 0);
  }
  for (size_t k = 0; k < hits.size(); ++k) {
    assert(hits[k] == (spare[k] != 0 ? 0 : 1));
  }
}

auto
check_pencil_views(const ddd::Span& s) -> void {
  const int64_3d n = s.logic_extent();
  const int64_t ly = n.y;
  const int64_t lz = n.z;

  const int64_3d es = s.iter_extent(DataView::Standard);
  assert(eq3(es, n));
  for (int64_t iz = 0; iz <= es.z; ++iz) {
    for (int64_t iy = 0; iy < es.y; ++iy) {
      for (int64_t ix = 0; ix < es.x; ++ix) {
        const auto v = s.set_and_validate(DataView::Standard, ix, iy, iz);
        assert(v.valid == (iz < es.z));
        if (v.valid) {
          assert(same_idx(v.idx, s.make_idx(ix, iy + 1, iz + 1)));
        }
      }
    }
  }

  assert(eq3(s.iter_extent(DataView::Internal), {n.x, ly - 2, lz - 2}));
  assert(eq3(s.iter_extent(DataView::Boundary), {n.x, 2 * ly + 2 * (lz - 2), 1}));
  std::vector<int> owned_hits(static_cast<size_t>(n.x * ly * lz), 0);
  for (const DataView view : {DataView::Internal, DataView::Boundary}) {
    const int64_3d e = s.iter_extent(view);
    for (int64_t iz = 0; iz <= e.z; ++iz) {
      for (int64_t iy = 0; iy <= e.y; ++iy) {
        for (int64_t ix = 0; ix < e.x; ++ix) {
          const auto v = s.set_and_validate(view, ix, iy, iz);
          assert(v.valid == (iy < e.y && iz < e.z));
          if (!v.valid) {
            continue;
          }
          const int64_3d c = v.idx.cell();
          assert(c.x >= 0 && c.x < n.x && c.y >= 1 && c.y <= ly && c.z >= 1 && c.z <= lz);
          assert(same_idx(v.idx, s.make_idx(c)));
          const bool internal =
              expected_placement(c.y, c.z, ly, lz).region == ddd::Region::Internal;
          assert(internal == (view == DataView::Internal));
          ++owned_hits[static_cast<size_t>(c.x + n.x * ((c.y - 1) + ly * (c.z - 1)))];
        }
      }
    }
  }
  for (const int k : owned_hits) {
    assert(k == 1);
  }
}

auto
check_pencil_neighbours(const ddd::Span& s) -> void {
  const int64_3d n = s.logic_extent();
  for (int64_t bz = 1; bz <= n.z; ++bz) {
    for (int64_t by = 1; by <= n.y; ++by) {
      for (int64_t bx = 0; bx < n.x; ++bx) {
        const ddd::Idx idx = s.make_idx(bx, by, bz);
        for (int8_t dz = -1; dz <= 1; ++dz) {
          for (int8_t dy = -1; dy <= 1; ++dy) {
            for (int8_t dx = -1; dx <= 1; ++dx) {
              const int64_3d nb{bx + dx, by + dy, bz + dz};
              bool want = nb.x >= 0 && nb.x < n.x && nb.y >= 1 - s.halo_y() &&
                          nb.y <= n.y + s.halo_y() && nb.z >= 1 - s.halo_z() &&
                          nb.z <= n.z + s.halo_z();
              int64_3d w{};
              want = want &&
                     wrap_global({nb.x + s.x_offset(), nb.y + s.y_offset(), nb.z + s.z_offset()},
                                 s.global_extent(), s.boundary_mode(), w);
              const auto r = ddd::neighbour_idx(s, idx, int8_3d{dx, dy, dz});
              assert(r.valid == want);
              if (want) {
                assert(same_idx(r.idx, s.make_idx(nb)));
              }
            }
          }
        }
      }
    }
  }
}

template <int Components>
auto
check_pencil_read_pull(const ddd::Span& s) -> void {
  const int64_3d n = s.logic_extent();
  const int64_3d h = s.haloed_extent();
  std::vector<int64_t> buf(static_cast<size_t>(s.linear_storage_elements(Components)));
  for (size_t k = 0; k < buf.size(); ++k) {
    buf[k] = static_cast<int64_t>(k);
  }
  const ddd::Partition<int64_t, Components> part(buf.data());
  for (int64_t bz = 1; bz <= n.z; ++bz) {
    for (int64_t by = 1; by <= n.y; ++by) {
      for (int64_t bx = 0; bx < n.x; ++bx) {
        const ddd::Idx idx = s.make_idx(bx, by, bz);
        for (int8_t dz = -1; dz <= 1; ++dz) {
          for (int8_t dy = -1; dy <= 1; ++dy) {
            for (int8_t dx = -1; dx <= 1; ++dx) {
              const int64_3d nb{bx + dx, by + dy, bz + dz};
              if (nb.x < 0 || nb.x >= h.x) {
                continue;
              }
              const ddd::Idx want = s.make_idx(nb);
              for (int c = 0; c < Components; ++c) {
                assert(part.read_pull(s, idx, int8_3d{dx, dy, dz}, c) ==
                       buf[static_cast<size_t>(s.linear_offset(want, Components, c))]);
              }
            }
          }
        }
      }
    }
  }
}

auto
part_d_pencil_storage() -> size_t {
  size_t chunks = 0;
  for (const PencilCase& pc : pencil_cases()) {
    const ProcessGrid pg = pencil_grid(pc);
    for (const int64_t halo : {int64_t{0}, int64_t{1}}) {
      const auto spans = make_world(pc.global, pg, pc.mode, halo);
      std::vector<int> covered(static_cast<size_t>(pc.global.x * pc.global.y * pc.global.z), 0);
      for (const ddd::Span& s : spans) {
        assert(s.halo_y() == halo && s.halo_z() == halo && s.halo_x() == 0);
        check_pencil_neighbours(s);
        if (halo == 0) {
          continue;
        }
        for (const int components : {1, 3, 19}) {
          check_pencil_storage(s, components);
        }
        check_pencil_views(s);
        check_pencil_read_pull<1>(s);
        check_pencil_read_pull<3>(s);
        const int64_3d n = s.logic_extent();
        for (int64_t iz = 0; iz < n.z; ++iz) {
          for (int64_t iy = 0; iy < n.y; ++iy) {
            for (int64_t ix = 0; ix < n.x; ++ix) {
              const auto v = s.set_and_validate(DataView::Standard, ix, iy, iz);
              const int64_3d g = s.global_location(v.idx);
              ++covered[static_cast<size_t>(g.x + pc.global.x * (g.y + pc.global.y * g.z))];
            }
          }
        }
        ++chunks;
      }
      if (halo != 0) {
        for (const int k : covered) {
          assert(k == 1);
        }
      }
    }
  }
  return chunks;
}

auto
crosses(carbon::StencilSemantic semantic,
        int32_t sign,
        int32_t c) -> bool {
  if (semantic == carbon::StencilSemantic::Standard) {
    return true;
  }
  if (semantic == carbon::StencilSemantic::LatticeDistributed) {
    return c != 0;
  }
  return sign > 0 ? c > 0 : c < 0;
}

/** Is component @p c of ghost cell @p (by, bz) one the halo must deliver under @p semantic ? */
auto
ghost_component_expected(carbon::StencilSemantic semantic,
                         const carbon::Stencil& stencil,
                         int c,
                         int64_t by,
                         int64_t bz,
                         int64_t ly,
                         int64_t lz) -> bool {
  if (semantic == carbon::StencilSemantic::Standard) {
    return true;
  }
  const int8_3d p = stencil.points()[static_cast<size_t>(c)];
  bool ok = true;
  if (by == 0 || by == ly + 1) {
    ok = ok && crosses(semantic, by == 0 ? 1 : -1, p.y);
  }
  if (bz == 0 || bz == lz + 1) {
    ok = ok && crosses(semantic, bz == 0 ? 1 : -1, p.z);
  }
  return ok;
}

/** Cells of the face / edge sent in direction @p d , in the order they must sit in the message. */
auto
send_cells(const ddd::Span& s,
           ddd::PencilDir d) -> std::vector<int64_3d> {
  const int64_3d n = s.logic_extent();
  std::vector<int64_3d> cells;
  if (d.dy == 0) {
    const int64_t bz = d.dz > 0 ? n.z : 1;
    for (int64_t by = 1; by <= n.y; ++by) {
      for (int64_t bx = 0; bx < n.x; ++bx) {
        cells.push_back({bx, by, bz});
      }
    }
    return cells;
  }
  const int64_t by = d.dy > 0 ? n.y : 1;
  if (d.dz == 0) {
    for (int64_t bz = 1; bz <= n.z; ++bz) {
      for (int64_t bx = 0; bx < n.x; ++bx) {
        cells.push_back({bx, by, bz});
      }
    }
    return cells;
  }
  const int64_t bz = d.dz > 0 ? n.z : 1;
  for (int64_t bx = 0; bx < n.x; ++bx) {
    cells.push_back({bx, by, bz});
  }
  return cells;
}

template <int Components>
auto
part_e_exchange(const PencilCase& pc,
                carbon::StencilSemantic semantic,
                const carbon::Stencil& stencil) -> size_t {
  constexpr int64_t kSentinel = -1;
  const ProcessGrid pg = pencil_grid(pc);
  const auto spans = make_world(pc.global, pg, pc.mode, 1);
  const size_t world = spans.size();

  std::vector<std::vector<int64_t>> bufs(world);
  for (size_t r = 0; r < world; ++r) {
    const ddd::Span& s = spans[r];
    bufs[r].assign(static_cast<size_t>(s.linear_storage_elements(Components)), kSentinel);
    const int64_3d n = s.logic_extent();
    for (int64_t iz = 0; iz < n.z; ++iz) {
      for (int64_t iy = 0; iy < n.y; ++iy) {
        for (int64_t ix = 0; ix < n.x; ++ix) {
          const auto v = s.set_and_validate(DataView::Standard, ix, iy, iz);
          for (int c = 0; c < Components; ++c) {
            bufs[r][static_cast<size_t>(s.linear_offset(v.idx, Components, c))] =
                encode(s.global_location(v.idx), c);
          }
        }
      }
    }
  }

  std::vector<ddd::RowCopyArgs> refresh(world);
  for (size_t r = 0; r < world; ++r) {
    const int32_3d coords = pg.coords_of(static_cast<int32_t>(r));
    for (const ddd::PencilDir d : ddd::kPencilDirs) {
      if (!ddd::pencil_peer(pg, coords, d, pc.mode).has_value()) {
        continue;
      }
      for (const auto& run : ddd::pencil_runs<Components>(semantic, stencil, d)) {
        ddd::append_send_refresh(spans[r], Components, d, run, refresh[r]);
      }
    }
    assert(refresh[r].job_count <= ddd::kMaxRowCopyJobs);
    ddd::row_copy_host(bufs[r].data(), refresh[r]);
  }

  if (semantic == carbon::StencilSemantic::Standard) {
    for (size_t r = 0; r < world; ++r) {
      const ddd::Span& s = spans[r];
      const int32_3d coords = pg.coords_of(static_cast<int32_t>(r));
      for (const ddd::PencilDir d : ddd::kPencilDirs) {
        if (!ddd::pencil_peer(pg, coords, d, pc.mode).has_value()) {
          continue;
        }
        const auto runs = ddd::pencil_runs<Components>(semantic, stencil, d);
        assert(runs.size() == 1U && runs[0].j_start == 0 && runs[0].j_len == Components);
        const ddd::PencilMessage msg = ddd::region_run(s, ddd::send_region(d), Components, runs[0]);
        const auto cells = send_cells(s, d);
        assert(msg.count == cells.size() * static_cast<size_t>(Components));
        size_t k = msg.first;
        for (int c = 0; c < Components; ++c) {
          for (const int64_3d& cell : cells) {
            assert(bufs[r][k] == encode(s.global_location(s.make_idx(cell)), c));
            ++k;
          }
        }
      }
    }
  }

  size_t messages = 0;
  for (size_t r = 0; r < world; ++r) {
    const int32_3d coords = pg.coords_of(static_cast<int32_t>(r));
    size_t rank_messages = 0;
    for (const ddd::PencilDir d : ddd::kPencilDirs) {
      const auto to = ddd::pencil_peer(pg, coords, d, pc.mode);
      if (!to.has_value()) {
        continue;
      }
      const auto peer = static_cast<size_t>(pg.rank_of(*to));
      for (const auto& run : ddd::pencil_runs<Components>(semantic, stencil, d)) {
        const ddd::PencilMessage src =
            ddd::region_run(spans[r], ddd::send_region(d), Components, run);
        const ddd::PencilMessage dst =
            ddd::region_run(spans[peer], ddd::recv_region(d), Components, run);
        assert(src.count == dst.count);
        std::copy_n(bufs[r].data() + src.first, src.count, bufs[peer].data() + dst.first);
        ++rank_messages;
      }
    }
    if (semantic == carbon::StencilSemantic::Standard) {
      assert(rank_messages <= 8U);
    }
    messages += rank_messages;
  }

  for (size_t r = 0; r < world; ++r) {
    const ddd::Span& s = spans[r];
    const int64_3d n = s.logic_extent();
    const int64_3d h = s.haloed_extent();
    for (int64_t bz = 0; bz < h.z; ++bz) {
      for (int64_t by = 0; by < h.y; ++by) {
        for (int64_t bx = 0; bx < h.x; ++bx) {
          const ddd::Idx idx = s.make_idx(bx, by, bz);
          const int64_3d g = s.global_location(idx);
          const bool owned = by >= 1 && by <= n.y && bz >= 1 && bz <= n.z;
          int64_3d w{};
          const bool inside = wrap_global(g, pc.global, pc.mode, w);
          for (int c = 0; c < Components; ++c) {
            const int64_t got = bufs[r][static_cast<size_t>(s.linear_offset(idx, Components, c))];
            if (owned) {
              assert(got == encode(g, c));
            } else if (!inside) {
              assert(got == kSentinel);
            } else if (ghost_component_expected(semantic, stencil, c, by, bz, n.y, n.z)) {
              if (got != encode(w, c)) {
                std::fprintf(stderr,
                             "part E [%s]: rank %zu ghost (%lld,%lld,%lld) comp %d got %lld want "
                             "%lld\n",
                             pc.label, r, static_cast<long long>(bx), static_cast<long long>(by),
                             static_cast<long long>(bz), c, static_cast<long long>(got),
                             static_cast<long long>(encode(w, c)));
              }
              assert(got == encode(w, c));
            }
          }
        }
      }
    }
  }
  return messages;
}

auto
part_e_pencil_exchange() -> size_t {
  size_t messages = 0;
  const carbon::Stencil q19 = d3q19_stencil();
  for (const PencilCase& pc : pencil_cases()) {
    messages += part_e_exchange<1>(pc, carbon::StencilSemantic::Standard, radius1_stencil());
    messages += part_e_exchange<3>(pc, carbon::StencilSemantic::Standard, radius1_stencil());
    messages += part_e_exchange<19>(pc, carbon::StencilSemantic::Lattice, q19);
    messages += part_e_exchange<19>(pc, carbon::StencilSemantic::LatticeDistributed, q19);
  }
  return messages;
}

struct BrickCase {
  const char* label;
  int32_t px;
  int32_t py;
  int32_t pz;
  BoundaryModeDistributed mode;
  int64_3d global;
};

auto
brick_cases() -> std::vector<BrickCase> {
  constexpr auto kO = BoundaryModeDistributed::Open;
  constexpr auto kY = BoundaryModeDistributed::PeriodicY;
  constexpr auto kZ = BoundaryModeDistributed::PeriodicZ;
  return {
      {"2x1x1-open", 2, 1, 1, kO, {6, 3, 4}},
      {"2x2x2-open", 2, 2, 2, kO, {7, 6, 8}},
      {"2x2x2-periodicYZ", 2, 2, 2, kY | kZ, {9, 5, 6}},
      {"2x2x1-periodicYZ", 2, 2, 1, kY | kZ, {5, 6, 4}},
      {"3x1x2-periodicZ", 3, 1, 2, kZ, {9, 4, 6}},
      {"2x1x1-periodicY", 2, 1, 1, kY, {6, 4, 3}},
      {"2x3x1-open", 2, 3, 1, kO, {4, 9, 2}},
      {"4x1x1-open", 4, 1, 1, kO, {8, 2, 3}},
      {"2x1x1-wide", 2, 1, 1, kO, {80, 4, 4}},
  };
}

auto
brick_grid(const BrickCase& bc) -> ProcessGrid {
  return ProcessGrid::custom(bc.px * bc.py * bc.pz, bc.px, bc.py, bc.pz);
}

struct BrickGroup {
  ddd::Region region;
  int64_t size;
  bool pad{false};
};

/** Brick storage order, from the layout spec ( @c pad marks the alignment gap before Internal). */
auto
brick_groups(const int64_3d& n,
             int64_t pitch) -> std::vector<BrickGroup> {
  using R = ddd::Region;
  const int64_t x = n.x;
  const int64_t y = n.y;
  const int64_t z = n.z;
  const int64_t iy = y - 2 > 0 ? y - 2 : 0;
  const int64_t iz = z - 2 > 0 ? z - 2 : 0;
  const auto plane = [&](bool hi) -> std::vector<BrickGroup> {
    const auto corner = [hi](bool xhi, bool yhi) {
      return ddd::corner_region(R::CornerGhostXLoYLoZLo, xhi, yhi, hi);
    };
    return {{corner(false, false), 1},
            {hi ? R::UpperGhostYLo : R::LowerGhostYLo, x},
            {corner(true, false), 1},
            {hi ? R::UpperGhostXLo : R::LowerGhostXLo, y},
            {hi ? R::UpperGhost : R::LowerGhost, x * y},
            {hi ? R::UpperGhostXHi : R::LowerGhostXHi, y},
            {corner(false, true), 1},
            {hi ? R::UpperGhostYHi : R::LowerGhostYHi, x},
            {corner(true, true), 1}};
  };
  std::vector<BrickGroup> g = plane(false);
  const std::vector<BrickGroup> lower_tail = {
      {R::DownBoundary, x * y},  {R::YLowerGhostXLo, z},    {R::YLowerGhost, x * z},
      {R::YLowerGhostXHi, z},    {R::YDownBoundary, x * z}, {R::XLowerGhost, y * z},
      {R::XDownBoundary, y * z},
  };
  g.insert(g.end(), lower_tail.begin(), lower_tail.end());
  int64_t prefix = 0;
  for (const BrickGroup& e : g) {
    prefix += e.size;
  }
  const int64_t pad = (prefix + 31) / 32 * 32 - prefix;
  g.push_back({R::Internal, pad, true});
  const std::vector<BrickGroup> middle = {
      {R::Internal, pitch * iy * iz}, {R::XUpBoundary, y * z}, {R::XUpperGhost, y * z},
      {R::YUpBoundary, x * z},        {R::YUpperGhostXLo, z},  {R::YUpperGhost, x * z},
      {R::YUpperGhostXHi, z},         {R::UpBoundary, x * y},
  };
  g.insert(g.end(), middle.begin(), middle.end());
  const auto upper = plane(true);
  g.insert(g.end(), upper.begin(), upper.end());
  for (const R r : {R::EdgeSendYLoZLo, R::EdgeSendYHiZLo, R::EdgeSendYLoZHi, R::EdgeSendYHiZHi}) {
    g.push_back({r, x});
  }
  for (const R r : {R::EdgeSendXLoZLo, R::EdgeSendXHiZLo, R::EdgeSendXLoZHi, R::EdgeSendXHiZHi}) {
    g.push_back({r, y});
  }
  for (const R r : {R::EdgeSendXLoYLo, R::EdgeSendXHiYLo, R::EdgeSendXLoYHi, R::EdgeSendXHiYHi}) {
    g.push_back({r, z});
  }
  for (int k = 0; k < 8; ++k) {
    g.push_back({static_cast<R>(static_cast<int>(R::CornerSendXLoYLoZLo) + k), 1});
  }
  return g;
}

struct BrickPlacement {
  ddd::Region region;
  int64_t local;
};

/** Region and per-component local offset of buffer cell @p (bx, by, bz) , from the layout spec. */
auto
brick_placement(int64_t bx,
                int64_t by,
                int64_t bz,
                const int64_3d& n,
                int64_t pitch) -> BrickPlacement {
  using R = ddd::Region;
  const bool xlo = bx == 0;
  const bool xhi = bx == n.x + 1;
  const bool ylo = by == 0;
  const bool yhi = by == n.y + 1;
  if (bz == 0 || bz == n.z + 1) {
    const bool hi = bz != 0;
    if ((ylo || yhi) && (xlo || xhi)) {
      return {ddd::corner_region(R::CornerGhostXLoYLoZLo, xhi, yhi, hi), 0};
    }
    if (ylo || yhi) {
      return {hi ? (yhi ? R::UpperGhostYHi : R::UpperGhostYLo)
                 : (yhi ? R::LowerGhostYHi : R::LowerGhostYLo),
              bx - 1};
    }
    if (xlo || xhi) {
      return {hi ? (xhi ? R::UpperGhostXHi : R::UpperGhostXLo)
                 : (xhi ? R::LowerGhostXHi : R::LowerGhostXLo),
              by - 1};
    }
    return {hi ? R::UpperGhost : R::LowerGhost, (by - 1) * n.x + bx - 1};
  }
  if (ylo || yhi) {
    if (xlo) {
      return {yhi ? R::YUpperGhostXLo : R::YLowerGhostXLo, bz - 1};
    }
    if (xhi) {
      return {yhi ? R::YUpperGhostXHi : R::YLowerGhostXHi, bz - 1};
    }
    return {yhi ? R::YUpperGhost : R::YLowerGhost, (bz - 1) * n.x + bx - 1};
  }
  if (xlo || xhi) {
    return {xhi ? R::XUpperGhost : R::XLowerGhost, (bz - 1) * n.y + by - 1};
  }
  if (bz == 1 || bz == n.z) {
    return {bz == 1 ? R::DownBoundary : R::UpBoundary, (by - 1) * n.x + bx - 1};
  }
  if (by == 1 || by == n.y) {
    return {by == 1 ? R::YDownBoundary : R::YUpBoundary, (bz - 1) * n.x + bx - 1};
  }
  if (bx == 1 || bx == n.x) {
    return {bx == 1 ? R::XDownBoundary : R::XUpBoundary, (bz - 1) * n.y + by - 1};
  }
  return {R::Internal, ((bz - 2) * (n.y - 2) + by - 2) * pitch + bx - 2};
}

auto
check_brick_storage(const ddd::Span& s,
                    int components) -> void {
  const int64_3d n = s.logic_extent();
  const int64_3d h = s.haloed_extent();
  const int64_t cc = components;
  assert(s.ghost_x() == 1 && s.ghost_y() == 1);
  assert(eq3(h, {n.x + 2, n.y + 2, n.z + 2}));
  assert(s.storage_unit() == 1);
  const int64_t pitch = s.internal_pitch();
  assert(pitch == (n.x > 2 ? (n.x - 2 + 31) / 32 * 32 : 0));
  assert(pitch % 32 == 0);

  std::array<int64_t, ddd::kRegionCount> first{};
  std::array<int64_t, ddd::kRegionCount> dim{};
  std::array<int, ddd::kRegionCount> seen{};
  int64_t prefix = 0;
  for (const BrickGroup& e : brick_groups(n, pitch)) {
    if (!e.pad) {
      const auto ri = static_cast<size_t>(e.region);
      ++seen[ri];
      first[ri] = prefix;
      dim[ri] = e.size;
      assert(s.region_first_z(e.region) == prefix);
      assert(s.region_z_dim(e.region) == e.size);
      assert(s.region_storage_offset(e.region, components) == prefix * cc);
      assert(s.region_storage_elements(e.region, components) == e.size * cc);
    }
    prefix += e.size;
  }
  for (const int k : seen) {
    assert(k == 1);
  }
  assert(first[static_cast<size_t>(ddd::Region::Internal)] % 32 == 0);
  const int64_t total = s.linear_storage_elements(components);
  assert(total == prefix * cc);

  std::vector<int> hits(static_cast<size_t>(total), 0);
  for (int64_t bz = 0; bz < h.z; ++bz) {
    for (int64_t by = 0; by < h.y; ++by) {
      for (int64_t bx = 0; bx < h.x; ++bx) {
        const BrickPlacement p = brick_placement(bx, by, bz, n, pitch);
        const auto ri = static_cast<size_t>(p.region);
        const ddd::Idx idx = s.make_idx(bx, by, bz);
        assert(eq3(idx.cell(), {bx, by, bz}));
        for (int c = 0; c < components; ++c) {
          const int64_t want = first[ri] * cc + c * dim[ri] + p.local;
          const int64_t got = s.linear_offset(idx, components, c);
          assert(got == want);
          ++hits[static_cast<size_t>(got)];
        }
      }
    }
  }

  std::vector<char> spare(static_cast<size_t>(total), 0);
  const auto mark = [&](ddd::Region r, int64_t local) {
    const auto ri = static_cast<size_t>(r);
    for (int c = 0; c < components; ++c) {
      spare[static_cast<size_t>(first[ri] * cc + c * dim[ri] + local)] = 1;
    }
  };
  const int64_t pad_first = first[static_cast<size_t>(ddd::Region::XDownBoundary)] + n.y * n.z;
  const int64_t pad_cells = first[static_cast<size_t>(ddd::Region::Internal)] - pad_first;
  for (int64_t k = 0; k < pad_cells * cc; ++k) {
    spare[static_cast<size_t>(pad_first * cc + k)] = 1;
  }
  for (int64_t row = 0; row < (n.y - 2) * (n.z - 2); ++row) {
    for (int64_t x = n.x - 2; x < pitch; ++x) {
      mark(ddd::Region::Internal, row * pitch + x);
    }
  }
  for (const ddd::Region r : {ddd::Region::YDownBoundary, ddd::Region::YUpBoundary}) {
    for (int64_t x = 0; x < n.x; ++x) {
      mark(r, x);
      mark(r, (n.z - 1) * n.x + x);
    }
  }
  for (const ddd::Region r : {ddd::Region::XDownBoundary, ddd::Region::XUpBoundary}) {
    for (int64_t bz = 1; bz <= n.z; ++bz) {
      for (int64_t by = 1; by <= n.y; ++by) {
        if (bz == 1 || bz == n.z || by == 1 || by == n.y) {
          mark(r, (bz - 1) * n.y + by - 1);
        }
      }
    }
  }
  for (int ri = static_cast<int>(ddd::Region::EdgeSendXLoZLo); ri < ddd::kRegionCount; ++ri) {
    for (int64_t k = 0; k < dim[static_cast<size_t>(ri)]; ++k) {
      mark(static_cast<ddd::Region>(ri), k);
    }
  }
  for (const ddd::Region r : {ddd::Region::EdgeSendYLoZLo, ddd::Region::EdgeSendYHiZLo,
                              ddd::Region::EdgeSendYLoZHi, ddd::Region::EdgeSendYHiZHi}) {
    for (int64_t k = 0; k < n.x; ++k) {
      mark(r, k);
    }
  }
  for (size_t k = 0; k < hits.size(); ++k) {
    assert(hits[k] == (spare[k] != 0 ? 0 : 1));
  }

  for (const ddd::Region r : {ddd::Region::XLowerGhost, ddd::Region::XDownBoundary,
                              ddd::Region::XUpBoundary, ddd::Region::XUpperGhost}) {
    const int64_t bx = r == ddd::Region::XLowerGhost     ? 0
                       : r == ddd::Region::XDownBoundary ? 1
                       : r == ddd::Region::XUpBoundary   ? n.x
                                                         : n.x + 1;
    for (int c = 0; c < components; ++c) {
      int64_t expect = s.region_storage_offset(r, components) + c * n.y * n.z;
      for (int64_t bz = 1; bz <= n.z; ++bz) {
        for (int64_t by = 1; by <= n.y; ++by) {
          const ddd::Idx idx = s.make_idx(bx, by, bz);
          if (r == ddd::Region::XDownBoundary || r == ddd::Region::XUpBoundary) {
            if (bz == 1 || bz == n.z || by == 1 || by == n.y) {
              ++expect;
              continue;
            }
          }
          assert(s.linear_offset(idx, components, c) == expect);
          ++expect;
        }
      }
    }
  }
}

auto
check_brick_views(const ddd::Span& s) -> void {
  const int64_3d n = s.logic_extent();
  const int64_3d es = s.iter_extent(DataView::Standard);
  assert(eq3(es, n));
  for (int64_t iz = 0; iz <= es.z; ++iz) {
    for (int64_t iy = 0; iy < es.y; ++iy) {
      for (int64_t ix = 0; ix < es.x; ++ix) {
        const auto v = s.set_and_validate(DataView::Standard, ix, iy, iz);
        assert(v.valid == (iz < es.z));
        if (v.valid) {
          assert(same_idx(v.idx, s.make_idx(ix + 1, iy + 1, iz + 1)));
        }
      }
    }
  }

  const int64_3d inner{n.x - 2, n.y - 2, n.z - 2};
  assert(eq3(s.iter_extent(DataView::Internal), inner));
  const int64_t nb = n.x * n.y * n.z - inner.x * inner.y * inner.z;
  assert(s.boundary_cell_count() == nb);
  assert(
      eq3(s.iter_extent(DataView::Boundary),
          {ddd::kBoundaryRowWidth, (nb + ddd::kBoundaryRowWidth - 1) / ddd::kBoundaryRowWidth, 1}));
  std::vector<int> owned_hits(static_cast<size_t>(n.x * n.y * n.z), 0);
  for (const DataView view : {DataView::Internal, DataView::Boundary}) {
    const int64_3d e = s.iter_extent(view);
    int prev_region = -1;
    int64_t prev = -1;
    int64_t valid_count = 0;
    for (int64_t iz = 0; iz <= e.z; ++iz) {
      for (int64_t iy = 0; iy <= e.y; ++iy) {
        for (int64_t ix = 0; ix <= e.x; ++ix) {
          const auto v = s.set_and_validate(view, ix, iy, iz);
          const bool in_box = ix < e.x && iy < e.y && iz < e.z;
          if (view == DataView::Internal) {
            assert(v.valid == in_box);
          } else {
            assert(v.valid == (in_box && ix + ddd::kBoundaryRowWidth * iy < nb));
          }
          if (!v.valid) {
            continue;
          }
          ++valid_count;
          const int64_3d c = v.idx.cell();
          assert(c.x >= 1 && c.x <= n.x && c.y >= 1 && c.y <= n.y && c.z >= 1 && c.z <= n.z);
          assert(same_idx(v.idx, s.make_idx(c)));
          const BrickPlacement p = brick_placement(c.x, c.y, c.z, n, s.internal_pitch());
          assert((p.region == ddd::Region::Internal) == (view == DataView::Internal));
          if (view == DataView::Boundary) {
            const int64_t off = s.linear_offset(v.idx, 1, 0);
            if (static_cast<int>(p.region) == prev_region) {
              assert(off > prev && off - prev <= 3);
            }
            prev_region = static_cast<int>(p.region);
            prev = off;
          }
          ++owned_hits[static_cast<size_t>((c.x - 1) + n.x * ((c.y - 1) + n.y * (c.z - 1)))];
        }
      }
    }
    if (view == DataView::Boundary) {
      assert(valid_count == nb);
    }
  }
  for (const int k : owned_hits) {
    assert(k == 1);
  }
}

auto
check_brick_neighbours(const ddd::Span& s) -> void {
  const int64_3d n = s.logic_extent();
  for (int64_t bz = 1; bz <= n.z; ++bz) {
    for (int64_t by = 1; by <= n.y; ++by) {
      for (int64_t bx = 1; bx <= n.x; ++bx) {
        const ddd::Idx idx = s.make_idx(bx, by, bz);
        for (int8_t dz = -1; dz <= 1; ++dz) {
          for (int8_t dy = -1; dy <= 1; ++dy) {
            for (int8_t dx = -1; dx <= 1; ++dx) {
              const int64_3d nb{bx + dx, by + dy, bz + dz};
              bool want = nb.x >= 1 - s.halo_x() && nb.x <= n.x + s.halo_x() &&
                          nb.y >= 1 - s.halo_y() && nb.y <= n.y + s.halo_y() &&
                          nb.z >= 1 - s.halo_z() && nb.z <= n.z + s.halo_z();
              int64_3d w{};
              want = want &&
                     wrap_global({nb.x + s.x_offset(), nb.y + s.y_offset(), nb.z + s.z_offset()},
                                 s.global_extent(), s.boundary_mode(), w);
              const auto r = ddd::neighbour_idx(s, idx, int8_3d{dx, dy, dz});
              assert(r.valid == want);
              if (want) {
                assert(same_idx(r.idx, s.make_idx(nb)));
              }
            }
          }
        }
      }
    }
  }
}

template <int Components>
auto
check_brick_read_pull(const ddd::Span& s) -> void {
  const int64_3d n = s.logic_extent();
  std::vector<int64_t> buf(static_cast<size_t>(s.linear_storage_elements(Components)));
  for (size_t k = 0; k < buf.size(); ++k) {
    buf[k] = static_cast<int64_t>(k);
  }
  const ddd::Partition<int64_t, Components> part(buf.data());
  for (int64_t bz = 1; bz <= n.z; ++bz) {
    for (int64_t by = 1; by <= n.y; ++by) {
      for (int64_t bx = 1; bx <= n.x; ++bx) {
        const ddd::Idx idx = s.make_idx(bx, by, bz);
        for (int8_t dz = -1; dz <= 1; ++dz) {
          for (int8_t dy = -1; dy <= 1; ++dy) {
            for (int8_t dx = -1; dx <= 1; ++dx) {
              const ddd::Idx want = s.make_idx(bx + dx, by + dy, bz + dz);
              for (int c = 0; c < Components; ++c) {
                assert(part.read_pull(s, idx, int8_3d{dx, dy, dz}, c) ==
                       buf[static_cast<size_t>(s.linear_offset(want, Components, c))]);
              }
            }
          }
        }
      }
    }
  }
}

auto
part_f_brick_storage() -> size_t {
  size_t chunks = 0;
  for (const BrickCase& bc : brick_cases()) {
    const ProcessGrid pg = brick_grid(bc);
    for (const int64_t halo : {int64_t{0}, int64_t{1}}) {
      const auto spans = make_world(bc.global, pg, bc.mode, halo);
      std::vector<int> covered(static_cast<size_t>(bc.global.x * bc.global.y * bc.global.z), 0);
      for (const ddd::Span& s : spans) {
        assert(s.halo_x() == halo && s.halo_y() == halo && s.halo_z() == halo);
        check_brick_neighbours(s);
        if (halo == 0) {
          continue;
        }
        for (const int components : {1, 3, 19}) {
          check_brick_storage(s, components);
        }
        check_brick_views(s);
        check_brick_read_pull<1>(s);
        check_brick_read_pull<3>(s);
        const int64_3d n = s.logic_extent();
        for (int64_t iz = 0; iz < n.z; ++iz) {
          for (int64_t iy = 0; iy < n.y; ++iy) {
            for (int64_t ix = 0; ix < n.x; ++ix) {
              const auto v = s.set_and_validate(DataView::Standard, ix, iy, iz);
              const int64_3d g = s.global_location(v.idx);
              ++covered[static_cast<size_t>(g.x + bc.global.x * (g.y + bc.global.y * g.z))];
            }
          }
        }
        ++chunks;
      }
      if (halo != 0) {
        for (const int k : covered) {
          assert(k == 1);
        }
      }
    }
  }
  return chunks;
}

/** Is component @p c of buffer cell @p cell one the halo must deliver under @p semantic ? */
auto
ghost_expected_xyz(carbon::StencilSemantic semantic,
                   const carbon::Stencil& stencil,
                   int c,
                   const int64_3d& cell,
                   const int64_3d& n,
                   int64_t gx,
                   int64_t gy) -> bool {
  if (semantic == carbon::StencilSemantic::Standard) {
    return true;
  }
  const int8_3d p = stencil.points()[static_cast<size_t>(c)];
  bool ok = true;
  if (gx != 0 && (cell.x == 0 || cell.x == n.x + 1)) {
    ok = ok && crosses(semantic, cell.x == 0 ? 1 : -1, p.x);
  }
  if (gy != 0 && (cell.y == 0 || cell.y == n.y + 1)) {
    ok = ok && crosses(semantic, cell.y == 0 ? 1 : -1, p.y);
  }
  if (cell.z == 0 || cell.z == n.z + 1) {
    ok = ok && crosses(semantic, cell.z == 0 ? 1 : -1, p.z);
  }
  return ok;
}

/** Cells of the face / edge / corner sent in direction @p d , in their send-block order. */
auto
brick_send_cells(const ddd::Span& s,
                 ddd::BrickDir d) -> std::vector<int64_3d> {
  const int64_3d n = s.logic_extent();
  const auto range = [](int32_t dv, int64_t len) -> std::pair<int64_t, int64_t> {
    if (dv == 0) {
      return {1, len};
    }
    return dv > 0 ? std::pair<int64_t, int64_t>{len, len} : std::pair<int64_t, int64_t>{1, 1};
  };
  const auto [x0, x1] = range(d.dx, n.x);
  const auto [y0, y1] = range(d.dy, n.y);
  const auto [z0, z1] = range(d.dz, n.z);
  std::vector<int64_3d> cells;
  if (d.dx != 0 && d.dy == 0 && d.dz == 0) {
    for (int64_t bz = z0; bz <= z1; ++bz) {
      for (int64_t by = y0; by <= y1; ++by) {
        cells.push_back({x0, by, bz});
      }
    }
    return cells;
  }
  for (int64_t bz = z0; bz <= z1; ++bz) {
    for (int64_t by = y0; by <= y1; ++by) {
      for (int64_t bx = x0; bx <= x1; ++bx) {
        cells.push_back({bx, by, bz});
      }
    }
  }
  return cells;
}

template <int Components>
auto
part_g_exchange(const BrickCase& bc,
                carbon::StencilSemantic semantic,
                const carbon::Stencil& stencil) -> size_t {
  constexpr int64_t kSentinel = -1;
  const ProcessGrid pg = brick_grid(bc);
  const auto spans = make_world(bc.global, pg, bc.mode, 1);
  const size_t world = spans.size();

  std::vector<std::vector<int64_t>> bufs(world);
  for (size_t r = 0; r < world; ++r) {
    const ddd::Span& s = spans[r];
    bufs[r].assign(static_cast<size_t>(s.linear_storage_elements(Components)), kSentinel);
    const int64_3d n = s.logic_extent();
    for (int64_t iz = 0; iz < n.z; ++iz) {
      for (int64_t iy = 0; iy < n.y; ++iy) {
        for (int64_t ix = 0; ix < n.x; ++ix) {
          const auto v = s.set_and_validate(DataView::Standard, ix, iy, iz);
          for (int c = 0; c < Components; ++c) {
            bufs[r][static_cast<size_t>(s.linear_offset(v.idx, Components, c))] =
                encode(s.global_location(v.idx), c);
          }
        }
      }
    }
  }

  for (size_t r = 0; r < world; ++r) {
    const int32_3d coords = pg.coords_of(static_cast<int32_t>(r));
    ddd::RowCopyArgs refresh;
    for (const ddd::BrickDir d : ddd::kBrickDirs) {
      const auto run = ddd::brick_run<Components>(semantic, stencil, d);
      if (!run.has_value() || !ddd::brick_peer(pg, coords, d, bc.mode).has_value()) {
        continue;
      }
      ddd::append_brick_send_refresh(spans[r], Components, d, *run, refresh);
    }
    assert(refresh.job_count <= ddd::kMaxRowCopyJobs);
    ddd::row_copy_host(bufs[r].data(), refresh);

    const ddd::Span& s = spans[r];
    for (const ddd::BrickDir d : ddd::kBrickDirs) {
      const auto run = ddd::brick_run<Components>(semantic, stencil, d);
      if (!run.has_value() || !ddd::brick_peer(pg, coords, d, bc.mode).has_value()) {
        continue;
      }
      const ddd::PencilMessage msg = ddd::region_run(s, ddd::send_region(d), Components, *run);
      const auto cells = brick_send_cells(s, d);
      assert(msg.count == cells.size() * static_cast<size_t>(run->j_len));
      size_t k = msg.first;
      for (int c = run->j_start; c < run->j_start + run->j_len; ++c) {
        for (const int64_3d& cell : cells) {
          assert(bufs[r][k] == encode(s.global_location(s.make_idx(cell)), c));
          ++k;
        }
      }
    }
  }

  size_t messages = 0;
  for (size_t r = 0; r < world; ++r) {
    const int32_3d coords = pg.coords_of(static_cast<int32_t>(r));
    size_t rank_messages = 0;
    for (const ddd::BrickDir d : ddd::kBrickDirs) {
      const auto to = ddd::brick_peer(pg, coords, d, bc.mode);
      const auto run = ddd::brick_run<Components>(semantic, stencil, d);
      if (!to.has_value() || !run.has_value()) {
        continue;
      }
      if (semantic != carbon::StencilSemantic::Standard && Components == 19) {
        assert(ddd::brick_dir_rank(d) < 3);
      }
      const auto peer = static_cast<size_t>(pg.rank_of(*to));
      const ddd::PencilMessage src =
          ddd::region_run(spans[r], ddd::send_region(d), Components, *run);
      const ddd::PencilMessage dst =
          ddd::region_run(spans[peer], ddd::recv_region(d), Components, *run);
      assert(src.count == dst.count);
      std::copy_n(bufs[r].data() + src.first, src.count, bufs[peer].data() + dst.first);
      ++rank_messages;
    }
    assert(rank_messages <= 26U);
    messages += rank_messages;
  }

  for (size_t r = 0; r < world; ++r) {
    const ddd::Span& s = spans[r];
    const int64_3d n = s.logic_extent();
    const int64_3d h = s.haloed_extent();
    for (int64_t bz = 0; bz < h.z; ++bz) {
      for (int64_t by = 0; by < h.y; ++by) {
        for (int64_t bx = 0; bx < h.x; ++bx) {
          const ddd::Idx idx = s.make_idx(bx, by, bz);
          const int64_3d g = s.global_location(idx);
          const bool owned = bx >= 1 && bx <= n.x && by >= 1 && by <= n.y && bz >= 1 && bz <= n.z;
          int64_3d w{};
          const bool inside = wrap_global(g, bc.global, bc.mode, w);
          for (int c = 0; c < Components; ++c) {
            const int64_t got = bufs[r][static_cast<size_t>(s.linear_offset(idx, Components, c))];
            if (owned) {
              assert(got == encode(g, c));
            } else if (!inside) {
              assert(got == kSentinel);
            } else if (ghost_expected_xyz(semantic, stencil, c, {bx, by, bz}, n, 1, 1)) {
              if (got != encode(w, c)) {
                std::fprintf(stderr,
                             "part G [%s]: rank %zu ghost (%lld,%lld,%lld) comp %d got %lld want "
                             "%lld\n",
                             bc.label, r, static_cast<long long>(bx), static_cast<long long>(by),
                             static_cast<long long>(bz), c, static_cast<long long>(got),
                             static_cast<long long>(encode(w, c)));
              }
              assert(got == encode(w, c));
            } else {
              assert(got == kSentinel || got == encode(w, c));
            }
          }
        }
      }
    }
  }
  return messages;
}

auto
part_g_brick_exchange() -> size_t {
  size_t messages = 0;
  const carbon::Stencil q19 = d3q19_stencil();
  for (const BrickCase& bc : brick_cases()) {
    messages += part_g_exchange<1>(bc, carbon::StencilSemantic::Standard, radius1_stencil());
    messages += part_g_exchange<3>(bc, carbon::StencilSemantic::Standard, radius1_stencil());
    messages += part_g_exchange<19>(bc, carbon::StencilSemantic::Lattice, q19);
    messages += part_g_exchange<19>(bc, carbon::StencilSemantic::LatticeDistributed, q19);
  }
  return messages;
}

template <int Components>
auto
check_grid_halo(carbon::System& sys,
                const carbon::DenseDisgDistributed& grid,
                carbon::StencilSemantic semantic,
                const carbon::Stencil& stencil,
                bool device) -> void {
  constexpr float kSentinel = -1.0F;
  const auto& spans = grid.spans();
  const size_t n_gpus = spans.size();
  auto f = grid.newField<float, Components>(carbon::MemorySide::HostDevice, "halo");
  for (size_t i = 0; i < n_gpus; ++i) {
    std::fill_n(f.cpu_buffers()[i].data(), f.cpu_buffers()[i].size(), kSentinel);
  }
  f.fill_owned_host([](int64_3d g, int c) -> float {
    return static_cast<float>(encode(g, c));
  });

  carbon::Stream stream = sys.stream();
  const carbon::HaloDisgDistributed<float, Components> halo(f, stencil, semantic);
  assert(halo.dist_entry_count() == 0U);
  assert(halo.host_intra_count() == halo.pair_count());
  if (device) {
    carbon::Transfer<float>(sys, f.cpu_buffers(), f.gpu_buffers())(stream);
    halo(stream);
    stream.wait();
    for (size_t i = 0; i < n_gpus; ++i) {
      std::fill_n(f.cpu_buffers()[i].data(), f.cpu_buffers()[i].size(), -2.0F);
    }
    carbon::Transfer<float>(sys, f.gpu_buffers(), f.cpu_buffers())(stream);
    stream.wait();
  } else {
    halo.launch_on_host(stream);
  }

  const int64_3d global = grid.global_extent();
  const BoundaryModeDistributed mode = grid.boundary_mode();
  for (size_t i = 0; i < n_gpus; ++i) {
    const ddd::Span& s = spans[i];
    const int64_3d n = s.logic_extent();
    const int64_3d h = s.haloed_extent();
    const int64_t gx = s.ghost_x();
    const int64_t gy = s.ghost_y();
    const float* buf = f.cpu_buffers()[i].data();
    for (int64_t bz = 0; bz < h.z; ++bz) {
      for (int64_t by = 0; by < h.y; ++by) {
        for (int64_t bx = 0; bx < h.x; ++bx) {
          const ddd::Idx idx = s.make_idx(bx, by, bz);
          const int64_3d g = s.global_location(idx);
          const bool owned =
              bx >= gx && bx < gx + n.x && by >= gy && by < gy + n.y && bz >= 1 && bz <= n.z;
          int64_3d w{};
          const bool inside = wrap_global(g, global, mode, w);
          for (int c = 0; c < Components; ++c) {
            const float got = buf[s.linear_offset(idx, Components, c)];
            if (owned) {
              assert(got == static_cast<float>(encode(g, c)));
            } else if (!inside) {
              assert(got == kSentinel);
            } else if (ghost_expected_xyz(semantic, stencil, c, {bx + 1 - gx, by + 1 - gy, bz}, n,
                                          gx, gy)) {
              if (got != static_cast<float>(encode(w, c))) {
                std::fprintf(stderr,
                             "part C halo (%s): gpu %zu ghost (%lld,%lld,%lld) comp %d got %g "
                             "want %lld\n",
                             device ? "device" : "host", i, static_cast<long long>(bx),
                             static_cast<long long>(by), static_cast<long long>(bz), c,
                             static_cast<double>(got), static_cast<long long>(encode(w, c)));
              }
              assert(got == static_cast<float>(encode(w, c)));
            }
          }
        }
      }
    }
  }
}

/**
 * Owned-cell values after fill_owned_host must equal a DenseDistributed on the same ProcessGrid
 * (compared by global cell, not by layout). Single-process DenseDistributed only accepts a
 * one-GPU pencil, so this runs on @p sys1 ; the multi-rank comparison lives in
 * test_dense_disg_distributed_dist.
 */
auto
check_dense_value_parity(carbon::System& sys1,
                         BoundaryModeDistributed bmd) -> void {
  const int64_3d global{6, 9, 11};
  const carbon::DenseDisgDistributed grid(sys1, global, DecompositionMode::Pencil,
                                          radius1_stencil(), bmd);
  const ProcessGrid& pg = grid.process_grid();
  const carbon::DenseDistributed dense(sys1, global, pg, radius1_stencil(), bmd);
  auto fa = grid.newField<float, 3>(carbon::MemorySide::Host, "a");
  auto fb = dense.newField<float, 3>(carbon::MemorySide::Host, "b");
  const auto fn = [](int64_3d g, int c) -> float {
    return static_cast<float>(encode(g, c));
  };
  fa.fill_owned_host(fn);
  fb.fill_owned_host(fn);
  assert(grid.spans().size() == dense.spans().size());
  for (size_t i = 0; i < grid.spans().size(); ++i) {
    const ddd::Span& sa = grid.spans()[i];
    const auto& sb = dense.spans()[i];
    assert(eq3(sa.logic_extent(), sb.logic_extent()));
    const int64_3d n = sa.logic_extent();
    const int64_3d hb = sb.halo();
    for (int64_t z = 0; z < n.z; ++z) {
      for (int64_t y = 0; y < n.y; ++y) {
        for (int64_t x = 0; x < n.x; ++x) {
          const ddd::Idx ia = sa.set_and_validate(DataView::Standard, x, y, z).idx;
          const auto ib = sb.make_idx(hb.x + x, hb.y + y, hb.z + z);
          assert(eq3(sa.global_location(ia), sb.global_location(ib)));
          for (int c = 0; c < 3; ++c) {
            assert(fa.cpu_buffers()[i].data()[sa.linear_offset(ia, 3, c)] ==
                   fb.cpu_buffers()[i].data()[sb.linear_offset(ib, c)]);
          }
        }
      }
    }
  }
}

auto
check_pencil_grid(carbon::System& sys,
                  BoundaryModeDistributed bmd) -> void {
  const int64_3d global{6, 9, 11};
  const carbon::DenseDisgDistributed grid(sys, global, DecompositionMode::Pencil, radius1_stencil(),
                                          bmd);
  const ProcessGrid& pg = grid.process_grid();
  assert(pg.dims().x == 1);
  const bool pencil_storage = pg.dims().y > 1 || carbon::is_periodic(bmd, Axis::Y);
  assert(grid.halo_z() == 1);
  assert(grid.halo_y() == (pencil_storage ? 1 : 0));
  assert(grid.local_rank_offset() == 0);

  auto f = grid.newField<float, 3>(carbon::MemorySide::Host, "a");
  f.fill_owned_host([](int64_3d g, int c) -> float {
    return static_cast<float>(encode(g, c));
  });
  std::vector<int> covered(static_cast<size_t>(global.x * global.y * global.z), 0);
  for (size_t i = 0; i < grid.spans().size(); ++i) {
    const ddd::Span& s = grid.spans()[i];
    assert(s.ghost_y() == (pencil_storage ? 1 : 0));
    const int64_3d n = s.logic_extent();
    for (int64_t z = 0; z < n.z; ++z) {
      for (int64_t y = 0; y < n.y; ++y) {
        for (int64_t x = 0; x < n.x; ++x) {
          const ddd::Idx idx = s.set_and_validate(DataView::Standard, x, y, z).idx;
          const int64_3d g = s.global_location(idx);
          ++covered[static_cast<size_t>(g.x + global.x * (g.y + global.y * g.z))];
          for (int c = 0; c < 3; ++c) {
            assert(f.cpu_buffers()[i].data()[s.linear_offset(idx, 3, c)] ==
                   static_cast<float>(encode(g, c)));
          }
        }
      }
    }
  }
  for (const int k : covered) {
    assert(k == 1);
  }

  const carbon::Stencil q19 = d3q19_stencil();
  for (const bool device : {false, true}) {
    check_grid_halo<1>(sys, grid, carbon::StencilSemantic::Standard, radius1_stencil(), device);
    check_grid_halo<3>(sys, grid, carbon::StencilSemantic::Standard, radius1_stencil(), device);
    check_grid_halo<19>(sys, grid, carbon::StencilSemantic::Lattice, q19, device);
    check_grid_halo<19>(sys, grid, carbon::StencilSemantic::LatticeDistributed, q19, device);
  }
  std::printf("pencil grid (%d,%d,%d) mode=%d storage=%s: checked\n", pg.dims().x, pg.dims().y,
              pg.dims().z, static_cast<int>(bmd), pencil_storage ? "pencil" : "slab");
}

auto
check_brick_grid(carbon::System& sys,
                 const ProcessGrid& pg,
                 BoundaryModeDistributed bmd) -> void {
  const int32_3d dims = pg.dims();
  const int64_3d global{3 * dims.x + 1, 3 * dims.y + 2, 3 * dims.z + 2};
  const carbon::DenseDisgDistributed grid(sys, global, pg, radius1_stencil(), bmd);
  assert(grid.halo_x() == 1 && grid.halo_y() == 1 && grid.halo_z() == 1);

  auto f = grid.newField<float, 3>(carbon::MemorySide::Host, "a");
  f.fill_owned_host([](int64_3d g, int c) -> float {
    return static_cast<float>(encode(g, c));
  });
  std::vector<int> covered(static_cast<size_t>(global.x * global.y * global.z), 0);
  for (size_t i = 0; i < grid.spans().size(); ++i) {
    const ddd::Span& s = grid.spans()[i];
    assert(s.ghost_x() == 1 && s.ghost_y() == 1 && s.storage_unit() == 1);
    const auto dense = carbon::detail::dense_distributed::rank_chunk(
        global, pg, pg.coords_of(static_cast<int32_t>(i)), int64_3d{1, 1, 1});
    assert(eq3(s.logic_extent(), dense.owned));
    const int64_3d n = s.logic_extent();
    for (int64_t z = 0; z < n.z; ++z) {
      for (int64_t y = 0; y < n.y; ++y) {
        for (int64_t x = 0; x < n.x; ++x) {
          const ddd::Idx idx = s.set_and_validate(DataView::Standard, x, y, z).idx;
          const int64_3d g = s.global_location(idx);
          assert(eq3(g, {dense.offset.x + 1 + x, dense.offset.y + 1 + y, dense.offset.z + 1 + z}));
          ++covered[static_cast<size_t>(g.x + global.x * (g.y + global.y * g.z))];
          for (int c = 0; c < 3; ++c) {
            assert(f.cpu_buffers()[i].data()[s.linear_offset(idx, 3, c)] ==
                   static_cast<float>(encode(g, c)));
          }
        }
      }
    }
  }
  for (const int k : covered) {
    assert(k == 1);
  }

  const carbon::Stencil q19 = d3q19_stencil();
  for (const bool device : {false, true}) {
    check_grid_halo<1>(sys, grid, carbon::StencilSemantic::Standard, radius1_stencil(), device);
    check_grid_halo<3>(sys, grid, carbon::StencilSemantic::Standard, radius1_stencil(), device);
    check_grid_halo<19>(sys, grid, carbon::StencilSemantic::Lattice, q19, device);
    check_grid_halo<19>(sys, grid, carbon::StencilSemantic::LatticeDistributed, q19, device);
  }
  std::printf("brick grid (%d,%d,%d) mode=%d: checked\n", dims.x, dims.y, dims.z,
              static_cast<int>(bmd));
}

auto
part_c_grid_parity() -> bool {
  carbon::init();
  if (carbon::device_count() == 0) {
    std::puts("test_dense_disg_distributed_host: no devices; skipping grid parity (Part C)");
    return false;
  }
  carbon::System sys(carbon::get_all_devices());
  const size_t n = sys.gpu_group().size();
  const int64_3d global{6, 5, static_cast<int64_t>(3 * n + 2)};

  for (const BoundaryModeDistributed bmd :
       {BoundaryModeDistributed::Open, BoundaryModeDistributed::PeriodicZ}) {
    const BoundaryMode bm = ddd::to_boundary_mode(bmd);
    const carbon::DenseDisgDistributed grid(sys, global, radius1_stencil(), bmd);
    const carbon::DenseDisg disg(sys, global, /*halo_z=*/1, bm);

    assert(grid.process_grid().is_slab());
    assert(grid.gpu_count() == disg.gpu_count());
    assert(grid.global_slab_count() == disg.global_slab_count());
    assert(grid.local_slab_offset() == disg.local_slab_offset());
    assert(grid.halo_z() == disg.halo_z());
    assert(grid.spans().size() == disg.spans().size());
    for (size_t i = 0; i < n; ++i) {
      const auto& s = grid.spans()[i];
      const auto& d = disg.spans()[i];
      assert(eq3(s.logic_extent(), d.logic_extent()));
      assert(s.linear_storage_elements(1) == d.linear_storage_elements(1));
      assert(s.linear_storage_elements(3) == d.linear_storage_elements(3));
      assert(eq3(s.haloed_extent(), d.haloed_extent()));
      assert(s.z_offset() == d.z_offset());
      assert(grid.local_z(i) == disg.local_z(i));
      assert(grid.z_offset_global(i) == disg.z_offset_global(i));
      for (const DataView view : {DataView::Standard, DataView::Internal, DataView::Boundary}) {
        const auto& ga = grid.gpu_grid(view)[i];
        const auto& gb = disg.gpu_grid(view)[i];
        assert(ga.x == gb.x && ga.y == gb.y && ga.z == gb.z);
      }
    }

    auto fa = grid.newField<float, 3>(carbon::MemorySide::Host, "a");
    auto fb = disg.newField<float, 3>(carbon::MemorySide::Host, "b");
    for (size_t i = 0; i < n; ++i) {
      const size_t total = static_cast<size_t>(grid.spans()[i].linear_storage_elements(3));
      assert(fa.cpu_buffers()[i].size() == total);
      assert(fb.cpu_buffers()[i].size() == total);
      std::fill_n(fa.cpu_buffers()[i].data(), total, -1.0F);
      std::fill_n(fb.cpu_buffers()[i].data(), total, -1.0F);
    }
    const auto fn = [](int64_3d g, int c) -> float {
      return static_cast<float>(g.x + 10 * g.y + 100 * g.z + 100000 * c) + 0.25F;
    };
    fa.fill_owned_host(fn);
    fb.fill_owned_host(fn);
    for (size_t i = 0; i < n; ++i) {
      const size_t total = static_cast<size_t>(grid.spans()[i].linear_storage_elements(3));
      assert(std::memcmp(fa.cpu_buffers()[i].data(), fb.cpu_buffers()[i].data(),
                         total * sizeof(float)) == 0);
    }
  }

  carbon::System sys1(std::vector<carbon::detail::Gpu>{carbon::get_all_devices().front()});
  for (const BoundaryModeDistributed bmd :
       {BoundaryModeDistributed::Open, BoundaryModeDistributed::PeriodicY,
        BoundaryModeDistributed::PeriodicY | BoundaryModeDistributed::PeriodicZ}) {
    check_pencil_grid(sys, bmd);
    check_pencil_grid(sys1, bmd);
    check_dense_value_parity(sys1, bmd);
  }
  const ProcessGrid pencil = ProcessGrid::make(static_cast<int32_t>(n), DecompositionMode::Pencil);
  CARBON_EXPECT_THROW(
      carbon::DenseDisgDistributed(sys, int64_3d{6, pencil.dims().y, 11}, pencil, radius1_stencil(),
                                   BoundaryModeDistributed::PeriodicY),
      std::invalid_argument);
  if (n >= 2U) {
    const auto ni = static_cast<int32_t>(n);
    std::vector<ProcessGrid> bricks{ProcessGrid::custom(ni, ni, 1, 1)};
    const ProcessGrid made = ProcessGrid::make(ni, DecompositionMode::Brick);
    if (made.dims().x > 1 && made.dims().x != ni) {
      bricks.push_back(made);
    }
    for (const ProcessGrid& pg : bricks) {
      for (const BoundaryModeDistributed bmd :
           {BoundaryModeDistributed::Open,
            BoundaryModeDistributed::PeriodicY | BoundaryModeDistributed::PeriodicZ}) {
        check_brick_grid(sys, pg, bmd);
      }
    }
    CARBON_EXPECT_THROW(
        carbon::DenseDisgDistributed(sys, int64_3d{static_cast<int64_t>(n), 5, 6},
                                     ProcessGrid::custom(ni, ni, 1, 1), radius1_stencil()),
        std::invalid_argument);
  }
  CARBON_EXPECT_THROW(carbon::DenseDisgDistributed(sys, global, radius1_stencil(),
                                                   BoundaryModeDistributed::PeriodicX),
                      std::invalid_argument);
  CARBON_EXPECT_THROW(
      carbon::DenseDisgDistributed(sys, int64_3d{6, 5, static_cast<int64_t>(n)}, radius1_stencil()),
      std::invalid_argument);
  return true;
}

}  // namespace

int
main() {
  part_a_layout();
  const size_t slabs = part_b_span_parity();
  std::printf("span parity: slabs checked=%zu\n", slabs);
  const size_t chunks = part_d_pencil_storage();
  std::printf("pencil storage: chunks checked=%zu\n", chunks);
  const size_t messages = part_e_pencil_exchange();
  std::printf("pencil exchange: messages emulated=%zu\n", messages);
  const size_t bricks = part_f_brick_storage();
  std::printf("brick storage: chunks checked=%zu\n", bricks);
  const size_t brick_messages = part_g_brick_exchange();
  std::printf("brick exchange: messages emulated=%zu\n", brick_messages);
  const bool ran_grid = part_c_grid_parity();
  std::printf("grid parity: %s\n", ran_grid ? "checked" : "skipped");
  std::puts("OK");
  return 0;
}
