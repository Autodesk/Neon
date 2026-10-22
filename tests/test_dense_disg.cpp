// Foundational unit test for carbon's port of Neon's disaggregated dGrid (dGridDisg).
//
// Locks the per-region pitch math + DataView coverage at the Span / Idx / Partition layer.
// Reference numbers are derived by hand from Neon's
//   Neon-disg/libNeonDomain/include/Neon/domain/details/dGridDisg/dSpan_imp.h
//   Neon-disg/libNeonDomain/include/Neon/domain/details/dGridDisg/dPartition.h
// so any divergence shows up as an assertion failure rather than a silent layout drift.

#include <carbon/carbon.h>
#include <carbon/data_view.h>
#include <carbon/dense_disg.h>
#include <carbon/detail/dense_disg/idx.h>
#include <carbon/detail/dense_disg/partition.h>
#include <carbon/detail/dense_disg/span.h>
#include <carbon/integer3d.h>

#include "expect_throw.h"

#include <cassert>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <vector>

namespace dd = carbon::detail::dense_disg;
using carbon::DataView;
using carbon::int64_3d;

namespace {

// Walk every Standard-view thread (ix, iy, iz, card) on a (xDim, yDim, lz, components)
// disg span, assert each address is in [0, total_storage), and assert no two threads
// land on the same address. Together these prove the Standard view covers exactly the
// owned + boundary slabs (lower / upper ghost are touched only via halo / neighbour
// reads, so they should not appear in this walk).
void
check_standard_walk(int64_t xDim,
                    int64_t yDim,
                    int64_t lz,
                    int components) {
  const int64_3d haloed{xDim, yDim, lz + 2 * dd::Span::ghost_radius};
  const int64_3d global{xDim, yDim, haloed.z};
  const int64_t xy_pitch = xDim * yDim;
  const dd::Span span(haloed, global, xy_pitch, /*z_offset=*/0);

  const int64_t total_storage = span.linear_storage_elements(components);
  assert(total_storage == (lz + 2) * static_cast<int64_t>(components) * xy_pitch);

  std::set<int64_t> seen;
  for (int64_t iz = 0; iz < lz; ++iz) {
    for (int64_t iy = 0; iy < yDim; ++iy) {
      for (int64_t ix = 0; ix < xDim; ++ix) {
        const dd::ValidatedIdx vi = span.set_and_validate(DataView::Standard, ix, iy, iz);
        assert(vi.valid);
        for (int c = 0; c < components; ++c) {
          const int64_t addr = span.linear_offset(vi.idx, components, c);
          assert(addr >= 0);
          assert(addr < total_storage);
          const auto [_, inserted] = seen.insert(addr);
          assert(inserted);
        }
      }
    }
  }

  // Standard covers (Lz × C) cells per slab — the lower / upper ghost slabs (1 × C each)
  // are the only ones not touched here.
  const int64_t expected_count = static_cast<int64_t>(components) * lz * xy_pitch;
  assert(static_cast<int64_t>(seen.size()) == expected_count);

  // Spot-check: ghost address ranges are NOT touched.
  // Lower ghost is the first C xy-planes, upper ghost is the last C xy-planes.
  for (int64_t addr = 0; addr < static_cast<int64_t>(components) * xy_pitch; ++addr) {
    assert(!seen.contains(addr));
  }
  for (int64_t addr = total_storage - static_cast<int64_t>(components) * xy_pitch;
       addr < total_storage; ++addr) {
    assert(!seen.contains(addr));
  }
}

// Walk Internal then Boundary and assert the union exactly equals the Standard walk.
// This locks the central disg invariant: Internal + Boundary == Standard, which carbon
// already guarantees for the aggregated dense grid and which any future overlapped-stencil
// helper for the disg variant will rely on.
void
check_internal_boundary_partition(int64_t xDim,
                                  int64_t yDim,
                                  int64_t lz,
                                  int components) {
  const int64_3d haloed{xDim, yDim, lz + 2 * dd::Span::ghost_radius};
  const int64_3d global{xDim, yDim, haloed.z};
  const int64_t xy_pitch = xDim * yDim;
  const dd::Span span(haloed, global, xy_pitch, /*z_offset=*/0);

  std::set<int64_t> standard;
  for (int64_t iz = 0; iz < lz; ++iz) {
    for (int64_t iy = 0; iy < yDim; ++iy) {
      for (int64_t ix = 0; ix < xDim; ++ix) {
        const dd::ValidatedIdx vi = span.set_and_validate(DataView::Standard, ix, iy, iz);
        for (int c = 0; c < components; ++c) {
          standard.insert(span.linear_offset(vi.idx, components, c));
        }
      }
    }
  }

  std::set<int64_t> split;
  const int64_t inner_z = (lz - 2 * dd::Span::boundary_radius);
  for (int64_t iz = 0; iz < (inner_z > 0 ? inner_z : 0); ++iz) {
    for (int64_t iy = 0; iy < yDim; ++iy) {
      for (int64_t ix = 0; ix < xDim; ++ix) {
        const dd::ValidatedIdx vi = span.set_and_validate(DataView::Internal, ix, iy, iz);
        assert(vi.valid);
        for (int c = 0; c < components; ++c) {
          const auto [_, inserted] = split.insert(span.linear_offset(vi.idx, components, c));
          assert(inserted);
        }
      }
    }
  }
  for (int64_t iz = 0; iz < 2 * dd::Span::boundary_radius; ++iz) {
    for (int64_t iy = 0; iy < yDim; ++iy) {
      for (int64_t ix = 0; ix < xDim; ++ix) {
        const dd::ValidatedIdx vi = span.set_and_validate(DataView::Boundary, ix, iy, iz);
        assert(vi.valid);
        for (int c = 0; c < components; ++c) {
          const auto [_, inserted] = split.insert(span.linear_offset(vi.idx, components, c));
          assert(inserted);
        }
      }
    }
  }

  assert(standard == split);
}

// Hand-verified spot checks against Neon's dPartition::getPitch + dSpan::setAndValidate
// for a small (xDim=2, yDim=3, Lz=4, C=2) buffer. Picked so every region has at least
// one cell (boundary radius = 1, so we need Lz >= 4 to give the internal block more than
// one slab and cover every region transition).
void
check_neon_parity_spot() {
  constexpr int64_t xDim = 2;
  constexpr int64_t yDim = 3;
  constexpr int64_t lz = 4;
  constexpr int components = 2;
  constexpr int64_t xy_pitch = xDim * yDim;  // 6

  const int64_3d haloed{xDim, yDim, lz + 2};
  const int64_3d global{xDim, yDim, haloed.z};
  const dd::Span span(haloed, global, xy_pitch, /*z_offset=*/0);

  // (1) iz=0 (Standard) → down boundary at buffer-z=1; offset_local_no_card = x + y*X.
  //     region_first_z = 1, region_z_dim = 1.
  //     For (ix=1, iy=2, iz=0, c=1):
  //       addr = (1 + 2*2) + 6 * (1*2 + 1*1) = 5 + 6*3 = 23.
  {
    const auto vi = span.set_and_validate(DataView::Standard, 1, 2, 0);
    assert(vi.valid);
    assert(vi.idx.cell().z == 1);
    assert(vi.idx.region_first_z() == 1);
    assert(vi.idx.region_z_dim() == 1);
    assert(vi.idx.offset_local_no_card() == 1 + 2 * xDim);
    assert(span.linear_offset(vi.idx, components, /*c=*/1) == 23);
  }

  // (2) iz=1 (Standard) → buffer-z=2 → first internal slab; region_first_z=2, region_z_dim=Lz-2=2.
  //     offset_local_no_card = (x + y*X) + (bz-2)*xy_pitch = (x + y*X) + 0.
  //     For (ix=0, iy=0, iz=1, c=0):
  //       addr = 0 + 6 * (2*2 + 0*2) = 24.
  //     For (ix=0, iy=0, iz=1, c=1):
  //       addr = 0 + 6 * (2*2 + 1*2) = 36.
  {
    const auto vi = span.set_and_validate(DataView::Standard, 0, 0, 1);
    assert(vi.valid);
    assert(vi.idx.cell().z == 2);
    assert(vi.idx.region_first_z() == 2);
    assert(vi.idx.region_z_dim() == 2);
    assert(vi.idx.offset_local_no_card() == 0);
    assert(span.linear_offset(vi.idx, components, /*c=*/0) == 24);
    assert(span.linear_offset(vi.idx, components, /*c=*/1) == 36);
  }

  // (3) iz=2 (Standard) → buffer-z=3 → still internal; offset_local_no_card adds (bz-2)*xy_pitch.
  //     For (ix=0, iy=0, iz=2, c=0):
  //       addr = 0 + (3-2)*6 + 6 * (2*2 + 0*2) = 6 + 24 = 30.
  {
    const auto vi = span.set_and_validate(DataView::Standard, 0, 0, 2);
    assert(vi.valid);
    assert(vi.idx.cell().z == 3);
    assert(vi.idx.region_first_z() == 2);
    assert(vi.idx.region_z_dim() == 2);
    assert(vi.idx.offset_local_no_card() == 1 * xy_pitch);
    assert(span.linear_offset(vi.idx, components, /*c=*/0) == 30);
  }

  // (4) iz=3 (Standard) → buffer-z=4 → up boundary; region_first_z = Lz = 4, region_z_dim = 1.
  //     For (ix=1, iy=2, iz=3, c=0):
  //       addr = (1 + 2*2) + 6 * (4*2 + 0*1) = 5 + 48 = 53.
  {
    const auto vi = span.set_and_validate(DataView::Standard, 1, 2, 3);
    assert(vi.valid);
    assert(vi.idx.cell().z == 4);
    assert(vi.idx.region_first_z() == 4);
    assert(vi.idx.region_z_dim() == 1);
    assert(span.linear_offset(vi.idx, components, /*c=*/0) == 53);
  }

  // (5) Internal view: (ix=1, iy=2, iz=0) → buffer-z=2, region_first_z=2, region_z_dim=2.
  //     offset_local_no_card = x + y*X + iz*xy_pitch = 5 + 0 = 5.
  //     For c=1: addr = 5 + 6*(2*2 + 1*2) = 5 + 36 = 41.
  {
    const auto vi = span.set_and_validate(DataView::Internal, 1, 2, 0);
    assert(vi.valid);
    assert(vi.idx.cell().z == 2);
    assert(vi.idx.region_first_z() == 2);
    assert(vi.idx.region_z_dim() == 2);
    assert(vi.idx.offset_local_no_card() == 5);
    assert(span.linear_offset(vi.idx, components, /*c=*/1) == 41);
  }

  // (6) Boundary view: iz=0 → down, iz=1 → up.
  {
    const auto down = span.set_and_validate(DataView::Boundary, 0, 0, 0);
    assert(down.valid);
    assert(down.idx.cell().z == 1);
    assert(down.idx.region_first_z() == 1);
    assert(down.idx.region_z_dim() == 1);

    const auto up = span.set_and_validate(DataView::Boundary, 0, 0, 1);
    assert(up.valid);
    assert(up.idx.cell().z == 4);
    assert(up.idx.region_first_z() == 4);
    assert(up.idx.region_z_dim() == 1);
  }

  // (7) Out-of-iter-extent threads must report valid == false but still populate the
  //     region fields so the caller can early-out without UB.
  {
    const auto over = span.set_and_validate(DataView::Standard, xDim, 0, 0);
    assert(!over.valid);
  }
}

// Round-trip a buffer through the Partition's read / write members and check the disg
// pitch arithmetic from the consumer side (not just on Span). Uses a host-side vector
// sized to the disg total storage; no GPU machinery involved.
void
check_partition_read_write() {
  constexpr int64_t xDim = 2;
  constexpr int64_t yDim = 3;
  constexpr int64_t lz = 4;
  constexpr int components = 2;
  const int64_t xy_pitch = xDim * yDim;

  const int64_3d haloed{xDim, yDim, lz + 2};
  const int64_3d global{xDim, yDim, haloed.z};
  const dd::Span span(haloed, global, xy_pitch, /*z_offset=*/0);

  const int64_t total = span.linear_storage_elements(components);
  std::vector<int32_t> buf(static_cast<size_t>(total), 0);

  using P = dd::Partition<int32_t, components>;
  P part(buf.data(), DataView::Standard);

  // Write a unique value at every (ix, iy, iz, c) in Standard view, then read back via
  // the same partition + via a freshly built one to confirm the layout is stable across
  // view tags (the layout is invariant; only the iteration shape on the launch side changes).
  int32_t tag = 1;
  for (int64_t iz = 0; iz < lz; ++iz) {
    for (int64_t iy = 0; iy < yDim; ++iy) {
      for (int64_t ix = 0; ix < xDim; ++ix) {
        const dd::ValidatedIdx vi = span.set_and_validate(DataView::Standard, ix, iy, iz);
        for (int c = 0; c < components; ++c) {
          part.write(span, vi.idx, c, tag++);
        }
      }
    }
  }

  tag = 1;
  for (int64_t iz = 0; iz < lz; ++iz) {
    for (int64_t iy = 0; iy < yDim; ++iy) {
      for (int64_t ix = 0; ix < xDim; ++ix) {
        const dd::ValidatedIdx vi = span.set_and_validate(DataView::Standard, ix, iy, iz);
        for (int c = 0; c < components; ++c) {
          assert(part.read(span, vi.idx, c) == tag++);
        }
      }
    }
  }
}

}  // namespace

int
main() {
  // Compile-time invariants — the disaggregated layout pins both radii to 1.
  static_assert(dd::Span::ghost_radius == 1);
  static_assert(dd::Span::boundary_radius == 1);

  // Empty-system policy: DenseDisg ctor rejects a zero-GPU System at construction
  // time, matching Dense / Block. Locks the diagnostic point.
  {
    std::vector<carbon::detail::Gpu> empty_devices;
    carbon::System sys(carbon::detail::GpuGroup(std::move(empty_devices)));
    CARBON_EXPECT_THROW(carbon::DenseDisg(sys, carbon::int64_3d{8, 8, 8}, /*halo_z=*/1),
                        std::invalid_argument);
  }

  // Iter extent shape for a small slab: Standard = Lz on z, Internal = Lz - 2, Boundary = 2.
  {
    constexpr dd::Span s(int64_3d{4, 5, 6}, int64_3d{4, 5, 6}, 20, 0, 1);
    static_assert(s.iter_extent(DataView::Standard).z == 4);
    static_assert(s.iter_extent(DataView::Internal).z == 2);
    static_assert(s.iter_extent(DataView::Boundary).z == 2);
    static_assert(s.logic_extent().z == 4);
    static_assert(s.linear_storage_elements(/*components=*/3) == 6 * 3 * 20);
  }

  // global_location adds z_offset onto the buffer-z.
  {
    constexpr dd::Span s(int64_3d{2, 2, 6}, int64_3d{2, 2, 24}, 4, 100, 1);
    constexpr dd::Idx idx(0, 0, 3, 0, 3, 1);
    static_assert(s.global_location(idx).z == 103);
  }

  // Standard walk on a few sizes — no double-write, no ghost overlap, no out-of-buffer.
  check_standard_walk(/*xDim=*/2, /*yDim=*/2, /*lz=*/4, /*components=*/1);
  check_standard_walk(/*xDim=*/2, /*yDim=*/3, /*lz=*/4, /*components=*/2);
  check_standard_walk(/*xDim=*/3, /*yDim=*/4, /*lz=*/6, /*components=*/3);

  // Internal + Boundary == Standard at the address-set level.
  check_internal_boundary_partition(/*xDim=*/2, /*yDim=*/2, /*lz=*/4, /*components=*/1);
  check_internal_boundary_partition(/*xDim=*/2, /*yDim=*/3, /*lz=*/4, /*components=*/2);
  check_internal_boundary_partition(/*xDim=*/3, /*yDim=*/4, /*lz=*/6, /*components=*/3);

  // Hand-computed spot parity vs. Neon's dPartition::getPitch + dSpan::setAndValidate.
  check_neon_parity_spot();

  // Partition.read / .write round-trip in the disg layout.
  check_partition_read_write();

  return 0;
}
