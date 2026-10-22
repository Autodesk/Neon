// Host-only tests for the carbon multi-resolution (mGrid) primitives that don't need a GPU:
// the MultiResDescriptor level math (refinement factor / spacing / per-level extents) and the
// cross-level get_parent / get_child round-trip + table agreement on a small hand-built
// refined domain. Locks the level-layout conventions (level 0 finest, voxel_spacing = R^level,
// the ceil-div extent identity) and the parent<->child arithmetic the device path relies on.

#include <carbon/carbon.h>
#include <carbon/detail/mgrid/descriptor.h>
#include <carbon/devices.h>
#include <carbon/mgrid.h>

#include <cstdint>
#include <iostream>

namespace {

using carbon::int64_3d;
using carbon::detail::mgrid::MultiResDescriptor;

namespace mg = carbon::detail::mgrid;
using SBlock = carbon::DefaultMultiResBlock;  // 8³ memory brick / 2³ user block (octree).
using Span = carbon::detail::block::Span<SBlock>;
using Idx = carbon::detail::block::Idx;

[[nodiscard]] auto
test_descriptor_uniform_octree() -> int {
  // Depth-3 octree (R = 2): voxel_spacing = R^level, spacing = R^(level+1).
  const MultiResDescriptor d{/*depth=*/3, /*ref_factor=*/2};
  if (d.depth() != 3) {
    std::cerr << "test_descriptor_uniform_octree: depth\n";
    return 1;
  }
  if (d.ref_factor() != 2 || d.ref_factor(2) != 2) {
    std::cerr << "test_descriptor_uniform_octree: ref_factor\n";
    return 1;
  }
  if (d.voxel_spacing(0) != 1 || d.voxel_spacing(1) != 2 || d.voxel_spacing(2) != 4) {
    std::cerr << "test_descriptor_uniform_octree: voxel_spacing\n";
    return 1;
  }
  if (d.spacing(0) != 2 || d.spacing(1) != 4 || d.spacing(2) != 8) {
    std::cerr << "test_descriptor_uniform_octree: spacing\n";
    return 1;
  }
  return 0;
}

[[nodiscard]] auto
test_descriptor_level_extent() -> int {
  // Clean power-of-two domain: each coarser level halves per axis exactly.
  const MultiResDescriptor d{/*depth=*/3, /*ref_factor=*/2};
  const int64_3d finest{32, 16, 8};
  const int64_3d l0 = d.level_extent(finest, 0);
  const int64_3d l1 = d.level_extent(finest, 1);
  const int64_3d l2 = d.level_extent(finest, 2);
  if (l0 != int64_3d{32, 16, 8} || l1 != int64_3d{16, 8, 4} || l2 != int64_3d{8, 4, 2}) {
    std::cerr << "test_descriptor_level_extent: clean extents\n";
    return 1;
  }
  // Non-multiple domain: ceil-div, and the ceil identity keeps levels consistent
  // (ceil(ceil(n/R)/R) == ceil(n/R^2)).
  const int64_3d odd{10, 7, 5};
  const int64_3d o1 = d.level_extent(odd, 1);
  const int64_3d o2 = d.level_extent(odd, 2);
  if (o1 != int64_3d{5, 4, 3}) {  // ceil(10/2)=5, ceil(7/2)=4, ceil(5/2)=3
    std::cerr << "test_descriptor_level_extent: odd level 1\n";
    return 1;
  }
  if (o2 != int64_3d{3, 2, 2}) {  // ceil(10/4)=3, ceil(7/4)=2, ceil(5/4)=2
    std::cerr << "test_descriptor_level_extent: odd level 2\n";
    return 1;
  }
  // Identity check: ceil(ceil(n/R)/R) == level_extent(.,2).
  const int64_3d o1_then_1 = d.level_extent(o1, 1);
  if (o1_then_1 != o2) {
    std::cerr << "test_descriptor_level_extent: ceil-div identity broken\n";
    return 1;
  }
  return 0;
}

// Builds a 2-level octree over a fully-active 16³ finest domain and exercises the cross-level
// arithmetic on the host: for every (level-1 voxel, octant) pair, get_child must land on a
// real level-0 voxel whose get_parent recovers the original level-1 voxel. Since the domain is
// fully active, the 1 level-1 block × 512 voxels × 8 octants form a bijection onto the
// 8 level-0 blocks × 512 voxels, so the round-trip touches every finest voxel exactly once.
// Needs a (single) GPU because MultiResBlock construction allocates per-level Block storage and
// uploads the cross-level tables; skips cleanly when no device is visible.
[[nodiscard]] auto
test_cross_level_host() -> int {
  carbon::init();
  auto devices = carbon::get_all_devices();
  if (devices.empty()) {
    std::cout << "test_mgrid_static: no CUDA devices visible -- skipping cross-level host test\n";
    return 0;
  }
  if (devices.size() > 1U) {
    devices.erase(devices.begin() + 1, devices.end());
  }
  carbon::System system(carbon::detail::GpuGroup{std::move(devices)});

  const int64_3d finest{16, 16, 16};
  const auto is_active = [](int /*level*/, const int64_3d& /*v*/) -> bool {
    return true;
  };
  carbon::MultiResBlock<SBlock> mblock(system, finest, /*depth=*/2, is_active, /*ref_factor=*/2,
                                       "mgrid_test");

  const int R = mblock.ref_factor();
  constexpr int Mem = SBlock::MemX;
  if (mblock.depth() != 2 || R != 2) {
    std::cerr << "test_cross_level_host: depth/ref_factor\n";
    return 1;
  }
  if (mblock.grid(0).block_count() != 8U || mblock.grid(1).block_count() != 1U) {
    std::cerr << "test_cross_level_host: per-level block counts (got L0="
              << mblock.grid(0).block_count() << " L1=" << mblock.grid(1).block_count() << ")\n";
    return 1;
  }

  const auto host_span = [&](int l) -> Span {
    const auto& bm = mblock.grid(l).host_bitmasks(0);
    const auto& cn = mblock.grid(l).host_connectivity(0);
    const auto& og = mblock.grid(l).host_origins(0);
    return Span(bm.data(), static_cast<uint32_t>(bm.size()), cn.data(), og.data());
  };
  const Span s0 = host_span(0);
  const Span s1 = host_span(1);

  // Dummy non-null storage so has_parent / has_children pass their null-pointer gate; the
  // index-only methods (get_parent / get_child) never dereference it.
  static int dummy = 0;
  int* mem = &dummy;

  // Level 0 (finest): no child level; parent is level 1.
  const mg::Partition<int, 1, SBlock> p0{
      mem, R, mem, s1, mblock.parent_block_id_host(0).data(), nullptr, Span{}, nullptr};
  // Level 1 (coarsest): no parent level; child is level 0.
  const mg::Partition<int, 1, SBlock> p1{
      mem, R, nullptr, Span{}, nullptr, mem, s0, mblock.child_block_id_host(1).data()};

  int64_t checked = 0;
  for (uint32_t b = 0; b < s1.block_count(); ++b) {
    for (int iz = 0; iz < Mem; ++iz) {
      for (int iy = 0; iy < Mem; ++iy) {
        for (int ix = 0; ix < Mem; ++ix) {
          const auto v = s1.set_and_validate(b, ix, iy, iz);
          if (!v.valid) {
            std::cerr << "test_cross_level_host: level-1 voxel unexpectedly inactive\n";
            return 1;
          }
          if (!p1.has_children(s1, v.idx)) {
            std::cerr << "test_cross_level_host: has_children false on active level-1 voxel\n";
            return 1;
          }
          for (int8_t oz = 0; oz < R; ++oz) {
            for (int8_t oy = 0; oy < R; ++oy) {
              for (int8_t ox = 0; ox < R; ++ox) {
                const Idx c = p1.get_child(s1, v.idx, carbon::int8_3d{ox, oy, oz});
                if (!c.valid_block()) {
                  std::cerr << "test_cross_level_host: get_child returned invalid\n";
                  return 1;
                }
                if (!p0.has_parent(s0, c)) {
                  std::cerr << "test_cross_level_host: child has no parent\n";
                  return 1;
                }
                const Idx p = p0.get_parent(s0, c);
                if (p.block_idx() != v.idx.block_idx() || p.intra() != v.idx.intra()) {
                  std::cerr << "test_cross_level_host: parent(child) round-trip mismatch\n";
                  return 1;
                }
                ++checked;
              }
            }
          }
        }
      }
    }
  }
  // 1 level-1 block × 512 voxels × 2³ octants.
  if (checked != 1LL * 512LL * 8LL) {
    std::cerr << "test_cross_level_host: checked count " << checked << " != 4096\n";
    return 1;
  }
  return 0;
}

}  // namespace

int
main() {
  if (test_descriptor_uniform_octree() != 0)
    return 1;
  if (test_descriptor_level_extent() != 0)
    return 1;
  if (test_cross_level_host() != 0)
    return 1;
  std::cout << "test_mgrid_static: OK\n";
  return 0;
}
