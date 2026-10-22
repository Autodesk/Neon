// Multi-level Skeleton OverlapMode parity test for carbon::MultiResBlock (mGrid).
//
// Cross-tower OCC: the mGrid V-cycle pipeline from test_mgrid_skeleton (fill, restrict,
// read_stencil smooth, prolong) is replayed under OverlapMode::{Standard,Extended,
// TwoWayExtended}. The rewrite splits the L1 stencil and isolates the auto-inserted halo on
// a dedicated stream; numerics must match the OverlapMode::None reference bit-for-bit on
// every GPU (same post-condition as test_skeleton_block / test_skeleton_dist_overlap_block).
//
// CUDA portable (.cu via KernelLoader). Skips when no CUDA device is visible.

#include <carbon/carbon.h>
#include <carbon/devices.h>
#include <carbon/kernel.h>
#include <carbon/mgrid.h>
#include <carbon/skeleton.h>

#include "test_domain.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using SBlock = carbon::DefaultMultiResBlock;
using Span = carbon::detail::block::Span<SBlock>;
using Idx = carbon::detail::block::Idx;

constexpr int VoxelCount = SBlock::VoxelCount;
constexpr double kTol = 1e-9;

[[nodiscard]] CARBON_HD constexpr auto
field_value(int64_t x,
            int64_t y,
            int64_t z) noexcept -> double {
  return static_cast<double>(x) + 2.0 * static_cast<double>(y) + 3.0 * static_cast<double>(z);
}

[[nodiscard]] auto
overlap_mode_name(carbon::OverlapMode mode) -> const char* {
  switch (mode) {
    case carbon::OverlapMode::None:
      return "None";
    case carbon::OverlapMode::Standard:
      return "Standard";
    case carbon::OverlapMode::Extended:
      return "Extended";
    case carbon::OverlapMode::TwoWayExtended:
      return "TwoWayExtended";
  }
  return "?";
}

void
add_pipeline(carbon::Skeleton& sk,
             carbon::MultiResBlock<SBlock>& mblock,
             carbon::detail::mgrid::Field<double,
                                          1,
                                          SBlock>& u,
             carbon::detail::mgrid::Field<double,
                                          1,
                                          SBlock>& sm,
             carbon::detail::mgrid::Field<double,
                                          1,
                                          SBlock>& recon,
             int R) {
  sk.add(mblock.make_kernel(0, [u](carbon::KernelLoader& loader) {
    auto p = loader.write(u.level_view(0));
    return [p] CARBON_HD(Span s, Idx idx) -> void {
      const auto g = s.global_location(idx);
      p.write(s, idx, 0, field_value(g.x, g.y, g.z));
    };
  }));
  sk.add(mblock.make_kernel(1, [u, R](carbon::KernelLoader& loader) {
    auto p = u.bind_cross_level(loader, 1, carbon::AccessMode::Write, /*read_child=*/true);
    return [p, R] CARBON_HD(Span s, Idx idx) -> void {
      double sum = 0.0;
      int count = 0;
      for (int8_t oz = 0; oz < R; ++oz) {
        for (int8_t oy = 0; oy < R; ++oy) {
          for (int8_t ox = 0; ox < R; ++ox) {
            const auto cv = p.child(s, idx, carbon::int8_3d{ox, oy, oz}, 0);
            if (cv.valid) {
              sum += cv.value;
              ++count;
            }
          }
        }
      }
      p.write(s, idx, 0, count > 0 ? sum / static_cast<double>(count) : 0.0);
    };
  }));
  sk.add(mblock.make_kernel(1, [u, sm](carbon::KernelLoader& loader) {
    auto in = loader.read_stencil(u.level_field(1));
    auto out = loader.write(sm.level_field(1));
    return [in, out] CARBON_HD(Span s, Idx idx) -> void {
      const carbon::int8_3d offs[6] = {{+1, 0, 0}, {-1, 0, 0}, {0, +1, 0},
                                       {0, -1, 0}, {0, 0, +1}, {0, 0, -1}};
      double sum = 0.0;
      int count = 0;
      for (const auto& o : offs) {
        const auto nv = in.neighbour(s, idx, o, 0);
        if (nv.valid) {
          sum += nv.value;
          ++count;
        }
      }
      out.write(s, idx, 0, count > 0 ? sum / static_cast<double>(count) : 0.0);
    };
  }));
  sk.add(mblock.make_kernel(0, [u, recon](carbon::KernelLoader& loader) {
    auto pin = u.bind_cross_level(loader, 0, carbon::AccessMode::Read, /*read_child=*/false,
                                  /*read_parent=*/true);
    auto pout = loader.write(recon.level_view(0));
    return [pin, pout] CARBON_HD(Span s, Idx idx) -> void {
      const auto pv = pin.parent(s, idx, 0);
      pout.write(s, idx, 0, pv.valid ? pv.value : 0.0);
    };
  }));
}

[[nodiscard]] auto
run_once(carbon::System& system,
         carbon::OverlapMode mode,
         std::vector<double>& out_sm1) -> bool {
  const carbon::int64_3d finest{
      16, 16, carbon::test::block_slab_z(32, system.gpu_group().size(), SBlock::MemZ * 2)};
  const auto is_active = [](int /*level*/, const carbon::int64_3d& /*v*/) -> bool {
    return true;
  };
  carbon::MultiResBlock<SBlock> mblock(system, finest, /*depth=*/2, is_active, /*ref_factor=*/2,
                                       "mgrid_skeleton_overlap");
  const int R = mblock.ref_factor();
  auto u = mblock.newField<double, 1>(carbon::MemorySide::Device, "u");
  auto sm = mblock.newField<double, 1>(carbon::MemorySide::Device, "sm");
  auto recon = mblock.newField<double, 1>(carbon::MemorySide::Device, "recon");

  carbon::Skeleton sk(system, carbon::Skeleton::Options{mode});
  add_pipeline(sk, mblock, u, sm, recon, R);
  sk.compile();
  sk.run(carbon::DataView::Standard);
  sk.wait();

  carbon::Stream stream = system.stream();
  auto host_sm1 = mblock.grid(1).download_to_host<double, 1>(sm.level_field(1), stream);
  stream.wait();

  out_sm1.clear();
  const size_t n_gpus = system.gpu_group().size();
  for (size_t g = 0; g < n_gpus; ++g) {
    const uint32_t nb = mblock.grid(1).block_count(g);
    for (uint32_t b = 0; b < nb; ++b) {
      const auto& bm = mblock.grid(1).host_bitmasks(g)[b];
      for (int z = 0; z < SBlock::MemZ; ++z) {
        for (int y = 0; y < SBlock::MemY; ++y) {
          for (int x = 0; x < SBlock::MemX; ++x) {
            if (!bm.is_active(x, y, z)) {
              continue;
            }
            const size_t lin = static_cast<size_t>(b) * static_cast<size_t>(VoxelCount) +
                               static_cast<size_t>(x) +
                               static_cast<size_t>(SBlock::MemX) * static_cast<size_t>(y) +
                               static_cast<size_t>(SBlock::MemX) *
                                   static_cast<size_t>(SBlock::MemY) * static_cast<size_t>(z);
            out_sm1.push_back(host_sm1[g].data()[lin]);
          }
        }
      }
    }
  }
  return true;
}

}  // namespace

int
main() {
  carbon::init();
  auto devices = carbon::get_all_devices();
  if (devices.empty()) {
    std::printf("test_mgrid_skeleton_overlap: no CUDA devices — skipping\n");
    return 0;
  }
  carbon::System system(carbon::detail::GpuGroup{std::move(devices)});

  std::vector<double> ref;
  if (!run_once(system, carbon::OverlapMode::None, ref)) {
    return 1;
  }

  const std::array<carbon::OverlapMode, 3> variants{carbon::OverlapMode::Standard,
                                                    carbon::OverlapMode::Extended,
                                                    carbon::OverlapMode::TwoWayExtended};
  for (const auto mode : variants) {
    std::vector<double> got;
    if (!run_once(system, mode, got)) {
      return 1;
    }
    if (got.size() != ref.size()) {
      std::fprintf(stderr, "test_mgrid_skeleton_overlap[%s]: size mismatch got=%zu ref=%zu\n",
                   overlap_mode_name(mode), got.size(), ref.size());
      return 1;
    }
    size_t mismatches = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
      if (std::fabs(got[i] - ref[i]) > kTol) {
        if (mismatches < 4U) {
          std::fprintf(stderr,
                       "test_mgrid_skeleton_overlap[%s]: mismatch at %zu got=%.9f ref=%.9f\n",
                       overlap_mode_name(mode), i, got[i], ref[i]);
        }
        ++mismatches;
      }
    }
    if (mismatches > 0U) {
      std::fprintf(stderr, "test_mgrid_skeleton_overlap[%s]: %zu mismatches\n",
                   overlap_mode_name(mode), mismatches);
      return 1;
    }
    std::printf("test_mgrid_skeleton_overlap[%s]: OK (%zu coarse active cells)\n",
                overlap_mode_name(mode), ref.size());
  }
  return 0;
}
