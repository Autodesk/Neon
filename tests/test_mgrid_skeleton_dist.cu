// Cross-rank (MPI / one-rank-per-GPU) multi-level Skeleton test for carbon::MultiResBlock (mGrid).
// Launched under `mpiexec -n 2 ./test_mgrid_skeleton_dist`.
//
// The distributed sibling of test_mgrid_skeleton: that test proves a multi-level pipeline rides
// the grid-agnostic Skeleton intra-process; this one proves the Skeleton's *auto-inserted*
// per-level halo carries the cross-rank DistTransfer companion — i.e. the same-level neighbour
// exchange the Skeleton wires in front of a stencil read actually delivers the neighbour rank's
// owned bricks into this rank's ghost bricks, on every level of the tower.
//
// Pipeline (one Skeleton, both levels), per level l in {0, 1}:
//   fill  : u@L_l = encode(l, global)                                   (write u@L_l)
//   diag  : read_stencil u@L_l, check +z / -z same-level neighbour      (read_stencil, write diag)
// The read_stencil makes the Skeleton auto-insert u@L_l's HaloBlock (intra-process Transfer +
// cross-rank DistTransfer) between fill and diag. Verification is kernel-side (a rank cannot peek
// the neighbour rank's buffer, test_halo_block Invariant 2): at the rank cut the +z/-z neighbour
// resolves to a ghost brick the dist halo just filled, and must equal encode(l, global±z); a
// broken halo leaves the ghost at its memset-0 value and the read mismatches (diag := 2).
//
// Domain: fully-active 16x16x64 finest, depth-2 octree (finest.z a multiple of MemZ*R^(depth-1) =
// 16, so the z-alignment precondition holds and both levels carry ghost bricks at the rank cut).
// Skipped (returns 0) when world_size < 2 or no CUDA device. CUDA-only (matches the mGrid suite).

#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/distributed.h>
#include <carbon/kernel.h>
#include <carbon/mgrid.h>
#include <carbon/skeleton.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

namespace {

using SBlock = carbon::DefaultMultiResBlock;  // 8^3 memory brick / 2^3 user block (octree).
using Span = carbon::detail::block::Span<SBlock>;
using Idx = carbon::detail::block::Idx;

constexpr int VoxelCount = SBlock::VoxelCount;

[[nodiscard]] CARBON_HD constexpr auto
encode(int level,
       int x,
       int y,
       int z) noexcept -> int32_t {
  return level * 1000000 + (x + 100 * y + 10000 * z);
}

}  // namespace

int
main() {
  carbon::init();

  auto distributed = std::make_shared<const carbon::Distributed>();
  if (distributed->world_size() < 2) {
    CARBON_LOG_INFO("test_mgrid_skeleton_dist: world_size={} < 2, skipping",
                    distributed->world_size());
    return 0;
  }
  if (carbon::device_count() < 1) {
    CARBON_LOG_INFO("test_mgrid_skeleton_dist: no CUDA devices, skipping");
    return 0;
  }

  const int rank = distributed->world_rank();
  carbon::System system(distributed);
  if (system.gpu_group().size() != 1U) {
    CARBON_LOG_ERROR(
        "test_mgrid_skeleton_dist: expected gpu_group size 1 (one rank per GPU), got {}",
        system.gpu_group().size());
    return 1;
  }

  const carbon::int64_3d finest{16, 16, 64};
  const auto is_active = [](int /*level*/, const carbon::int64_3d& /*v*/) -> bool {
    return true;
  };
  carbon::MultiResBlock<SBlock> mblock(system, finest, /*depth=*/2, is_active, /*ref_factor=*/2,
                                       "mgrid_skeleton_dist");

  auto u = mblock.newField<int32_t, 1>(carbon::MemorySide::Device, "u");
  auto diag = mblock.newField<int32_t, 1>(carbon::MemorySide::Device, "diag");
  carbon::Stream stream = system.stream();

  // Build the whole-tower pipeline once: per level a fill then a stencil-read diag kernel; the
  // Skeleton auto-inserts each level's HaloBlock (with its cross-rank DistTransfer companion)
  // between them.
  carbon::Skeleton sk(system, stream);
  for (int l = 0; l < mblock.depth(); ++l) {
    sk.add(mblock.make_kernel(l, [u, l](carbon::KernelLoader& loader) {
      auto p = loader.write(u.level_view(l));
      return [p, l] CARBON_HD(Span s, Idx idx) -> void {
        const auto g = s.global_location(idx);
        p.write(s, idx, 0,
                encode(l, static_cast<int>(g.x), static_cast<int>(g.y), static_cast<int>(g.z)));
      };
    }));
    sk.add(mblock.make_kernel(l, [u, diag, l](carbon::KernelLoader& loader) {
      auto field = loader.read_stencil(u.level_field(l));  // block::Partition (has neighbour()).
      auto out = loader.write(diag.level_field(l));
      return [field, out, l] CARBON_HD(Span s, Idx idx) -> void {
        const auto g = s.global_location(idx);
        const int gx = static_cast<int>(g.x);
        const int gy = static_cast<int>(g.y);
        const int gz = static_cast<int>(g.z);
        const auto zp = field.neighbour(s, idx, carbon::int8_3d{0, 0, +1}, 0);
        const auto zm = field.neighbour(s, idx, carbon::int8_3d{0, 0, -1}, 0);
        int32_t zp_ok = 1;
        if (zp.valid) {
          zp_ok = (zp.value == encode(l, gx, gy, gz + 1)) ? 1 : 2;
        }
        int32_t zm_ok = 1;
        if (zm.valid) {
          zm_ok = (zm.value == encode(l, gx, gy, gz - 1)) ? 1 : 2;
        }
        out.write(s, idx, 0, (zp_ok == 2 || zm_ok == 2) ? 2 : 1);
      };
    }));
  }
  sk.compile();

  // One auto-halo per level's stencil read.
  size_t auto_halos = 0;
  for (size_t e : sk.effective_to_user()) {
    if (e == carbon::Skeleton::kNoKernel) {
      ++auto_halos;
    }
  }
  if (auto_halos != static_cast<size_t>(mblock.depth())) {
    std::fprintf(stderr, "test_mgrid_skeleton_dist: rank %d expected %d auto-halos, got %zu\n",
                 rank, mblock.depth(), auto_halos);
    return 1;
  }

  // Zero u (owned + ghost) so an undelivered halo surfaces as a neighbour mismatch, and diag so
  // the host sweep starts from a known state — then replay the compiled pipeline.
  for (int l = 0; l < mblock.depth(); ++l) {
    const size_t slots = static_cast<size_t>(mblock.grid(l).storage_block_count(0)) *
                         static_cast<size_t>(VoxelCount);
    if (slots != 0U) {
      system.async_memset(u.level_field(l).gpu_buffers()[0], 0, slots * sizeof(int32_t), stream);
      system.async_memset(diag.level_field(l).gpu_buffers()[0], 0, slots * sizeof(int32_t), stream);
    }
  }
  stream.wait();

  sk.run(carbon::DataView::Standard);
  sk.wait();

  // Sweep diag (owned bricks only) on the host: any 2 is a cross-rank halo failure.
  int64_t total_checked = 0;
  for (int l = 0; l < mblock.depth(); ++l) {
    auto host = mblock.grid(l).download_to_host<int32_t, 1>(diag.level_field(l), stream);
    stream.wait();
    const uint32_t nb = mblock.grid(l).block_count(0);
    for (uint32_t b = 0; b < nb; ++b) {
      const auto& bm = mblock.grid(l).host_bitmasks(0)[b];
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
            if (host[0].data()[lin] == 2) {
              std::fprintf(stderr,
                           "test_mgrid_skeleton_dist: rank %d L%d cross-rank neighbour mismatch at "
                           "brick %u voxel (%d,%d,%d)\n",
                           rank, l, b, x, y, z);
              return 1;
            }
            ++total_checked;
          }
        }
      }
    }
  }

  CARBON_LOG_INFO(
      "test_mgrid_skeleton_dist: rank {} of {} ok (auto-halos={}, per-level cross-rank neighbour "
      "cells checked: {})",
      rank, distributed->world_size(), auto_halos, total_checked);
  return 0;
}
