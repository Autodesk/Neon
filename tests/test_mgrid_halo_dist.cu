// Cross-rank (MPI / one-rank-per-GPU) intra-level halo test for carbon::MultiResBlock (mGrid).
// Launched under `mpiexec -n 2 ./test_mgrid_halo_dist`.
//
// The co-resident sibling of test_mgrid_halo: that test proves the projected co-resident cut
// yields a correct per-level halo plan intra-process (byte-level ghost compare against the remote
// GPU's owned brick); this one proves the *cross-rank* path — the DistTransfer companion of each
// level's HaloBlock — actually delivers the neighbour rank's data into this rank's ghost bricks.
//
// Because a rank cannot peek the neighbour rank's buffer, verification is kernel-side
// (test_halo_block Invariant 2, per level): after the per-level halo, a Block compute kernel on
// each level walks owned voxels and reads their +z / -z same-level neighbour through
// block::Partition::neighbour; at the rank boundary that neighbour resolves to a ghost brick that
// the dist halo just filled, and its value must equal encode(level, global±z). A broken halo
// leaves the ghost at its memset-0 value and the read mismatches.
//
// Domain: fully-active 16×16×64 finest, depth-2 octree (finest.z a multiple of MemZ·R^(depth-1) =
// 16, so the multi-slab z-alignment precondition holds and both levels carry ghost bricks at the
// rank cut). Skipped (returns 0) when world_size < 2 or no CUDA device is visible. CUDA-only,
// matching the rest of the mGrid suite (SYCL parity is a tracked follow-up).

#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/distributed.h>
#include <carbon/kernel.h>
#include <carbon/mgrid.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

namespace {

using SBlock = carbon::DefaultMultiResBlock;  // 8³ memory brick / 2³ user block (octree).
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
    CARBON_LOG_INFO("test_mgrid_halo_dist: world_size={} < 2, skipping", distributed->world_size());
    return 0;
  }
  if (carbon::device_count() < 1) {
    CARBON_LOG_INFO("test_mgrid_halo_dist: no CUDA devices, skipping");
    return 0;
  }

  const int rank = distributed->world_rank();
  carbon::System system(distributed);
  if (system.gpu_group().size() != 1U) {
    CARBON_LOG_ERROR("test_mgrid_halo_dist: expected gpu_group size 1 (one rank per GPU), got {}",
                     system.gpu_group().size());
    return 1;
  }

  const carbon::int64_3d finest{16, 16, 64};
  const auto is_active = [](int /*level*/, const carbon::int64_3d& /*v*/) -> bool {
    return true;
  };
  carbon::MultiResBlock<SBlock> mblock(system, finest, /*depth=*/2, is_active, /*ref_factor=*/2,
                                       "mgrid_halo_dist");

  auto u = mblock.newField<int32_t, 1>(carbon::MemorySide::Device, "u");
  auto diag = mblock.newField<int32_t, 1>(carbon::MemorySide::Device, "diag");
  carbon::Stream stream = system.stream();

  int64_t total_checked = 0;
  for (int l = 0; l < mblock.depth(); ++l) {
    // Zero u (owned + ghost) so an undelivered halo surfaces as a neighbour mismatch, and diag so
    // the host sweep starts from a known state.
    const size_t slots = static_cast<size_t>(mblock.grid(l).storage_block_count(0)) *
                         static_cast<size_t>(VoxelCount);
    if (slots != 0U) {
      system.async_memset(u.level_field(l).gpu_buffers()[0], 0, slots * sizeof(int32_t), stream);
      system.async_memset(diag.level_field(l).gpu_buffers()[0], 0, slots * sizeof(int32_t), stream);
    }
    stream.wait();

    // Fill every active owned voxel with encode(level, global) through the mGrid kernel layer.
    carbon::Kernel fill = mblock.make_kernel(l, [u, l](carbon::KernelLoader& loader) {
      auto p = loader.write(u.level_view(l));
      return [p, l] CARBON_HD(Span s, Idx idx) -> void {
        const auto g = s.global_location(idx);
        p.write(s, idx, 0,
                encode(l, static_cast<int>(g.x), static_cast<int>(g.y), static_cast<int>(g.z)));
      };
    });
    fill.launch(stream);
    stream.wait();

    // Cross-rank intra-level halo (DistTransfer companion of the level's HaloBlock).
    carbon::Kernel halo = u.make_halo_kernel(l);
    halo.launch(stream);
    stream.wait();

    // Verify same-level +z / -z neighbour reads (resolving to dist-filled ghost bricks at the rank
    // cut). Driven through the level's Block grid because block::Partition has neighbour() (the
    // cross-level mgrid::Partition intentionally does not). Writes 1 = ok (match or no neighbour),
    // 2 = mismatch.
    carbon::Kernel diag_kernel =
        mblock.grid(l).make_kernel([u, diag, l](carbon::KernelLoader& loader) {
          auto field = loader.read(u.level_field(l));
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
        });
    diag_kernel.launch(stream);
    stream.wait();

    // Sweep diag (owned bricks only) on the host: any 2 is a cross-rank halo failure.
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
                           "test_mgrid_halo_dist: rank %d L%d cross-rank neighbour mismatch at "
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
      "test_mgrid_halo_dist: rank {} of {} ok (per-level cross-rank neighbour cells "
      "checked: {})",
      rank, distributed->world_size(), total_checked);
  return 0;
}
