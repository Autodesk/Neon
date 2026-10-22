// Multi-GPU intra-level halo test for carbon::MultiResBlock (mGrid).
//
// mGrid's cross-level (parent/child) path is transfer-free by co-residency, so the only data
// movement a multi-resolution stencil needs is the *same-level* neighbour exchange. This test
// proves that exchange works on the projected co-resident cut at EVERY level of the tower:
//
//   - builds a fully-active depth-2 octree over a 16×16×N domain split across every visible GPU
//     (skipped if < 2). finest.z is a multiple of MemZ·R^(depth-1) = 16, so the multi-slab
//     z-alignment precondition holds and both levels carry ghost bricks at the slab cut.
//   - for each level: zero the per-GPU storage (owned + ghost slots), fill every active voxel of
//     every owned brick with encode(level, global) via MultiResBlock::make_kernel, then run the
//     intra-level halo through MultiResField::make_halo_kernel(level).
//   - validates (test_halo_block Invariant 1, per level): every ghost slot on each GPU holds the
//     byte-for-byte payload of the matching owned brick on the adjacent GPU, walking
//     Block::halo_pairs_upward / halo_pairs_downward on that level's grid.
//
// CUDA-only registration: per-brick D2H readback uses Allocation::unmanaged + System::async_copy
// (backend-portable), but the mGrid test suite is CUDA-only this pass (SYCL parity is a tracked
// follow-up).

#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/devices.h>
#include <carbon/kernel.h>
#include <carbon/mgrid.h>

#include "test_domain.h"

#include <cstdint>
#include <cstdio>
#include <utility>
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

// Per-level byte-level ghost-content check: each ghost slot must equal the matching owned brick
// on the adjacent GPU (mirrors test_halo_block Invariant 1).
template <typename FieldT>
[[nodiscard]] auto
check_level_ghosts(carbon::System& system,
                   const carbon::Block<SBlock>& grid,
                   FieldT& u,
                   int level,
                   size_t n_gpus,
                   carbon::Stream& stream) -> int {
  const auto& buffers = u.level_field(level).gpu_buffers();
  auto check = [&](size_t src_gpu, size_t dst_gpu, uint32_t src_id, uint32_t dst_id,
                   const char* dir) -> int {
    auto src_host = carbon::Cpu::buffer<int32_t>(static_cast<size_t>(VoxelCount));
    auto dst_host = carbon::Cpu::buffer<int32_t>(static_cast<size_t>(VoxelCount));
    const auto& src_buf = buffers[src_gpu];
    const auto& dst_buf = buffers[dst_gpu];
    auto src_view = carbon::detail::Allocation<int32_t>::unmanaged(
        src_buf.location(),
        src_buf.data() + static_cast<size_t>(src_id) * static_cast<size_t>(VoxelCount),
        static_cast<size_t>(VoxelCount), src_buf.gpu_id(), src_buf.relative_gpu_id());
    auto dst_view = carbon::detail::Allocation<int32_t>::unmanaged(
        dst_buf.location(),
        dst_buf.data() + static_cast<size_t>(dst_id) * static_cast<size_t>(VoxelCount),
        static_cast<size_t>(VoxelCount), dst_buf.gpu_id(), dst_buf.relative_gpu_id());
    system.async_copy(src_view, src_host, sizeof(int32_t) * static_cast<size_t>(VoxelCount),
                      stream);
    system.async_copy(dst_view, dst_host, sizeof(int32_t) * static_cast<size_t>(VoxelCount),
                      stream);
    stream.wait();
    for (size_t i = 0; i < static_cast<size_t>(VoxelCount); ++i) {
      if (src_host.data()[i] != dst_host.data()[i]) {
        std::fprintf(stderr,
                     "test_mgrid_halo: L%d %s mismatch src_gpu=%zu src_id=%u dst_gpu=%zu "
                     "dst_id=%u voxel=%zu got=%d want=%d\n",
                     level, dir, src_gpu, src_id, dst_gpu, dst_id, i, dst_host.data()[i],
                     src_host.data()[i]);
        return 1;
      }
    }
    return 0;
  };

  size_t pairs_checked = 0;
  for (size_t adj = 0; adj + 1U < n_gpus; ++adj) {
    for (const auto& p : grid.halo_pairs_upward(adj)) {
      if (check(adj, adj + 1, p.src_local_id, p.dst_local_id, "upward")) {
        return -1;
      }
      ++pairs_checked;
    }
    for (const auto& p : grid.halo_pairs_downward(adj)) {
      if (check(adj + 1, adj, p.src_local_id, p.dst_local_id, "downward")) {
        return -1;
      }
      ++pairs_checked;
    }
  }
  return static_cast<int>(pairs_checked);
}

[[nodiscard]] auto
run_test() -> int {
  carbon::init();
  auto devices = carbon::get_all_devices();
  if (devices.size() < 2U) {
    std::printf("test_mgrid_halo: fewer than 2 CUDA devices visible (%zu) — skipping\n",
                devices.size());
    return 0;
  }
  const size_t n_gpus = devices.size();
  carbon::System system(carbon::detail::GpuGroup{std::move(devices)});

  const carbon::int64_3d finest{16, 16, carbon::test::block_slab_z(64, n_gpus, SBlock::MemZ * 2)};
  const auto is_active = [](int /*level*/, const carbon::int64_3d& /*v*/) -> bool {
    return true;
  };
  carbon::MultiResBlock<SBlock> mblock(system, finest, /*depth=*/2, is_active, /*ref_factor=*/2,
                                       "mgrid_halo");

  auto u = mblock.newField<int32_t, 1>(carbon::MemorySide::Device, "u");
  carbon::Stream stream = system.stream();

  size_t total_pairs = 0;
  for (int l = 0; l < mblock.depth(); ++l) {
    // Zero every per-GPU buffer (owned + ghost slots) so an absent halo surfaces as a mismatch.
    for (size_t g = 0; g < n_gpus; ++g) {
      const size_t slots = static_cast<size_t>(mblock.grid(l).storage_block_count(g)) *
                           static_cast<size_t>(VoxelCount);
      if (slots == 0U) {
        continue;
      }
      system.async_memset(u.level_field(l).gpu_buffers()[g], 0, slots * sizeof(int32_t), stream);
    }
    stream.wait();

    // Fill every active owned voxel with encode(level, global) — own-level write.
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

    // Intra-level halo exchange via the new convenience.
    carbon::Kernel halo = u.make_halo_kernel(l);
    halo.launch(stream);
    stream.wait();

    const int pairs = check_level_ghosts(system, mblock.grid(l), u, l, n_gpus, stream);
    if (pairs < 0) {
      return 1;
    }
    if (pairs == 0) {
      std::fprintf(stderr,
                   "test_mgrid_halo: level %d had no halo pairs across %zu GPUs — the projected "
                   "co-resident cut should leave ghost bricks at the slab boundary\n",
                   l, n_gpus);
      return 1;
    }
    total_pairs += static_cast<size_t>(pairs);
  }

  std::printf("test_mgrid_halo: OK (%zu GPU(s), intra-level halo pairs checked: %zu)\n", n_gpus,
              total_pairs);
  return 0;
}

}  // namespace

int
main() {
  return run_test();
}
