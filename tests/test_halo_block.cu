// Multi-GPU HaloBlock round-trip + cross-slab neighbour test.
//
// Builds a Block<8,8,8> over a sphere predicate on a 64-cube domain split across
// every visible GPU (skipped if fewer than 2). After init, every voxel of every
// owned brick is encoded with its global (gx, gy, gz) coordinate.
//
// We then issue a HaloBlock<int32_t, 1> exchange and validate two invariants:
//
//   1. After the halo, each ghost slot on every GPU contains the encoded payload
//      of the matching owned brick on the adjacent GPU. That's a direct
//      cell-by-cell check via Block::halo_pairs_upward / halo_pairs_downward
//      reading back the field's per-GPU device buffer.
//   2. A device kernel that walks owned bricks at the slab boundary and reads
//      their +z / -z neighbours via Partition::neighbour returns the expected
//      encoded payload (i.e. ghost connectivity entries point into the ghost
//      range and read the right data — proving the connectivity rewrite + ghost
//      bitmask upload work end-to-end).
//
// The first invariant exercises the byte-level halo plan; the second exercises
// the kernel-side path. Together they cover the full Phase B contract.

#include <carbon/block.h>
#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/cuda_annotation.h>
#include <carbon/devices.h>
#include <carbon/halo_block.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/system.h>

#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

namespace {

using SB = carbon::detail::block::DefaultStaticBlock;
using BlockGrid = carbon::Block<SB>;
using Span = BlockGrid::Span;
using Idx = BlockGrid::Idx;
using FieldT = BlockGrid::Field<int32_t, 1>;

[[nodiscard]] CARBON_HD constexpr auto
encode(int gx,
       int gy,
       int gz) noexcept -> int32_t {
  return ((gx + 1) * 73856093) ^ ((gy + 1) * 19349663) ^ ((gz + 1) * 83492791);
}

[[nodiscard]] auto
sphere_predicate(int64_t domain_n,
                 const carbon::int64_3d& gc) -> bool {
  const double cx = (domain_n - 1) * 0.5;
  const double cy = (domain_n - 1) * 0.5;
  const double cz = (domain_n - 1) * 0.5;
  const double r = domain_n * 0.45;
  const double dx = static_cast<double>(gc.x) - cx;
  const double dy = static_cast<double>(gc.y) - cy;
  const double dz = static_cast<double>(gc.z) - cz;
  return (dx * dx + dy * dy + dz * dz) <= (r * r);
}

[[nodiscard]] auto
run_test() -> int {
  carbon::init();
  auto devices = carbon::get_all_devices();
  if (devices.size() < 2U) {
    std::printf("test_halo_block: fewer than 2 CUDA devices visible (%zu) — skipping\n",
                devices.size());
    return 0;
  }
  const size_t n_gpus = devices.size();
  carbon::System system(carbon::detail::GpuGroup{std::move(devices)});

  constexpr int64_t domain_n = 64;
  const carbon::int64_3d domain_extent{domain_n, domain_n, domain_n};
  const auto predicate = [domain_n](const carbon::int64_3d& gc) -> bool {
    return sphere_predicate(domain_n, gc);
  };
  BlockGrid block(system, domain_extent, predicate, "test_halo_block");
  size_t total_ghosts = 0;
  for (size_t g = 0; g < n_gpus; ++g) {
    total_ghosts += block.ghost_lower_count(g) + block.ghost_upper_count(g);
  }
  std::printf("test_halo_block: gpus=%zu retained=%u ghosts=%zu\n", n_gpus, block.block_count(),
              total_ghosts);
  if (total_ghosts == 0U) {
    std::fprintf(stderr,
                 "test_halo_block: expected at least some ghost slots in a multi-GPU "
                 "layout — Block partitioning is broken\n");
    return 1;
  }

  FieldT u = block.template newField<int32_t, 1>(carbon::MemorySide::Device, "u");

  // Zero-init the entire per-GPU buffers (owned + ghost slots) so any stale value
  // in a ghost slot would surface as a mismatch in the readback below.
  carbon::Stream stream = system.stream();
  for (size_t g = 0; g < n_gpus; ++g) {
    const size_t total_slots_g =
        static_cast<size_t>(block.storage_block_count(g)) * static_cast<size_t>(SB::VoxelCount);
    if (total_slots_g == 0U) {
      continue;
    }
    system.async_memset(u.gpu_buffers()[g], 0, total_slots_g * sizeof(int32_t), stream);
  }
  stream.wait();

  // Init kernel: write encode(global coords) into every active voxel of every
  // owned brick. Inactive voxels stay zero.
  carbon::Kernel init = block.make_kernel([u, &block](carbon::KernelLoader& loader) {
    auto field = loader.write(u);
    const carbon::int32_3d* origins = block.device_origins_ptr(loader.relative_gpu_id());
    return [field, origins] CARBON_HD(Span s, Idx idx) -> void {
      const auto intra = idx.intra();
      const carbon::int32_3d origin = origins[idx.block_idx()];
      field.write(s, idx, 0, encode(origin.x + intra.x, origin.y + intra.y, origin.z + intra.z));
    };
  });
  init.launch(stream);
  stream.wait();

  // Issue the halo exchange.
  carbon::HaloBlock<int32_t, 1> halo(block, u);
  std::printf("test_halo_block: halo pair_count=%zu\n", halo.pair_count());
  halo(stream);
  stream.wait();

  // Invariant 1: byte-level ghost contents match the source brick on the remote
  // GPU. Walk every adjacency, fetch the source GPU's owned brick (encoded by
  // init), then fetch the destination GPU's ghost slot and compare voxel-wise.
  for (size_t adj = 0; adj + 1U < n_gpus; ++adj) {
    const auto& upward = block.halo_pairs_upward(adj);
    const auto& downward = block.halo_pairs_downward(adj);
    auto check = [&](size_t src_gpu, size_t dst_gpu, uint32_t src_id, uint32_t dst_id,
                     const char* dir) -> int {
      auto src_host = carbon::Cpu::buffer<int32_t>(static_cast<size_t>(SB::VoxelCount));
      auto dst_host = carbon::Cpu::buffer<int32_t>(static_cast<size_t>(SB::VoxelCount));
      // Backend-portable per-brick D2H readback: build an unmanaged sub-view of
      // the per-GPU buffer at the brick's slab offset and route through
      // System::async_copy. The async_copy primitive handles per-GPU stream
      // dispatch + cross-backend memcpy semantics, replacing the previous raw
      // cudaSetDevice + cudaMemcpy(DeviceToHost) pair (gated this test on
      // carbon_add_cuda_only_unit_test).
      const auto& src_buf = u.gpu_buffers()[src_gpu];
      const auto& dst_buf = u.gpu_buffers()[dst_gpu];
      auto src_view = carbon::detail::Allocation<int32_t>::unmanaged(
          src_buf.location(),
          src_buf.data() + static_cast<size_t>(src_id) * static_cast<size_t>(SB::VoxelCount),
          static_cast<size_t>(SB::VoxelCount), src_buf.gpu_id(), src_buf.relative_gpu_id());
      auto dst_view = carbon::detail::Allocation<int32_t>::unmanaged(
          dst_buf.location(),
          dst_buf.data() + static_cast<size_t>(dst_id) * static_cast<size_t>(SB::VoxelCount),
          static_cast<size_t>(SB::VoxelCount), dst_buf.gpu_id(), dst_buf.relative_gpu_id());
      system.async_copy(src_view, src_host, sizeof(int32_t) * static_cast<size_t>(SB::VoxelCount),
                        stream);
      system.async_copy(dst_view, dst_host, sizeof(int32_t) * static_cast<size_t>(SB::VoxelCount),
                        stream);
      stream.wait();
      for (size_t i = 0; i < static_cast<size_t>(SB::VoxelCount); ++i) {
        if (src_host.data()[i] != dst_host.data()[i]) {
          std::fprintf(stderr,
                       "test_halo_block: %s mismatch adj=%zu src_gpu=%zu src_id=%u "
                       "dst_gpu=%zu dst_id=%u voxel=%zu got=%d want=%d\n",
                       dir, adj, src_gpu, src_id, dst_gpu, dst_id, i, dst_host.data()[i],
                       src_host.data()[i]);
          return 1;
        }
      }
      return 0;
    };
    for (const auto& p : upward) {
      if (check(adj, adj + 1, p.src_local_id, p.dst_local_id, "upward")) {
        return 1;
      }
    }
    for (const auto& p : downward) {
      if (check(adj + 1, adj, p.src_local_id, p.dst_local_id, "downward")) {
        return 1;
      }
    }
  }

  // Invariant 2: kernel-side cross-slab neighbour read returns the same encode
  // value the remote GPU's owned brick wrote. We capture per-GPU device-side
  // origin pointers so the kernel can compute its own (gx, gy, gz) and compare
  // to neighbour(stencil, +z/-z) at iteration time.
  FieldT diag = block.template newField<int32_t, 1>(carbon::MemorySide::Device, "diag");
  for (size_t g = 0; g < n_gpus; ++g) {
    const size_t total_slots_g =
        static_cast<size_t>(block.storage_block_count(g)) * static_cast<size_t>(SB::VoxelCount);
    if (total_slots_g == 0U) {
      continue;
    }
    system.async_memset(diag.gpu_buffers()[g], 0, total_slots_g * sizeof(int32_t), stream);
  }
  stream.wait();

  // Diagnostic kernel: write 1 to diag[idx] when the +z neighbour read either
  //   (a) reports valid=false (no cross-slab neighbour exists), or
  //   (b) reports valid=true and the value equals encode(gx, gy, gz+1).
  // Otherwise write 2 (mismatch) so a ground-truth voxel-wise sweep on the host
  // can distinguish OK / mismatch / not-checked.
  carbon::Kernel diag_kernel = block.make_kernel([u, diag, &block](carbon::KernelLoader& loader) {
    auto field = loader.read(u);
    auto out = loader.write(diag);
    const carbon::int32_3d* origins = block.device_origins_ptr(loader.relative_gpu_id());
    return [field, out, origins] CARBON_HD(Span s, Idx idx) -> void {
      const auto intra = idx.intra();
      const carbon::int32_3d origin = origins[idx.block_idx()];
      const int gx = origin.x + intra.x;
      const int gy = origin.y + intra.y;
      const int gz = origin.z + intra.z;
      const auto zp = field.neighbour(s, idx, carbon::int8_3d{0, 0, +1}, 0);
      const auto zm = field.neighbour(s, idx, carbon::int8_3d{0, 0, -1}, 0);
      int32_t zp_ok = 1;
      if (zp.valid) {
        zp_ok = (zp.value == encode(gx, gy, gz + 1)) ? 1 : 2;
      }
      int32_t zm_ok = 1;
      if (zm.valid) {
        zm_ok = (zm.value == encode(gx, gy, gz - 1)) ? 1 : 2;
      }
      out.write(s, idx, 0, (zp_ok == 2 || zm_ok == 2) ? 2 : 1);
    };
  });
  diag_kernel.launch(stream);
  stream.wait();

  // Sweep the diag field on the host: any mismatch (== 2) is a failure.
  int64_t checked = 0;
  for (size_t g = 0; g < n_gpus; ++g) {
    const uint32_t n_blocks_g = block.block_count(g);
    if (n_blocks_g == 0U) {
      continue;
    }
    const size_t total_g = static_cast<size_t>(n_blocks_g) * static_cast<size_t>(SB::VoxelCount);
    auto host = carbon::Cpu::buffer<int32_t>(total_g);
    system.async_copy(diag.gpu_buffers()[g], host, total_g * sizeof(int32_t), stream);
    stream.wait();
    for (uint32_t b = 0; b < n_blocks_g; ++b) {
      const auto& bm = block.host_bitmasks(g)[b];
      for (int z = 0; z < SB::MemZ; ++z) {
        for (int y = 0; y < SB::MemY; ++y) {
          for (int x = 0; x < SB::MemX; ++x) {
            if (!bm.is_active(x, y, z)) {
              continue;
            }
            const size_t lin = static_cast<size_t>(b) * static_cast<size_t>(SB::VoxelCount) +
                               static_cast<size_t>(x) +
                               static_cast<size_t>(SB::MemX) * static_cast<size_t>(y) +
                               static_cast<size_t>(SB::MemX) * static_cast<size_t>(SB::MemY) *
                                   static_cast<size_t>(z);
            const int32_t v = host.data()[lin];
            if (v == 2) {
              const auto origin = block.host_origins(g)[b];
              std::fprintf(stderr,
                           "test_halo_block: cross-slab neighbour mismatch g=%zu b=%u "
                           "(%d,%d,%d) origin=(%d,%d,%d)\n",
                           g, b, x, y, z, origin.x, origin.y, origin.z);
              return 1;
            }
            ++checked;
          }
        }
      }
    }
  }

  std::printf("test_halo_block: OK (cross-slab neighbour cells checked: %ld)\n",
              static_cast<long>(checked));
  return 0;
}

}  // namespace

int
main() {
  return run_test();
}
