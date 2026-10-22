// Host-side intra-process Block halo exchange: the @c std::memcpy twin of
// test_halo_block.cu, locking the @c HaloBlock::launch_on_host /
// @c BlockHaloKernelImpl::launch_on_host path closed under P0.α of the
// third-pass audit (@c docs/audit_2026_05_third_pass.md ).
//
// Builds a Block<8,8,8> over a sphere predicate on a 32-cube split across
// every visible CUDA device (skipped if fewer than 2). The field is built
// with @c MemorySide::Host on pass 1 and @c MemorySide::HostDevice on pass
// 2 — both should expose a fully-populated @c cpu_buffers and the host
// halo should land identical data in the ghost slots regardless of which
// side carries the writable device mirror. Pass 3 drives the same halo
// through the polymorphic @c Kernel handle.
//
// The verification mirrors @c test_halo_block 's "Invariant 1": for every
// adjacency, walk @c halo_pairs_upward / @c halo_pairs_downward and confirm
// that the ghost-slot brick on the destination GPU's host mirror is a
// voxel-by-voxel copy of the owned-brick on the source GPU's host mirror.
//
// The body of this test never touches the GPU — all writes go through
// @c field.cpu_buffers() , the halo runs through @c std::memcpy , and the
// reads pull from the host mirror.

#include <carbon/block.h>
#include <carbon/carbon.h>
#include <carbon/devices.h>
#include <carbon/halo_block.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/memory_side.h>
#include <carbon/system.h>

#include "test_domain.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <utility>
#include <vector>

namespace {

using SB = carbon::detail::block::DefaultStaticBlock;
using BlockGrid = carbon::Block<SB>;
using FieldT = BlockGrid::Field<int32_t, 1>;

constexpr int32_t kInactiveSentinel = -7777;
constexpr int32_t kGhostInitSentinel = -8888;

[[nodiscard]] constexpr auto
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

// Fill the host mirror: each owned brick's active voxels get encode(global),
// inactive voxels get @c kInactiveSentinel , and every ghost slot is seeded
// with @c kGhostInitSentinel so we can tell "halo overwrote me" from
// "halo no-op'd me".
auto
fill_host_owned_and_seed_ghosts(const BlockGrid& block,
                                FieldT& field) -> void {
  constexpr int32_t voxel_count = static_cast<int32_t>(SB::VoxelCount);
  const size_t n_gpus = block.gpu_count();
  for (size_t g = 0; g < n_gpus; ++g) {
    const uint32_t n_owned = block.block_count(g);
    const uint32_t storage = block.storage_block_count(g);
    int32_t* const data = field.cpu_buffers()[g].data();
    for (uint32_t b = 0; b < n_owned; ++b) {
      const auto& bm = block.host_bitmasks(g)[b];
      const auto origin = block.host_origins(g)[b];
      for (int z = 0; z < SB::MemZ; ++z) {
        for (int y = 0; y < SB::MemY; ++y) {
          for (int x = 0; x < SB::MemX; ++x) {
            const size_t voxel_idx = static_cast<size_t>(x) +
                                     static_cast<size_t>(SB::MemX) * static_cast<size_t>(y) +
                                     static_cast<size_t>(SB::MemX) * static_cast<size_t>(SB::MemY) *
                                         static_cast<size_t>(z);
            const size_t lin =
                static_cast<size_t>(b) * static_cast<size_t>(voxel_count) + voxel_idx;
            data[lin] = bm.is_active(x, y, z) ? encode(origin.x + x, origin.y + y, origin.z + z)
                                              : kInactiveSentinel;
          }
        }
      }
    }
    for (uint32_t b = n_owned; b < storage; ++b) {
      for (int32_t v = 0; v < voxel_count; ++v) {
        data[static_cast<size_t>(b) * static_cast<size_t>(voxel_count) + static_cast<size_t>(v)] =
            kGhostInitSentinel;
      }
    }
  }
}

// For each adjacency, walk every (src_local_id, dst_local_id) halo pair and
// confirm voxel-by-voxel that the destination ghost slot equals the source
// owned brick.
[[nodiscard]] auto
verify_host_ghosts(const BlockGrid& block,
                   const FieldT& field,
                   const char* tag) -> bool {
  constexpr size_t voxel_count = static_cast<size_t>(SB::VoxelCount);
  const size_t n_gpus = block.gpu_count();
  size_t mismatches = 0;
  size_t checked_bricks = 0;
  auto compare = [&](size_t src_gpu, size_t dst_gpu, uint32_t src_id, uint32_t dst_id,
                     const char* dir) -> void {
    const int32_t* src =
        field.cpu_buffers()[src_gpu].data() + static_cast<size_t>(src_id) * voxel_count;
    const int32_t* dst =
        field.cpu_buffers()[dst_gpu].data() + static_cast<size_t>(dst_id) * voxel_count;
    for (size_t v = 0; v < voxel_count; ++v) {
      if (src[v] != dst[v]) {
        if (mismatches < 4U) {
          std::fprintf(stderr,
                       "test_halo_block_host[%s]: %s mismatch src_gpu=%zu src_id=%u "
                       "dst_gpu=%zu dst_id=%u voxel=%zu got=%d want=%d\n",
                       tag, dir, src_gpu, src_id, dst_gpu, dst_id, v, dst[v], src[v]);
        }
        ++mismatches;
      }
    }
    ++checked_bricks;
  };
  for (size_t adj = 0; adj + 1U < n_gpus; ++adj) {
    for (const auto& p : block.halo_pairs_upward(adj)) {
      compare(adj, adj + 1U, p.src_local_id, p.dst_local_id, "upward");
    }
    for (const auto& p : block.halo_pairs_downward(adj)) {
      compare(adj + 1U, adj, p.src_local_id, p.dst_local_id, "downward");
    }
  }
  if (mismatches > 0U) {
    std::fprintf(stderr, "test_halo_block_host[%s]: %zu mismatches over %zu bricks\n", tag,
                 mismatches, checked_bricks);
    return false;
  }
  std::printf("test_halo_block_host[%s]: ok (%zu bricks compared)\n", tag, checked_bricks);
  return true;
}

[[nodiscard]] auto
run_pass(carbon::System& system,
         const BlockGrid& block,
         carbon::MemorySide side,
         const char* tag) -> bool {
  FieldT field = block.template newField<int32_t, 1>(side, "u_host_halo");
  if (field.cpu_buffers().size() != block.gpu_count()) {
    std::fprintf(stderr,
                 "test_halo_block_host[%s]: field cpu_buffers().size()=%zu != gpu_count=%zu\n", tag,
                 field.cpu_buffers().size(), block.gpu_count());
    return false;
  }
  fill_host_owned_and_seed_ghosts(block, field);

  carbon::HaloBlock<int32_t, 1> halo(block, field);
  const size_t n_gpus = block.gpu_count();
  const size_t expected_device_pairs = (n_gpus >= 2U) ? 2U * (n_gpus - 1U) : 0U;
  if (halo.pair_count() != expected_device_pairs) {
    std::fprintf(stderr,
                 "test_halo_block_host[%s]: pair_count=%zu != expected=%zu (coalesced device "
                 "transfers per adjacency+direction)\n",
                 tag, halo.pair_count(), expected_device_pairs);
    return false;
  }
  if (halo.host_intra_count() != halo.pair_count()) {
    std::fprintf(stderr,
                 "test_halo_block_host[%s]: host_intra_count=%zu != pair_count=%zu (coalesced "
                 "host/device face copies must match)\n",
                 tag, halo.host_intra_count(), halo.pair_count());
    return false;
  }
  if (halo.host_intra_count() == 0U) {
    std::fprintf(stderr,
                 "test_halo_block_host[%s]: empty host_intra (n_gpus=%zu, retained=%u) — sphere "
                 "predicate should retain enough bricks to span every slab boundary\n",
                 tag, block.gpu_count(), block.block_count());
    return false;
  }

  carbon::Stream stream = system.stream();
  try {
    halo.launch_on_host(stream);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "test_halo_block_host[%s]: launch_on_host failed: %s\n", tag, e.what());
    return false;
  }
  return verify_host_ghosts(block, field, tag);
}

[[nodiscard]] auto
run_kernel_pass(carbon::System& system,
                const BlockGrid& block) -> bool {
  FieldT field = block.template newField<int32_t, 1>(carbon::MemorySide::Host, "u_kernel_host");
  fill_host_owned_and_seed_ghosts(block, field);

  carbon::Kernel halo_k = field.make_halo_kernel("halo_block_host_k");
  if (!halo_k.runs_on_host()) {
    std::fprintf(stderr,
                 "test_halo_block_host[Kernel]: halo kernel does not advertise runs_on_host()\n");
    return false;
  }
  carbon::Stream stream = system.stream();
  try {
    halo_k.launch_on_host(stream);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "test_halo_block_host[Kernel]: launch_on_host failed: %s\n", e.what());
    return false;
  }
  return verify_host_ghosts(block, field, "Kernel");
}

[[nodiscard]] auto
run_test() -> int {
  carbon::init();
  auto devices = carbon::get_all_devices();
  if (devices.size() < 2U) {
    std::printf("test_halo_block_host: fewer than 2 CUDA devices visible (%zu) — skipping\n",
                devices.size());
    return 0;
  }
  carbon::System system(carbon::detail::GpuGroup{std::move(devices)});

  // Kept cubic as it grows, so the sphere scales with the domain instead of leaving the extra
  // brick rows empty.
  const int64_t domain_n = carbon::test::block_slab_z(32, system.gpu_group().size(), SB::MemZ);
  const carbon::int64_3d domain_extent{domain_n, domain_n, domain_n};
  const auto predicate = [domain_n](const carbon::int64_3d& gc) -> bool {
    return sphere_predicate(domain_n, gc);
  };
  BlockGrid block(system, domain_extent, predicate, "test_halo_block_host");
  std::printf("test_halo_block_host: gpus=%zu retained=%u\n", system.gpu_group().size(),
              block.block_count());

  if (!run_pass(system, block, carbon::MemorySide::Host, "Host")) {
    return EXIT_FAILURE;
  }
  if (!run_pass(system, block, carbon::MemorySide::HostDevice, "HostDevice")) {
    return EXIT_FAILURE;
  }
  if (!run_kernel_pass(system, block)) {
    return EXIT_FAILURE;
  }

  std::printf("test_halo_block_host: PASS\n");
  return EXIT_SUCCESS;
}

}  // namespace

int
main() {
  return run_test();
}
