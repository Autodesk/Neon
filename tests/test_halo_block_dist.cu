// Cross-rank HaloBlock<T, Components, SBlock> exchange test (NCCL DistTransfer path).
//
// Launched under `mpiexec -n 2 ./test_halo_block_dist`. Each rank constructs a distributed
// Block on a small dense-cube domain (every voxel active so the brick layout is regular and
// the cross-rank ghost reservation kicks in for every (bx, by) at the slab boundary), fills
// the owned bricks of an int field with encode(global_x, global_y, global_z), runs HaloBlock
// on the system stream, and verifies:
//
//   - Owned bricks: unchanged (still hold the encoded global coordinates).
//   - Ghost slots: now hold encode(...) for the corresponding remote-owned brick voxels —
//     proving the cross-rank Send / Recv pairing matches up positionally and the byte counts
//     are correct on every rank.
//
// We also sanity-check Block::halo_dist_*_upward / *_downward sizes against the expected
// counts (each rank participates in at most 2 adjacencies — one above, one below — and every
// (bx, by) at the boundary contributes one Send + one Recv per direction).
//
// Skipped (returns 0) when world_size < 2 or when no CUDA devices are visible.

#include <carbon/block.h>
#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/dist_transfer.h>
#include <carbon/distributed.h>
#include <carbon/halo_block.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <utility>
#include <vector>

namespace {

using SB = carbon::detail::block::DefaultStaticBlock;
using BlockGrid = carbon::Block<SB>;
using FieldT = BlockGrid::Field<int, 1>;

[[nodiscard]] auto
encode(int64_t gx,
       int64_t gy,
       int64_t gz) noexcept -> int {
  return static_cast<int>(gz * 1000000 + gy * 1000 + gx);
}

[[nodiscard]] auto
sync_all(const carbon::Stream& stream,
         const char* tag) -> bool {
  try {
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_halo_block_dist: sync after {} failed: {}", tag, e.what());
    return false;
  }
  return true;
}

}  // namespace

int
main() {
  carbon::init();

  auto distributed = std::make_shared<const carbon::Distributed>();
  if (distributed->world_size() < 2) {
    CARBON_LOG_INFO("test_halo_block_dist: world_size={} < 2, skipping", distributed->world_size());
    return 0;
  }
  if (carbon::device_count() < 1) {
    CARBON_LOG_INFO("test_halo_block_dist: no CUDA devices, skipping");
    return 0;
  }

  const int rank = distributed->world_rank();
  const int world_size = distributed->world_size();

  carbon::System sys(distributed);
  if (sys.gpu_group().size() != 1U) {
    CARBON_LOG_ERROR("test_halo_block_dist: expected gpu_group size 1 (one rank per GPU), got {}",
                     sys.gpu_group().size());
    return 1;
  }

  // Cube domain divisible by both MemZ and world_size so the brick partition is even
  // and every (bx, by) at the slab boundary contributes one ghost on each side.
  const int64_t bricks_per_dim = 4;
  const int64_t domain = bricks_per_dim * static_cast<int64_t>(SB::MemX);
  const carbon::int64_3d global{domain, domain, domain};
  // Every voxel active.
  const auto predicate = [](const carbon::int64_3d&) -> bool {
    return true;
  };

  BlockGrid block(sys, global, predicate, "halo_block_dist");
  if (block.global_slab_count() != static_cast<size_t>(world_size) ||
      block.local_slab_offset() != static_cast<size_t>(rank)) {
    CARBON_LOG_ERROR(
        "test_halo_block_dist: layout mismatch (global_slab_count={}, local_slab_offset={}, "
        "world_size={}, rank={})",
        block.global_slab_count(), block.local_slab_offset(), world_size, rank);
    return 1;
  }
  if (block.gpu_count() != 1U) {
    CARBON_LOG_ERROR("test_halo_block_dist: expected gpu_count 1, got {}", block.gpu_count());
    return 1;
  }

  // Expected counts: every (bx, by) plane contributes one ghost row per active boundary.
  const size_t per_plane =
      static_cast<size_t>(bricks_per_dim) * static_cast<size_t>(bricks_per_dim);
  const size_t exp_lower = (rank > 0) ? per_plane : 0U;
  const size_t exp_upper = (rank + 1 < world_size) ? per_plane : 0U;
  if (block.ghost_lower_count(0) != exp_lower) {
    CARBON_LOG_ERROR("test_halo_block_dist: ghost_lower_count {} != {} (rank {})",
                     block.ghost_lower_count(0), exp_lower, rank);
    return 1;
  }
  if (block.ghost_upper_count(0) != exp_upper) {
    CARBON_LOG_ERROR("test_halo_block_dist: ghost_upper_count {} != {} (rank {})",
                     block.ghost_upper_count(0), exp_upper, rank);
    return 1;
  }
  if (block.halo_dist_send_upward().size() != exp_upper ||
      block.halo_dist_recv_downward().size() != exp_upper ||
      block.halo_dist_send_downward().size() != exp_lower ||
      block.halo_dist_recv_upward().size() != exp_lower) {
    CARBON_LOG_ERROR(
        "test_halo_block_dist: dist halo plan sizes mismatch (rank {}, send_up={}, recv_up={}, "
        "send_down={}, recv_down={}, exp_lower={}, exp_upper={})",
        rank, block.halo_dist_send_upward().size(), block.halo_dist_recv_upward().size(),
        block.halo_dist_send_downward().size(), block.halo_dist_recv_downward().size(), exp_lower,
        exp_upper);
    return 1;
  }

  FieldT u = block.template newField<int, 1>(carbon::MemorySide::Device, "u");
  carbon::Stream stream = sys.stream();

  // Build the host-side initial buffer: owned bricks get encode(global_x, global_y, global_z)
  // per voxel; ghost slots stay at a sentinel that HaloBlock must overwrite.
  constexpr int kGhostSentinel = -1234567;
  const uint32_t n_owned = block.block_count(0);
  const uint32_t n_total = block.storage_block_count(0);
  const size_t total_elements = static_cast<size_t>(n_total) * static_cast<size_t>(SB::VoxelCount);
  auto host_u = carbon::Cpu::buffer<int>(total_elements);
  for (size_t i = 0; i < total_elements; ++i) {
    host_u.data()[i] = kGhostSentinel;
  }
  for (uint32_t b = 0; b < n_owned; ++b) {
    const auto origin = block.host_origins(0)[b];
    for (int z = 0; z < SB::MemZ; ++z) {
      for (int y = 0; y < SB::MemY; ++y) {
        for (int x = 0; x < SB::MemX; ++x) {
          const size_t lin = static_cast<size_t>(b) * static_cast<size_t>(SB::VoxelCount) +
                             static_cast<size_t>(x) +
                             static_cast<size_t>(SB::MemX) * static_cast<size_t>(y) +
                             static_cast<size_t>(SB::MemX) * static_cast<size_t>(SB::MemY) *
                                 static_cast<size_t>(z);
          host_u.data()[lin] =
              encode(static_cast<int64_t>(origin.x + x), static_cast<int64_t>(origin.y + y),
                     static_cast<int64_t>(origin.z + z));
        }
      }
    }
  }
  try {
    sys.async_copy(host_u, u.gpu_buffers()[0], total_elements * sizeof(int), stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_halo_block_dist: upload failed: {}", e.what());
    return 1;
  }
  if (!sync_all(stream, "upload")) {
    return 1;
  }

  // Run the halo exchange across ranks.
  carbon::HaloBlock<int, 1> halo(block, u);
  if (halo.pair_count() != 0U) {
    CARBON_LOG_ERROR(
        "test_halo_block_dist: pair_count {} != 0 (expected zero intra entries in MPI mode)",
        halo.pair_count());
    return 1;
  }
  const size_t expected_dist_entries = 2U * (exp_lower + exp_upper);
  if (halo.dist_entry_count() != expected_dist_entries) {
    CARBON_LOG_ERROR("test_halo_block_dist: dist_entry_count {} != expected {} (rank {})",
                     halo.dist_entry_count(), expected_dist_entries, rank);
    return 1;
  }

  try {
    halo(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_halo_block_dist: halo() failed: {}", e.what());
    return 1;
  }
  if (!sync_all(stream, "halo")) {
    return 1;
  }

  // Download and verify. Owned bricks must be unchanged; ghost slots must hold the encoded
  // global coordinates of the matching remote brick (which we can compute locally because
  // the brick origins are stored alongside each ghost reservation: the lower-ghost on rank
  // r mirrors the upper-edge brick of rank r-1, with origin.z = bz_begin_global[r-1+1]-1
  // worth of bricks below; similarly for the upper-ghost). We use the bitmask-stored origin
  // approximation: ghost_lower at index k corresponds to the kth (bx, by) at bz = bz_begin-1
  // in row-major order, and the encode() function only depends on global coordinates.
  auto host_dl = carbon::Cpu::buffer<int>(total_elements);
  for (size_t i = 0; i < total_elements; ++i) {
    host_dl.data()[i] = kGhostSentinel - 1;
  }
  try {
    sys.async_copy(u.gpu_buffers()[0], host_dl, total_elements * sizeof(int), stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_halo_block_dist: download failed: {}", e.what());
    return 1;
  }
  if (!sync_all(stream, "download")) {
    return 1;
  }

  size_t mismatches = 0;
  // Owned bricks must be unchanged.
  for (uint32_t b = 0; b < n_owned; ++b) {
    const auto origin = block.host_origins(0)[b];
    for (int z = 0; z < SB::MemZ; ++z) {
      for (int y = 0; y < SB::MemY; ++y) {
        for (int x = 0; x < SB::MemX; ++x) {
          const size_t lin = static_cast<size_t>(b) * static_cast<size_t>(SB::VoxelCount) +
                             static_cast<size_t>(x) +
                             static_cast<size_t>(SB::MemX) * static_cast<size_t>(y) +
                             static_cast<size_t>(SB::MemX) * static_cast<size_t>(SB::MemY) *
                                 static_cast<size_t>(z);
          const int got = host_dl.data()[lin];
          const int want =
              encode(static_cast<int64_t>(origin.x + x), static_cast<int64_t>(origin.y + y),
                     static_cast<int64_t>(origin.z + z));
          if (got != want) {
            if (mismatches < 4U) {
              CARBON_LOG_ERROR(
                  "test_halo_block_dist: owned brick {} voxel ({},{},{}) got {} want {} (rank {})",
                  b, x, y, z, got, want, rank);
            }
            ++mismatches;
          }
        }
      }
    }
  }

  // Ghost slots must hold encode(...) of the *remote* owned brick voxels. Reconstruct the
  // expected origin by walking the same row-major (bx, by) sweep at the boundary bz.
  auto verify_ghost_row = [&](int64_t bz_row, uint32_t base_id) -> uint32_t {
    uint32_t id = base_id;
    if (bz_row < 0 || bz_row >= block.candidate_block_extent().z) {
      return id;
    }
    for (int64_t by = 0; by < block.candidate_block_extent().y; ++by) {
      for (int64_t bx = 0; bx < block.candidate_block_extent().x; ++bx) {
        // Predicate is "all active", so every (bx, by) contributes a ghost.
        const int64_t origin_x = bx * SB::MemX;
        const int64_t origin_y = by * SB::MemY;
        const int64_t origin_z = bz_row * SB::MemZ;
        for (int z = 0; z < SB::MemZ; ++z) {
          for (int y = 0; y < SB::MemY; ++y) {
            for (int x = 0; x < SB::MemX; ++x) {
              const size_t lin = static_cast<size_t>(id) * static_cast<size_t>(SB::VoxelCount) +
                                 static_cast<size_t>(x) +
                                 static_cast<size_t>(SB::MemX) * static_cast<size_t>(y) +
                                 static_cast<size_t>(SB::MemX) * static_cast<size_t>(SB::MemY) *
                                     static_cast<size_t>(z);
              const int got = host_dl.data()[lin];
              const int want = encode(origin_x + x, origin_y + y, origin_z + z);
              if (got != want) {
                if (mismatches < 4U) {
                  CARBON_LOG_ERROR(
                      "test_halo_block_dist: ghost slot {} voxel ({},{},{}) global=({},{},{}) "
                      "got {} want {} (rank {})",
                      id, x, y, z, origin_x + x, origin_y + y, origin_z + z, got, want, rank);
                }
                ++mismatches;
              }
            }
          }
        }
        ++id;
      }
    }
    return id;
  };
  uint32_t cursor = n_owned;
  if (rank > 0) {
    cursor = verify_ghost_row(block.bz_begin(0) - 1, cursor);
  }
  if (rank + 1 < world_size) {
    cursor = verify_ghost_row(block.bz_end(0), cursor);
  }
  (void)cursor;

  if (mismatches > 0U) {
    CARBON_LOG_ERROR("test_halo_block_dist: rank {} {} mismatches", rank, mismatches);
    return 1;
  }
  CARBON_LOG_INFO("test_halo_block_dist: rank {} of {} ok ({} dist entries, {} ghosts checked)",
                  rank, world_size, halo.dist_entry_count(), exp_lower + exp_upper);
  return 0;
}
