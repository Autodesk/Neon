// Cross-rank HaloBlock host halo exchange test (MPI_Sendrecv / run_host path).
//
// Launched under `mpiexec -n 2 ./test_halo_block_host_dist`. Host sibling of
// test_halo_block_dist: same domain, predicate, and ghost verification, but the
// field is built with MemorySide::HostDevice, owned bricks are seeded directly in
// cpu_buffers(), and HaloBlock::launch_on_host replays the cross-rank exchange via
// dist_transfer_detail::run_host instead of NCCL on the device path.
//
// Skipped (returns 0) when world_size < 2 or when no CUDA devices are visible.

#include <carbon/block.h>
#include <carbon/carbon.h>
#include <carbon/cpu.h>
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

}  // namespace

int
main() {
  carbon::init();

  auto distributed = std::make_shared<const carbon::Distributed>();
  if (distributed->world_size() < 2) {
    CARBON_LOG_INFO("test_halo_block_host_dist: world_size={} < 2, skipping",
                    distributed->world_size());
    return 0;
  }
  if (carbon::device_count() < 1) {
    CARBON_LOG_INFO("test_halo_block_host_dist: no CUDA devices, skipping");
    return 0;
  }

  const int rank = distributed->world_rank();
  const int world_size = distributed->world_size();

  carbon::System sys(distributed);
  if (sys.gpu_group().size() != 1U) {
    CARBON_LOG_ERROR(
        "test_halo_block_host_dist: expected gpu_group size 1 (one rank per GPU), got {}",
        sys.gpu_group().size());
    return 1;
  }

  const int64_t bricks_per_dim = 4;
  const int64_t domain = bricks_per_dim * static_cast<int64_t>(SB::MemX);
  const carbon::int64_3d global{domain, domain, domain};
  const auto predicate = [](const carbon::int64_3d&) -> bool {
    return true;
  };

  BlockGrid block(sys, global, predicate, "halo_block_host_dist");
  if (block.global_slab_count() != static_cast<size_t>(world_size) ||
      block.local_slab_offset() != static_cast<size_t>(rank)) {
    CARBON_LOG_ERROR(
        "test_halo_block_host_dist: layout mismatch (global_slab_count={}, "
        "local_slab_offset={}, world_size={}, rank={})",
        block.global_slab_count(), block.local_slab_offset(), world_size, rank);
    return 1;
  }

  const size_t per_plane =
      static_cast<size_t>(bricks_per_dim) * static_cast<size_t>(bricks_per_dim);
  const size_t exp_lower = (rank > 0) ? per_plane : 0U;
  const size_t exp_upper = (rank + 1 < world_size) ? per_plane : 0U;

  FieldT u = block.template newField<int, 1>(carbon::MemorySide::HostDevice, "u");
  carbon::Stream stream = sys.stream();

  constexpr int kGhostSentinel = -1234567;
  const uint32_t n_owned = block.block_count(0);
  const uint32_t n_total = block.storage_block_count(0);
  const size_t total_elements = static_cast<size_t>(n_total) * static_cast<size_t>(SB::VoxelCount);

  auto& host_u = u.cpu_buffers()[0];
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

  carbon::HaloBlock<int, 1> halo(block, u);
  const size_t expected_dist_entries = 2U * (exp_lower + exp_upper);
  if (halo.dist_entry_count() != expected_dist_entries) {
    CARBON_LOG_ERROR("test_halo_block_host_dist: dist_entry_count {} != expected {} (rank {})",
                     halo.dist_entry_count(), expected_dist_entries, rank);
    return 1;
  }

  try {
    halo.launch_on_host(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_halo_block_host_dist: launch_on_host failed: {}", e.what());
    return 1;
  }

  size_t mismatches = 0;
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
          const int got = host_u.data()[lin];
          const int want =
              encode(static_cast<int64_t>(origin.x + x), static_cast<int64_t>(origin.y + y),
                     static_cast<int64_t>(origin.z + z));
          if (got != want) {
            if (mismatches < 4U) {
              CARBON_LOG_ERROR(
                  "test_halo_block_host_dist: owned brick {} voxel ({},{},{}) got {} want {} "
                  "(rank {})",
                  b, x, y, z, got, want, rank);
            }
            ++mismatches;
          }
        }
      }
    }
  }

  auto verify_ghost_row = [&](int64_t bz_row, uint32_t base_id) -> uint32_t {
    uint32_t id = base_id;
    if (bz_row < 0 || bz_row >= block.candidate_block_extent().z) {
      return id;
    }
    for (int64_t by = 0; by < block.candidate_block_extent().y; ++by) {
      for (int64_t bx = 0; bx < block.candidate_block_extent().x; ++bx) {
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
              const int got = host_u.data()[lin];
              const int want = encode(origin_x + x, origin_y + y, origin_z + z);
              if (got != want) {
                if (mismatches < 4U) {
                  CARBON_LOG_ERROR(
                      "test_halo_block_host_dist: ghost slot {} voxel ({},{},{}) global=({},{},{}) "
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
    CARBON_LOG_ERROR("test_halo_block_host_dist: rank {} {} mismatches", rank, mismatches);
    return 1;
  }
  CARBON_LOG_INFO(
      "test_halo_block_host_dist: rank {} of {} ok ({} dist entries, {} ghosts checked)", rank,
      world_size, halo.dist_entry_count(), exp_lower + exp_upper);
  return 0;
}
