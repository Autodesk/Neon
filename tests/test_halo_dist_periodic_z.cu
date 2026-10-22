// Cross-rank Halo<T, Components> exchange with PeriodicZ (NCCL DistTransfer path).
//
// Same encoder + verify shape as test_halo_dist, but the Dense is built with
// BoundaryMode::PeriodicZ. Compared to Open, PeriodicZ adds the wrap pair (rank N-1 <-> rank 0)
// so every halo cell on every rank carries a real wrap source:
//
// - Rank 0's lower halo represents global z = global.z - halo_z .. global.z - 1 (wrapped from
//   rank N-1's last halo_z owned cells).
// - Rank N-1's upper halo represents global z = 0 .. halo_z - 1 (wrapped from rank 0's first
//   halo_z owned cells).
// - Interior halos (every other rank) carry their straight-adjacent neighbour as in Open.
//
// dist_entry_count is therefore uniform across ranks: every rank has exactly two peer
// directions (up and down), so 2 * Components * 2 NCCL group entries per rank.
//
// Skipped (returns 0) when world_size < 2 or when no CUDA devices are visible.

#include <carbon/carbon.h>
#include <carbon/dist_transfer.h>
#include <carbon/distributed.h>
#include <carbon/halo.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <vector>

namespace {

constexpr int kSentinel = -7777;

[[nodiscard]] auto
encode(int64_t x,
       int64_t y,
       int64_t global_z) noexcept -> int {
  return static_cast<int>(global_z * 10000 + y * 100 + x);
}

[[nodiscard]] auto
sync_all(const carbon::Stream& stream,
         const char* tag) -> bool {
  try {
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_halo_dist_periodic_z: sync after {} failed: {}", tag, e.what());
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
    CARBON_LOG_INFO("test_halo_dist_periodic_z: world_size={} < 2, skipping",
                    distributed->world_size());
    return 0;
  }
  if (carbon::device_count() < 1) {
    CARBON_LOG_INFO("test_halo_dist_periodic_z: no CUDA devices, skipping");
    return 0;
  }

  const int rank = distributed->world_rank();
  const int world_size = distributed->world_size();

  carbon::System sys(distributed);
  if (sys.gpu_group().size() != 1U) {
    CARBON_LOG_ERROR(
        "test_halo_dist_periodic_z: expected gpu_group size 1 (one rank per GPU), got {}",
        sys.gpu_group().size());
    return 1;
  }

  carbon::Stencil stencil(std::vector<carbon::int32_3d>{{0, 0, -1}, {0, 0, 1}},
                          /*remove_center=*/true);

  const int64_t z_per_slab = 4;
  const carbon::int64_3d global{4, 4, z_per_slab * static_cast<int64_t>(world_size)};
  carbon::Dense dense(sys, global, std::move(stencil), carbon::BoundaryMode::PeriodicZ);

  if (dense.boundary_mode() != carbon::BoundaryMode::PeriodicZ) {
    CARBON_LOG_ERROR("test_halo_dist_periodic_z: expected PeriodicZ boundary mode");
    return 1;
  }
  if (dense.gpu_count() != 1U || dense.global_slab_count() != static_cast<size_t>(world_size) ||
      dense.local_slab_offset() != static_cast<size_t>(rank)) {
    CARBON_LOG_ERROR(
        "test_halo_dist_periodic_z: layout mismatch (gpu_count={}, global_slab_count={}, "
        "local_slab_offset={}, world_size={}, rank={})",
        dense.gpu_count(), dense.global_slab_count(), dense.local_slab_offset(), world_size, rank);
    return 1;
  }

  auto field = dense.newField<int, 1>(carbon::MemorySide::HostDevice);

  // Fill: owned -> encode(x, y, global_z), halos -> sentinel.
  {
    const auto& span = dense.spans()[0];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t halo_z = span.halo_z();
    const int64_t z_origin = span.z_offset();
    const int64_t lim_z = span.logic_extent().z;
    int* h = field.cpu_buffers()[0].data();
    for (int64_t z = 0; z < haloed.z; ++z) {
      const bool is_halo = (z < halo_z) || (z >= halo_z + lim_z);
      const int64_t global_z = z_origin + z;
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = z * pitch + y * haloed.x + x;
          h[off] = is_halo ? kSentinel : encode(x, y, global_z);
        }
      }
    }
  }

  carbon::Stream stream = sys.stream();

  try {
    carbon::Transfer<int> upload(sys, field.cpu_buffers(), field.gpu_buffers());
    upload(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_halo_dist_periodic_z: upload failed: {}", e.what());
    return 1;
  }
  if (!sync_all(stream, "upload")) {
    return 1;
  }

  carbon::Halo halo(field);

  // PeriodicZ: every rank has two peer directions (up + down, with wrap), so 2 * 2 * Components
  // entries per rank. Components = 1.
  if (halo.pair_count() != 0U) {
    CARBON_LOG_ERROR(
        "test_halo_dist_periodic_z: pair_count {} != 0 (expected zero intra entries in MPI mode)",
        halo.pair_count());
    return 1;
  }
  const size_t expected_dist_entries = 2U * 1U * 2U;  // 2 peers * 1 component * (Send+Recv)
  if (halo.dist_entry_count() != expected_dist_entries) {
    CARBON_LOG_ERROR(
        "test_halo_dist_periodic_z: dist_entry_count {} != expected {} (rank {}, world_size {})",
        halo.dist_entry_count(), expected_dist_entries, rank, world_size);
    return 1;
  }

  try {
    halo(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_halo_dist_periodic_z: halo() failed: {}", e.what());
    return 1;
  }
  if (!sync_all(stream, "halo")) {
    return 1;
  }

  constexpr int kDownloadSentinel = kSentinel - 1;
  {
    int* h = field.cpu_buffers()[0].data();
    const size_t n = field.cpu_buffers()[0].size();
    for (size_t j = 0; j < n; ++j) {
      h[j] = kDownloadSentinel;
    }
  }

  try {
    carbon::Transfer<int> download(sys, field.gpu_buffers(), field.cpu_buffers());
    download(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_halo_dist_periodic_z: download failed: {}", e.what());
    return 1;
  }
  if (!sync_all(stream, "download")) {
    return 1;
  }

  // Verify every cell on this rank's slab. With PeriodicZ every halo cell wraps to a real
  // owned cell on the global grid, so the expected value is encode(x, y, wrapped_global_z).
  const int64_t global_z_extent = global.z;
  size_t mismatches = 0;
  {
    const auto& span = dense.spans()[0];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t z_origin = span.z_offset();
    const int* h = field.cpu_buffers()[0].data();
    for (int64_t z = 0; z < haloed.z; ++z) {
      const int64_t raw_global_z = z_origin + z;
      int64_t wrapped = raw_global_z % global_z_extent;
      if (wrapped < 0) {
        wrapped += global_z_extent;
      }
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = z * pitch + y * haloed.x + x;
          const int got = h[off];
          const int want = encode(x, y, wrapped);
          if (got != want) {
            if (mismatches < 4U) {
              CARBON_LOG_ERROR(
                  "test_halo_dist_periodic_z: rank {} buffer (x,y,z)=({},{},{}) raw_global_z={} "
                  "wrapped={} got={} want={}",
                  rank, x, y, z, raw_global_z, wrapped, got, want);
            }
            ++mismatches;
          }
        }
      }
    }
  }

  if (mismatches > 0U) {
    CARBON_LOG_ERROR("test_halo_dist_periodic_z: rank {} {} mismatches", rank, mismatches);
    return 1;
  }
  CARBON_LOG_INFO("test_halo_dist_periodic_z: rank {} of {} ok ({} dist entries)", rank, world_size,
                  halo.dist_entry_count());
  return 0;
}
