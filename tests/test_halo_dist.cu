// Cross-rank Halo<T, Components> exchange test (NCCL DistTransfer path).
//
// Launched under `mpiexec -n 2 ./test_halo_dist`. Each rank constructs a distributed Dense
// (one slab per rank, halo_z = 1), allocates a HostDevice Field, fills its owned cells with
// encode(x, y, global_z) and its halo cells with a sentinel, uploads, runs Halo on a stream,
// downloads, and verifies:
//
// - Owned cells: unchanged (still hold encode(x, y, global_z)).
// - Interior halo regions (rank 0's upper, rank N-1's lower, and both halos on every rank in
//   between): now hold encode(x, y, global_z) for the global cell each halo location
//   represents - proving the cross-rank Send / Recv pairing, the byte counts, and the
//   element offsets are correct on every rank.
// - Exterior halo regions (rank 0's lower halo, rank (N-1)'s upper halo): untouched, still
//   the host-side sentinel - Halo posts no NCCL ops to a non-existent peer in Open mode.
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
  // Same encoder as test_halo_single_node: each (x, y, gz) triple produces a unique int as
  // long as x < 100, y < 100, |gz| stays modest (well within int's range for the extents
  // chosen below).
  return static_cast<int>(global_z * 10000 + y * 100 + x);
}

[[nodiscard]] auto
sync_all(const carbon::Stream& stream,
         const char* tag) -> bool {
  try {
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_halo_dist: sync after {} failed: {}", tag, e.what());
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
    CARBON_LOG_INFO("test_halo_dist: world_size={} < 2, skipping", distributed->world_size());
    return 0;
  }
  if (carbon::device_count() < 1) {
    CARBON_LOG_INFO("test_halo_dist: no CUDA devices, skipping");
    return 0;
  }

  const int rank = distributed->world_rank();
  const int world_size = distributed->world_size();

  carbon::System sys(distributed);
  if (sys.gpu_group().size() != 1U) {
    CARBON_LOG_ERROR("test_halo_dist: expected gpu_group size 1 (one rank per GPU), got {}",
                     sys.gpu_group().size());
    return 1;
  }

  // Z-only stencil so halo_z == 1 and the exchange is along the partitioning axis.
  carbon::Stencil stencil(std::vector<carbon::int32_3d>{{0, 0, -1}, {0, 0, 1}},
                          /*remove_center=*/true);
  if (stencil.radius() != 1) {
    CARBON_LOG_ERROR("test_halo_dist: unexpected stencil radius {}", stencil.radius());
    return 1;
  }

  // Modest x/y; global_z divisible by world_size keeps each rank's slab arithmetic compact.
  const int64_t z_per_slab = 4;
  const carbon::int64_3d global{4, 4, z_per_slab * static_cast<int64_t>(world_size)};
  carbon::Dense dense(sys, global, std::move(stencil));

  if (dense.gpu_count() != 1U || dense.global_slab_count() != static_cast<size_t>(world_size) ||
      dense.local_slab_offset() != static_cast<size_t>(rank)) {
    CARBON_LOG_ERROR(
        "test_halo_dist: distributed Dense layout mismatch (gpu_count={}, global_slab_count={}, "
        "local_slab_offset={}, world_size={}, rank={})",
        dense.gpu_count(), dense.global_slab_count(), dense.local_slab_offset(), world_size, rank);
    return 1;
  }

  auto field = dense.newField<int, 1>(carbon::MemorySide::HostDevice);

  // Fill the host mirror of this rank's single slab: owned cells encode(x, y, global_z); halo
  // cells the sentinel value.
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
    CARBON_LOG_ERROR("test_halo_dist: upload failed: {}", e.what());
    return 1;
  }
  if (!sync_all(stream, "upload")) {
    return 1;
  }

  // Build + run the halo exchange across ranks.
  carbon::Halo halo(field);

  // Carbon's one-rank-per-GPU model: zero intra-process pairs, all entries cross-rank.
  if (halo.pair_count() != 0U) {
    CARBON_LOG_ERROR("test_halo_dist: pair_count {} != 0 (expected zero intra entries in MPI mode)",
                     halo.pair_count());
    return 1;
  }
  // Each peer contributes 2 NCCL group entries per component (Send + Recv); Components == 1.
  const size_t expected_peer_count = ((rank > 0) ? 1U : 0U) + ((rank + 1 < world_size) ? 1U : 0U);
  const size_t expected_dist_entries = 2U * 1U * expected_peer_count;
  if (halo.dist_entry_count() != expected_dist_entries) {
    CARBON_LOG_ERROR("test_halo_dist: dist_entry_count {} != expected {} (rank {}, world_size {})",
                     halo.dist_entry_count(), expected_dist_entries, rank, world_size);
    return 1;
  }

  try {
    halo(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_halo_dist: halo() failed: {}", e.what());
    return 1;
  }
  if (!sync_all(stream, "halo")) {
    return 1;
  }

  // Sentinel-overwrite the host mirror so any cell we forget to download stays catchable.
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
    CARBON_LOG_ERROR("test_halo_dist: download failed: {}", e.what());
    return 1;
  }
  if (!sync_all(stream, "download")) {
    return 1;
  }

  // Verify every cell on this rank's slab.
  size_t mismatches = 0;
  {
    const auto& span = dense.spans()[0];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t halo_z = span.halo_z();
    const int64_t z_origin = span.z_offset();
    const int64_t lim_z = span.logic_extent().z;
    const int* h = field.cpu_buffers()[0].data();
    for (int64_t z = 0; z < haloed.z; ++z) {
      const bool is_lower_halo = (z < halo_z);
      const bool is_upper_halo = (z >= halo_z + lim_z);
      const bool is_exterior_lower = is_lower_halo && (rank == 0);
      const bool is_exterior_upper = is_upper_halo && (rank + 1 == world_size);
      const int64_t global_z = z_origin + z;
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = z * pitch + y * haloed.x + x;
          const int got = h[off];
          const int want =
              (is_exterior_lower || is_exterior_upper) ? kSentinel : encode(x, y, global_z);
          if (got != want) {
            if (mismatches < 4U) {
              CARBON_LOG_ERROR(
                  "test_halo_dist: rank {} buffer (x,y,z)=({},{},{}) global_z={} got={} want={}",
                  rank, x, y, z, global_z, got, want);
            }
            ++mismatches;
          }
        }
      }
    }
  }

  if (mismatches > 0U) {
    CARBON_LOG_ERROR("test_halo_dist: rank {} {} mismatches", rank, mismatches);
    return 1;
  }
  CARBON_LOG_INFO("test_halo_dist: rank {} of {} ok ({} dist entries)", rank, world_size,
                  halo.dist_entry_count());
  return 0;
}
