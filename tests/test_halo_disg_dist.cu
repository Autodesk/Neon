// Cross-rank HaloDisg<T, Components> exchange test (NCCL DistTransfer path).
//
// Launched under `mpiexec -n 2 ./test_halo_disg_dist`. Each rank constructs a distributed
// DenseDisg (one slab per rank, halo_z = 1), allocates a HostDevice Field, fills its
// owned regions (down-boundary, interior, up-boundary) with encode(x, y, global_z) and
// its ghost regions (lower-ghost, upper-ghost) with a sentinel, uploads, runs HaloDisg
// on a stream, downloads, and verifies:
//
// - Owned regions: unchanged (still hold encode(x, y, global_z)).
// - Interior ghost regions (rank r > 0 lower-ghost, rank r+1 < world_size upper-ghost):
//   now hold encode(x, y, peer_global_z) — proving the cross-rank Send/Recv pairing,
//   byte counts, and offsets are correct on every rank.
// - Exterior ghost regions (rank 0's lower, rank N-1's upper): untouched sentinel —
//   HaloDisg posts no NCCL ops to a non-existent peer in Open mode.
//
// This is the @c HaloDisg counterpart to @c test_halo_dist . Pre-fix HaloDisg silently
// dropped every cross-rank pair (host_dist_pending_++ but no transfer queued), so this
// test would have caught the limitation immediately.
//
// Skipped (returns 0) when world_size < 2 or when no CUDA devices are visible.

#include <carbon/carbon.h>
#include <carbon/dense_disg.h>
#include <carbon/distributed.h>
#include <carbon/halo_disg.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>

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
    CARBON_LOG_ERROR("test_halo_disg_dist: sync after {} failed: {}", tag, e.what());
    return false;
  }
  return true;
}

// Map (buffer_z, x, y) into the disg single-component buffer offset for a 1-component
// field with halo_z = 1. Layout (Lz = logic_extent().z, pitch = xy_slab_elements):
//   bz == 0       -> lower-ghost,  offset = y * xdim + x
//   bz == 1       -> down-boundary, offset = pitch + y * xdim + x
//   bz in [2, Lz) -> interior,     offset = 2*pitch + (bz - 2)*pitch + y*xdim + x
//   bz == Lz      -> up-boundary,  offset = Lz*pitch + y*xdim + x
//   bz == Lz + 1  -> upper-ghost,  offset = (Lz + 1)*pitch + y*xdim + x
[[nodiscard]] auto
disg_offset_1c(const carbon::DenseDisg::Span& span,
               int64_t bz,
               int64_t x,
               int64_t y) -> int64_t {
  const int64_t lz = span.logic_extent().z;
  const int64_t xdim = span.haloed_extent().x;
  const int64_t pitch = span.xy_slab_elements();
  if (bz <= 1) {
    return bz * pitch + y * xdim + x;
  }
  if (bz >= lz) {
    return bz * pitch + y * xdim + x;
  }
  return 2 * pitch + (bz - 2) * pitch + y * xdim + x;
}

}  // namespace

int
main() {
  carbon::init();

  auto distributed = std::make_shared<const carbon::Distributed>();
  if (distributed->world_size() < 2) {
    CARBON_LOG_INFO("test_halo_disg_dist: world_size={} < 2, skipping", distributed->world_size());
    return 0;
  }
  if (carbon::device_count() < 1) {
    CARBON_LOG_INFO("test_halo_disg_dist: no CUDA devices, skipping");
    return 0;
  }

  const int rank = distributed->world_rank();
  const int world_size = distributed->world_size();

  carbon::System sys(distributed);
  if (sys.gpu_group().size() != 1U) {
    CARBON_LOG_ERROR("test_halo_disg_dist: expected gpu_group size 1 (one rank per GPU), got {}",
                     sys.gpu_group().size());
    return 1;
  }

  const int64_t z_per_slab = 4;
  const carbon::int64_3d global{4, 4, z_per_slab * static_cast<int64_t>(world_size)};
  carbon::DenseDisg dense(sys, global, /*halo_z=*/1);
  if (dense.gpu_count() != 1U || dense.global_slab_count() != static_cast<size_t>(world_size) ||
      dense.local_slab_offset() != static_cast<size_t>(rank)) {
    CARBON_LOG_ERROR(
        "test_halo_disg_dist: distributed DenseDisg layout mismatch (gpu_count={}, "
        "global_slab_count={}, local_slab_offset={}, world_size={}, rank={})",
        dense.gpu_count(), dense.global_slab_count(), dense.local_slab_offset(), world_size, rank);
    return 1;
  }

  auto field = dense.newField<int, 1>(carbon::MemorySide::HostDevice);

  // Fill the host mirror: every owned region cell holds encode(x, y, global_z); ghosts
  // hold the sentinel so we can prove the halo overwrote them.
  {
    const auto& span = dense.spans()[0];
    const auto haloed = span.haloed_extent();
    const int64_t halo_z = span.halo_z();
    const int64_t z_origin = span.z_offset();
    const int64_t lim_z = span.logic_extent().z;
    int* h = field.cpu_buffers()[0].data();
    // Zero-fill via sentinel as a baseline; sentinel value is harmless if it leaks.
    const size_t n = field.cpu_buffers()[0].size();
    for (size_t i = 0; i < n; ++i) {
      h[i] = kSentinel;
    }
    for (int64_t bz = 0; bz < haloed.z; ++bz) {
      const bool is_halo = (bz < halo_z) || (bz >= halo_z + lim_z);
      const int64_t global_z = z_origin + bz;
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = disg_offset_1c(span, bz, x, y);
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
    CARBON_LOG_ERROR("test_halo_disg_dist: upload failed: {}", e.what());
    return 1;
  }
  if (!sync_all(stream, "upload")) {
    return 1;
  }

  // Build the halo. One-rank-per-GPU: no intra-process pairs; all cross-rank.
  carbon::HaloDisg<int, 1> halo(field);
  if (halo.pair_count() != 0U) {
    CARBON_LOG_ERROR(
        "test_halo_disg_dist: pair_count {} != 0 (expected zero intra entries in MPI mode)",
        halo.pair_count());
    return 1;
  }
  // Each peer contributes 2 NCCL entries per component (one Send + one Recv), with
  // Components = 1 and Standard semantic (1 run-up + 1 run-down).
  const size_t expected_peers = ((rank > 0) ? 1U : 0U) + ((rank + 1 < world_size) ? 1U : 0U);
  const size_t expected_dist = 2U * expected_peers;
  if (halo.dist_entry_count() != expected_dist) {
    CARBON_LOG_ERROR("test_halo_disg_dist: dist_entry_count {} != expected {} (rank {})",
                     halo.dist_entry_count(), expected_dist, rank);
    return 1;
  }

  try {
    halo(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_halo_disg_dist: halo() failed: {}", e.what());
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
    for (size_t i = 0; i < n; ++i) {
      h[i] = kDownloadSentinel;
    }
  }
  try {
    carbon::Transfer<int> download(sys, field.gpu_buffers(), field.cpu_buffers());
    download(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_halo_disg_dist: download failed: {}", e.what());
    return 1;
  }
  if (!sync_all(stream, "download")) {
    return 1;
  }

  // Verify every cell.
  size_t mismatches = 0;
  {
    const auto& span = dense.spans()[0];
    const auto haloed = span.haloed_extent();
    const int64_t halo_z = span.halo_z();
    const int64_t z_origin = span.z_offset();
    const int64_t lim_z = span.logic_extent().z;
    const int* h = field.cpu_buffers()[0].data();
    for (int64_t bz = 0; bz < haloed.z; ++bz) {
      const bool is_lower_halo = (bz < halo_z);
      const bool is_upper_halo = (bz >= halo_z + lim_z);
      const bool is_exterior_lower = is_lower_halo && (rank == 0);
      const bool is_exterior_upper = is_upper_halo && (rank + 1 == world_size);
      const int64_t global_z = z_origin + bz;
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = disg_offset_1c(span, bz, x, y);
          const int got = h[off];
          const int want =
              (is_exterior_lower || is_exterior_upper) ? kSentinel : encode(x, y, global_z);
          if (got != want) {
            if (mismatches < 4U) {
              CARBON_LOG_ERROR(
                  "test_halo_disg_dist: rank {} buffer (x,y,bz)=({},{},{}) global_z={} got={} "
                  "want={}",
                  rank, x, y, bz, global_z, got, want);
            }
            ++mismatches;
          }
        }
      }
    }
  }

  if (mismatches > 0U) {
    CARBON_LOG_ERROR("test_halo_disg_dist: rank {} {} mismatches", rank, mismatches);
    return 1;
  }
  CARBON_LOG_INFO("test_halo_disg_dist: rank {} of {} ok ({} dist entries)", rank, world_size,
                  halo.dist_entry_count());
  return 0;
}
