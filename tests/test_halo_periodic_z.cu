// Single-process PeriodicZ halo exchange test (no MPI).
//
// Same encoder-and-verify shape as test_halo_single_node, but the Dense is constructed with
// BoundaryMode::PeriodicZ. With PeriodicZ the wrap pair (slab N-1 -> slab 0) is added on top
// of the regular adjacent pairs, so every halo region on every slab carries a real source:
//
// - Lower halo on slab i represents global z range
//     [z_offset(i), z_offset(i) + halo_z) when measured against the global domain;
//   PeriodicZ remaps any negative global z by + global.z, so slab 0's lower halo represents
//   global z = global.z - halo_z .. global.z - 1 (= the last halo_z cells of the global
//   domain). Either way the halo cell's "represented" global z is (z_origin + z + global.z)
//   mod global.z, and after a successful wrap-aware exchange the buffer there should hold
//   encode(x, y, that_global_z).
// - Upper halo on slab N-1 wraps to global z = 0 .. halo_z - 1.
// - Single-slab (n_gpus == 1) special case: the slab wraps to itself; its lower and upper
//   halos copy from its own owned region. The test exercises this branch when run on a
//   single visible CUDA device.
//
// All cells (owned + halos) are checked against the same encoder, so off-by-one wrap edges,
// swapped wrap directions, or a missing self-pair would all produce visible mismatches.

#include <carbon/carbon.h>
#include <carbon/halo.h>

#include <cstddef>
#include <cstdint>
#include <exception>
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
    CARBON_LOG_ERROR("test_halo_periodic_z: sync after {} failed: {}", tag, e.what());
    return false;
  }
  return true;
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() < 1) {
    CARBON_LOG_WARN("test_halo_periodic_z: no visible CUDA devices; skipping");
    return 0;
  }

  carbon::System sys(carbon::get_all_devices());
  const size_t n_gpus = sys.gpu_group().size();

  carbon::Stencil stencil(std::vector<carbon::int32_3d>{{0, 0, -1}, {0, 0, 1}},
                          /*remove_center=*/true);

  // z >= 2 * halo_z per slab so the self-wrap source ranges (first/last halo_z owned) don't
  // overlap with the corresponding halo destinations (carbon's user contract).
  const int64_t z_per_slab = 4;
  const carbon::int64_3d global{4, 4, z_per_slab * static_cast<int64_t>(n_gpus)};
  carbon::Dense dense(sys, global, std::move(stencil), carbon::BoundaryMode::PeriodicZ);

  if (dense.boundary_mode() != carbon::BoundaryMode::PeriodicZ) {
    CARBON_LOG_ERROR("test_halo_periodic_z: expected PeriodicZ boundary mode");
    return 1;
  }

  auto field = dense.newField<int, 1>(carbon::MemorySide::HostDevice);

  // Fill: owned -> encode(x, y, global_z), halos -> sentinel.
  for (size_t i = 0; i < n_gpus; ++i) {
    const auto& span = dense.spans()[i];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t halo_z = span.halo_z();
    const int64_t z_origin = span.z_offset();
    const int64_t lim_z = span.logic_extent().z;
    int* h = field.cpu_buffers()[i].data();
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
    CARBON_LOG_ERROR("test_halo_periodic_z: upload failed: {}", e.what());
    return 1;
  }
  if (!sync_all(stream, "upload")) {
    return 1;
  }

  carbon::Halo halo(field);
  // Adjacency pair count: (n_gpus - 1) regular + 1 wrap = n_gpus pairs. Each pair contributes
  // 2 entries per component (up + down). Components = 1 here.
  const size_t expected_pairs = 2U * n_gpus;
  CARBON_LOG_INFO("test_halo_periodic_z: pair_count = {} (expect {})", halo.pair_count(),
                  expected_pairs);
  if (halo.pair_count() != expected_pairs) {
    CARBON_LOG_ERROR(
        "test_halo_periodic_z: pair_count {} != expected {} (= 2 * n_gpus for "
        "PeriodicZ wrap)",
        halo.pair_count(), expected_pairs);
    return 1;
  }

  try {
    halo(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_halo_periodic_z: halo() failed: {}", e.what());
    return 1;
  }
  if (!sync_all(stream, "halo")) {
    return 1;
  }

  // Reset host mirror to a fresh sentinel before downloading.
  constexpr int kDownloadSentinel = kSentinel - 1;
  for (size_t i = 0; i < n_gpus; ++i) {
    int* h = field.cpu_buffers()[i].data();
    const size_t n = field.cpu_buffers()[i].size();
    for (size_t j = 0; j < n; ++j) {
      h[j] = kDownloadSentinel;
    }
  }

  try {
    carbon::Transfer<int> download(sys, field.gpu_buffers(), field.cpu_buffers());
    download(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_halo_periodic_z: download failed: {}", e.what());
    return 1;
  }
  if (!sync_all(stream, "download")) {
    return 1;
  }

  // Verify every cell. With PeriodicZ every halo cell wraps to a real owned cell somewhere,
  // so the expected value is encode(x, y, wrapped_global_z) uniformly.
  const int64_t global_z_extent = global.z;
  size_t mismatches = 0;
  for (size_t i = 0; i < n_gpus; ++i) {
    const auto& span = dense.spans()[i];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t z_origin = span.z_offset();
    const int* h = field.cpu_buffers()[i].data();
    for (int64_t z = 0; z < haloed.z; ++z) {
      const int64_t raw_global_z = z_origin + z;
      // Wrap raw_global_z into [0, global_z_extent) (it can be negative for slab 0's lower
      // halo and >= extent for slab N-1's upper halo).
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
                  "test_halo_periodic_z: slab {} buffer (x,y,z)=({},{},{}) raw_global_z={} "
                  "wrapped={} got={} want={}",
                  i, x, y, z, raw_global_z, wrapped, got, want);
            }
            ++mismatches;
          }
        }
      }
    }
  }

  if (mismatches > 0U) {
    CARBON_LOG_ERROR("test_halo_periodic_z: {} mismatches", mismatches);
    return 1;
  }
  CARBON_LOG_INFO("test_halo_periodic_z: ok ({} GPUs, {} pairs)", n_gpus, halo.pair_count());
  return 0;
}
