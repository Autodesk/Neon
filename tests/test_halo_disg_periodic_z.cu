// Single-process PeriodicZ halo regression test for the disaggregated dense grid.
// Twin of test_halo_periodic_z.cu but on carbon::DenseDisg / carbon::HaloDisg.
//
// With PeriodicZ every halo region wraps to a real owned cell on some slab. We fill
// owned cells on the host with encode(x, y, gz), upload, run HaloDisg on a stream,
// download, and verify every owned cell + halo cell against encode(x, y, gz_wrapped)
// where gz_wrapped = ((z_origin + bz) mod global.z + global.z) mod global.z. The disg
// layout's five-region store is exercised through the disg_addr helper (same one used
// in test_dense_disg_multi_gpu.cu).

#include <carbon/carbon.h>
#include <carbon/dense_disg.h>
#include <carbon/halo_disg.h>
#include <carbon/transfer.h>

#include <cstddef>
#include <cstdint>
#include <exception>

namespace {

constexpr int kSentinel = -7777;
constexpr int kComponents = 1;

[[nodiscard]] auto
encode(int64_t x,
       int64_t y,
       int64_t global_z) noexcept -> int {
  return static_cast<int>(global_z * 10000 + y * 100 + x);
}

[[nodiscard]] auto
disg_addr(const carbon::DenseDisg::Span& span,
          int64_t bz,
          int64_t x,
          int64_t y,
          int c,
          int components) -> int64_t {
  const int64_t lz = span.logic_extent().z;
  const int64_t xdim = span.haloed_extent().x;
  const int64_t xy = span.xy_slab_elements();
  int64_t region_first_z = 0;
  int64_t region_z_dim = 1;
  int64_t offset_local_no_card = x + y * xdim;
  if (bz <= 1 || bz >= lz) {
    region_first_z = bz;
    region_z_dim = 1;
  } else {
    region_first_z = 2;
    region_z_dim = lz - 2;
    offset_local_no_card += (bz - 2) * xy;
  }
  return offset_local_no_card + xy * (region_first_z * static_cast<int64_t>(components) +
                                      static_cast<int64_t>(c) * region_z_dim);
}

[[nodiscard]] auto
sync_all(const carbon::Stream& stream,
         const char* tag) -> bool {
  try {
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_halo_disg_periodic_z: sync after {} failed: {}", tag, e.what());
    return false;
  }
  return true;
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() < 1) {
    CARBON_LOG_WARN("test_halo_disg_periodic_z: no visible CUDA devices; skipping");
    return 0;
  }

  carbon::System sys(carbon::get_all_devices());
  const size_t n_gpus = sys.gpu_group().size();

  // z must be ≥ 2 * halo_z * n_gpus so Lz ≥ 2 on every slab (the Internal / Boundary
  // contract); the self-wrap source ranges (first/last halo_z owned cells) also need
  // to be disjoint from their halo destinations on a single-slab layout, which is
  // already implied by Lz ≥ 2.
  const int64_t z_per_slab = 4;
  const carbon::int64_3d global{3, 3, z_per_slab * static_cast<int64_t>(n_gpus)};
  const carbon::DenseDisg dense(sys, global, /*halo_z=*/1, carbon::BoundaryMode::PeriodicZ);

  if (dense.boundary_mode() != carbon::BoundaryMode::PeriodicZ) {
    CARBON_LOG_ERROR("test_halo_disg_periodic_z: expected PeriodicZ boundary mode");
    return 1;
  }

  auto field = dense.newField<int, kComponents>(carbon::MemorySide::HostDevice);

  // Fill: owned -> encode(x, y, gz); halos -> sentinel.
  for (size_t i = 0; i < n_gpus; ++i) {
    const auto& span = dense.spans()[i];
    const auto haloed = span.haloed_extent();
    const int64_t halo_z = span.halo_z();
    const int64_t z_origin = span.z_offset();
    const int64_t lim_z = span.logic_extent().z;
    int* h = field.cpu_buffers()[i].data();
    for (int64_t bz = 0; bz < haloed.z; ++bz) {
      const bool is_halo = (bz < halo_z) || (bz >= halo_z + lim_z);
      const int64_t gz = z_origin + bz;
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = disg_addr(span, bz, x, y, 0, kComponents);
          h[off] = is_halo ? kSentinel : encode(x, y, gz);
        }
      }
    }
  }

  carbon::Stream stream = sys.stream();
  try {
    carbon::Transfer<int> upload(sys, field.cpu_buffers(), field.gpu_buffers());
    upload(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_halo_disg_periodic_z: upload failed: {}", e.what());
    return 1;
  }
  if (!sync_all(stream, "upload")) {
    return 1;
  }

  carbon::HaloDisg halo(field);
  // PeriodicZ wraps slab N-1 to slab 0, so the adjacency list has one extra pair on
  // top of the (n_gpus - 1) regular ones — total n_gpus pairs, each contributing
  // 2 directions; HaloDisg collapses Components into a single transfer per direction.
  const size_t expected_pairs = 2U * n_gpus;
  CARBON_LOG_INFO("test_halo_disg_periodic_z: pair_count = {} (expect {})", halo.pair_count(),
                  expected_pairs);
  if (halo.pair_count() != expected_pairs) {
    CARBON_LOG_ERROR("test_halo_disg_periodic_z: pair_count {} != expected {}", halo.pair_count(),
                     expected_pairs);
    return 1;
  }

  try {
    halo(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_halo_disg_periodic_z: halo() failed: {}", e.what());
    return 1;
  }
  if (!sync_all(stream, "halo")) {
    return 1;
  }

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
    CARBON_LOG_ERROR("test_halo_disg_periodic_z: download failed: {}", e.what());
    return 1;
  }
  if (!sync_all(stream, "download")) {
    return 1;
  }

  const int64_t global_z_extent = global.z;
  size_t mismatches = 0;
  for (size_t i = 0; i < n_gpus; ++i) {
    const auto& span = dense.spans()[i];
    const auto haloed = span.haloed_extent();
    const int64_t z_origin = span.z_offset();
    const int* h = field.cpu_buffers()[i].data();
    for (int64_t bz = 0; bz < haloed.z; ++bz) {
      const int64_t raw_gz = z_origin + bz;
      int64_t wrapped = raw_gz % global_z_extent;
      if (wrapped < 0) {
        wrapped += global_z_extent;
      }
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = disg_addr(span, bz, x, y, 0, kComponents);
          const int got = h[off];
          const int want = encode(x, y, wrapped);
          if (got != want) {
            if (mismatches < 4U) {
              CARBON_LOG_ERROR(
                  "test_halo_disg_periodic_z: slab {} (bz={}, x={}, y={}) raw_gz={} wrapped={} "
                  "got={} want={}",
                  i, bz, x, y, raw_gz, wrapped, got, want);
            }
            ++mismatches;
          }
        }
      }
    }
  }
  if (mismatches > 0U) {
    CARBON_LOG_ERROR("test_halo_disg_periodic_z: {} mismatches", mismatches);
    return 1;
  }

  CARBON_LOG_INFO("test_halo_disg_periodic_z: ok ({} GPUs, {} pairs)", n_gpus, halo.pair_count());
  return 0;
}
