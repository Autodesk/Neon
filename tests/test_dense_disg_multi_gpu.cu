// Multi-GPU halo round-trip test for the disaggregated dense grid.
//
// Mirrors test_halo_single_node.cu but on carbon::DenseDisg. Builds a multi-GPU
// DenseDisg over a small global extent with halo_z = 1, allocates a HostDevice
// Field<int, 2> (two cardinality components — the disg layout's per-region SoA
// across components is the central invariant being exercised), fills every owned
// cell on the host with encode(x, y, global_z, c), uploads, runs HaloDisg on a
// stream, downloads the whole haloed buffer, and verifies:
//
// - Owned cells (down boundary, internal, up boundary) — unchanged.
// - Interior halo regions (lower ghost on slab i with i > 0; upper ghost on slab
//   i with i < N - 1) — now hold encode(x, y, global_z, c) where global_z is the
//   global coordinate the halo cell *represents* (computed from the slab's
//   z_offset).
// - Exterior halo regions (lower ghost on slab 0; upper ghost on slab N - 1) —
//   untouched (still kSentinel) under Open boundary mode.
//
// Skipped (returns 0) when fewer than 2 visible CUDA devices are available.

#include <carbon/carbon.h>
#include <carbon/dense_disg.h>
#include <carbon/halo_disg.h>
#include <carbon/transfer.h>

#include <cstddef>
#include <cstdint>
#include <exception>

namespace {

constexpr int kSentinel = -7777;
constexpr int kComponents = 2;

[[nodiscard]] auto
encode(int64_t x,
       int64_t y,
       int64_t global_z,
       int c) noexcept -> int {
  // Each (x, y, gz, c) tuple maps to a unique int as long as |gz| < 100000, x < 100,
  // y < 100, c < 10. The test extents and cardinality stay well within int's range.
  return static_cast<int>(global_z * 1'000'000 + y * 10'000 + x * 100 + c);
}

// Address for cell (bz, x, y, c) in slab `span`'s disaggregated buffer. The disaggregated
// pitch formula is:
//   addr = offset_local_no_card + xy_pitch * (region_first_z * Components + c * region_z_dim)
// with the five-region map driving the (region_first_z, region_z_dim, offset_local_no_card)
// triple. See carbon::detail::dense_disg::Span class docs.
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
    CARBON_LOG_ERROR("test_dense_disg_multi_gpu: sync after {} failed: {}", tag, e.what());
    return false;
  }
  return true;
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() < 2) {
    CARBON_LOG_WARN(
        "test_dense_disg_multi_gpu: needs >=2 visible CUDA devices for a meaningful exchange; "
        "have {} — skipping",
        carbon::device_count());
    return 0;
  }

  carbon::System sys(carbon::get_all_devices());
  const size_t n_gpus = sys.gpu_group().size();

  // Modest global extent. z must be split into at least 2 owned layers per slab so the
  // Internal / Boundary view split is well-defined (Lz >= 2). We pick z = 4 * n_gpus to
  // keep the per-slab arithmetic compact (4 layers each with z divisible by n_gpus).
  const int64_t z_per_slab = 4;
  const carbon::int64_3d global{4, 5, z_per_slab * static_cast<int64_t>(n_gpus)};

  carbon::DenseDisg dense(sys, global, /*halo_z=*/1);

  if (dense.boundary_mode() != carbon::BoundaryMode::Open) {
    CARBON_LOG_ERROR("test_dense_disg_multi_gpu: expected default Open boundary mode");
    return 1;
  }
  if (dense.gpu_count() != n_gpus) {
    CARBON_LOG_ERROR("test_dense_disg_multi_gpu: gpu_count {} != n_gpus {}", dense.gpu_count(),
                     n_gpus);
    return 1;
  }

  auto field = dense.newField<int, kComponents>(carbon::MemorySide::HostDevice);

  // Host fill: sentinel everywhere first, then encode() on every owned cell. This relies on
  // the disg layout's storage being a single contiguous array per slab — the regions are
  // disjoint within that array, so a flat sentinel + per-region overwrite is well-defined.
  for (size_t i = 0; i < n_gpus; ++i) {
    int* h = field.cpu_buffers()[i].data();
    const size_t total = field.cpu_buffers()[i].size();
    for (size_t j = 0; j < total; ++j) {
      h[j] = kSentinel;
    }
  }
  for (size_t i = 0; i < n_gpus; ++i) {
    const auto& span = dense.spans()[i];
    const auto haloed = span.haloed_extent();
    const int64_t halo_z = span.halo_z();
    const int64_t z_origin = span.z_offset();
    const int64_t lim_z = span.logic_extent().z;
    int* h = field.cpu_buffers()[i].data();
    // Owned cells live at buffer-z [halo_z, halo_z + lim_z); their global z is
    // z_origin + bz (Span::z_offset is the global z of buffer-local z=0, the start
    // of the lower ghost). Same convention as the aggregated dense field.
    for (int64_t bz = halo_z; bz < halo_z + lim_z; ++bz) {
      const int64_t gz = z_origin + bz;
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          for (int c = 0; c < kComponents; ++c) {
            const int64_t off = disg_addr(span, bz, x, y, c, kComponents);
            h[off] = encode(x, y, gz, c);
          }
        }
      }
    }
  }

  carbon::Stream stream = sys.stream();

  try {
    carbon::Transfer<int> upload(sys, field.cpu_buffers(), field.gpu_buffers());
    upload(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dense_disg_multi_gpu: upload failed: {}", e.what());
    return 1;
  }
  if (!sync_all(stream, "upload")) {
    return 1;
  }

  // Build + run the disg halo exchange on the device side.
  carbon::HaloDisg halo(field);
  CARBON_LOG_INFO("test_dense_disg_multi_gpu: pair_count = {}", halo.pair_count());

  // The disaggregated layout collapses all components of a region into a single
  // contiguous transfer per direction, so pair_count is independent of Components
  // and equals 2 * (n_gpus - 1) for a non-toroidal n-GPU layout.
  const size_t expected_pairs = 2U * (n_gpus - 1U);
  if (halo.pair_count() != expected_pairs) {
    CARBON_LOG_ERROR("test_dense_disg_multi_gpu: pair_count {} != expected {} (= 2 * (n_gpus - 1))",
                     halo.pair_count(), expected_pairs);
    return 1;
  }

  try {
    halo(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dense_disg_multi_gpu: halo() failed: {}", e.what());
    return 1;
  }
  if (!sync_all(stream, "halo")) {
    return 1;
  }

  // Sentinel-overwrite host then download to confirm we read back what's on device.
  constexpr int kDownloadSentinel = kSentinel - 1;
  for (size_t i = 0; i < n_gpus; ++i) {
    int* h = field.cpu_buffers()[i].data();
    const size_t total = field.cpu_buffers()[i].size();
    for (size_t j = 0; j < total; ++j) {
      h[j] = kDownloadSentinel;
    }
  }

  try {
    carbon::Transfer<int> download(sys, field.gpu_buffers(), field.cpu_buffers());
    download(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dense_disg_multi_gpu: download failed: {}", e.what());
    return 1;
  }
  if (!sync_all(stream, "download")) {
    return 1;
  }

  // Verify every cell on every slab.
  size_t mismatches = 0;
  for (size_t i = 0; i < n_gpus; ++i) {
    const auto& span = dense.spans()[i];
    const auto haloed = span.haloed_extent();
    const int64_t halo_z = span.halo_z();
    const int64_t z_origin = span.z_offset();
    const int64_t lim_z = span.logic_extent().z;
    const int* h = field.cpu_buffers()[i].data();
    for (int64_t bz = 0; bz < haloed.z; ++bz) {
      const bool is_lower_halo = (bz < halo_z);
      const bool is_upper_halo = (bz >= halo_z + lim_z);
      const bool is_exterior_lower = is_lower_halo && (i == 0U);
      const bool is_exterior_upper = is_upper_halo && (i + 1U == n_gpus);
      const int64_t gz = z_origin + bz;
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          for (int c = 0; c < kComponents; ++c) {
            const int64_t off = disg_addr(span, bz, x, y, c, kComponents);
            const int got = h[off];
            const int want =
                (is_exterior_lower || is_exterior_upper) ? kSentinel : encode(x, y, gz, c);
            if (got != want) {
              if (mismatches < 8U) {
                CARBON_LOG_ERROR(
                    "test_dense_disg_multi_gpu: slab {} (bz={}, x={}, y={}, c={}, gz={}) "
                    "got={} want={}",
                    i, bz, x, y, c, gz, got, want);
              }
              ++mismatches;
            }
          }
        }
      }
    }
  }

  if (mismatches > 0U) {
    CARBON_LOG_ERROR("test_dense_disg_multi_gpu: {} mismatches", mismatches);
    return 1;
  }
  CARBON_LOG_INFO("test_dense_disg_multi_gpu: ok ({} GPUs, {} pairs, {} components)", n_gpus,
                  halo.pair_count(), kComponents);
  return 0;
}
