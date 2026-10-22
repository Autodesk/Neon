// Single-node intra-process halo exchange test (no MPI, no DistTransfer).
//
// Builds a multi-GPU Dense over a small global extent with halo_z = 1 (radius-1 stencil),
// allocates a HostDevice Field, fills every owned cell on the host with an encoder of the
// cell's global (x, y, z), uploads, runs Halo on a stream, downloads the whole haloed buffer
// back, and verifies:
//
// - Owned cells: unchanged (still hold encode(x, y, global_z)).
// - Interior halo regions (lower halo on slab i with i > 0; upper halo on slab i with
//   i < N - 1): now hold encode(x, y, global_z), where global_z is the global coordinate the
//   halo cell *represents* (computed from buffer-z + z_offset()). The encoder maps each global
//   cell to a unique value, so a successful halo copy of the right neighbour cells produces
//   exactly the value the destination cell would hold "as if it owned itself". This catches
//   off-by-one slab pairings, swapped up/down directions, and wrong source/destination
//   offsets in a single uniform check.
// - Exterior halo regions (lower halo on slab 0; upper halo on slab N - 1): untouched (still
//   the host-side sentinel) — Open boundary mode: out-of-domain cells are not written.
//
// Skipped (returns 0) when fewer than 2 visible CUDA devices are available — the single-GPU
// case has no inter-slab boundaries and is already covered by Halo's no-op early return.

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
  // Each (x, y, gz) triple maps to a unique int as long as x < 100, y < 100, |gz| < 200000.
  // Both the test extents below and the encoder check stay well within int's range.
  return static_cast<int>(global_z * 10000 + y * 100 + x);
}

[[nodiscard]] auto
sync_all(const carbon::Stream& stream,
         const char* tag) -> bool {
  try {
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_halo_single_node: sync after {} failed: {}", tag, e.what());
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
        "test_halo_single_node: needs >=2 visible CUDA devices for a meaningful exchange; have {} "
        "— skipping",
        carbon::device_count());
    return 0;
  }

  carbon::System sys(carbon::get_all_devices());
  const size_t n_gpus = sys.gpu_group().size();

  // Z-only stencil so halo_z == 1 and the exchange is along the partitioning axis.
  carbon::Stencil stencil(std::vector<carbon::int32_3d>{{0, 0, -1}, {0, 0, 1}},
                          /*remove_center=*/true);
  if (stencil.radius() != 1) {
    CARBON_LOG_ERROR("test_halo_single_node: unexpected stencil radius {}", stencil.radius());
    return 1;
  }

  // Modest size; z divisible by gpu count keeps the per-slab arithmetic compact (carbon would
  // otherwise spread the remainder over the first few GPUs, which the test below tolerates
  // because every check is driven off span.z_offset() / logic_extent().z directly).
  const int64_t z_per_slab = 4;
  const carbon::int64_3d global{4, 4, z_per_slab * static_cast<int64_t>(n_gpus)};
  carbon::Dense dense(sys, global, std::move(stencil));

  if (dense.boundary_mode() != carbon::BoundaryMode::Open) {
    CARBON_LOG_ERROR("test_halo_single_node: expected default Open boundary mode");
    return 1;
  }

  auto field = dense.newField<int, 1>(carbon::MemorySide::HostDevice);

  // Fill the host mirror: owned cells get encode(x, y, global_z); halo cells get sentinel.
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

  // Upload host -> device, then sync so the halo can read deterministic source data.
  try {
    carbon::Transfer<int> upload(sys, field.cpu_buffers(), field.gpu_buffers());
    upload(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_halo_single_node: upload failed: {}", e.what());
    return 1;
  }
  if (!sync_all(stream, "upload")) {
    return 1;
  }

  // Build + run the halo exchange.
  carbon::Halo halo(field);
  CARBON_LOG_INFO("test_halo_single_node: pair_count = {}", halo.pair_count());
  const size_t expected_pairs = 2U * (n_gpus - 1U);  // Components == 1.
  if (halo.pair_count() != expected_pairs) {
    CARBON_LOG_ERROR("test_halo_single_node: pair_count {} != expected {} (= 2 * (n_gpus - 1))",
                     halo.pair_count(), expected_pairs);
    return 1;
  }

  try {
    halo(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_halo_single_node: halo() failed: {}", e.what());
    return 1;
  }
  if (!sync_all(stream, "halo")) {
    return 1;
  }

  // Sentinel-overwrite the host mirror (different value from the halo sentinel) so any cell
  // we forget to download stays catchable; then download the whole device buffer.
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
    CARBON_LOG_ERROR("test_halo_single_node: download failed: {}", e.what());
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
    const int64_t pitch = span.xy_slab_elements();
    const int64_t halo_z = span.halo_z();
    const int64_t z_origin = span.z_offset();
    const int64_t lim_z = span.logic_extent().z;
    const int* h = field.cpu_buffers()[i].data();
    for (int64_t z = 0; z < haloed.z; ++z) {
      const bool is_lower_halo = (z < halo_z);
      const bool is_upper_halo = (z >= halo_z + lim_z);
      const bool is_exterior_lower = is_lower_halo && (i == 0U);
      const bool is_exterior_upper = is_upper_halo && (i + 1U == n_gpus);
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
                  "test_halo_single_node: slab {} buffer (x,y,z)=({},{},{}) global_z={} got={} "
                  "want={}",
                  i, x, y, z, global_z, got, want);
            }
            ++mismatches;
          }
        }
      }
    }
  }

  if (mismatches > 0U) {
    CARBON_LOG_ERROR("test_halo_single_node: {} mismatches", mismatches);
    return 1;
  }
  CARBON_LOG_INFO("test_halo_single_node: ok ({} GPUs, {} pairs)", n_gpus, halo.pair_count());
  return 0;
}
