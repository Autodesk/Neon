// Field::fill_owned_host smoke test.
//
// Allocates a HostDevice scalar field over a multi-GPU Dense (with a small radius-1 stencil
// so each slab carries z halos), calls fill_owned_host with an encoder of (global_x, global_y,
// global_z), uploads, downloads the whole haloed buffer, and verifies:
//
// - Owned cells: hold encode(x, y, global_z).
// - Halo cells: untouched (still the pre-fill sentinel) -- fill_owned_host must not write
//   into the halo region. (After Halo() runs the halos would be filled too, but this test
//   does not invoke Halo so any halo write would be visible.)
//
// Also exercises the throw path: a Device-only field rejects fill_owned_host with
// std::invalid_argument because there is no host mirror to write into.

#include <carbon/carbon.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <stdexcept>
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
    CARBON_LOG_ERROR("test_field_fill_host: sync after {} failed: {}", tag, e.what());
    return false;
  }
  return true;
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() < 1) {
    CARBON_LOG_WARN("test_field_fill_host: no CUDA devices, skipping");
    return 0;
  }

  carbon::System sys(carbon::get_all_devices());
  const size_t n_gpus = sys.gpu_group().size();

  carbon::Stencil stencil(std::vector<carbon::int32_3d>{{0, 0, -1}, {0, 0, 1}},
                          /*remove_center=*/true);
  const int64_t z_per_slab = 4;
  const carbon::int64_3d global{4, 4, z_per_slab * static_cast<int64_t>(n_gpus)};
  carbon::Dense dense(sys, global, std::move(stencil));

  auto field = dense.newField<int, 1>(carbon::MemorySide::HostDevice);

  // Pre-fill the entire host buffer (owned + halos) with a sentinel so we can detect any
  // halo write fill_owned_host accidentally makes.
  for (size_t g = 0; g < n_gpus; ++g) {
    int* h = field.cpu_buffers()[g].data();
    const size_t n = field.cpu_buffers()[g].size();
    for (size_t j = 0; j < n; ++j) {
      h[j] = kSentinel;
    }
  }

  // Fill owned cells with the global encoder.
  field.fill_owned_host([](carbon::int64_3d global_cell, int /*c*/) {
    return encode(global_cell.x, global_cell.y, global_cell.z);
  });

  // Verify on the host directly: owned holds encode, halos still kSentinel.
  size_t mismatches_pre = 0;
  for (size_t g = 0; g < n_gpus; ++g) {
    const auto& span = dense.spans()[g];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t halo_z = span.halo_z();
    const int64_t z_origin = span.z_offset();
    const int64_t lim_z = span.logic_extent().z;
    const int* h = field.cpu_buffers()[g].data();
    for (int64_t z = 0; z < haloed.z; ++z) {
      const bool is_halo = (z < halo_z) || (z >= halo_z + lim_z);
      const int64_t global_z = z_origin + z;
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = z * pitch + y * haloed.x + x;
          const int got = h[off];
          const int want = is_halo ? kSentinel : encode(x, y, global_z);
          if (got != want) {
            if (mismatches_pre < 4U) {
              CARBON_LOG_ERROR(
                  "test_field_fill_host: pre-upload slab {} (x,y,z)=({},{},{}) gz={} got={} "
                  "want={} (is_halo={})",
                  g, x, y, z, global_z, got, want, is_halo);
            }
            ++mismatches_pre;
          }
        }
      }
    }
  }
  if (mismatches_pre > 0U) {
    CARBON_LOG_ERROR("test_field_fill_host: {} pre-upload mismatches", mismatches_pre);
    return 1;
  }

  // Round-trip through the device to confirm the host data we computed is uploadable and
  // downloadable as-is (catches accidental SoA / pitch mismatches in fill_owned_host).
  carbon::Stream stream = sys.stream();
  try {
    carbon::Transfer<int> upload(sys, field.cpu_buffers(), field.gpu_buffers());
    upload(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_field_fill_host: upload failed: {}", e.what());
    return 1;
  }
  if (!sync_all(stream, "upload")) {
    return 1;
  }

  // Wipe the host mirror to a different sentinel and re-download.
  constexpr int kDownloadSentinel = kSentinel - 1;
  for (size_t g = 0; g < n_gpus; ++g) {
    int* h = field.cpu_buffers()[g].data();
    const size_t n = field.cpu_buffers()[g].size();
    for (size_t j = 0; j < n; ++j) {
      h[j] = kDownloadSentinel;
    }
  }
  try {
    carbon::Transfer<int> download(sys, field.gpu_buffers(), field.cpu_buffers());
    download(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_field_fill_host: download failed: {}", e.what());
    return 1;
  }
  if (!sync_all(stream, "download")) {
    return 1;
  }

  size_t mismatches_post = 0;
  for (size_t g = 0; g < n_gpus; ++g) {
    const auto& span = dense.spans()[g];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t halo_z = span.halo_z();
    const int64_t z_origin = span.z_offset();
    const int64_t lim_z = span.logic_extent().z;
    const int* h = field.cpu_buffers()[g].data();
    for (int64_t z = 0; z < haloed.z; ++z) {
      const bool is_halo = (z < halo_z) || (z >= halo_z + lim_z);
      const int64_t global_z = z_origin + z;
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = z * pitch + y * haloed.x + x;
          const int got = h[off];
          const int want = is_halo ? kSentinel : encode(x, y, global_z);
          if (got != want) {
            if (mismatches_post < 4U) {
              CARBON_LOG_ERROR(
                  "test_field_fill_host: post-roundtrip slab {} (x,y,z)=({},{},{}) gz={} got={} "
                  "want={} (is_halo={})",
                  g, x, y, z, global_z, got, want, is_halo);
            }
            ++mismatches_post;
          }
        }
      }
    }
  }
  if (mismatches_post > 0U) {
    CARBON_LOG_ERROR("test_field_fill_host: {} post-roundtrip mismatches", mismatches_post);
    return 1;
  }

  // Negative path: device-only field has no host mirror; fill_owned_host must throw.
  auto device_only = dense.newField<int, 1>(carbon::MemorySide::Device);
  bool threw = false;
  try {
    device_only.fill_owned_host([](carbon::int64_3d, int) {
      return 0;
    });
  } catch (const std::invalid_argument&) {
    threw = true;
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_field_fill_host: device-only field threw unexpected exception: {}",
                     e.what());
    return 1;
  }
  if (!threw) {
    CARBON_LOG_ERROR("test_field_fill_host: device-only fill_owned_host did not throw");
    return 1;
  }

  CARBON_LOG_INFO("test_field_fill_host: ok ({} GPUs)", n_gpus);
  return 0;
}
