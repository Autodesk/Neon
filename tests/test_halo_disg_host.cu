// Host-side intra-process halo exchange on carbon::DenseDisg / carbon::HaloDisg.
//
// Twin of test_halo_host.cu for the disaggregated layout (Components = 1,
// Standard stencil semantic). Builds a multi-GPU DenseDisg with halo_z = 1,
// allocates a Host-only field, fills owned cells with encode(x, y, gz), runs
// HaloDisg::launch_on_host, and verifies interior z halos hold the neighbour
// encode while Open exterior halos stay at the sentinel.
//
// Skipped (returns 0) when fewer than two GPUs are visible.

#include <carbon/carbon.h>
#include <carbon/dense_disg.h>
#include <carbon/halo_disg.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <vector>

namespace {

constexpr int kComponents = 1;
constexpr int kSentinel = -7777;

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
fill_owned(const carbon::DenseDisg& dense,
           carbon::DenseDisg::Field<int,
                                    kComponents>& field) -> void {
  const size_t n_gpus = dense.gpu_count();
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
}

[[nodiscard]] auto
verify_after_halo(const carbon::DenseDisg& dense,
                  const carbon::DenseDisg::Field<int,
                                                 kComponents>& field,
                  const char* tag) -> bool {
  const size_t n_gpus = dense.gpu_count();
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
      const int64_t local_global_z = z_origin + bz;
      const bool expect_sentinel = is_exterior_lower || is_exterior_upper;
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = disg_addr(span, bz, x, y, 0, kComponents);
          const int got = h[off];
          const int want = expect_sentinel ? kSentinel : encode(x, y, local_global_z);
          if (got != want) {
            if (mismatches < 4U) {
              CARBON_LOG_ERROR(
                  "test_halo_disg_host[{}]: slab {} (x,y,bz)=({},{},{}) gz={} got={} want={}", tag,
                  i, x, y, bz, local_global_z, got, want);
            }
            ++mismatches;
          }
        }
      }
    }
  }
  if (mismatches > 0U) {
    CARBON_LOG_ERROR("test_halo_disg_host[{}]: {} mismatches", tag, mismatches);
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
        "test_halo_disg_host: needs >=2 visible devices for a meaningful exchange; have {} — "
        "skipping",
        carbon::device_count());
    return EXIT_SUCCESS;
  }

  carbon::System sys(carbon::get_all_devices());
  const size_t n_gpus = sys.gpu_group().size();

  const int64_t z_per_slab = 4;
  const carbon::int64_3d global{4, 4, z_per_slab * static_cast<int64_t>(n_gpus)};

  carbon::Stream stream = sys.stream();

  // ---- Pass 1: Open boundary, Host-only field ----
  {
    const carbon::DenseDisg dense(sys, global, /*halo_z=*/1, carbon::BoundaryMode::Open);
    auto field = dense.newField<int, kComponents>(carbon::MemorySide::Host);
    fill_owned(dense, field);

    carbon::HaloDisg halo(field);
    if (halo.host_intra_count() != 2U * (n_gpus - 1U)) {
      CARBON_LOG_ERROR("test_halo_disg_host[Open]: host_intra_count {} != expected {}",
                       halo.host_intra_count(), 2U * (n_gpus - 1U));
      return EXIT_FAILURE;
    }

    try {
      halo.launch_on_host(stream);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_halo_disg_host[Open]: launch_on_host failed: {}", e.what());
      return EXIT_FAILURE;
    }

    if (!verify_after_halo(dense, field, "Open")) {
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_halo_disg_host[Open]: ok ({} GPUs, {} host intra entries)", n_gpus,
                    halo.host_intra_count());
  }

  // ---- Pass 2: HostDevice field — lazy host alloc + halo on host ----
  {
    const carbon::DenseDisg dense(sys, global, /*halo_z=*/1, carbon::BoundaryMode::Open);
    auto field = dense.newField<int, kComponents>(carbon::MemorySide::HostDevice);
    fill_owned(dense, field);

    carbon::HaloDisg halo(field);
    halo.launch_on_host(stream);
    if (!verify_after_halo(dense, field, "HostDevice")) {
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_halo_disg_host[HostDevice]: ok ({} GPUs)", n_gpus);
  }

  CARBON_LOG_INFO("test_halo_disg_host: PASS");
  return EXIT_SUCCESS;
}
