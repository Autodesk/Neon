// Host-side intra-process halo exchange: the @c std::memcpy twin of test_halo_single_node.
//
// Builds a multi-GPU Dense over a small global extent with halo_z = 1 (radius-1
// stencil), allocates a Host-only Field (no device side at all), fills every owned
// cell with an encoder of the cell's global (x, y, z), runs Halo::launch_on_host on
// the single-process layout, and verifies the same property the device test does:
// every interior halo cell has been overwritten with the matching neighbour-cell
// encode value, and every exterior halo cell (top/bottom Open boundary) is
// untouched. The body of the field never touches the GPU.
//
// Also runs a second pass with PeriodicZ on the SAME slab layout to cover the wrap
// pair. Under PeriodicZ the exterior halos at the global top/bottom hold the
// encoded value of the wrapped global cell — exactly the contract Halo / Span /
// Partition::neighbour share with the device path.
//
// A third pass exercises the host halo through the carbon::Kernel polymorphic
// handle: Field::make_halo_kernel + Kernel::launch_on_host on the heterogeneous
// pipeline. This locks the HaloKernelImpl::launch_on_host override.

#include <carbon/carbon.h>
#include <carbon/halo.h>
#include <carbon/kernel.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <string>
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
fill_owned(carbon::Dense& dense,
           carbon::Dense::Field<int,
                                1>& field) -> void {
  const size_t n_gpus = dense.gpu_count();
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
}

[[nodiscard]] auto
verify_after_halo(const carbon::Dense& dense,
                  const carbon::Dense::Field<int,
                                             1>& field,
                  bool periodic_z,
                  const char* tag) -> bool {
  const size_t n_gpus = dense.gpu_count();
  const int64_t global_z_extent = dense.global_extent().z;
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
      const int64_t local_global_z = z_origin + z;
      // For interior halos the buffer's local global_z == the source cell's global
      // z (the halo is filled with the neighbour's owned data). For PeriodicZ
      // exterior halos the source comes from the wrap, so the global z to encode
      // against is the wrapped value (mod global_z_extent). For Open exterior
      // halos the cell is untouched and must still be the sentinel.
      int64_t source_global_z = local_global_z;
      bool expect_sentinel = false;
      if (is_exterior_lower || is_exterior_upper) {
        if (periodic_z) {
          source_global_z =
              ((local_global_z % global_z_extent) + global_z_extent) % global_z_extent;
        } else {
          expect_sentinel = true;
        }
      }
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = z * pitch + y * haloed.x + x;
          const int got = h[off];
          const int want = expect_sentinel ? kSentinel : encode(x, y, source_global_z);
          if (got != want) {
            if (mismatches < 4U) {
              CARBON_LOG_ERROR(
                  "test_halo_host[{}]: slab {} (x,y,z)=({},{},{}) gz={} got={} want={}", tag, i, x,
                  y, z, local_global_z, got, want);
            }
            ++mismatches;
          }
        }
      }
    }
  }
  if (mismatches > 0U) {
    CARBON_LOG_ERROR("test_halo_host[{}]: {} mismatches", tag, mismatches);
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
        "test_halo_host: needs >=2 visible CUDA devices for a meaningful exchange; have {} — "
        "skipping",
        carbon::device_count());
    return 0;
  }

  carbon::System sys(carbon::get_all_devices());
  const size_t n_gpus = sys.gpu_group().size();

  carbon::Stencil stencil(std::vector<carbon::int32_3d>{{0, 0, -1}, {0, 0, 1}},
                          /*remove_center=*/true);

  const int64_t z_per_slab = 4;
  const carbon::int64_3d global{4, 4, z_per_slab * static_cast<int64_t>(n_gpus)};

  carbon::Stream stream = sys.stream();

  // ---- Pass 1: Open boundary, Host-only field ----
  {
    carbon::Dense dense(sys, global, stencil, carbon::BoundaryMode::Open);
    auto field = dense.newField<int, 1>(carbon::MemorySide::Host);
    fill_owned(dense, field);

    carbon::Halo halo(field);
    if (halo.host_intra_count() != 2U * (n_gpus - 1U)) {
      CARBON_LOG_ERROR("test_halo_host[Open]: host_intra_count {} != expected {}",
                       halo.host_intra_count(), 2U * (n_gpus - 1U));
      return EXIT_FAILURE;
    }

    try {
      halo.launch_on_host(stream);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_halo_host[Open]: launch_on_host failed: {}", e.what());
      return EXIT_FAILURE;
    }

    if (!verify_after_halo(dense, field, /*periodic_z=*/false, "Open")) {
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_halo_host[Open]: ok ({} GPUs, {} host intra entries)", n_gpus,
                    halo.host_intra_count());
  }

  // ---- Pass 2: PeriodicZ boundary on a Host-only field ----
  {
    carbon::Dense dense(sys, global, stencil, carbon::BoundaryMode::PeriodicZ);
    auto field = dense.newField<int, 1>(carbon::MemorySide::Host);
    fill_owned(dense, field);

    carbon::Halo halo(field);
    // PeriodicZ adds the wrap pair (last, first) — one extra adjacency pair → +2
    // per-component memcpy entries. With Components == 1 the host_intra_count grows
    // from 2*(n-1) to 2*n.
    const size_t expected = 2U * n_gpus;
    if (halo.host_intra_count() != expected) {
      CARBON_LOG_ERROR("test_halo_host[PeriodicZ]: host_intra_count {} != expected {}",
                       halo.host_intra_count(), expected);
      return EXIT_FAILURE;
    }

    try {
      halo.launch_on_host(stream);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_halo_host[PeriodicZ]: launch_on_host failed: {}", e.what());
      return EXIT_FAILURE;
    }
    if (!verify_after_halo(dense, field, /*periodic_z=*/true, "PeriodicZ")) {
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_halo_host[PeriodicZ]: ok ({} GPUs, {} host intra entries)", n_gpus,
                    halo.host_intra_count());
  }

  // ---- Pass 3: HostDevice field — lazy host alloc + halo on host ----
  // Builds a HostDevice field, fills the (lazily-allocated) host mirror, runs
  // launch_on_host. Locks the lazy-mirror + host-halo interaction.
  {
    carbon::Dense dense(sys, global, stencil, carbon::BoundaryMode::Open);
    auto field = dense.newField<int, 1>(carbon::MemorySide::HostDevice);
    fill_owned(dense, field);  // triggers ensure_host_storage_

    carbon::Halo halo(field);
    halo.launch_on_host(stream);
    if (!verify_after_halo(dense, field, /*periodic_z=*/false, "HostDevice")) {
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_halo_host[HostDevice]: ok ({} GPUs)", n_gpus);
  }

  // ---- Pass 4: through the polymorphic Kernel handle ----
  // make_halo_kernel + launch_on_host should produce identical results to a direct
  // Halo::launch_on_host. Confirms HaloKernelImpl::runs_on_host + launch_on_host
  // are wired correctly.
  {
    carbon::Dense dense(sys, global, stencil, carbon::BoundaryMode::Open);
    auto field = dense.newField<int, 1>(carbon::MemorySide::Host);
    fill_owned(dense, field);

    carbon::Kernel halo_k = field.make_halo_kernel("halo_host");
    if (!halo_k.runs_on_host()) {
      CARBON_LOG_ERROR("test_halo_host[Kernel]: halo kernel does not advertise runs_on_host()");
      return EXIT_FAILURE;
    }
    try {
      halo_k.launch_on_host(stream);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_halo_host[Kernel]: launch_on_host failed: {}", e.what());
      return EXIT_FAILURE;
    }
    if (!verify_after_halo(dense, field, /*periodic_z=*/false, "Kernel")) {
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_halo_host[Kernel]: ok");
  }

  CARBON_LOG_INFO("test_halo_host: PASS");
  return EXIT_SUCCESS;
}
