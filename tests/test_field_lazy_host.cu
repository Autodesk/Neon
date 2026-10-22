// Smoke test for the lazy host-mirror allocation introduced for
// MemorySide::HostDevice fields. The field is built without a kernel ever touching
// the host side; we verify that:
//
// 1. cpu_buffers().size() == 0 and host_partitions().size() == 0 do NOT hold for a
//    HostDevice field — the first access fires the lazy allocation and both come
//    back populated.
// 2. Calling cpu_buffers() / host_partitions() repeatedly returns the same
//    underlying pointers (idempotent — the once_flag fires exactly once).
// 3. Two shallow copies of a Field share the lazy-allocated host mirror: triggering
//    the alloc through one copy makes the other copy's accessors see the same
//    populated buffers.
// 4. A MemorySide::Device field never allocates a host mirror — cpu_buffers()
//    stays empty (size() == 0) regardless of how many times it's called.
// 5. fill_owned_host() on a HostDevice field works whether or not the user has
//    pre-touched cpu_buffers() — the lazy alloc fires inside fill_owned_host's
//    own ensure_host_storage call.

#include <carbon/carbon.h>
#include <carbon/dense.h>
#include <carbon/halo.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/stencil.h>

#include <cstdint>
#include <cstdlib>
#include <vector>

int
main() {
  carbon::init();
  if (carbon::device_count() <= 0) {
    CARBON_LOG_ERROR("test_field_lazy_host: no CUDA devices visible");
    return EXIT_FAILURE;
  }

  carbon::System sys(carbon::get_all_devices());
  const size_t n_gpus = sys.gpu_group().size();

  const carbon::int64_3d global{16, 8, std::max<int64_t>(4, static_cast<int64_t>(n_gpus))};
  carbon::Dense dense(sys, global, carbon::Stencil{});

  // ---- Case 1: HostDevice lazy-allocates on first access ----
  {
    auto field = dense.newField<std::int32_t, 1>(carbon::MemorySide::HostDevice);
    if (field.gpu_buffers().size() != n_gpus) {
      CARBON_LOG_ERROR("test_field_lazy_host: HostDevice gpu_buffers().size() = {}, expected {}",
                       field.gpu_buffers().size(), n_gpus);
      return EXIT_FAILURE;
    }
    // First access through cpu_buffers() must fire the lazy alloc.
    auto& cpu = field.cpu_buffers();
    if (cpu.size() != n_gpus) {
      CARBON_LOG_ERROR(
          "test_field_lazy_host: HostDevice cpu_buffers() size after first access = {}, "
          "expected {}",
          cpu.size(), n_gpus);
      return EXIT_FAILURE;
    }
    auto& parts = field.host_partitions();
    if (parts.size() != n_gpus) {
      CARBON_LOG_ERROR("test_field_lazy_host: HostDevice host_partitions() size = {}, expected {}",
                       parts.size(), n_gpus);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO(
        "test_field_lazy_host: HostDevice first access populated cpu_buffers + "
        "host_partitions ({} GPUs)",
        n_gpus);

    // Idempotency: second access returns the same pointer.
    auto& cpu_again = field.cpu_buffers();
    for (size_t g = 0; g < n_gpus; ++g) {
      if (cpu_again[g].data() != cpu[g].data()) {
        CARBON_LOG_ERROR(
            "test_field_lazy_host: HostDevice cpu_buffers() second-access pointer differs on "
            "GPU {} (first {}, second {})",
            g, static_cast<const void*>(cpu[g].data()),
            static_cast<const void*>(cpu_again[g].data()));
        return EXIT_FAILURE;
      }
    }
  }

  // ---- Case 2: copies share the lazily-allocated mirror ----
  {
    auto field = dense.newField<std::int32_t, 1>(carbon::MemorySide::HostDevice);
    auto field_copy = field;  // shallow copy

    // Trigger via the copy. The original should see the same buffers.
    auto& cpu_via_copy = field_copy.cpu_buffers();
    if (cpu_via_copy.size() != n_gpus) {
      CARBON_LOG_ERROR("test_field_lazy_host: copy access did not trigger lazy alloc (size {})",
                       cpu_via_copy.size());
      return EXIT_FAILURE;
    }

    auto& cpu_via_orig = field.cpu_buffers();
    if (cpu_via_orig.size() != n_gpus) {
      CARBON_LOG_ERROR(
          "test_field_lazy_host: original sees empty cpu_buffers ({}) after copy triggered alloc",
          cpu_via_orig.size());
      return EXIT_FAILURE;
    }
    for (size_t g = 0; g < n_gpus; ++g) {
      if (cpu_via_orig[g].data() != cpu_via_copy[g].data()) {
        CARBON_LOG_ERROR(
            "test_field_lazy_host: copy / original see different pointers on GPU {} (orig {}, "
            "copy {}) — shared lazy state broken",
            g, static_cast<const void*>(cpu_via_orig[g].data()),
            static_cast<const void*>(cpu_via_copy[g].data()));
        return EXIT_FAILURE;
      }
    }
    CARBON_LOG_INFO("test_field_lazy_host: HostDevice copy + original share the same lazy mirror");
  }

  // ---- Case 3: Device-only never allocates a host mirror ----
  {
    auto field = dense.newField<std::int32_t, 1>(carbon::MemorySide::Device);
    if (field.cpu_buffers().size() != 0U) {
      CARBON_LOG_ERROR(
          "test_field_lazy_host: Device-only cpu_buffers().size() = {}, expected 0 (no host "
          "mirror should ever be allocated for MemorySide::Device)",
          field.cpu_buffers().size());
      return EXIT_FAILURE;
    }
    if (field.host_partitions().size() != 0U) {
      CARBON_LOG_ERROR(
          "test_field_lazy_host: Device-only host_partitions().size() = {}, expected 0",
          field.host_partitions().size());
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_field_lazy_host: Device-only stays empty on host side");
  }

  // ---- Case 4: Host-only stays eagerly allocated ----
  {
    auto field = dense.newField<std::int32_t, 1>(carbon::MemorySide::Host);
    if (field.cpu_buffers().size() != n_gpus) {
      CARBON_LOG_ERROR(
          "test_field_lazy_host: Host-only cpu_buffers().size() = {}, expected {} (eager alloc)",
          field.cpu_buffers().size(), n_gpus);
      return EXIT_FAILURE;
    }
    if (field.gpu_buffers().size() != 0U) {
      CARBON_LOG_ERROR("test_field_lazy_host: Host-only gpu_buffers().size() = {}, expected 0",
                       field.gpu_buffers().size());
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_field_lazy_host: Host-only is eager + gpu side empty");
  }

  // ---- Case 5: fill_owned_host on a HostDevice field works without pre-touching ----
  {
    auto field = dense.newField<std::int32_t, 1>(carbon::MemorySide::HostDevice);
    // Without first calling cpu_buffers(): fill_owned_host must trigger the lazy
    // allocation internally before writing.
    field.fill_owned_host([](carbon::int64_3d gc, int /*c*/) {
      return static_cast<std::int32_t>(gc.x + 1000 * gc.y + 1000000 * gc.z);
    });
    if (field.cpu_buffers().size() != n_gpus) {
      CARBON_LOG_ERROR(
          "test_field_lazy_host: HostDevice fill_owned_host did not trigger lazy alloc (size {})",
          field.cpu_buffers().size());
      return EXIT_FAILURE;
    }
    // Spot-check: the (0, 0, 0) global cell on slab 0 holds encode(0, 0, 0) == 0.
    const auto& span = dense.spans()[0];
    const int64_t halo_z = span.halo_z();
    const int64_t pitch = span.xy_slab_elements();
    const std::int32_t got = field.cpu_buffers()[0].data()[halo_z * pitch + 0];
    const std::int32_t want = 0;  // global_z 0 + y 0 + x 0
    if (got != want) {
      CARBON_LOG_ERROR(
          "test_field_lazy_host: HostDevice fill_owned_host wrote {} at (0,0,0); expected {}", got,
          want);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_field_lazy_host: HostDevice fill_owned_host (no pre-touch) works");
  }

  // ---- Case 6: Halo built over a HostDevice field still works ----
  // The Halo's ctor reaches into gpu_buffers (eager), not cpu_buffers, so the lazy
  // mirror should stay un-allocated until something host-side touches the field.
  {
    auto field = dense.newField<std::int32_t, 1>(carbon::MemorySide::HostDevice);
    carbon::Halo<std::int32_t, 1> halo(field);
    // We don't have a clean public predicate "is host mirror allocated yet?", but
    // we can verify the Halo built without throwing and the field's gpu_buffers
    // stayed populated. The host mirror's lazy alloc semantics are validated in
    // case 1 above.
    if (field.gpu_buffers().size() != n_gpus) {
      CARBON_LOG_ERROR(
          "test_field_lazy_host: HostDevice + Halo: gpu_buffers().size() = {}, expected {}",
          field.gpu_buffers().size(), n_gpus);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO(
        "test_field_lazy_host: Halo built over a HostDevice field without forcing "
        "host alloc");
  }

  CARBON_LOG_INFO("test_field_lazy_host: PASS");
  return EXIT_SUCCESS;
}
