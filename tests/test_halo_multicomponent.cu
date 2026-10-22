// Probe: multi-component intra-process halo. Mirrors test_halo_single_node
// but with Components=4 so we exercise the per-component plane offset
// arithmetic. Seed each owned cell with a deterministic value that encodes
// (gpu, component, buf_z), run the halo, then read back from each slab's
// upper/lower halo and verify the per-component plane carries the matching
// neighbour's data.

#include <carbon/carbon.h>
#include <carbon/dense.h>
#include <carbon/halo.h>
#include <carbon/integer3d.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/transfer.h>

#include <cstdint>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kComponents = 4;

[[nodiscard]] auto
encode(int g,
       int c,
       int64_t buf_z) -> int32_t {
  // Pack (gpu, component, buf_z) into one int — small enough to fit and
  // unique per (gpu, component, buf_z) for our test sizes.
  return (g * 100 + c) * 10000 + static_cast<int32_t>(buf_z);
}

}  // namespace

auto
main() -> int {
  carbon::init();
  carbon::System sys(carbon::get_all_devices());
  const size_t n_gpus = sys.gpu_group().size();
  if (n_gpus < 2U) {
    CARBON_LOG_INFO("test_halo_multicomponent: SKIP — needs >= 2 GPUs (have {})", n_gpus);
    return 0;
  }

  const carbon::int64_3d global{8, 8, 16};  // 16 / n_gpus must be >= 2 with halo_z=1.
  const carbon::Stencil stencil({{0, 0, -1}, {0, 0, 1}}, /*remove_center=*/true);
  carbon::Dense dense(sys, global, stencil);
  auto field = dense.newField<int32_t, kComponents>(carbon::MemorySide::HostDevice);

  // Seed every owned cell with the encoded (gpu, component, buf_z) value.
  field.fill_owned_host([&](carbon::int64_3d gc, int c) -> int32_t {
    // Reverse-engineer (gpu, buf_z) from the global z. Per-slab lim_z assumes
    // an even split; the test harness ensures global.z is divisible by n_gpus.
    const int64_t slab = static_cast<int64_t>(gc.z * static_cast<int64_t>(n_gpus) / global.z);
    const int64_t lim_z = global.z / static_cast<int64_t>(n_gpus);
    const int64_t buf_z = (gc.z - slab * lim_z) + 1;  // halo_z=1, owned starts at buf_z=1
    return encode(static_cast<int>(slab), c, buf_z);
  });

  carbon::Stream upload = sys.stream();
  carbon::Transfer<int32_t>(sys, field.cpu_buffers(), field.gpu_buffers())(upload);
  upload.wait();

  carbon::Halo<int32_t, kComponents> halo(field);
  CARBON_LOG_INFO("test_halo_multicomponent: pair_count = {} (expected {} = 2 * (n_gpus-1) * C)",
                  halo.pair_count(), 2U * (n_gpus - 1U) * kComponents);
  if (halo.pair_count() != 2U * (n_gpus - 1U) * kComponents) {
    CARBON_LOG_ERROR("test_halo_multicomponent: pair_count mismatch");
    return EXIT_FAILURE;
  }

  carbon::Stream halo_stream = sys.stream();
  halo(halo_stream);
  halo_stream.wait();

  // Download and inspect each GPU's halo planes.
  carbon::Stream dl = sys.stream();
  carbon::Transfer<int32_t>(sys, field.gpu_buffers(), field.cpu_buffers())(dl);
  dl.wait();

  const auto& spans = dense.spans();
  int errors = 0;
  for (size_t g = 0; g < n_gpus; ++g) {
    const auto& span = spans[g];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t halo_z = span.halo_z();
    const int64_t lim_z = span.logic_extent().z;
    const int64_t per_comp = span.linear_storage_elements();
    const int32_t* h = field.cpu_buffers()[g].data();

    for (int c = 0; c < kComponents; ++c) {
      const int64_t plane = static_cast<int64_t>(c) * per_comp;

      // Lower halo (buf z=0) on GPU g should contain GPU (g-1)'s last owned
      // (its buf_z = halo_z + lim_z - 1 = lim_z when halo_z=1).
      if (g > 0U) {
        const int64_t off = plane + 0 * pitch + 0 * haloed.x + 0;
        const int32_t got = h[off];
        const int32_t want = encode(static_cast<int>(g) - 1, c, halo_z + lim_z - 1);
        if (got != want) {
          CARBON_LOG_ERROR(
              "test_halo_multicomponent: GPU {} c={} lower-halo (buf z=0) got {} want {}", g, c,
              got, want);
          ++errors;
        }
      }

      // Upper halo (buf z = halo_z + lim_z) on GPU g should contain GPU (g+1)'s first owned
      // (its buf_z = halo_z = 1).
      if (g + 1U < n_gpus) {
        const int64_t bz = halo_z + lim_z;
        const int64_t off = plane + bz * pitch + 0 * haloed.x + 0;
        const int32_t got = h[off];
        const int32_t want = encode(static_cast<int>(g) + 1, c, halo_z);
        if (got != want) {
          CARBON_LOG_ERROR(
              "test_halo_multicomponent: GPU {} c={} upper-halo (buf z={}) got {} want {}", g, c,
              bz, got, want);
          ++errors;
        }
      }
    }
  }

  if (errors > 0) {
    CARBON_LOG_ERROR("test_halo_multicomponent: FAIL — {} mismatches", errors);
    return EXIT_FAILURE;
  }
  CARBON_LOG_INFO("test_halo_multicomponent: OK");
  return 0;
}
