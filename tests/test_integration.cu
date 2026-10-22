// End-to-end integration: Dense + Field + Transfer + Kernel.
//
// Allocates a multi-GPU Dense grid (no halo), allocates a host+device Field via newField,
// initializes the host mirror with a deterministic per-cell pattern, uploads it to the
// device side via a Transfer, doubles every cell in-place via a Kernel that uses the
// Field's Partition + Span, downloads the result via a second Transfer, and verifies
// every cell equals 2 * pattern. Exercises every public abstraction touched by a typical
// stencil application's per-step data motion.

#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/kernel.h>
#include <carbon/memory_side.h>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <vector>

namespace {

using Field0 = carbon::Dense::Field<float, 1>;

[[nodiscard]] auto
expected_value(int64_t global_x,
               int64_t global_y,
               int64_t global_z) -> float {
  return 1.0F + static_cast<float>(global_x) + 7.0F * static_cast<float>(global_y) +
         53.0F * static_cast<float>(global_z);
}

auto
fill_host_mirror(Field0& field,
                 const carbon::Dense& dense) -> void {
  const size_t n_gpu = dense.gpu_count();
  for (size_t g = 0; g < n_gpu; ++g) {
    const auto& span = dense.spans()[g];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t z_origin = span.z_offset();
    float* h = field.cpu_buffers()[g].data();
    for (int64_t z = 0; z < haloed.z; ++z) {
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = z * pitch + y * haloed.x + x;
          h[off] = expected_value(x, y, z + z_origin);
        }
      }
    }
  }
}

auto
verify_host_mirror(const Field0& field,
                   const carbon::Dense& dense) -> void {
  const size_t n_gpu = dense.gpu_count();
  for (size_t g = 0; g < n_gpu; ++g) {
    const auto& span = dense.spans()[g];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t z_origin = span.z_offset();
    const float* h = field.cpu_buffers()[g].data();
    for (int64_t z = 0; z < haloed.z; ++z) {
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = z * pitch + y * haloed.x + x;
          const float got = h[off];
          const float want = 2.0F * expected_value(x, y, z + z_origin);
          if (got != want) {
            CARBON_LOG_ERROR(
                "test_integration: mismatch at gpu={} (x,y,z)=({},{},{}) got={} want={}", g, x, y,
                z + z_origin, got, want);
            std::abort();
          }
        }
      }
    }
  }
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() <= 0) {
    CARBON_LOG_ERROR("test_integration: no CUDA devices");
    return 1;
  }

  carbon::System sys(carbon::get_all_devices());
  if (sys.gpu_group().empty()) {
    CARBON_LOG_ERROR("test_integration: empty GpuGroup");
    return 1;
  }

  // No-halo grid so xy_slab_elements == haloed.x * haloed.y and z_offset is the global
  // z origin of the slab. nz is divisible by the GPU count for the most common cases.
  const carbon::Dense dense(sys, carbon::int64_3d{4, 4, 8});

  auto field = dense.newField<float, 1>(carbon::MemorySide::HostDevice);
  if (field.gpu_buffers().size() != dense.gpu_count() ||
      field.cpu_buffers().size() != dense.gpu_count()) {
    CARBON_LOG_ERROR("test_integration: HostDevice field missing one of host/device buffers");
    return 1;
  }

  fill_host_mirror(field, dense);

  carbon::Transfer<float> upload(sys, field.cpu_buffers(), field.gpu_buffers());
  carbon::Transfer<float> download(sys, field.gpu_buffers(), field.cpu_buffers());

  carbon::Stream stream = sys.stream();
  try {
    upload(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_integration: upload failed: {}", e.what());
    return 1;
  }

  // Double every owned cell in-place via the field's Partition. The kernel iterates owned
  // cells only (driven by Span::logic_extent inside the Dense launch); halos stay untouched.
  const auto kernel = dense.make_kernel(
      [&field](carbon::KernelLoader& loader) {
        const auto part = loader(field);
        return [part] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
          const float v = part.read(s, idx, 0);
          part.write(s, idx, 0, v * 2.0F);
        };
      },
      0U);

  try {
    kernel.launch(stream);
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_integration: kernel launch / wait failed: {}", e.what());
    return 1;
  }

  // Refill host with an obvious sentinel pattern so a faulty download is caught immediately.
  for (size_t g = 0; g < dense.gpu_count(); ++g) {
    auto& cpu = field.cpu_buffers()[g];
    for (size_t i = 0; i < cpu.size(); ++i) {
      cpu.data()[i] = -1.0F;
    }
  }

  try {
    download(stream);
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_integration: download / wait failed: {}", e.what());
    return 1;
  }

  // The kernel only touches owned cells, but with halo_z == 0 the haloed extent is the
  // logic extent, so every host cell must equal 2 * expected_value.
  verify_host_mirror(field, dense);
  return 0;
}
