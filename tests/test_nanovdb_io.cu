// carbon::io::write_nanovdb / read_nanovdb round-trip (single-process).
//
// Mirrors test_hdf5_io: writes a scalar f32 and a 3-component i32 vector field to a
// multi-grid .nvdb file (one nanoVDB grid per (field, component) — single-component
// fields drop the _c0 suffix), reads them back into fresh fields, and verifies bit-
// for-bit equality on every owned cell. Halo cells are not touched by I/O so they keep
// the sentinel they were pre-filled with, exercising the same contract as the HDF5 and
// native backends.
//
// Compile-time skipped (returns 0) when CARBON_HAVE_NANOVDB is not defined; runtime
// skipped when no CUDA devices are visible.

#include <carbon/carbon.h>
#include <carbon/dense.h>
#include <carbon/memory_side.h>
#include <carbon/stencil.h>

#ifdef CARBON_HAVE_NANOVDB
#include <carbon/io.h>
#endif

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kSentinel = -7777;

[[nodiscard]] auto
encode_scalar(int64_t x,
              int64_t y,
              int64_t global_z) noexcept -> float {
  return 0.125F * static_cast<float>(global_z * 10000 + y * 100 + x);
}

[[nodiscard]] auto
encode_vec(int64_t x,
           int64_t y,
           int64_t global_z,
           int c) noexcept -> std::int32_t {
  return static_cast<std::int32_t>(global_z * 100000 + y * 1000 + x * 10 + c);
}

template <typename T,
          int N>
void
fill_halo_sentinel(carbon::Dense::Field<T,
                                        N>& field,
                   T sentinel) {
  for (size_t g = 0; g < field.cpu_buffers().size(); ++g) {
    T* h = field.cpu_buffers()[g].data();
    const size_t n = field.cpu_buffers()[g].size();
    for (size_t j = 0; j < n; ++j) {
      h[j] = sentinel;
    }
  }
}

}  // namespace

int
main() {
#ifndef CARBON_HAVE_NANOVDB
  CARBON_LOG_INFO("test_nanovdb_io: built without CARBON_ENABLE_NANOVDB; skipping");
  return 0;
#else
  carbon::init();
  if (carbon::device_count() < 1) {
    CARBON_LOG_WARN("test_nanovdb_io: no CUDA devices, skipping");
    return 0;
  }

  carbon::System sys(carbon::get_all_devices());
  const size_t n_gpus = sys.gpu_group().size();
  carbon::Stencil stencil(std::vector<carbon::int32_3d>{{0, 0, -1}, {0, 0, 1}},
                          /*remove_center=*/true);
  const int64_t z_per_slab = 4;
  const carbon::int64_3d global{4, 4, z_per_slab * static_cast<int64_t>(n_gpus)};
  carbon::Dense dense(sys, global, std::move(stencil));

  auto scalar = dense.newField<float, 1>(carbon::MemorySide::Host);
  auto vec = dense.newField<std::int32_t, 3>(carbon::MemorySide::Host);
  fill_halo_sentinel(scalar, static_cast<float>(kSentinel));
  fill_halo_sentinel(vec, static_cast<std::int32_t>(kSentinel));
  scalar.fill_owned_host([](carbon::int64_3d gc, int) {
    return encode_scalar(gc.x, gc.y, gc.z);
  });
  vec.fill_owned_host([](carbon::int64_3d gc, int c) {
    return encode_vec(gc.x, gc.y, gc.z, c);
  });

  const std::string path = "carbon_nanovdb_io_test.nvdb";
  std::remove(path.c_str());

  try {
    carbon::io::write_nanovdb(path, dense, scalar, vec);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_nanovdb_io: write_nanovdb failed: {}", e.what());
    return 1;
  }

  auto scalar2 = dense.newField<float, 1>(carbon::MemorySide::Host);
  auto vec2 = dense.newField<std::int32_t, 3>(carbon::MemorySide::Host);
  fill_halo_sentinel(scalar2, static_cast<float>(kSentinel));
  fill_halo_sentinel(vec2, static_cast<std::int32_t>(kSentinel));

  try {
    carbon::io::read_nanovdb(path, dense, scalar2, vec2);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_nanovdb_io: read_nanovdb failed: {}", e.what());
    return 1;
  }

  size_t mismatches = 0;
  for (size_t g = 0; g < n_gpus; ++g) {
    const auto& span = dense.spans()[g];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t halo_z = span.halo_z();
    const int64_t z_origin = span.z_offset();
    const int64_t lim_z = span.logic_extent().z;
    const float* hs = scalar2.cpu_buffers()[g].data();
    for (int64_t z = 0; z < haloed.z; ++z) {
      const bool is_halo = (z < halo_z) || (z >= halo_z + lim_z);
      const int64_t global_z = z_origin + z;
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = z * pitch + y * haloed.x + x;
          const float got = hs[off];
          const float want =
              is_halo ? static_cast<float>(kSentinel) : encode_scalar(x, y, global_z);
          if (got != want) {
            if (mismatches < 4U) {
              CARBON_LOG_ERROR(
                  "test_nanovdb_io: scalar slab {} (x,y,z)=({},{},{}) gz={} got={} want={} "
                  "is_halo={}",
                  g, x, y, z, global_z, got, want, is_halo);
            }
            ++mismatches;
          }
        }
      }
    }
  }
  if (mismatches > 0U) {
    CARBON_LOG_ERROR("test_nanovdb_io: {} scalar mismatches", mismatches);
    return 1;
  }

  for (size_t g = 0; g < n_gpus; ++g) {
    const auto& span = dense.spans()[g];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t halo_z = span.halo_z();
    const int64_t z_origin = span.z_offset();
    const int64_t lim_z = span.logic_extent().z;
    const int64_t per_comp = span.linear_storage_elements();
    const std::int32_t* hv = vec2.cpu_buffers()[g].data();
    for (int c = 0; c < 3; ++c) {
      const int64_t plane = static_cast<int64_t>(c) * per_comp;
      for (int64_t z = 0; z < haloed.z; ++z) {
        const bool is_halo = (z < halo_z) || (z >= halo_z + lim_z);
        const int64_t global_z = z_origin + z;
        for (int64_t y = 0; y < haloed.y; ++y) {
          for (int64_t x = 0; x < haloed.x; ++x) {
            const int64_t off = plane + z * pitch + y * haloed.x + x;
            const std::int32_t got = hv[off];
            const std::int32_t want =
                is_halo ? static_cast<std::int32_t>(kSentinel) : encode_vec(x, y, global_z, c);
            if (got != want) {
              if (mismatches < 8U) {
                CARBON_LOG_ERROR(
                    "test_nanovdb_io: vec slab {} c={} (x,y,z)=({},{},{}) gz={} got={} want={} "
                    "is_halo={}",
                    g, c, x, y, z, global_z, got, want, is_halo);
              }
              ++mismatches;
            }
          }
        }
      }
    }
  }
  if (mismatches > 0U) {
    CARBON_LOG_ERROR("test_nanovdb_io: {} total mismatches", mismatches);
    return 1;
  }

  std::remove(path.c_str());
  CARBON_LOG_INFO("test_nanovdb_io: ok ({} GPUs)", n_gpus);
  return 0;
#endif
}
