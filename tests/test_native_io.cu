// carbon::io::write_native / read_native round-trip test (single-process).
//
// Builds a multi-GPU Dense, allocates two HostDevice fields with different (T, Components)
// (a scalar float and a 3-component int32 vector), fills them on the host with
// well-defined per-cell encoders, and exercises the full write -> read -> verify cycle:
//
//   1. write_native(path, dense, scalar, vec)              -- writes one shard per rank.
//   2. (clear the host mirrors so a missed read is catchable)
//   3. read_native (path, dense, scalar2, vec2)            -- repopulates the host mirrors.
//   4. Verify owned cells match the original encoder values; verify halo cells were not
//      touched (the I/O pair operates on owned cells only).
//   5. Negative paths: dtype mismatch and component-count mismatch must throw.
//
// Single-process scope (CARBON_ENABLE_MPI is unrelated here): the distributed shard layout
// is exercised by the dist test when carbon is built with MPI; this test focuses on the
// payload bit-for-bit correctness across multiple slabs and multiple component layouts.

#include <carbon/carbon.h>
#include <carbon/dense.h>
#include <carbon/io.h>
#include <carbon/memory_side.h>
#include <carbon/stencil.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
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

template <typename T>
void
fill_halo_sentinel(carbon::Dense::Field<T,
                                        1>& field,
                   T sentinel) {
  for (size_t g = 0; g < field.cpu_buffers().size(); ++g) {
    T* h = field.cpu_buffers()[g].data();
    const size_t n = field.cpu_buffers()[g].size();
    for (size_t j = 0; j < n; ++j) {
      h[j] = sentinel;
    }
  }
}

template <typename T,
          int N>
void
fill_halo_sentinel_n(carbon::Dense::Field<T,
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
  carbon::init();
  if (carbon::device_count() < 1) {
    CARBON_LOG_WARN("test_native_io: no CUDA devices, skipping");
    return 0;
  }

  carbon::System sys(carbon::get_all_devices());
  const size_t n_gpus = sys.gpu_group().size();

  carbon::Stencil stencil(std::vector<carbon::int32_3d>{{0, 0, -1}, {0, 0, 1}},
                          /*remove_center=*/true);
  const int64_t z_per_slab = 4;
  const carbon::int64_3d global{4, 4, z_per_slab * static_cast<int64_t>(n_gpus)};
  carbon::Dense dense(sys, global, std::move(stencil));

  // Scalar f32 field + 3-component i32 field on the same Dense layout.
  auto scalar = dense.newField<float, 1>(carbon::MemorySide::Host);
  auto vec = dense.newField<std::int32_t, 3>(carbon::MemorySide::Host);

  // Fill halos with sentinels first; fill_owned_host then writes only owned cells, and we
  // can verify after the round-trip that no halo cell was touched by I/O.
  fill_halo_sentinel(scalar, static_cast<float>(kSentinel));
  fill_halo_sentinel_n(vec, static_cast<std::int32_t>(kSentinel));
  scalar.fill_owned_host([](carbon::int64_3d gc, int /*c*/) -> float {
    return encode_scalar(gc.x, gc.y, gc.z);
  });
  vec.fill_owned_host([](carbon::int64_3d gc, int c) -> std::int32_t {
    return encode_vec(gc.x, gc.y, gc.z, c);
  });

  const std::string path = "carbon_native_io_test.bin";
  std::remove(path.c_str());

  try {
    carbon::io::write_native(path, dense, scalar, vec);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_native_io: write_native failed: {}", e.what());
    return 1;
  }

  // Read back into fresh fields. Pre-fill them entirely with a different sentinel so any
  // missed write is detectable, and so the halo regions remain at that sentinel after read
  // (read_native must not touch halos either).
  auto scalar2 = dense.newField<float, 1>(carbon::MemorySide::Host);
  auto vec2 = dense.newField<std::int32_t, 3>(carbon::MemorySide::Host);
  fill_halo_sentinel(scalar2, static_cast<float>(kSentinel));
  fill_halo_sentinel_n(vec2, static_cast<std::int32_t>(kSentinel));

  try {
    carbon::io::read_native(path, dense, scalar2, vec2);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_native_io: read_native failed: {}", e.what());
    return 1;
  }

  // Verify scalar field.
  size_t mismatches = 0;
  for (size_t g = 0; g < n_gpus; ++g) {
    const auto& span = dense.spans()[g];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t halo_z = span.halo_z();
    const int64_t z_origin = span.z_offset();
    const int64_t lim_z = span.logic_extent().z;
    const float* h = scalar2.cpu_buffers()[g].data();
    for (int64_t z = 0; z < haloed.z; ++z) {
      const bool is_halo = (z < halo_z) || (z >= halo_z + lim_z);
      const int64_t global_z = z_origin + z;
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = z * pitch + y * haloed.x + x;
          const float got = h[off];
          const float want =
              is_halo ? static_cast<float>(kSentinel) : encode_scalar(x, y, global_z);
          if (got != want) {
            if (mismatches < 4U) {
              CARBON_LOG_ERROR(
                  "test_native_io: scalar mismatch slab {} (x,y,z)=({},{},{}) gz={} got={} "
                  "want={} is_halo={}",
                  g, x, y, z, global_z, got, want, is_halo);
            }
            ++mismatches;
          }
        }
      }
    }
  }
  if (mismatches > 0U) {
    CARBON_LOG_ERROR("test_native_io: {} scalar mismatches", mismatches);
    return 1;
  }

  // Verify vec field (Components = 3, SoA per-component).
  for (size_t g = 0; g < n_gpus; ++g) {
    const auto& span = dense.spans()[g];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t halo_z = span.halo_z();
    const int64_t z_origin = span.z_offset();
    const int64_t lim_z = span.logic_extent().z;
    const int64_t per_comp = span.linear_storage_elements();
    const std::int32_t* h = vec2.cpu_buffers()[g].data();
    for (int c = 0; c < 3; ++c) {
      const int64_t plane = static_cast<int64_t>(c) * per_comp;
      for (int64_t z = 0; z < haloed.z; ++z) {
        const bool is_halo = (z < halo_z) || (z >= halo_z + lim_z);
        const int64_t global_z = z_origin + z;
        for (int64_t y = 0; y < haloed.y; ++y) {
          for (int64_t x = 0; x < haloed.x; ++x) {
            const int64_t off = plane + z * pitch + y * haloed.x + x;
            const std::int32_t got = h[off];
            const std::int32_t want =
                is_halo ? static_cast<std::int32_t>(kSentinel) : encode_vec(x, y, global_z, c);
            if (got != want) {
              if (mismatches < 8U) {
                CARBON_LOG_ERROR(
                    "test_native_io: vec mismatch slab {} c={} (x,y,z)=({},{},{}) gz={} got={} "
                    "want={} is_halo={}",
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
    CARBON_LOG_ERROR("test_native_io: {} total mismatches (after vec check)", mismatches);
    return 1;
  }

  // Negative path: dtype mismatch -- bind a double field to the slot the writer used for
  // float; read_native must throw std::invalid_argument.
  {
    auto wrong_dtype = dense.newField<double, 1>(carbon::MemorySide::Host);
    auto vec3 = dense.newField<std::int32_t, 3>(carbon::MemorySide::Host);
    bool threw = false;
    try {
      carbon::io::read_native(path, dense, wrong_dtype, vec3);
    } catch (const std::invalid_argument&) {
      threw = true;
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_native_io: dtype-mismatch read threw unexpected exception: {}",
                       e.what());
      return 1;
    }
    if (!threw) {
      CARBON_LOG_ERROR("test_native_io: dtype-mismatch read did not throw");
      return 1;
    }
  }

  // Negative path: components mismatch -- bind a 1-component int field where the writer
  // had 3 components.
  {
    auto scalar3 = dense.newField<float, 1>(carbon::MemorySide::Host);
    auto wrong_components = dense.newField<std::int32_t, 1>(carbon::MemorySide::Host);
    bool threw = false;
    try {
      carbon::io::read_native(path, dense, scalar3, wrong_components);
    } catch (const std::invalid_argument&) {
      threw = true;
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_native_io: components-mismatch read threw unexpected exception: {}",
                       e.what());
      return 1;
    }
    if (!threw) {
      CARBON_LOG_ERROR("test_native_io: components-mismatch read did not throw");
      return 1;
    }
  }

  // Negative path: a truncated shard. read_native has always documented a throw when a payload
  // read short-circuits, but until the check was added failbit simply latched on the first short
  // row and every remaining read became a no-op that left its destination untouched — so a
  // half-written checkpoint loaded as a silent mix of file data and stale host-mirror contents
  // and the restart continued on it. Four bytes off the end keeps the header and both field
  // metas intact, isolating the payload check.
  {
    const std::string truncated = path + ".truncated";
    std::filesystem::copy_file(path, truncated, std::filesystem::copy_options::overwrite_existing);
    const auto full_size = std::filesystem::file_size(truncated);
    std::filesystem::resize_file(truncated, full_size - 4U);

    auto scalar4 = dense.newField<float, 1>(carbon::MemorySide::Host);
    auto vec4 = dense.newField<std::int32_t, 3>(carbon::MemorySide::Host);
    bool threw = false;
    try {
      carbon::io::read_native(truncated, dense, scalar4, vec4);
    } catch (const std::runtime_error&) {
      threw = true;
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_native_io: truncated read threw unexpected exception: {}", e.what());
      std::filesystem::remove(truncated);
      return 1;
    }
    std::filesystem::remove(truncated);
    if (!threw) {
      CARBON_LOG_ERROR("test_native_io: truncated read did not throw (short read went unnoticed)");
      return 1;
    }
  }

  // Negative path: trailing bytes. The writer emits nothing past the last payload, so a longer
  // file means the field list being read does not describe this file.
  {
    const std::string extended = path + ".extended";
    std::filesystem::copy_file(path, extended, std::filesystem::copy_options::overwrite_existing);
    {
      std::ofstream app(extended, std::ios::binary | std::ios::app);
      const char junk[8] = {};
      app.write(junk, sizeof(junk));
    }

    auto scalar5 = dense.newField<float, 1>(carbon::MemorySide::Host);
    auto vec5 = dense.newField<std::int32_t, 3>(carbon::MemorySide::Host);
    bool threw = false;
    try {
      carbon::io::read_native(extended, dense, scalar5, vec5);
    } catch (const std::runtime_error&) {
      threw = true;
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_native_io: trailing-bytes read threw unexpected exception: {}",
                       e.what());
      std::filesystem::remove(extended);
      return 1;
    }
    std::filesystem::remove(extended);
    if (!threw) {
      CARBON_LOG_ERROR("test_native_io: trailing-bytes read did not throw");
      return 1;
    }
  }

  std::remove(path.c_str());
  CARBON_LOG_INFO("test_native_io: ok ({} GPUs, scalar f32 + vec i32x3 round-trip)", n_gpus);
  return 0;
}
