// carbon::io::write_native / read_native round-trip for the DenseDisg layout
// (single-process).
//
// Mirrors test_native_io.cu but exercises the templated-on-Grid I/O path against the
// disaggregated layout. The on-disk format is identical to Dense's (Cartesian
// global_extent, z-only partitioned), so this test also opens a Dense-written shard with
// matching global_extent and reads it back into a DenseDisg field — the cross-family
// round-trip is the headline guarantee of the P1.epsilon DenseDisg closure.
//
// Steps:
//   1. Build a multi-GPU DenseDisg over a chosen global_extent.
//   2. Allocate two HostDevice fields with different (T, Components) and seed the owned
//      cells via the disg_addr offset formula (cloned from test_dense_disg_host.cu so
//      the layout knowledge stays in one place).
//   3. write_native(path, dense_disg, scalar, vec) and read_native(path, dense_disg,
//      scalar2, vec2) — verify owned cells + halo sentinels.
//   4. Cross-family read: build a Dense over the same global_extent / system, write a
//      shard with it, then read the shard back into a fresh DenseDisg field and verify.
//   5. Negative paths: dtype mismatch and component-count mismatch must throw.

#include <carbon/carbon.h>
#include <carbon/dense.h>
#include <carbon/dense_disg.h>
#include <carbon/io.h>
#include <carbon/memory_side.h>
#include <carbon/stencil.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kSentinel = -7777;

// disg_addr: per-cell offset into a DenseDisg field's host mirror. Cloned from
// test_dense_disg_host.cu — bit-equal with the I/O path's row_offset overload, lifted
// to per-cell granularity for convenient verification.
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
  int64_t off = x + y * xdim;
  if (bz <= 1 || bz >= lz) {
    region_first_z = bz;
    region_z_dim = 1;
  } else {
    region_first_z = 2;
    region_z_dim = lz - 2;
    off += (bz - 2) * xy;
  }
  return off + xy * (region_first_z * static_cast<int64_t>(components) +
                     static_cast<int64_t>(c) * region_z_dim);
}

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
fill_buffer_with_sentinel(carbon::DenseDisg::Field<T,
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

template <typename T,
          int N>
void
fill_buffer_with_sentinel_dense(carbon::Dense::Field<T,
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
    CARBON_LOG_WARN("test_native_io_disg: no CUDA devices, skipping");
    return 0;
  }

  carbon::System sys(carbon::get_all_devices());
  const size_t n_gpus = sys.gpu_group().size();

  const int64_t z_per_slab = 4;
  const carbon::int64_3d global{4, 4, z_per_slab * static_cast<int64_t>(n_gpus)};
  carbon::DenseDisg disg(sys, global, /*halo_z=*/1);

  // Allocate two fields with different (T, Components) so the per-field meta path is
  // exercised across the disg's three-region layout.
  auto scalar = disg.newField<float, 1>(carbon::MemorySide::Host);
  auto vec = disg.newField<std::int32_t, 3>(carbon::MemorySide::Host);

  // Pre-fill everything (owned + halo cards) with the sentinel; then overwrite owned
  // cells via disg_addr. After read_native, halo cards must remain at the sentinel.
  fill_buffer_with_sentinel(scalar, static_cast<float>(kSentinel));
  fill_buffer_with_sentinel(vec, static_cast<std::int32_t>(kSentinel));
  for (size_t g = 0; g < n_gpus; ++g) {
    const auto& span = disg.spans()[g];
    const auto haloed = span.haloed_extent();
    const int64_t halo_z = span.halo_z();
    const int64_t lim_z = span.logic_extent().z;
    const int64_t z_origin = span.z_offset();
    float* hs = scalar.cpu_buffers()[g].data();
    std::int32_t* hv = vec.cpu_buffers()[g].data();
    for (int64_t bz = halo_z; bz < halo_z + lim_z; ++bz) {
      const int64_t global_z = z_origin + bz;
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          hs[disg_addr(span, bz, x, y, 0, /*components=*/1)] = encode_scalar(x, y, global_z);
          for (int c = 0; c < 3; ++c) {
            hv[disg_addr(span, bz, x, y, c, /*components=*/3)] = encode_vec(x, y, global_z, c);
          }
        }
      }
    }
  }

  const std::string path = "carbon_native_io_disg_test.bin";
  std::remove(path.c_str());

  try {
    carbon::io::write_native(path, disg, scalar, vec);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_native_io_disg: write_native failed: {}", e.what());
    return 1;
  }

  // Read into fresh fields. Halo cards remain at the sentinel after read_native.
  auto scalar2 = disg.newField<float, 1>(carbon::MemorySide::Host);
  auto vec2 = disg.newField<std::int32_t, 3>(carbon::MemorySide::Host);
  fill_buffer_with_sentinel(scalar2, static_cast<float>(kSentinel));
  fill_buffer_with_sentinel(vec2, static_cast<std::int32_t>(kSentinel));

  try {
    carbon::io::read_native(path, disg, scalar2, vec2);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_native_io_disg: read_native failed: {}", e.what());
    return 1;
  }

  // Verify scalar field: owned cells match the encoder, halo cards stayed at the
  // sentinel. Halo cards are bz==0 and bz==lz+1 (the disaggregated front/back regions);
  // owned bz range is [halo_z, halo_z + lim_z).
  size_t mismatches = 0;
  for (size_t g = 0; g < n_gpus; ++g) {
    const auto& span = disg.spans()[g];
    const auto haloed = span.haloed_extent();
    const int64_t halo_z = span.halo_z();
    const int64_t z_origin = span.z_offset();
    const int64_t lim_z = span.logic_extent().z;
    const float* h = scalar2.cpu_buffers()[g].data();
    for (int64_t bz = 0; bz < haloed.z; ++bz) {
      const bool is_halo = (bz < halo_z) || (bz >= halo_z + lim_z);
      const int64_t global_z = z_origin + bz;
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = disg_addr(span, bz, x, y, 0, /*components=*/1);
          const float got = h[off];
          const float want =
              is_halo ? static_cast<float>(kSentinel) : encode_scalar(x, y, global_z);
          if (got != want) {
            if (mismatches < 4U) {
              CARBON_LOG_ERROR(
                  "test_native_io_disg: scalar slab {} (bz,y,x)=({},{},{}) gz={} got={} "
                  "want={} is_halo={}",
                  g, bz, y, x, global_z, got, want, is_halo);
            }
            ++mismatches;
          }
        }
      }
    }
  }
  if (mismatches > 0U) {
    CARBON_LOG_ERROR("test_native_io_disg: {} scalar mismatches", mismatches);
    return 1;
  }

  // Verify vec field (Components = 3, SoA per-region per-component).
  for (size_t g = 0; g < n_gpus; ++g) {
    const auto& span = disg.spans()[g];
    const auto haloed = span.haloed_extent();
    const int64_t halo_z = span.halo_z();
    const int64_t z_origin = span.z_offset();
    const int64_t lim_z = span.logic_extent().z;
    const std::int32_t* h = vec2.cpu_buffers()[g].data();
    for (int c = 0; c < 3; ++c) {
      for (int64_t bz = 0; bz < haloed.z; ++bz) {
        const bool is_halo = (bz < halo_z) || (bz >= halo_z + lim_z);
        const int64_t global_z = z_origin + bz;
        for (int64_t y = 0; y < haloed.y; ++y) {
          for (int64_t x = 0; x < haloed.x; ++x) {
            const int64_t off = disg_addr(span, bz, x, y, c, /*components=*/3);
            const std::int32_t got = h[off];
            const std::int32_t want =
                is_halo ? static_cast<std::int32_t>(kSentinel) : encode_vec(x, y, global_z, c);
            if (got != want) {
              if (mismatches < 8U) {
                CARBON_LOG_ERROR(
                    "test_native_io_disg: vec slab {} c={} (bz,y,x)=({},{},{}) gz={} got={} "
                    "want={} is_halo={}",
                    g, c, bz, y, x, global_z, got, want, is_halo);
              }
              ++mismatches;
            }
          }
        }
      }
    }
  }
  if (mismatches > 0U) {
    CARBON_LOG_ERROR("test_native_io_disg: {} total mismatches (after vec check)", mismatches);
    return 1;
  }

  // Cross-family round-trip: write from a Dense over the same global_extent / system,
  // then read into a DenseDisg field. The headline P1.epsilon guarantee — Cartesian
  // families share the same on-disk format.
  {
    carbon::Stencil stencil(std::vector<carbon::int32_3d>{{0, 0, -1}, {0, 0, 1}},
                            /*remove_center=*/true);
    carbon::Dense dense(sys, global, std::move(stencil));
    auto dense_scalar = dense.newField<float, 1>(carbon::MemorySide::Host);
    fill_buffer_with_sentinel_dense(dense_scalar, static_cast<float>(kSentinel));
    dense_scalar.fill_owned_host([](carbon::int64_3d gc, int /*c*/) -> float {
      return encode_scalar(gc.x, gc.y, gc.z);
    });

    const std::string cross_path = "carbon_native_io_cross_test.bin";
    std::remove(cross_path.c_str());

    try {
      carbon::io::write_native(cross_path, dense, dense_scalar);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_native_io_disg: cross-family write_native failed: {}", e.what());
      return 1;
    }

    auto disg_scalar3 = disg.newField<float, 1>(carbon::MemorySide::Host);
    fill_buffer_with_sentinel(disg_scalar3, static_cast<float>(kSentinel));
    try {
      carbon::io::read_native(cross_path, disg, disg_scalar3);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_native_io_disg: cross-family read_native failed: {}", e.what());
      return 1;
    }

    size_t cross_mismatches = 0;
    for (size_t g = 0; g < n_gpus; ++g) {
      const auto& span = disg.spans()[g];
      const auto haloed = span.haloed_extent();
      const int64_t halo_z = span.halo_z();
      const int64_t z_origin = span.z_offset();
      const int64_t lim_z = span.logic_extent().z;
      const float* h = disg_scalar3.cpu_buffers()[g].data();
      for (int64_t bz = halo_z; bz < halo_z + lim_z; ++bz) {
        const int64_t global_z = z_origin + bz;
        for (int64_t y = 0; y < haloed.y; ++y) {
          for (int64_t x = 0; x < haloed.x; ++x) {
            const int64_t off = disg_addr(span, bz, x, y, 0, /*components=*/1);
            const float got = h[off];
            const float want = encode_scalar(x, y, global_z);
            if (got != want) {
              if (cross_mismatches < 4U) {
                CARBON_LOG_ERROR(
                    "test_native_io_disg: cross-family slab {} (bz,y,x)=({},{},{}) gz={} "
                    "got={} want={}",
                    g, bz, y, x, global_z, got, want);
              }
              ++cross_mismatches;
            }
          }
        }
      }
    }
    if (cross_mismatches > 0U) {
      CARBON_LOG_ERROR("test_native_io_disg: cross-family {} mismatches", cross_mismatches);
      return 1;
    }
    std::remove(cross_path.c_str());
  }

  // Negative path: dtype mismatch -- bind a double field to the slot the writer used for
  // float; read_native must throw std::invalid_argument.
  {
    auto wrong_dtype = disg.newField<double, 1>(carbon::MemorySide::Host);
    auto vec3 = disg.newField<std::int32_t, 3>(carbon::MemorySide::Host);
    bool threw = false;
    try {
      carbon::io::read_native(path, disg, wrong_dtype, vec3);
    } catch (const std::invalid_argument&) {
      threw = true;
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_native_io_disg: dtype-mismatch threw unexpected: {}", e.what());
      return 1;
    }
    if (!threw) {
      CARBON_LOG_ERROR("test_native_io_disg: dtype-mismatch read did not throw");
      return 1;
    }
  }

  // Negative path: components mismatch.
  {
    auto scalar3 = disg.newField<float, 1>(carbon::MemorySide::Host);
    auto wrong_components = disg.newField<std::int32_t, 1>(carbon::MemorySide::Host);
    bool threw = false;
    try {
      carbon::io::read_native(path, disg, scalar3, wrong_components);
    } catch (const std::invalid_argument&) {
      threw = true;
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_native_io_disg: components-mismatch threw unexpected: {}", e.what());
      return 1;
    }
    if (!threw) {
      CARBON_LOG_ERROR("test_native_io_disg: components-mismatch read did not throw");
      return 1;
    }
  }

  std::remove(path.c_str());
  CARBON_LOG_INFO("test_native_io_disg: ok ({} GPUs, scalar f32 + vec i32x3 + cross-family)",
                  n_gpus);
  return 0;
}
