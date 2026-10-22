// Multi-GPU correctness test for lattice-semantic auto-halo on carbon::Dense via
// KernelLoader::read_stencil(field, stencil, StencilSemantic::Lattice) and the
// Skeleton auto-insertion path (TypedLatticeHaloFactory -> Field::make_halo_kernel).
//
// Mirrors test_lattice_halo_aggregated.cu but drives the exchange through a
// compiled Skeleton instead of a manually constructed carbon::Halo.
//
// Skipped (returns 0) when fewer than 2 visible CUDA devices are available.

#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/skeleton.h>
#include <carbon/stencil.h>
#include <carbon/transfer.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kComponents = 19;
constexpr int32_t kSentinelBase = -777'000;

[[nodiscard]] auto
encode(int g,
       int q,
       int64_t buf_z) noexcept -> int32_t {
  return static_cast<int32_t>(g * 1'000'000 + q * 10'000 + buf_z * 100);
}

[[nodiscard]] auto
sentinel(size_t g,
         int q,
         int64_t bz,
         int64_t y,
         int64_t x) noexcept -> int32_t {
  return kSentinelBase - static_cast<int32_t>(g * 100'000 + q * 1'000 + bz * 100 + y * 10 + x);
}

[[nodiscard]] auto
d3q19_natural() -> std::vector<carbon::int32_3d> {
  return {
      {-1, 0, 0},  {0, -1, 0}, {0, 0, -1}, {-1, -1, 0}, {-1, 1, 0}, {-1, 0, -1}, {-1, 0, 1},
      {0, -1, -1}, {0, -1, 1}, {0, 0, 0},  {1, 0, 0},   {0, 1, 0},  {0, 0, 1},   {1, 1, 0},
      {1, -1, 0},  {1, 0, 1},  {1, 0, -1}, {0, 1, 1},   {0, 1, -1},
  };
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() < 2) {
    CARBON_LOG_WARN(
        "test_lattice_halo_dense_auto: needs >=2 visible CUDA devices for a meaningful exchange; "
        "have {} -- skipping",
        carbon::device_count());
    return 0;
  }

  carbon::System sys(carbon::get_all_devices());
  const size_t n_gpus = sys.gpu_group().size();
  const carbon::int64_3d global{4, 5, 8 * static_cast<int64_t>(n_gpus)};

  const std::vector<carbon::int32_3d> velocities = d3q19_natural();
  carbon::Stencil grid_stencil(velocities);
  carbon::Stencil lattice_stencil(velocities);
  carbon::Dense dense(sys, global, grid_stencil);
  auto field = dense.newField<int32_t, kComponents>(carbon::MemorySide::HostDevice, "pop");

  const auto profile = lattice_stencil.lattice_profile();
  CARBON_LOG_INFO("test_lattice_halo_dense_auto: profile +z={} -z={} 0z={}", profile.positive_z,
                  profile.negative_z, profile.zero_z);

  for (size_t g = 0; g < n_gpus; ++g) {
    const auto& span = dense.spans()[g];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t per_comp = span.linear_storage_elements();
    int32_t* h = field.cpu_buffers()[g].data();
    for (int q = 0; q < kComponents; ++q) {
      for (int64_t bz = 0; bz < haloed.z; ++bz) {
        for (int64_t y = 0; y < haloed.y; ++y) {
          for (int64_t x = 0; x < haloed.x; ++x) {
            const int64_t off = static_cast<int64_t>(q) * per_comp + bz * pitch + y * haloed.x + x;
            h[off] = sentinel(g, q, bz, y, x);
          }
        }
      }
    }
  }
  for (size_t g = 0; g < n_gpus; ++g) {
    const auto& span = dense.spans()[g];
    const auto haloed = span.haloed_extent();
    const int64_t halo_z = span.halo_z();
    const int64_t lim_z = span.logic_extent().z;
    const int64_t pitch = span.xy_slab_elements();
    const int64_t per_comp = span.linear_storage_elements();
    int32_t* h = field.cpu_buffers()[g].data();
    for (int q = 0; q < kComponents; ++q) {
      for (int64_t bz = halo_z; bz < halo_z + lim_z; ++bz) {
        for (int64_t y = 0; y < haloed.y; ++y) {
          for (int64_t x = 0; x < haloed.x; ++x) {
            const int64_t off = static_cast<int64_t>(q) * per_comp + bz * pitch + y * haloed.x + x;
            h[off] = encode(static_cast<int>(g), q, bz);
          }
        }
      }
    }
  }

  carbon::Stream stream = sys.stream();
  try {
    carbon::Transfer<int32_t>(sys, field.cpu_buffers(), field.gpu_buffers())(stream);
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_lattice_halo_dense_auto: upload failed: {}", e.what());
    return 1;
  }

  carbon::Skeleton sk(sys);
  sk.add(carbon::Kernel::compute(
      dense,
      [&field, lattice_stencil](carbon::KernelLoader& loader) {
        const auto pop =
            loader.read_stencil(field, lattice_stencil, carbon::StencilSemantic::Lattice);
        return [pop] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
          (void)pop.read(s, idx, 0);
        };
      },
      "touch_pop"));
  try {
    sk.compile();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_lattice_halo_dense_auto: skeleton compile failed: {}", e.what());
    return 1;
  }

  const size_t expected_kernels = 2U;
  if (sk.effective_kernels().size() != expected_kernels) {
    CARBON_LOG_ERROR(
        "test_lattice_halo_dense_auto: expected {} effective kernels (halo + compute), "
        "got {}",
        expected_kernels, sk.effective_kernels().size());
    return 1;
  }

  try {
    sk.run();
    sk.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_lattice_halo_dense_auto: skeleton run failed: {}", e.what());
    return 1;
  }

  for (size_t g = 0; g < n_gpus; ++g) {
    int32_t* h = field.cpu_buffers()[g].data();
    const size_t total = field.cpu_buffers()[g].size();
    for (size_t j = 0; j < total; ++j) {
      h[j] = -42'424'242;
    }
  }
  try {
    carbon::Transfer<int32_t>(sys, field.gpu_buffers(), field.cpu_buffers())(stream);
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_lattice_halo_dense_auto: download failed: {}", e.what());
    return 1;
  }

  size_t mismatches = 0;
  const auto& s_pts = lattice_stencil.points();
  for (size_t g = 0; g < n_gpus; ++g) {
    const auto& span = dense.spans()[g];
    const auto haloed = span.haloed_extent();
    const int64_t halo_z = span.halo_z();
    const int64_t lim_z = span.logic_extent().z;
    const int64_t pitch = span.xy_slab_elements();
    const int64_t per_comp = span.linear_storage_elements();
    const int32_t* h = field.cpu_buffers()[g].data();

    for (int q = 0; q < kComponents; ++q) {
      const auto sp = s_pts[static_cast<size_t>(q)];
      for (int64_t bz = 0; bz < haloed.z; ++bz) {
        const bool is_lower_halo = (bz < halo_z);
        const bool is_upper_halo = (bz >= halo_z + lim_z);
        const bool is_owned = !is_lower_halo && !is_upper_halo;
        const bool is_exterior_lower = is_lower_halo && (g == 0U);
        const bool is_exterior_upper = is_upper_halo && (g + 1U == n_gpus);
        for (int64_t y = 0; y < haloed.y; ++y) {
          for (int64_t x = 0; x < haloed.x; ++x) {
            const int64_t off = static_cast<int64_t>(q) * per_comp + bz * pitch + y * haloed.x + x;
            const int32_t got = h[off];
            int32_t want = 0;

            if (is_owned) {
              want = encode(static_cast<int>(g), q, bz);
            } else if (is_exterior_lower || is_exterior_upper) {
              want = sentinel(g, q, bz, y, x);
            } else if (is_lower_halo) {
              if (sp.z > 0) {
                want = encode(static_cast<int>(g) - 1, q, halo_z + lim_z - 1);
              } else {
                want = sentinel(g, q, bz, y, x);
              }
            } else {
              if (sp.z < 0) {
                want = encode(static_cast<int>(g) + 1, q, halo_z);
              } else {
                want = sentinel(g, q, bz, y, x);
              }
            }

            if (got != want) {
              if (mismatches < 8U) {
                CARBON_LOG_ERROR(
                    "test_lattice_halo_dense_auto: gpu {} q={} bz={} y={} x={} got={} want={}", g,
                    q, bz, y, x, got, want);
              }
              ++mismatches;
            }
          }
        }
      }
    }
  }

  if (mismatches > 0U) {
    CARBON_LOG_ERROR("test_lattice_halo_dense_auto: {} mismatches", mismatches);
    return 1;
  }

  CARBON_LOG_INFO("test_lattice_halo_dense_auto: ok ({} slabs, auto-halo + compute)", n_gpus);
  return 0;
}
