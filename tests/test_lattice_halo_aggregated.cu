// Multi-GPU correctness test for the lattice-semantic halo on the aggregated
// (z-major-then-cardinality) dense grid. Mirrors test_lattice_halo_disg.cu
// against carbon::Dense / carbon::Halo.
//
// On the aggregated layout the cardinality planes live linear_storage_elements
// apart — there's no run-length fusion to do, so the lattice halo simply
// drops the per-component entries whose stencil[c].z does not match the
// transfer direction. The expected pair_count is therefore
//   2 * (n_gpus - 1) * (positive_z + negative_z)
// vs the standard halo's
//   2 * (n_gpus - 1) * Components .
//
// For D3Q19 that's 10 entries instead of 38 per pair, regardless of population
// ordering — the natural Carbon ordering already triggers the win. We still
// keep the "grouped" flavour as a control to confirm aggregated lattice halos
// are *insensitive* to ordering (no fusion to lose).
//
// Skipped (returns 0) when fewer than 2 visible CUDA devices are available.

#include <carbon/carbon.h>
#include <carbon/dense.h>
#include <carbon/halo.h>
#include <carbon/integer3d.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
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

[[nodiscard]] auto
d3q19_grouped() -> std::vector<carbon::int32_3d> {
  return {
      {0, 0, -1}, {-1, 0, -1}, {0, -1, -1}, {1, 0, -1}, {0, 1, -1}, {0, 0, 0}, {-1, 0, 0},
      {0, -1, 0}, {-1, -1, 0}, {-1, 1, 0},  {1, 0, 0},  {0, 1, 0},  {1, 1, 0}, {1, -1, 0},
      {0, 0, 1},  {-1, 0, 1},  {0, -1, 1},  {1, 0, 1},  {0, 1, 1},
  };
}

[[nodiscard]] auto
run_flavour(const std::string& label,
            const std::vector<carbon::int32_3d>& velocities) -> int {
  carbon::System sys(carbon::get_all_devices());
  const size_t n_gpus = sys.gpu_group().size();
  // 16 / n_gpus must be >= 2 with halo_z=1 (so Internal/Boundary view split is well-defined).
  const carbon::int64_3d global{4, 5, 8 * static_cast<int64_t>(n_gpus)};

  carbon::Stencil stencil(velocities);
  carbon::Dense dense(sys, global, stencil);
  auto field = dense.newField<int32_t, kComponents>(carbon::MemorySide::HostDevice);

  const auto profile = stencil.lattice_profile();
  CARBON_LOG_INFO("test_lattice_halo_aggregated[{}]: profile +z={} -z={} 0z={}", label,
                  profile.positive_z, profile.negative_z, profile.zero_z);

  // Flat sentinel fill, then overwrite owned cells with encode(g, q, buf_z).
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
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_lattice_halo_aggregated[{}]: upload failed: {}", label, e.what());
    return 1;
  }
  stream.wait();

  carbon::Halo<int32_t, kComponents> halo(field, stencil, carbon::StencilSemantic::Lattice);
  // Per pair: positive_z entries upward + negative_z entries downward (one cudaMemcpyAsync
  // per active cardinality plane on the aggregated layout — no fusion across cardinalities).
  const size_t expected =
      (n_gpus - 1U) * static_cast<size_t>(profile.positive_z + profile.negative_z);
  CARBON_LOG_INFO("test_lattice_halo_aggregated[{}]: pair_count = {} (expected {})", label,
                  halo.pair_count(), expected);
  if (halo.pair_count() != expected) {
    CARBON_LOG_ERROR("test_lattice_halo_aggregated[{}]: pair_count {} != expected {}", label,
                     halo.pair_count(), expected);
    return 1;
  }

  try {
    halo(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_lattice_halo_aggregated[{}]: halo() failed: {}", label, e.what());
    return 1;
  }
  stream.wait();

  // Sentinel-overwrite host then download.
  for (size_t g = 0; g < n_gpus; ++g) {
    int32_t* h = field.cpu_buffers()[g].data();
    const size_t total = field.cpu_buffers()[g].size();
    for (size_t j = 0; j < total; ++j) {
      h[j] = -42'424'242;
    }
  }
  try {
    carbon::Transfer<int32_t>(sys, field.gpu_buffers(), field.cpu_buffers())(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_lattice_halo_aggregated[{}]: download failed: {}", label, e.what());
    return 1;
  }
  stream.wait();

  // Verify.
  size_t mismatches = 0;
  const auto& s_pts = stencil.points();
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
              // Filled from neighbour (g-1)'s last owned at buf_z = halo_z + lim_z - 1.
              if (sp.z > 0) {
                want = encode(static_cast<int>(g) - 1, q, halo_z + lim_z - 1);
              } else {
                want = sentinel(g, q, bz, y, x);
              }
            } else {  // is_upper_halo
              if (sp.z < 0) {
                want = encode(static_cast<int>(g) + 1, q, halo_z);
              } else {
                want = sentinel(g, q, bz, y, x);
              }
            }

            if (got != want) {
              if (mismatches < 8U) {
                CARBON_LOG_ERROR(
                    "test_lattice_halo_aggregated[{}]: gpu {} q={} bz={} y={} x={} "
                    "got={} want={}",
                    label, g, q, bz, y, x, got, want);
              }
              ++mismatches;
            }
          }
        }
      }
    }
  }

  if (mismatches > 0U) {
    CARBON_LOG_ERROR("test_lattice_halo_aggregated[{}]: {} mismatches", label, mismatches);
    return 1;
  }
  CARBON_LOG_INFO("test_lattice_halo_aggregated[{}]: ok ({} slabs, {} entries)", label, n_gpus,
                  halo.pair_count());
  return 0;
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() < 2) {
    CARBON_LOG_WARN(
        "test_lattice_halo_aggregated: needs >=2 visible CUDA devices for a meaningful exchange; "
        "have {} -- skipping",
        carbon::device_count());
    return 0;
  }
  if (run_flavour("natural", d3q19_natural()) != 0) {
    return 1;
  }
  if (run_flavour("grouped", d3q19_grouped()) != 0) {
    return 1;
  }
  return 0;
}
