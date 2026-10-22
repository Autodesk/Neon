// Intra-process (single-rank) multi-GPU PeriodicZ halo with LATTICE stencil
// semantic — the direction-filtered population exchange that Skeleton auto-inserts
// for an LBM iteration kernel (loader.read_stencil(fIn, lattice_stencil, Lattice)).
//
// test_halo_periodic_z already covers the STANDARD semantic (every component moved
// both directions) for the same multi-GPU PeriodicZ layout. This test covers the
// LATTICE semantic, where a component q is only sent UP when c_q.z > 0 and only sent
// DOWN when c_q.z < 0 (c_q.z == 0 populations never cross a z boundary). That path
// was previously unexercised intra-process across a device split, and the
// render_wind_tunnel 3D wake diverged on 2 GPUs while being bit-stable on 1 — a
// wrong-halo-values signature. This test pins the exact contract:
//
//   - owned cells: unchanged.
//   - lower halo (buf z < halo_z): holds the wrap/neighbour source ONLY for
//     components with c_q.z > 0; every other component stays at the sentinel.
//   - upper halo (buf z >= halo_z + lim_z): holds the source ONLY for components
//     with c_q.z < 0; every other component stays at the sentinel.
//
// With PeriodicZ every halo cell maps to a real owned cell at global z
// (z_offset + z) mod global.z, so the expected filled value is
// encode(x, y, wrapped, c). Runs on 1 GPU (self-wrap) and >= 2 GPUs (the case that
// exposed the bug).

#include <carbon/carbon.h>
#include <carbon/dense.h>
#include <carbon/halo.h>
#include <carbon/integer3d.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/stencil.h>
#include <carbon/transfer.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kComponents = 6;
constexpr int kSentinel = -7777;

// Discrete velocities with a mix of z-signs: two up (c.z>0), two down (c.z<0), two
// in-plane (c.z==0). Index order is the population index q.
constexpr std::array<carbon::int32_3d, kComponents> kVel{{
    {0, 0, 1},     // q0 up
    {0, 0, -1},    // q1 down
    {1, 0, 0},     // q2 in-plane
    {-1, 0, 0},    // q3 in-plane
    {1, 1, 1},     // q4 up
    {-1, -1, -1},  // q5 down
}};

[[nodiscard]] auto
encode(int64_t x,
       int64_t y,
       int64_t global_z,
       int c) noexcept -> int {
  return static_cast<int>(((global_z * 100 + y) * 100 + x) * 8 + c);
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() < 1) {
    CARBON_LOG_WARN("test_halo_periodic_z_lattice: no visible CUDA devices; skipping");
    return 0;
  }

  carbon::System sys(carbon::get_all_devices());
  const size_t n_gpus = sys.gpu_group().size();

  std::vector<carbon::int32_3d> points(kVel.begin(), kVel.end());
  carbon::Stencil stencil(points, /*remove_center=*/false);

  const int64_t z_per_slab = 4;  // >= 2 * halo_z so self-wrap source/dest don't overlap
  const carbon::int64_3d global{4, 4, z_per_slab * static_cast<int64_t>(n_gpus)};
  carbon::Dense dense(sys, global, stencil, carbon::BoundaryMode::PeriodicZ);

  auto field = dense.newField<int, kComponents>(carbon::MemorySide::HostDevice);

  // Fill owned -> encode(x, y, global_z, c); halos -> sentinel.
  for (size_t i = 0; i < n_gpus; ++i) {
    const auto& span = dense.spans()[i];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t halo_z = span.halo_z();
    const int64_t z_origin = span.z_offset();
    const int64_t lim_z = span.logic_extent().z;
    const int64_t per_comp = span.linear_storage_elements();
    int* h = field.cpu_buffers()[i].data();
    for (int c = 0; c < kComponents; ++c) {
      for (int64_t z = 0; z < haloed.z; ++z) {
        const bool is_halo = (z < halo_z) || (z >= halo_z + lim_z);
        const int64_t global_z = z_origin + z;
        for (int64_t y = 0; y < haloed.y; ++y) {
          for (int64_t x = 0; x < haloed.x; ++x) {
            const int64_t off = static_cast<int64_t>(c) * per_comp + z * pitch + y * haloed.x + x;
            h[off] = is_halo ? kSentinel : encode(x, y, global_z, c);
          }
        }
      }
    }
  }

  carbon::Stream stream = sys.stream();
  carbon::Transfer<int>(sys, field.cpu_buffers(), field.gpu_buffers())(stream);
  stream.wait();

  // Lattice-semantic halo: pair the field's kComponents populations with the
  // stencil's discrete velocities and only move direction-matching populations.
  carbon::Halo<int, kComponents> halo(field, stencil, carbon::StencilSemantic::Lattice);

  // Entry accounting: 2 up-populations (q0,q4) + 2 down-populations (q1,q5) per
  // adjacency pair; PeriodicZ has n_gpus pairs (n_gpus-1 regular + 1 wrap).
  const size_t expected_pairs = 4U * n_gpus;
  CARBON_LOG_INFO("test_halo_periodic_z_lattice: pair_count = {} (expect {})", halo.pair_count(),
                  expected_pairs);
  if (halo.pair_count() != expected_pairs) {
    CARBON_LOG_ERROR("test_halo_periodic_z_lattice: pair_count {} != expected {}",
                     halo.pair_count(), expected_pairs);
    return 1;
  }

  halo(stream);
  stream.wait();

  // Reset host mirror, then download.
  constexpr int kDownloadSentinel = kSentinel - 1;
  for (size_t i = 0; i < n_gpus; ++i) {
    int* h = field.cpu_buffers()[i].data();
    const size_t n = field.cpu_buffers()[i].size();
    for (size_t j = 0; j < n; ++j) {
      h[j] = kDownloadSentinel;
    }
  }
  carbon::Transfer<int>(sys, field.gpu_buffers(), field.cpu_buffers())(stream);
  stream.wait();

  const int64_t global_z_extent = global.z;
  size_t mismatches = 0;
  for (size_t i = 0; i < n_gpus; ++i) {
    const auto& span = dense.spans()[i];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t halo_z = span.halo_z();
    const int64_t z_origin = span.z_offset();
    const int64_t lim_z = span.logic_extent().z;
    const int64_t per_comp = span.linear_storage_elements();
    const int* h = field.cpu_buffers()[i].data();

    for (int c = 0; c < kComponents; ++c) {
      const bool up_pop = kVel[static_cast<size_t>(c)].z > 0;
      const bool down_pop = kVel[static_cast<size_t>(c)].z < 0;
      for (int64_t z = 0; z < haloed.z; ++z) {
        const bool is_lower = z < halo_z;
        const bool is_upper = z >= halo_z + lim_z;
        int64_t wrapped = (z_origin + z) % global_z_extent;
        if (wrapped < 0) {
          wrapped += global_z_extent;
        }
        for (int64_t y = 0; y < haloed.y; ++y) {
          for (int64_t x = 0; x < haloed.x; ++x) {
            const int64_t off = static_cast<int64_t>(c) * per_comp + z * pitch + y * haloed.x + x;
            const int got = h[off];
            // Expected: owned unchanged; lower halo filled only for up-populations;
            // upper halo filled only for down-populations; everything else sentinel.
            int want = kSentinel;
            if (!is_lower && !is_upper) {
              want = encode(x, y, wrapped, c);
            } else if (is_lower && up_pop) {
              want = encode(x, y, wrapped, c);
            } else if (is_upper && down_pop) {
              want = encode(x, y, wrapped, c);
            }
            if (got != want) {
              if (mismatches < 8U) {
                CARBON_LOG_ERROR(
                    "test_halo_periodic_z_lattice: slab {} c={} (x,y,z)=({},{},{}) "
                    "lower={} upper={} wrapped={} got={} want={}",
                    i, c, x, y, z, is_lower, is_upper, wrapped, got, want);
              }
              ++mismatches;
            }
          }
        }
      }
    }
  }

  if (mismatches > 0U) {
    CARBON_LOG_ERROR("test_halo_periodic_z_lattice: FAIL — {} mismatches ({} GPUs)", mismatches,
                     n_gpus);
    return 1;
  }
  CARBON_LOG_INFO("test_halo_periodic_z_lattice: ok ({} GPUs, {} pairs)", n_gpus,
                  halo.pair_count());
  return 0;
}
