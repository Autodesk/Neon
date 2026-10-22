// Multi-GPU correctness test for the lattice-semantic halo on the disaggregated
// grid (StencilSemantic::Lattice).
//
// Runs two flavours of D3Q19 over carbon::DenseDisg with halo_z = 1, on >= 2
// visible CUDA devices:
//
//   1. "natural" ordering: textbook D3Q19 indexing as used by Carbon's LBM
//      benchmark (GO 0..8, center 9, BK 10..18). Up populations
//      {6, 8, 12, 15, 17} are not contiguous, so the disg lattice halo issues
//      multiple small transfers per direction.
//
//   2. "grouped" ordering: a permuted D3Q19 where the 5 c.z<0 populations come
//      first, then the 9 c.z=0 populations, then the 5 c.z>0 populations. This
//      makes the lattice halo "fully groupable" — each direction collapses
//      back to a single fused transfer per pair, same call count as the
//      Standard halo with 5/19 of the bytes.
//
// For each flavour the test:
//   - allocates a HostDevice Field<int, 19>,
//   - fills owned cells on the host with encode(x, y, gz, q),
//   - sentinels every halo cell to a unique value,
//   - uploads to device, runs the lattice halo, downloads,
//   - verifies that for every halo cell:
//       * if its population q is "active" in this direction
//         (c_q.z matches the transfer sign), it now holds encode(x, y, gz_neighbour, q),
//       * else the cell is still at its original sentinel — no over-copy.
//
// Skipped (returns 0) when fewer than 2 visible CUDA devices are available.

#include <carbon/carbon.h>
#include <carbon/dense_disg.h>
#include <carbon/halo_disg.h>
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
constexpr int kSentinelBase = -777'000;

[[nodiscard]] auto
encode(int64_t x,
       int64_t y,
       int64_t global_z,
       int q) noexcept -> int {
  return static_cast<int>(global_z * 1'000'000 + y * 10'000 + x * 100 + q);
}

// Address for cell (bz, x, y, q) in slab `span`'s disaggregated buffer.
[[nodiscard]] auto
disg_addr(const carbon::DenseDisg::Span& span,
          int64_t bz,
          int64_t x,
          int64_t y,
          int q,
          int components) -> int64_t {
  const int64_t lz = span.logic_extent().z;
  const int64_t xdim = span.haloed_extent().x;
  const int64_t xy = span.xy_slab_elements();
  int64_t region_first_z = 0;
  int64_t region_z_dim = 1;
  int64_t offset_local_no_card = x + y * xdim;
  if (bz <= 1 || bz >= lz) {
    region_first_z = bz;
    region_z_dim = 1;
  } else {
    region_first_z = 2;
    region_z_dim = lz - 2;
    offset_local_no_card += (bz - 2) * xy;
  }
  return offset_local_no_card + xy * (region_first_z * static_cast<int64_t>(components) +
                                      static_cast<int64_t>(q) * region_z_dim);
}

// D3Q19 discrete velocities in Carbon's natural ordering (matches
// benchmarks/lbm_lid_driven_cavity/src/d3q19.h byte-for-byte).
[[nodiscard]] auto
d3q19_natural() -> std::vector<carbon::int32_3d> {
  return {
      {-1, 0, 0},  {0, -1, 0}, {0, 0, -1}, {-1, -1, 0}, {-1, 1, 0}, {-1, 0, -1}, {-1, 0, 1},
      {0, -1, -1}, {0, -1, 1}, {0, 0, 0},  {1, 0, 0},   {0, 1, 0},  {0, 0, 1},   {1, 1, 0},
      {1, -1, 0},  {1, 0, 1},  {1, 0, -1}, {0, 1, 1},   {0, 1, -1},
  };
}

// D3Q19 grouped by c.z sign: 5 entries with c.z<0 first (positions 0..4),
// then 9 entries with c.z==0 (positions 5..13), then 5 entries with c.z>0
// (positions 14..18). Concrete permutation chosen so each bucket contains
// the same set of velocities as the natural ordering, just relocated.
[[nodiscard]] auto
d3q19_grouped() -> std::vector<carbon::int32_3d> {
  return {
      // c.z < 0  (5 entries)
      {0, 0, -1},
      {-1, 0, -1},
      {0, -1, -1},
      {1, 0, -1},
      {0, 1, -1},
      // c.z == 0 (9 entries)
      {0, 0, 0},
      {-1, 0, 0},
      {0, -1, 0},
      {-1, -1, 0},
      {-1, 1, 0},
      {1, 0, 0},
      {0, 1, 0},
      {1, 1, 0},
      {1, -1, 0},
      // c.z > 0  (5 entries)
      {0, 0, 1},
      {-1, 0, 1},
      {0, -1, 1},
      {1, 0, 1},
      {0, 1, 1},
  };
}

[[nodiscard]] auto
sync_all(const carbon::Stream& stream,
         const char* tag) -> bool {
  try {
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_lattice_halo_disg: sync after {} failed: {}", tag, e.what());
    return false;
  }
  return true;
}

[[nodiscard]] auto
run_flavour(const std::string& label,
            const std::vector<carbon::int32_3d>& velocities,
            int expected_run_count_per_dir) -> int {
  carbon::System sys(carbon::get_all_devices());
  const size_t n_gpus = sys.gpu_group().size();
  const int64_t z_per_slab = 4;
  const carbon::int64_3d global{4, 5, z_per_slab * static_cast<int64_t>(n_gpus)};

  carbon::DenseDisg dense(sys, global, /*halo_z=*/1);
  carbon::Stencil stencil(velocities);

  const auto profile = stencil.lattice_profile();
  CARBON_LOG_INFO(
      "test_lattice_halo_disg[{}]: profile +z={} -z={} 0z={} runs(+/-)=({},{}) "
      "fully_groupable={}",
      label, profile.positive_z, profile.negative_z, profile.zero_z, profile.run_count_positive,
      profile.run_count_negative, profile.fully_groupable);

  auto field = dense.newField<int, kComponents>(carbon::MemorySide::HostDevice);

  // Per-cell sentinel that's *unique per slab/cell/component* so we can detect any
  // accidental write and tell us where it came from.
  auto sentinel = [](size_t slab, int64_t bz, int64_t x, int64_t y, int q) noexcept -> int {
    return kSentinelBase - static_cast<int>(slab * 1'000'000 + bz * 10'000 + y * 100 + x * 10 + q);
  };

  // Host fill: every cell (owned + halo) starts at its slot-unique sentinel; then we
  // overwrite the OWNED cells with encode(...). That way a correct lattice halo will
  // overwrite only the active-population halo cells, leaving inactive halos at sentinel.
  for (size_t i = 0; i < n_gpus; ++i) {
    const auto& span = dense.spans()[i];
    const auto haloed = span.haloed_extent();
    int* h = field.cpu_buffers()[i].data();
    for (int64_t bz = 0; bz < haloed.z; ++bz) {
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          for (int q = 0; q < kComponents; ++q) {
            const int64_t off = disg_addr(span, bz, x, y, q, kComponents);
            h[off] = sentinel(i, bz, x, y, q);
          }
        }
      }
    }
  }
  for (size_t i = 0; i < n_gpus; ++i) {
    const auto& span = dense.spans()[i];
    const auto haloed = span.haloed_extent();
    const int64_t halo_z = span.halo_z();
    const int64_t z_origin = span.z_offset();
    const int64_t lim_z = span.logic_extent().z;
    int* h = field.cpu_buffers()[i].data();
    for (int64_t bz = halo_z; bz < halo_z + lim_z; ++bz) {
      const int64_t gz = z_origin + bz;
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          for (int q = 0; q < kComponents; ++q) {
            const int64_t off = disg_addr(span, bz, x, y, q, kComponents);
            h[off] = encode(x, y, gz, q);
          }
        }
      }
    }
  }

  carbon::Stream stream = sys.stream();

  try {
    carbon::Transfer<int> upload(sys, field.cpu_buffers(), field.gpu_buffers());
    upload(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_lattice_halo_disg[{}]: upload failed: {}", label, e.what());
    return 1;
  }
  if (!sync_all(stream, "upload")) {
    return 1;
  }

  carbon::HaloDisg<int, kComponents> halo(field, stencil, carbon::StencilSemantic::Lattice);
  if (halo.stencil_semantic() != carbon::StencilSemantic::Lattice) {
    CARBON_LOG_ERROR("test_lattice_halo_disg[{}]: stencil_semantic accessor mismatch", label);
    return 1;
  }
  // Expected pair_count per (n_gpus - 1) interfaces: runs_up + runs_down per pair.
  const int per_pair = profile.run_count_positive + profile.run_count_negative;
  const size_t expected_pairs = static_cast<size_t>(per_pair) * (n_gpus - 1U);
  CARBON_LOG_INFO("test_lattice_halo_disg[{}]: pair_count = {} (expected {}; per_pair runs = {})",
                  label, halo.pair_count(), expected_pairs, per_pair);
  if (halo.pair_count() != expected_pairs) {
    CARBON_LOG_ERROR("test_lattice_halo_disg[{}]: pair_count {} != expected {}", label,
                     halo.pair_count(), expected_pairs);
    return 1;
  }
  // Sanity: profile run_count_positive should equal `expected_run_count_per_dir` for
  // each direction (the test caller picks the flavour and predicts the run count).
  if (profile.run_count_positive != expected_run_count_per_dir ||
      profile.run_count_negative != expected_run_count_per_dir) {
    CARBON_LOG_ERROR("test_lattice_halo_disg[{}]: profile runs ({}, {}) do not match expected {}",
                     label, profile.run_count_positive, profile.run_count_negative,
                     expected_run_count_per_dir);
    return 1;
  }

  try {
    halo(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_lattice_halo_disg[{}]: halo() failed: {}", label, e.what());
    return 1;
  }
  if (!sync_all(stream, "halo")) {
    return 1;
  }

  // Sentinel-overwrite host with a different sentinel then download to confirm we
  // read back what's on device (not a stale host copy).
  for (size_t i = 0; i < n_gpus; ++i) {
    int* h = field.cpu_buffers()[i].data();
    const size_t total = field.cpu_buffers()[i].size();
    for (size_t j = 0; j < total; ++j) {
      h[j] = -424242;
    }
  }
  try {
    carbon::Transfer<int> download(sys, field.gpu_buffers(), field.cpu_buffers());
    download(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_lattice_halo_disg[{}]: download failed: {}", label, e.what());
    return 1;
  }
  if (!sync_all(stream, "download")) {
    return 1;
  }

  // Verify every cell on every slab.
  size_t mismatches = 0;
  const auto& s_pts = stencil.points();
  for (size_t i = 0; i < n_gpus; ++i) {
    const auto& span = dense.spans()[i];
    const auto haloed = span.haloed_extent();
    const int64_t halo_z = span.halo_z();
    const int64_t z_origin = span.z_offset();
    const int64_t lim_z = span.logic_extent().z;
    const int* h = field.cpu_buffers()[i].data();

    for (int64_t bz = 0; bz < haloed.z; ++bz) {
      const bool is_lower_halo = (bz < halo_z);
      const bool is_upper_halo = (bz >= halo_z + lim_z);
      const bool is_owned = !is_lower_halo && !is_upper_halo;
      const bool is_exterior_lower = is_lower_halo && (i == 0U);
      const bool is_exterior_upper = is_upper_halo && (i + 1U == n_gpus);
      const int64_t gz = z_origin + bz;

      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          for (int q = 0; q < kComponents; ++q) {
            const int64_t off = disg_addr(span, bz, x, y, q, kComponents);
            const int got = h[off];
            int want = 0;

            if (is_owned) {
              want = encode(x, y, gz, q);
            } else if (is_exterior_lower || is_exterior_upper) {
              want = sentinel(i, bz, x, y, q);  // Open boundary, no neighbour.
            } else {
              // Interior halo. Active iff stencil[q].z matches direction.
              const auto sp = s_pts[static_cast<size_t>(q)];
              const bool came_from_below = is_lower_halo;  // Lower halo is filled by upward xfer.
              const bool active = came_from_below ? (sp.z > 0) : (sp.z < 0);
              if (active) {
                want = encode(x, y, gz, q);
              } else {
                want = sentinel(i, bz, x, y, q);
              }
            }

            if (got != want) {
              if (mismatches < 8U) {
                CARBON_LOG_ERROR(
                    "test_lattice_halo_disg[{}]: slab {} (bz={}, x={}, y={}, q={}, "
                    "gz={}) got={} want={}",
                    label, i, bz, x, y, q, gz, got, want);
              }
              ++mismatches;
            }
          }
        }
      }
    }
  }

  if (mismatches > 0U) {
    CARBON_LOG_ERROR("test_lattice_halo_disg[{}]: {} mismatches", label, mismatches);
    return 1;
  }
  CARBON_LOG_INFO("test_lattice_halo_disg[{}]: ok ({} slabs, {} pair entries)", label, n_gpus,
                  halo.pair_count());
  return 0;
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() < 2) {
    CARBON_LOG_WARN(
        "test_lattice_halo_disg: needs >=2 visible CUDA devices for a meaningful exchange; have "
        "{} -- skipping",
        carbon::device_count());
    return 0;
  }

  // Natural ordering: D3Q19 up populations are {6, 8, 12, 15, 17} -- 5 separate runs.
  // Down populations are {2, 5, 7, 16, 18} -- also 5 separate runs.
  if (run_flavour("natural", d3q19_natural(), /*expected_run_count_per_dir=*/5) != 0) {
    return 1;
  }
  // Grouped ordering: down populations contiguous in [0,5), up populations contiguous in
  // [14,19) -- 1 run each per direction. This is the disg-lattice "fully groupable" case.
  if (run_flavour("grouped", d3q19_grouped(), /*expected_run_count_per_dir=*/1) != 0) {
    return 1;
  }
  return 0;
}
