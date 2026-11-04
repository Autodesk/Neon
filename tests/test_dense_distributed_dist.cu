// DenseDistributed on real GPUs: slab, pencil and brick decompositions, end to end.
//
// Launched under `mpiexec -n 4 ./test_dense_distributed_dist` (one rank per GPU). With 8 ranks it
// runs the full brick (2,2,2) instead; any other world size skips (returns 0), as does a node with
// fewer visible devices than ranks.
//
// Process grids (Px, Py, Pz) per world size, each in Open and periodic variants:
//   4 ranks : slab (1,1,4) | pencil (1,2,2) | brick (2,2,1) and (2,1,2)
//   8 ranks : brick (2,2,2)
//
// Two checks per case, both against ground truth computed on the host:
//
//  1. Raw HaloDistributed exchange. Owned cells hold encode(global cell), every halo cell holds a
//     sentinel; after one exchange every cell of every rank's buffer is compared:
//       - owned cells unchanged,
//       - interior halos == encode(the global cell they mirror),
//       - exterior halos on an Open axis still the sentinel (never clobbered),
//       - exterior halos on a periodic axis == encode(the wrapped global cell).
//     It also checks the per-rank layout (owned extent, offset, halo, haloed extent) against an
//     independent computation. Scalar and 3-component fields are run, each with the packed
//     X/Y/Z exchange (CARBON_HALO_PACK=1) and the run-based path (=0), so packed == run-based
//     follows from both matching ground truth.
//
//  2. Skeleton + Kernel path. A 6-point stencil kernel (sum of the six neighbours, 0 where a
//     neighbour is outside an open boundary) is compiled into a Skeleton, so the halo kernel is
//     auto-inserted from read_stencil and DenseDistributed::Span / Partition run on the device.
//     Run once without overlap and once with OverlapMode::Standard, which splits the kernel into
//     Internal + Boundary launches and therefore exercises those Span views for real. (For pencil
//     and brick the Boundary launch is the compact shell, not the full owned extent.)
//
//  3. VTK shard I/O. A 3-component float field goes host -> device -> host, is written with
//     io::write_vtk (one XML `.rank<N>.vti` shard per rank) plus the `.vtm` index, and read back
//     with io::read_vtk into a second field. Each rank checks its own shard WholeExtent / Origin
//     against its owned box, every owned cell after the round trip, that halo cells were not
//     touched, and rank 0 checks that the index lists every rank. Files are written to the working
//     directory (like test_io_dist) and removed afterwards.

#include <carbon/boundary_mode_distributed.h>
#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense_distributed.h>
#include <carbon/distributed.h>
#include <carbon/halo_distributed.h>
#include <carbon/integer3d.h>
#include <carbon/io.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/process_grid.h>
#include <carbon/skeleton.h>
#include <carbon/stencil.h>
#include <carbon/system.h>
#include <carbon/transfer.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using carbon::Axis;
using carbon::BoundaryModeDistributed;
using carbon::DenseDistributed;
using carbon::int32_3d;
using carbon::int64_3d;
using carbon::ProcessGrid;

constexpr int kSentinel = -7777;

[[nodiscard]] auto
encode(int64_t x,
       int64_t y,
       int64_t z,
       int comp = 0) noexcept -> int {
  return static_cast<int>(z * 10000 + y * 100 + x) + comp * 1'000'000;
}

[[nodiscard]] auto
split_extent(int64_t n,
             int32_t parts) -> std::vector<int64_t> {
  std::vector<int64_t> counts(static_cast<size_t>(parts), 0);
  for (int32_t i = 0; i < parts; ++i) {
    counts[static_cast<size_t>(i)] = n / parts + (static_cast<int64_t>(i) < n % parts ? 1 : 0);
  }
  return counts;
}

[[nodiscard]] auto
prefix_sum(const std::vector<int64_t>& counts) -> std::vector<int64_t> {
  std::vector<int64_t> prefix(counts.size() + 1U, 0);
  for (size_t i = 0; i < counts.size(); ++i) {
    prefix[i + 1U] = prefix[i] + counts[i];
  }
  return prefix;
}

[[nodiscard]] auto
sync_all(const carbon::Stream& stream,
         const char* tag) -> bool {
  try {
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dense_distributed_dist: sync after {} failed: {}", tag, e.what());
    return false;
  }
  return true;
}

[[nodiscard]] auto
six_point() -> carbon::Stencil {
  return carbon::Stencil(
      std::vector<int32_3d>{{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);
}

[[nodiscard]] auto
xml_attribute(std::string_view tag,
              std::string_view key) -> std::string {
  const std::string needle = std::string(key) + "=\"";
  const size_t begin = tag.find(needle);
  if (begin == std::string_view::npos) {
    return {};
  }
  const size_t value_begin = begin + needle.size();
  const size_t end = tag.find('"', value_begin);
  if (end == std::string_view::npos) {
    return {};
  }
  return std::string(tag.substr(value_begin, end - value_begin));
}

/** Wrap @p v into [0, n) if @p periodic; return false when it is outside an open axis. */
[[nodiscard]] auto
wrap_axis(int64_t v,
          int64_t n,
          bool periodic,
          int64_t& out) noexcept -> bool {
  if (v >= 0 && v < n) {
    out = v;
    return true;
  }
  if (!periodic) {
    return false;
  }
  out = ((v % n) + n) % n;
  return true;
}

// ------------------------------------------------------------------------------------------------
// Check 1: raw HaloDistributed exchange, every buffer cell compared with ground truth.
// ------------------------------------------------------------------------------------------------

template <int Components>
[[nodiscard]] auto
run_halo_once(carbon::System& sys,
              int rank,
              const ProcessGrid& pg,
              BoundaryModeDistributed mode,
              const char* label,
              const int64_3d& global,
              carbon::StencilSemantic semantic = carbon::StencilSemantic::Standard) -> bool {
  const int32_3d dims = pg.dims();
  const int32_3d coords = pg.coords_of(rank);
  const carbon::Stencil stencil = six_point();

  // Expected geometry, derived independently of the library.
  const bool slab_shape = dims.x == 1 && dims.y == 1;
  const bool legacy_slab =
      slab_shape && !carbon::is_periodic(mode, Axis::X) && !carbon::is_periodic(mode, Axis::Y);
  const auto axis_halo = [&](int32_t parts, Axis axis) -> int64_t {
    return (parts > 1 || carbon::is_periodic(mode, axis)) ? 1 : 0;
  };
  const int64_3d expected_halo =
      legacy_slab ? int64_3d{0, 0, 1}
                  : int64_3d{axis_halo(dims.x, Axis::X), axis_halo(dims.y, Axis::Y),
                             slab_shape ? 1 : axis_halo(dims.z, Axis::Z)};
  const auto xs = split_extent(global.x, dims.x);
  const auto ys = split_extent(global.y, dims.y);
  const auto zs = split_extent(global.z, dims.z);
  const auto xp = prefix_sum(xs);
  const auto yp = prefix_sum(ys);
  const auto zp = prefix_sum(zs);
  const int64_3d expected_owned{xs[static_cast<size_t>(coords.x)],
                                ys[static_cast<size_t>(coords.y)],
                                zs[static_cast<size_t>(coords.z)]};
  const int64_3d expected_offset{xp[static_cast<size_t>(coords.x)] - expected_halo.x,
                                 yp[static_cast<size_t>(coords.y)] - expected_halo.y,
                                 zp[static_cast<size_t>(coords.z)] - expected_halo.z};

  DenseDistributed grid(sys, global, pg, stencil, mode);
  const auto& span = grid.spans()[0];
  const int64_3d halo = span.halo();
  const int64_3d owned = span.logic_extent();
  const int64_3d haloed = span.haloed_extent();
  const int64_3d offset = span.offset();

  const auto same = [](const int64_3d& a, const int64_3d& b) -> bool {
    return a.x == b.x && a.y == b.y && a.z == b.z;
  };
  if (grid.gpu_count() != 1U || grid.global_slab_count() != static_cast<size_t>(dims.z) ||
      grid.local_slab_offset() != static_cast<size_t>(coords.z) || !same(halo, expected_halo) ||
      !same(owned, expected_owned) || !same(offset, expected_offset) ||
      haloed.x != owned.x + 2 * halo.x || haloed.y != owned.y + 2 * halo.y ||
      haloed.z != owned.z + 2 * halo.z) {
    CARBON_LOG_ERROR(
        "test_dense_distributed_dist[{}]: rank {} layout mismatch: halo=({},{},{}) want ({},{},{}) "
        "owned=({},{},{}) want ({},{},{}) offset=({},{},{}) want ({},{},{})",
        label, rank, halo.x, halo.y, halo.z, expected_halo.x, expected_halo.y, expected_halo.z,
        owned.x, owned.y, owned.z, expected_owned.x, expected_owned.y, expected_owned.z, offset.x,
        offset.y, offset.z, expected_offset.x, expected_offset.y, expected_offset.z);
    return false;
  }

  auto field = grid.template newField<int, Components>(carbon::MemorySide::HostDevice);
  const int64_t pitch = span.xy_slab_elements();
  const int64_t per_comp = span.linear_storage_elements();
  const auto is_owned_cell = [&](int64_t x, int64_t y, int64_t z) -> bool {
    return x >= halo.x && x < halo.x + owned.x && y >= halo.y && y < halo.y + owned.y &&
           z >= halo.z && z < halo.z + owned.z;
  };
  {
    int* h = field.cpu_buffers()[0].data();
    for (int64_t z = 0; z < haloed.z; ++z) {
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_3d g = span.global_location(span.make_idx(x, y, z));
          const int64_t off = z * pitch + y * haloed.x + x;
          for (int c = 0; c < Components; ++c) {
            h[c * per_comp + off] = is_owned_cell(x, y, z) ? encode(g.x, g.y, g.z, c) : kSentinel;
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
    CARBON_LOG_ERROR("test_dense_distributed_dist[{}]: upload failed: {}", label, e.what());
    return false;
  }
  if (!sync_all(stream, "upload")) {
    return false;
  }

  carbon::HaloDistributed<int, Components> exchange(field, stencil, semantic);
  // A multi-rank layout always needs cross-rank transfers (X/Y faces fan out into several runs, so
  // the exact count is geometry dependent; correctness is enforced by the cell comparison below).
  if (exchange.pair_count() != 0U || exchange.dist_entry_count() == 0U ||
      (exchange.dist_entry_count() % 2U) != 0U) {
    CARBON_LOG_ERROR(
        "test_dense_distributed_dist[{}]: pair_count={} dist_entry_count={} (want 0 and a "
        "non-zero even count)",
        label, exchange.pair_count(), exchange.dist_entry_count());
    return false;
  }
  try {
    exchange(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dense_distributed_dist[{}]: halo exchange failed: {}", label, e.what());
    return false;
  }
  if (!sync_all(stream, "halo")) {
    return false;
  }

  {
    int* h = field.cpu_buffers()[0].data();
    for (size_t i = 0; i < field.cpu_buffers()[0].size(); ++i) {
      h[i] = kSentinel - 1;
    }
  }
  try {
    carbon::Transfer<int> download(sys, field.gpu_buffers(), field.cpu_buffers());
    download(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dense_distributed_dist[{}]: download failed: {}", label, e.what());
    return false;
  }
  if (!sync_all(stream, "download")) {
    return false;
  }

  size_t mismatches = 0;
  const int* h = field.cpu_buffers()[0].data();
  for (int64_t z = 0; z < haloed.z; ++z) {
    for (int64_t y = 0; y < haloed.y; ++y) {
      for (int64_t x = 0; x < haloed.x; ++x) {
        const int64_3d g = span.global_location(span.make_idx(x, y, z));
        int64_t wx = 0;
        int64_t wy = 0;
        int64_t wz = 0;
        const bool inside = wrap_axis(g.x, global.x, carbon::is_periodic(mode, Axis::X), wx) &&
                            wrap_axis(g.y, global.y, carbon::is_periodic(mode, Axis::Y), wy) &&
                            wrap_axis(g.z, global.z, carbon::is_periodic(mode, Axis::Z), wz);
        const bool owned_cell = is_owned_cell(x, y, z);
        const bool halo_x = x < halo.x || x >= halo.x + owned.x;
        const bool halo_y = y < halo.y || y >= halo.y + owned.y;
        const bool halo_z = z < halo.z || z >= halo.z + owned.z;
        const int halo_axes = static_cast<int>(halo_x) + static_cast<int>(halo_y) +
                              static_cast<int>(halo_z);
        const Axis halo_axis = halo_x ? Axis::X : (halo_y ? Axis::Y : Axis::Z);
        const int64_t off = z * pitch + y * haloed.x + x;
        for (int c = 0; c < Components; ++c) {
          bool check = true;
          int want = kSentinel;
          if (owned_cell) {
            want = encode(g.x, g.y, g.z, c);
          } else if (!inside) {
            want = kSentinel;
          } else if (semantic == carbon::StencilSemantic::LatticeDistributed) {
            // Corners are written by more than one face pass; only face-interior
            // halo cells have an unambiguous source for every active component.
            if (halo_axes != 1) {
              check = false;
            } else {
              const auto& p = stencil.points()[static_cast<size_t>(c)];
              const int64_t v =
                  halo_axis == Axis::X ? p.x : (halo_axis == Axis::Y ? p.y : p.z);
              want = v != 0 ? encode(wx, wy, wz, c) : kSentinel;
            }
          } else {
            want = encode(wx, wy, wz, c);
          }
          if (!check) {
            continue;
          }
          const int got = h[c * per_comp + off];
          if (got != want) {
            if (mismatches < 4U) {
              CARBON_LOG_ERROR(
                  "test_dense_distributed_dist[{}]: rank {} comp {} buffer ({},{},{}) global "
                  "({},{},{}) got {} want {}",
                  label, rank, c, x, y, z, g.x, g.y, g.z, got, want);
            }
            ++mismatches;
          }
        }
      }
    }
  }
  if (mismatches != 0U) {
    CARBON_LOG_ERROR("test_dense_distributed_dist[{}]: rank {} has {} wrong cells", label, rank,
                     mismatches);
    return false;
  }
  return true;
}

/**
 * Runs @ref run_halo_once with the packed exchange and with the run-based path.
 * CARBON_HALO_PACK is read when a HaloDistributed is constructed; each variant is verified against
 * ground truth, so packed == run-based follows. The environment is restored afterwards.
 */
template <int Components>
[[nodiscard]] auto
run_halo(carbon::System& sys,
         int rank,
         const ProcessGrid& pg,
         BoundaryModeDistributed mode,
         const std::string& label,
         const int64_3d& global,
         carbon::StencilSemantic semantic = carbon::StencilSemantic::Standard) -> bool {
  const char* saved = std::getenv("CARBON_HALO_PACK");
  const bool had = saved != nullptr;
  const std::string previous = had ? saved : "";
  bool ok = true;
  for (const bool pack : {true, false}) {
    ::setenv("CARBON_HALO_PACK", pack ? "1" : "0", 1);
    const std::string tagged = label + "/c" + std::to_string(Components) +
                               (pack ? "/packed" : "/runs") +
                               (semantic == carbon::StencilSemantic::LatticeDistributed
                                    ? "/lattice-distributed"
                                    : "");
    ok = run_halo_once<Components>(sys, rank, pg, mode, tagged.c_str(), global, semantic);
    if (!ok) {
      break;
    }
  }
  if (had) {
    ::setenv("CARBON_HALO_PACK", previous.c_str(), 1);
  } else {
    ::unsetenv("CARBON_HALO_PACK");
  }
  return ok;
}

// ------------------------------------------------------------------------------------------------
// Check 2: Skeleton + stencil kernel (auto-inserted halo, optional compute/communication overlap).
// ------------------------------------------------------------------------------------------------

[[nodiscard]] auto
run_stencil(carbon::System& sys,
            int rank,
            const ProcessGrid& pg,
            BoundaryModeDistributed mode,
            carbon::OverlapMode overlap,
            const std::string& label,
            const int64_3d& global) -> bool {
  DenseDistributed grid(sys, global, pg, six_point(), mode);
  const auto& span = grid.spans()[0];
  const int64_3d halo = span.halo();
  const int64_3d owned = span.logic_extent();
  const int64_3d haloed = span.haloed_extent();
  const int64_t pitch = span.xy_slab_elements();

  auto in = grid.newField<int, 1>(carbon::MemorySide::HostDevice, "in");
  auto out = grid.newField<int, 1>(carbon::MemorySide::HostDevice, "out");
  const auto is_owned_cell = [&](int64_t x, int64_t y, int64_t z) -> bool {
    return x >= halo.x && x < halo.x + owned.x && y >= halo.y && y < halo.y + owned.y &&
           z >= halo.z && z < halo.z + owned.z;
  };
  {
    int* hi = in.cpu_buffers()[0].data();
    int* ho = out.cpu_buffers()[0].data();
    for (int64_t z = 0; z < haloed.z; ++z) {
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_3d g = span.global_location(span.make_idx(x, y, z));
          const int64_t off = z * pitch + y * haloed.x + x;
          hi[off] = is_owned_cell(x, y, z) ? encode(g.x, g.y, g.z) : kSentinel;
          ho[off] = kSentinel;
        }
      }
    }
  }
  carbon::Stream stream = sys.stream();
  try {
    carbon::Transfer<int>(sys, in.cpu_buffers(), in.gpu_buffers())(stream);
    carbon::Transfer<int>(sys, out.cpu_buffers(), out.gpu_buffers())(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dense_distributed_dist[{}]: upload failed: {}", label, e.what());
    return false;
  }
  if (!sync_all(stream, "stencil upload")) {
    return false;
  }

  using Span = DenseDistributed::Span;
  using Idx = DenseDistributed::Idx;
  carbon::Skeleton sk(sys, carbon::Skeleton::Options(overlap));
  sk.add(carbon::Kernel::compute(
      grid,
      [in, out](carbon::KernelLoader& loader) {
        const auto pin = loader.read_stencil(in);
        const auto pout = loader.write(out);
        return [pin, pout] CARBON_HOST_DEVICE(Span s, Idx idx) {
          int acc = 0;
          acc += pin.neighbour(s, idx, carbon::int8_3d{1, 0, 0}, 0).value_or(0);
          acc += pin.neighbour(s, idx, carbon::int8_3d{-1, 0, 0}, 0).value_or(0);
          acc += pin.neighbour(s, idx, carbon::int8_3d{0, 1, 0}, 0).value_or(0);
          acc += pin.neighbour(s, idx, carbon::int8_3d{0, -1, 0}, 0).value_or(0);
          acc += pin.neighbour(s, idx, carbon::int8_3d{0, 0, 1}, 0).value_or(0);
          acc += pin.neighbour(s, idx, carbon::int8_3d{0, 0, -1}, 0).value_or(0);
          pout.write(s, idx, 0, acc);
        };
      },
      "neighbour_sum"));
  try {
    sk.compile();
    sk.run();
    sk.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dense_distributed_dist[{}]: skeleton failed: {}", label, e.what());
    return false;
  }

  {
    int* ho = out.cpu_buffers()[0].data();
    for (size_t i = 0; i < out.cpu_buffers()[0].size(); ++i) {
      ho[i] = kSentinel - 1;
    }
  }
  try {
    carbon::Transfer<int>(sys, out.gpu_buffers(), out.cpu_buffers())(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dense_distributed_dist[{}]: download failed: {}", label, e.what());
    return false;
  }
  if (!sync_all(stream, "stencil download")) {
    return false;
  }

  static constexpr int kDirs[6][3] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                                      {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
  size_t mismatches = 0;
  const int* ho = out.cpu_buffers()[0].data();
  for (int64_t z = 0; z < owned.z; ++z) {
    for (int64_t y = 0; y < owned.y; ++y) {
      for (int64_t x = 0; x < owned.x; ++x) {
        const int64_3d g = span.global_location(span.make_idx(halo.x + x, halo.y + y, halo.z + z));
        int want = 0;
        for (const auto& d : kDirs) {
          int64_t wx = 0;
          int64_t wy = 0;
          int64_t wz = 0;
          if (wrap_axis(g.x + d[0], global.x, carbon::is_periodic(mode, Axis::X), wx) &&
              wrap_axis(g.y + d[1], global.y, carbon::is_periodic(mode, Axis::Y), wy) &&
              wrap_axis(g.z + d[2], global.z, carbon::is_periodic(mode, Axis::Z), wz)) {
            want += encode(wx, wy, wz);
          }
        }
        const int got = ho[(halo.z + z) * pitch + (halo.y + y) * haloed.x + (halo.x + x)];
        if (got != want) {
          if (mismatches < 4U) {
            CARBON_LOG_ERROR(
                "test_dense_distributed_dist[{}]: rank {} global ({},{},{}) got {} want {}", label,
                rank, g.x, g.y, g.z, got, want);
          }
          ++mismatches;
        }
      }
    }
  }
  if (mismatches != 0U) {
    CARBON_LOG_ERROR("test_dense_distributed_dist[{}]: rank {} has {} wrong cells", label, rank,
                     mismatches);
    return false;
  }
  return true;
}

// ------------------------------------------------------------------------------------------------
// Check 3: VTK shard I/O for DenseDistributed (write_vtk / read_vtk / multiblock index).
// ------------------------------------------------------------------------------------------------

[[nodiscard]] auto
run_vtk(carbon::System& sys,
        int rank,
        int world,
        const ProcessGrid& pg,
        BoundaryModeDistributed mode,
        const std::string& label,
        const int64_3d& global) -> bool {
  const std::string base = "carbon_test_dense_distributed_vtk_" + label;
  const std::string shard = base + ".rank" + std::to_string(rank) + ".vti";
  const std::string index = base + ".vtm";
  bool ok = true;
  try {
    DenseDistributed grid(sys, global, pg, six_point(), mode);
    const auto& span = grid.spans()[0];
    const int64_3d halo = span.halo();
    const int64_3d owned = span.logic_extent();
    const int64_3d haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t per_comp = span.linear_storage_elements();
    constexpr int kComps = 3;
    constexpr float kSentinelIn = -1.0F;
    constexpr float kSentinelOut = -2.0F;

    auto f = grid.template newField<float, kComps>(carbon::MemorySide::HostDevice, "pop");
    auto g = grid.template newField<float, kComps>(carbon::MemorySide::HostDevice, "pop_in");
    const auto is_owned_cell = [&](int64_t x, int64_t y, int64_t z) -> bool {
      return x >= halo.x && x < halo.x + owned.x && y >= halo.y && y < halo.y + owned.y &&
             z >= halo.z && z < halo.z + owned.z;
    };
    {
      float* hf = f.cpu_buffers()[0].data();
      float* hg = g.cpu_buffers()[0].data();
      for (int64_t z = 0; z < haloed.z; ++z) {
        for (int64_t y = 0; y < haloed.y; ++y) {
          for (int64_t x = 0; x < haloed.x; ++x) {
            const int64_3d gl = span.global_location(span.make_idx(x, y, z));
            const int64_t off = z * pitch + y * haloed.x + x;
            for (int c = 0; c < kComps; ++c) {
              hf[c * per_comp + off] = is_owned_cell(x, y, z)
                                           ? static_cast<float>(encode(gl.x, gl.y, gl.z, c))
                                           : kSentinelIn;
              hg[c * per_comp + off] = kSentinelOut;
            }
          }
        }
      }
    }
    // host -> device -> host, so the field written is what a real run would download
    carbon::Stream stream = sys.stream();
    carbon::Transfer<float>(sys, f.cpu_buffers(), f.gpu_buffers())(stream);
    if (!sync_all(stream, "vtk upload")) {
      return false;
    }
    {
      float* hf = f.cpu_buffers()[0].data();
      for (size_t i = 0; i < f.cpu_buffers()[0].size(); ++i) {
        hf[i] = 0.0F;
      }
    }
    carbon::Transfer<float>(sys, f.gpu_buffers(), f.cpu_buffers())(stream);
    if (!sync_all(stream, "vtk download")) {
      return false;
    }

    carbon::io::write_vtk(base, grid, f);
    carbon::io::write_vtk_multiblock_index(base, grid);

    // --- this rank's shard header must describe its owned box ----------------------------------
    {
      std::ifstream in(shard, std::ios::binary);
      std::string line;
      int64_t whole[6] = {-1, -1, -1, -1, -1, -1};
      int64_t org[3] = {-1, -1, -1};
      while (std::getline(in, line)) {
        if (line.find("<ImageData") != std::string::npos) {
          const std::string extent = xml_attribute(line, "WholeExtent");
          std::istringstream es(extent);
          es >> whole[0] >> whole[1] >> whole[2] >> whole[3] >> whole[4] >> whole[5];
          const std::string origin = xml_attribute(line, "Origin");
          std::istringstream os(origin);
          os >> org[0] >> org[1] >> org[2];
          break;
        }
      }
      const int64_3d o0 = span.global_location(span.make_idx(halo.x, halo.y, halo.z));
      if (whole[0] != 0 || whole[1] != owned.x - 1 || whole[2] != 0 ||
          whole[3] != owned.y - 1 || whole[4] != 0 || whole[5] != owned.z - 1 ||
          org[0] != o0.x || org[1] != o0.y || org[2] != o0.z) {
        CARBON_LOG_ERROR(
            "test_dense_distributed_dist[{}]: rank {} shard WholeExtent=({},{},{},{},{},{}) "
            "Origin=({},{},{}) want owned=({},{},{}) origin=({},{},{})",
            label, rank, whole[0], whole[1], whole[2], whole[3], whole[4], whole[5], org[0],
            org[1], org[2], owned.x, owned.y, owned.z, o0.x, o0.y, o0.z);
        ok = false;
      }
      std::ifstream raw(shard, std::ios::binary);
      const std::string text((std::istreambuf_iterator<char>(raw)),
                             std::istreambuf_iterator<char>());
      if (text.find("format=\"binary\"") == std::string::npos ||
          text.find("<AppendedData") != std::string::npos) {
        CARBON_LOG_ERROR(
            "test_dense_distributed_dist[{}]: rank {} shard is not inline-binary VTK XML", label,
            rank);
        ok = false;
      }
    }

    // --- rank 0's index lists every rank (all own cells for this extent) -------------------------
    if (rank == 0) {
      std::ifstream in(index);
      const std::string text((std::istreambuf_iterator<char>(in)),
                             std::istreambuf_iterator<char>());
      if (text.find("vtkMultiBlockDataSet") == std::string::npos) {
        CARBON_LOG_ERROR("test_dense_distributed_dist[{}]: missing/invalid {}", label, index);
        ok = false;
      }
      for (int r = 0; r < world; ++r) {
        if (text.find("name=\"rank" + std::to_string(r) + "\"") == std::string::npos ||
            text.find(".rank" + std::to_string(r) + ".vti") == std::string::npos) {
          CARBON_LOG_ERROR("test_dense_distributed_dist[{}]: index does not list rank {}", label,
                           r);
          ok = false;
        }
      }
    }

    // --- read back into a fresh field: owned == written, halo untouched ----------------------------
    carbon::io::read_vtk(base, grid, g);
    size_t mismatches = 0;
    const float* hg = g.cpu_buffers()[0].data();
    for (int64_t z = 0; z < haloed.z; ++z) {
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_3d gl = span.global_location(span.make_idx(x, y, z));
          const int64_t off = z * pitch + y * haloed.x + x;
          for (int c = 0; c < kComps; ++c) {
            const float want = is_owned_cell(x, y, z)
                                   ? static_cast<float>(encode(gl.x, gl.y, gl.z, c))
                                   : kSentinelOut;
            if (hg[c * per_comp + off] != want) {
              ++mismatches;
            }
          }
        }
      }
    }
    if (mismatches != 0U) {
      CARBON_LOG_ERROR(
          "test_dense_distributed_dist[{}]: rank {} VTK round trip has {} wrong values", label,
          rank, mismatches);
      ok = false;
    }
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dense_distributed_dist[{}]: VTK I/O failed: {}", label, e.what());
    ok = false;
  }
  std::error_code ec;
  std::filesystem::remove(shard, ec);
  if (rank == 0) {
    std::filesystem::remove(index, ec);
  }
  return ok;
}

struct Case {
  ProcessGrid pg;
  BoundaryModeDistributed mode;
  const char* label;
  bool multi_component;  // also run the 3-component (packed-Z) exchange
};

}  // namespace

int
main() {
  carbon::init();

  auto distributed = std::make_shared<const carbon::Distributed>();
  const int world = distributed->world_size();
  if (world != 4 && world != 8) {
    CARBON_LOG_INFO("test_dense_distributed_dist: world_size={} not in {{4, 8}}, skipping", world);
    return 0;
  }
  if (carbon::device_count() < world) {
    CARBON_LOG_INFO("test_dense_distributed_dist: needs >= {} visible devices, have {}; skipping",
                    world, carbon::device_count());
    return 0;
  }
  const int rank = distributed->world_rank();
  carbon::System sys(distributed);
  if (sys.gpu_group().size() != 1U) {
    CARBON_LOG_ERROR("test_dense_distributed_dist: expected one GPU per rank, got {}",
                     sys.gpu_group().size());
    return 1;
  }

  constexpr auto kO = BoundaryModeDistributed::Open;
  constexpr auto kX = BoundaryModeDistributed::PeriodicX;
  constexpr auto kY = BoundaryModeDistributed::PeriodicY;
  constexpr auto kZ = BoundaryModeDistributed::PeriodicZ;
  const auto custom = [world](int x, int y, int z) {
    return ProcessGrid::custom(world, x, y, z);
  };

  std::vector<Case> cases;
  if (world == 4) {
    cases = {
        {custom(1, 1, 4), kO, "slab-open", true},
        {custom(1, 1, 4), kZ, "slab-periodicZ", false},
        {custom(1, 1, 4), kX, "slab-periodicX", false},
        {custom(1, 2, 2), kO, "pencil-open", true},
        {custom(1, 2, 2), kY | kZ, "pencil-periodicYZ", false},
        {custom(1, 2, 2), kX, "pencil-periodicX", false},
        {custom(2, 2, 1), kO, "brickXY-open", false},
        {custom(2, 2, 1), kX | kY, "brickXY-periodicXY", true},
        {custom(2, 1, 2), kO, "brickXZ-open", false},
        {custom(2, 1, 2), kX | kZ, "brickXZ-periodicXZ", false},
    };
  } else {
    cases = {
        {custom(2, 2, 2), kO, "brick-open", true},
        {custom(2, 2, 2), kX, "brick-periodicX", false},
        {custom(2, 2, 2), kX | kY | kZ, "brick-periodicXYZ", true},
    };
  }

  // Deliberately uneven extents so chunks differ in size across ranks.
  const int64_3d global{9, 8, 11};
  for (const Case& c : cases) {
    if (!run_halo<1>(sys, rank, c.pg, c.mode, c.label, global) ||
        (c.multi_component && !run_halo<3>(sys, rank, c.pg, c.mode, c.label, global)) ||
        !run_halo<6>(sys, rank, c.pg, c.mode, std::string(c.label) + "/lattice", global,
                     carbon::StencilSemantic::LatticeDistributed) ||
        !run_stencil(sys, rank, c.pg, c.mode, carbon::OverlapMode::None,
                     std::string(c.label) + "/stencil", global) ||
        !run_stencil(sys, rank, c.pg, c.mode, carbon::OverlapMode::Standard,
                     std::string(c.label) + "/stencil-overlap", global) ||
        !run_vtk(sys, rank, world, c.pg, c.mode, c.label, global)) {
      return 1;
    }
  }

  CARBON_LOG_INFO("test_dense_distributed_dist: rank {} of {} ok ({} cases)", rank, world,
                  cases.size());
  return 0;
}
