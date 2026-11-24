// DenseDisgDistributed on real GPUs, one rank per GPU: slab, pencil and brick process grids end to
// end.
//
// Launched under `mpiexec -n 4` or `mpiexec -n 8 ./test_dense_disg_distributed_dist`. Any other
// world size skips (returns 0), as does a node with fewer visible devices than ranks.
//
// Process grids (Px, Py, Pz) on 4 ranks: pencil (1,2,2) open and periodic YZ, y-only pencil
// (1,4,1) open and periodic Y, slab (1,1,4) open and periodic Z, bricks (2,2,1) open and periodic
// YZ (z wraps onto the rank itself), (2,1,2) open and periodic Z, (4,1,1) open. On 8 ranks: brick
// (2,2,2) open and periodic YZ, (4,2,1) open, (2,1,4) periodic Z, pencil (1,2,4), slab (1,1,8).
//
// Three checks per case:
//
//  1. Value parity with DenseDistributed on the same ProcessGrid: both fields are seeded with
//     fill_owned_host and every owned cell is compared by global coordinate (values, not layout).
//
//  2. Raw HaloDisgDistributed exchange (one NCCL group of face + edge (+ corner) messages, at most
//     one per direction: 26 send + 26 receive entries on a brick, 18 + 18 under D3Q19). Owned cells hold
//     encode(global cell), every other storage element a sentinel; after one exchange every cell
//     of the buffer is compared: owned unchanged, ghosts on an open edge still the sentinel, every
//     other ghost == encode(the wrapped global cell it mirrors). Standard semantic with 1 and 3
//     components, then D3Q19 under Lattice and LatticeDistributed, where only the populations
//     crossing into a ghost are required.
//
//  3. Skeleton + Kernel. A kernel summing the six face neighbours, the four yz-diagonal, two xy-
//     and two xz-diagonal neighbours and two corners (0 outside an open boundary) runs with the
//     auto-inserted halo, once without overlap and once with OverlapMode::Standard (Internal +
//     Boundary launches).

#include <carbon/boundary_mode_distributed.h>
#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense_disg_distributed.h>
#include <carbon/dense_distributed.h>
#include <carbon/distributed.h>
#include <carbon/halo_disg_distributed.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/process_grid.h>
#include <carbon/skeleton.h>
#include <carbon/stencil.h>
#include <carbon/system.h>
#include <carbon/transfer.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace {

using carbon::Axis;
using carbon::BoundaryModeDistributed;
using carbon::DenseDisgDistributed;
using carbon::int32_3d;
using carbon::int64_3d;
using carbon::ProcessGrid;
using carbon::StencilSemantic;

constexpr int kSentinel = -7777;

[[nodiscard]] auto
encode(const int64_3d& g,
       int comp = 0) noexcept -> int {
  return static_cast<int>(g.z * 10000 + g.y * 100 + g.x) + comp * 1'000'000;
}

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

[[nodiscard]] auto
wrap_global(const int64_3d& g,
            const int64_3d& global,
            BoundaryModeDistributed mode,
            int64_3d& out) noexcept -> bool {
  return wrap_axis(g.x, global.x, carbon::is_periodic(mode, Axis::X), out.x) &&
         wrap_axis(g.y, global.y, carbon::is_periodic(mode, Axis::Y), out.y) &&
         wrap_axis(g.z, global.z, carbon::is_periodic(mode, Axis::Z), out.z);
}

[[nodiscard]] auto
sync_all(const carbon::Stream& stream,
         const char* tag) -> bool {
  try {
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dense_disg_distributed_dist: sync after {} failed: {}", tag, e.what());
    return false;
  }
  return true;
}

[[nodiscard]] auto
neighbour_points() -> std::vector<int32_3d> {
  return {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},  {0, -1, 0},  {0, 0, 1}, {0, 0, -1},
          {0, 1, 1},  {0, 1, -1}, {0, -1, 1}, {0, -1, -1}, {1, 1, 0}, {-1, -1, 0},
          {1, 0, -1}, {-1, 0, 1}, {1, 1, 1},  {-1, -1, -1}};
}

[[nodiscard]] auto
d3q19() -> carbon::Stencil {
  std::vector<int32_3d> pts{{0, 0, 0}};
  for (int z = -1; z <= 1; ++z) {
    for (int y = -1; y <= 1; ++y) {
      for (int x = -1; x <= 1; ++x) {
        const int nz = (x != 0) + (y != 0) + (z != 0);
        if (nz >= 1 && nz <= 2) {
          pts.push_back({x, y, z});
        }
      }
    }
  }
  return carbon::Stencil(pts);
}

[[nodiscard]] auto
crosses(StencilSemantic semantic,
        int32_t sign,
        int32_t c) noexcept -> bool {
  if (semantic == StencilSemantic::LatticeDistributed) {
    return c != 0;
  }
  return sign > 0 ? c > 0 : c < 0;
}

// ------------------------------------------------------------------------------------------------
// Check 1: owned-cell value parity with DenseDistributed.
// ------------------------------------------------------------------------------------------------

[[nodiscard]] auto
run_value_parity(carbon::System& sys,
                 int rank,
                 const ProcessGrid& pg,
                 BoundaryModeDistributed mode,
                 const std::string& label,
                 const int64_3d& global) -> bool {
  const carbon::Stencil stencil(neighbour_points());
  const DenseDisgDistributed grid(sys, global, pg, stencil, mode);
  const carbon::DenseDistributed dense(sys, global, pg, stencil, mode);
  if (grid.gpu_count() != 1U || grid.local_rank_offset() != rank ||
      grid.local_slab_offset() != static_cast<size_t>(pg.coords_of(rank).z)) {
    CARBON_LOG_ERROR("test_dense_disg_distributed_dist[{}]: rank {} bad grid bookkeeping", label,
                     rank);
    return false;
  }
  const bool brick_storage = pg.dims().x > 1;
  const bool pencil_storage =
      brick_storage || pg.dims().y > 1 || carbon::is_periodic(mode, Axis::Y);
  if (grid.halo_y() != (pencil_storage ? 1 : 0) ||
      grid.spans()[0].ghost_y() != (pencil_storage ? 1 : 0) ||
      grid.halo_x() != (brick_storage ? 1 : 0) ||
      grid.spans()[0].ghost_x() != (brick_storage ? 1 : 0)) {
    CARBON_LOG_ERROR(
        "test_dense_disg_distributed_dist[{}]: rank {} halo_y={} ghost_y={} halo_x={} ghost_x={}",
        label, rank, grid.halo_y(), grid.spans()[0].ghost_y(), grid.halo_x(),
        grid.spans()[0].ghost_x());
    return false;
  }

  auto fa = grid.newField<float, 3>(carbon::MemorySide::Host, "a");
  auto fb = dense.newField<float, 3>(carbon::MemorySide::Host, "b");
  const auto fn = [](int64_3d g, int c) -> float {
    return static_cast<float>(encode(g, c));
  };
  fa.fill_owned_host(fn);
  fb.fill_owned_host(fn);

  const auto& sa = grid.spans()[0];
  const auto& sb = dense.spans()[0];
  const int64_3d n = sa.logic_extent();
  const int64_3d nb = sb.logic_extent();
  if (n.x != nb.x || n.y != nb.y || n.z != nb.z) {
    CARBON_LOG_ERROR(
        "test_dense_disg_distributed_dist[{}]: rank {} owned ({},{},{}) vs dense "
        "({},{},{})",
        label, rank, n.x, n.y, n.z, nb.x, nb.y, nb.z);
    return false;
  }
  const int64_3d hb = sb.halo();
  size_t mismatches = 0;
  for (int64_t z = 0; z < n.z; ++z) {
    for (int64_t y = 0; y < n.y; ++y) {
      for (int64_t x = 0; x < n.x; ++x) {
        const auto ia = sa.set_and_validate(carbon::DataView::Standard, x, y, z).idx;
        const auto ib = sb.make_idx(hb.x + x, hb.y + y, hb.z + z);
        const int64_3d ga = sa.global_location(ia);
        const int64_3d gb = sb.global_location(ib);
        if (ga.x != gb.x || ga.y != gb.y || ga.z != gb.z) {
          ++mismatches;
          continue;
        }
        for (int c = 0; c < 3; ++c) {
          if (fa.cpu_buffers()[0].data()[sa.linear_offset(ia, 3, c)] !=
              fb.cpu_buffers()[0].data()[sb.linear_offset(ib, c)]) {
            ++mismatches;
          }
        }
      }
    }
  }
  if (mismatches != 0U) {
    CARBON_LOG_ERROR(
        "test_dense_disg_distributed_dist[{}]: rank {} has {} owned-value mismatches "
        "against DenseDistributed",
        label, rank, mismatches);
    return false;
  }
  return true;
}

// ------------------------------------------------------------------------------------------------
// Check 2: raw HaloDisgDistributed exchange, every buffer cell compared with ground truth.
// ------------------------------------------------------------------------------------------------

template <int Components>
[[nodiscard]] auto
run_halo(carbon::System& sys,
         int rank,
         const ProcessGrid& pg,
         BoundaryModeDistributed mode,
         const std::string& label,
         const int64_3d& global,
         StencilSemantic semantic,
         const carbon::Stencil& stencil) -> bool {
  const DenseDisgDistributed grid(sys, global, pg, carbon::Stencil(neighbour_points()), mode);
  const auto& span = grid.spans()[0];
  const int64_3d n = span.logic_extent();
  const int64_3d h = span.haloed_extent();
  const int64_t gx = span.ghost_x();
  const int64_t gy = span.ghost_y();
  const int32_3d dims = pg.dims();
  const bool self_wrap = (dims.y == 1 && carbon::is_periodic(mode, Axis::Y) && gy != 0) ||
                         (dims.z == 1 && carbon::is_periodic(mode, Axis::Z));
  const size_t max_entries = gx != 0 ? (semantic == StencilSemantic::Standard ? 52U : 36U) : 20U;

  auto field = grid.newField<int, Components>(carbon::MemorySide::HostDevice, "halo");
  int* host = field.cpu_buffers()[0].data();
  std::fill_n(host, field.cpu_buffers()[0].size(), kSentinel);
  field.fill_owned_host([](int64_3d g, int c) -> int {
    return encode(g, c);
  });

  carbon::Stream stream = sys.stream();
  try {
    carbon::Transfer<int>(sys, field.cpu_buffers(), field.gpu_buffers())(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dense_disg_distributed_dist[{}]: upload failed: {}", label, e.what());
    return false;
  }
  if (!sync_all(stream, "upload")) {
    return false;
  }

  try {
    const carbon::HaloDisgDistributed<int, Components> exchange(field, stencil, semantic);
    if ((exchange.pair_count() != 0U && !self_wrap) || exchange.dist_entry_count() == 0U ||
        exchange.dist_entry_count() > max_entries) {
      CARBON_LOG_ERROR(
          "test_dense_disg_distributed_dist[{}]: pair_count={} dist_entry_count={} (want 0 "
          "pairs unless an axis wraps onto the rank, and 1..{} entries)",
          label, exchange.pair_count(), exchange.dist_entry_count(), max_entries);
      return false;
    }
    exchange(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dense_disg_distributed_dist[{}]: halo exchange failed: {}", label,
                     e.what());
    return false;
  }
  if (!sync_all(stream, "halo")) {
    return false;
  }

  std::fill_n(host, field.cpu_buffers()[0].size(), kSentinel - 1);
  try {
    carbon::Transfer<int>(sys, field.gpu_buffers(), field.cpu_buffers())(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dense_disg_distributed_dist[{}]: download failed: {}", label, e.what());
    return false;
  }
  if (!sync_all(stream, "download")) {
    return false;
  }

  size_t mismatches = 0;
  for (int64_t bz = 0; bz < h.z; ++bz) {
    for (int64_t by = 0; by < h.y; ++by) {
      for (int64_t bx = 0; bx < h.x; ++bx) {
        const auto idx = span.make_idx(bx, by, bz);
        const int64_3d g = span.global_location(idx);
        const bool ghost_x = bx < gx || bx >= gx + n.x;
        const bool ghost_y = by < gy || by >= gy + n.y;
        const bool ghost_z = bz < 1 || bz > n.z;
        int64_3d w{};
        const bool inside = wrap_global(g, global, mode, w);
        for (int c = 0; c < Components; ++c) {
          int want = kSentinel;
          if (!ghost_x && !ghost_y && !ghost_z) {
            want = encode(g, c);
          } else if (inside) {
            bool required = true;
            if (semantic != StencilSemantic::Standard) {
              const auto p = stencil.points()[static_cast<size_t>(c)];
              if (ghost_x) {
                required = required && crosses(semantic, bx < gx ? 1 : -1, p.x);
              }
              if (ghost_y) {
                required = required && crosses(semantic, by < gy ? 1 : -1, p.y);
              }
              if (ghost_z) {
                required = required && crosses(semantic, bz < 1 ? 1 : -1, p.z);
              }
            }
            if (!required) {
              continue;
            }
            want = encode(w, c);
          }
          const int got = host[span.linear_offset(idx, Components, c)];
          if (got != want) {
            if (mismatches < 4U) {
              CARBON_LOG_ERROR(
                  "test_dense_disg_distributed_dist[{}]: rank {} comp {} buffer ({},{},{}) global "
                  "({},{},{}) got {} want {}",
                  label, rank, c, bx, by, bz, g.x, g.y, g.z, got, want);
            }
            ++mismatches;
          }
        }
      }
    }
  }
  if (mismatches != 0U) {
    CARBON_LOG_ERROR("test_dense_disg_distributed_dist[{}]: rank {} has {} wrong cells", label,
                     rank, mismatches);
    return false;
  }
  return true;
}

// ------------------------------------------------------------------------------------------------
// Check 3: Skeleton + neighbour-read kernel (auto-inserted halo, optional overlap).
// ------------------------------------------------------------------------------------------------

[[nodiscard]] auto
run_stencil(carbon::System& sys,
            int rank,
            const ProcessGrid& pg,
            BoundaryModeDistributed mode,
            carbon::OverlapMode overlap,
            const std::string& label,
            const int64_3d& global) -> bool {
  const DenseDisgDistributed grid(sys, global, pg, carbon::Stencil(neighbour_points()), mode);
  const auto& span = grid.spans()[0];
  const int64_3d n = span.logic_extent();
  const int64_t gx = span.ghost_x();
  const int64_t gy = span.ghost_y();

  auto in = grid.newField<int, 1>(carbon::MemorySide::HostDevice, "in");
  auto out = grid.newField<int, 1>(carbon::MemorySide::HostDevice, "out");
  std::fill_n(in.cpu_buffers()[0].data(), in.cpu_buffers()[0].size(), kSentinel);
  std::fill_n(out.cpu_buffers()[0].data(), out.cpu_buffers()[0].size(), kSentinel);
  in.fill_owned_host([](int64_3d g, int) -> int {
    return encode(g);
  });

  carbon::Stream stream = sys.stream();
  try {
    carbon::Transfer<int>(sys, in.cpu_buffers(), in.gpu_buffers())(stream);
    carbon::Transfer<int>(sys, out.cpu_buffers(), out.gpu_buffers())(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dense_disg_distributed_dist[{}]: upload failed: {}", label, e.what());
    return false;
  }
  if (!sync_all(stream, "stencil upload")) {
    return false;
  }

  using Span = DenseDisgDistributed::Span;
  using Idx = DenseDisgDistributed::Idx;
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
          acc += pin.neighbour(s, idx, carbon::int8_3d{0, 1, 1}, 0).value_or(0);
          acc += pin.neighbour(s, idx, carbon::int8_3d{0, 1, -1}, 0).value_or(0);
          acc += pin.neighbour(s, idx, carbon::int8_3d{0, -1, 1}, 0).value_or(0);
          acc += pin.neighbour(s, idx, carbon::int8_3d{0, -1, -1}, 0).value_or(0);
          acc += pin.neighbour(s, idx, carbon::int8_3d{1, 1, 0}, 0).value_or(0);
          acc += pin.neighbour(s, idx, carbon::int8_3d{-1, -1, 0}, 0).value_or(0);
          acc += pin.neighbour(s, idx, carbon::int8_3d{1, 0, -1}, 0).value_or(0);
          acc += pin.neighbour(s, idx, carbon::int8_3d{-1, 0, 1}, 0).value_or(0);
          acc += pin.neighbour(s, idx, carbon::int8_3d{1, 1, 1}, 0).value_or(0);
          acc += pin.neighbour(s, idx, carbon::int8_3d{-1, -1, -1}, 0).value_or(0);
          pout.write(s, idx, 0, acc);
        };
      },
      "neighbour_sum"));
  try {
    sk.compile();
    sk.run();
    sk.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dense_disg_distributed_dist[{}]: skeleton failed: {}", label, e.what());
    return false;
  }

  std::fill_n(out.cpu_buffers()[0].data(), out.cpu_buffers()[0].size(), kSentinel - 1);
  try {
    carbon::Transfer<int>(sys, out.gpu_buffers(), out.cpu_buffers())(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dense_disg_distributed_dist[{}]: download failed: {}", label, e.what());
    return false;
  }
  if (!sync_all(stream, "stencil download")) {
    return false;
  }

  const auto points = neighbour_points();
  size_t mismatches = 0;
  const int* ho = out.cpu_buffers()[0].data();
  for (int64_t z = 0; z < n.z; ++z) {
    for (int64_t y = 0; y < n.y; ++y) {
      for (int64_t x = 0; x < n.x; ++x) {
        const auto idx = span.make_idx(gx + x, gy + y, 1 + z);
        const int64_3d g = span.global_location(idx);
        int want = 0;
        for (const int32_3d& d : points) {
          int64_3d w{};
          if (wrap_global({g.x + d.x, g.y + d.y, g.z + d.z}, global, mode, w)) {
            want += encode(w);
          }
        }
        const int got = ho[span.linear_offset(idx, 1, 0)];
        if (got != want) {
          if (mismatches < 4U) {
            CARBON_LOG_ERROR(
                "test_dense_disg_distributed_dist[{}]: rank {} global ({},{},{}) got {} want {}",
                label, rank, g.x, g.y, g.z, got, want);
          }
          ++mismatches;
        }
      }
    }
  }
  if (mismatches != 0U) {
    CARBON_LOG_ERROR("test_dense_disg_distributed_dist[{}]: rank {} has {} wrong cells", label,
                     rank, mismatches);
    return false;
  }
  return true;
}

struct Case {
  ProcessGrid pg;
  BoundaryModeDistributed mode;
  const char* label;
};

}  // namespace

int
main() {
  carbon::init();

  auto distributed = std::make_shared<const carbon::Distributed>();
  const int world = distributed->world_size();
  if (world != 4 && world != 8) {
    CARBON_LOG_INFO("test_dense_disg_distributed_dist: world_size={} is not 4 or 8, skipping",
                    world);
    return 0;
  }
  if (carbon::device_count() < world) {
    CARBON_LOG_INFO(
        "test_dense_disg_distributed_dist: needs >= {} visible devices, have {}; skipping", world,
        carbon::device_count());
    return 0;
  }
  const int rank = distributed->world_rank();
  carbon::System sys(distributed);
  if (sys.gpu_group().size() != 1U) {
    CARBON_LOG_ERROR("test_dense_disg_distributed_dist: expected one GPU per rank, got {}",
                     sys.gpu_group().size());
    return 1;
  }

  constexpr auto kO = BoundaryModeDistributed::Open;
  constexpr auto kY = BoundaryModeDistributed::PeriodicY;
  constexpr auto kZ = BoundaryModeDistributed::PeriodicZ;
  const auto custom = [world](int x, int y, int z) {
    return ProcessGrid::custom(world, x, y, z);
  };
  const std::vector<Case> cases =
      world == 4 ? std::vector<Case>{
                       {custom(1, 2, 2), kO, "pencil-open"},
                       {custom(1, 2, 2), kY | kZ, "pencil-periodicYZ"},
                       {custom(1, 4, 1), kO, "pencilY-open"},
                       {custom(1, 4, 1), kY, "pencilY-periodicY"},
                       {custom(1, 1, 4), kO, "slab-open"},
                       {custom(1, 1, 4), kZ, "slab-periodicZ"},
                       {custom(2, 2, 1), kO, "brick221-open"},
                       {custom(2, 2, 1), kY | kZ, "brick221-periodicYZ"},
                       {custom(2, 1, 2), kO, "brick212-open"},
                       {custom(2, 1, 2), kZ, "brick212-periodicZ"},
                       {custom(4, 1, 1), kO, "brick411-open"},
                   }
                 : std::vector<Case>{
                       {custom(2, 2, 2), kO, "brick222-open"},
                       {custom(2, 2, 2), kY | kZ, "brick222-periodicYZ"},
                       {custom(4, 2, 1), kO, "brick421-open"},
                       {custom(2, 1, 4), kZ, "brick214-periodicZ"},
                       {custom(1, 2, 4), kO, "pencil124-open"},
                       {custom(1, 1, 8), kZ, "slab118-periodicZ"},
                   };

  const int64_3d global = world == 4 ? int64_3d{9, 8, 11} : int64_3d{12, 10, 18};
  const carbon::Stencil q19 = d3q19();
  const carbon::Stencil faces(neighbour_points());
  for (const Case& c : cases) {
    const std::string label(c.label);
    if (!run_value_parity(sys, rank, c.pg, c.mode, label, global) ||
        !run_halo<1>(sys, rank, c.pg, c.mode, label + "/c1", global, StencilSemantic::Standard,
                     faces) ||
        !run_halo<3>(sys, rank, c.pg, c.mode, label + "/c3", global, StencilSemantic::Standard,
                     faces) ||
        !run_halo<19>(sys, rank, c.pg, c.mode, label + "/d3q19", global, StencilSemantic::Lattice,
                      q19) ||
        !run_halo<19>(sys, rank, c.pg, c.mode, label + "/d3q19-dist", global,
                      StencilSemantic::LatticeDistributed, q19) ||
        !run_stencil(sys, rank, c.pg, c.mode, carbon::OverlapMode::None, label + "/stencil",
                     global) ||
        !run_stencil(sys, rank, c.pg, c.mode, carbon::OverlapMode::Standard,
                     label + "/stencil-overlap", global)) {
      return 1;
    }
  }

  CARBON_LOG_INFO("test_dense_disg_distributed_dist: rank {} of {} ok ({} cases)", rank, world,
                  cases.size());
  return 0;
}
