// Host-only verification of DenseDistributed's decomposition and its packed halo exchange.
// No GPU, no MPI.
//
// Part A  per-thread pack/unpack index math (the exact code the CUDA/HIP kernels run).
// Part B  slab / pencil / brick geometry: for every rank of several process grids, the
//         production layout functions (detail/dense_distributed/layout.h) are compared with
//         hand-written expectations, the owned boxes must tile the global domain exactly once, and
//         the resulting Span must iterate (Standard, Internal + Boundary) and resolve
//         neighbours (open vs periodic) correctly.
// Part D  VTK shard I/O (io::detail stream-level writer/reader behind write_vtk / read_vtk for
//         DenseDistributed): byte-level check of the legacy BINARY payload with an independent
//         decoder, library round trip that must leave halo cells untouched, shard boxes that tile
//         the domain exactly once, and the placeholder shard of a rank that owns no cells.
// Part C  correctness of the full X -> Y -> Z staged halo exchange on an emulated multi-rank world
//         (corners/edges, periodic wraps, uneven splits, self-wrap), driven by the production
//         plan builder and checked against the run-based builder (kept as the oracle) and against
//         ground truth. Also covers `LatticeDistributed` face-interior halo components in both
//         directions, so the brick/pencil wall-bounce-back fix is exercised without GPUs.
//
// What it does NOT cover: stream ordering, NCCL/MPI transport and real kernel launches. That is the
// job of the dense_distributed_dist test, which runs the real exchange on GPUs.

#include <carbon/boundary_mode_distributed.h>
#include <carbon/detail/dense_distributed/layout.h>
#include <carbon/detail/dense_distributed/span.h>
#include <carbon/detail/halo_distributed_build.h>
#include <carbon/detail/halo_pack_plan.h>
#include <carbon/io.h>
#include <carbon/logging.h>
#include <carbon/process_grid.h>
#include <carbon/stencil.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using carbon::Axis;
using carbon::BoundaryModeDistributed;
using carbon::int32_3d;
using carbon::int64_3d;
using carbon::int8_3d;
using carbon::ProcessGrid;
namespace hb = carbon::detail::halo_distributed_build;
namespace dd = carbon::detail::dense_distributed;
namespace hp = carbon::detail::halo_pack;
using hb::AxisGeometry;
using hb::AxisSide;

#define CHECK(cond)                                                                       \
  do {                                                                                    \
    if (!(cond)) {                                                                        \
      std::fprintf(stderr, "CHECK failed: %s  (%s:%d)\n  context: %s\n", #cond, __FILE__, \
                   __LINE__, g_ctx.c_str());                                              \
      std::exit(1);                                                                       \
    }                                                                                     \
  } while (0)

std::string g_ctx;

constexpr uint64_t kSentinel = 0xDEADBEEFDEADBEEFULL;
constexpr uint64_t kMarker = 0xB0C0000000000000ULL;

struct Vel {
  int x, y, z;
};

auto
lattice(int q) -> std::vector<Vel> {
  std::vector<Vel> v;
  for (int z = -1; z <= 1; ++z) {
    for (int y = -1; y <= 1; ++y) {
      for (int x = -1; x <= 1; ++x) {
        const int nz = (x != 0) + (y != 0) + (z != 0);
        if (q == 27 || (q == 19 && nz <= 2)) {
          v.push_back({x, y, z});
        }
      }
    }
  }
  return v;
}

auto
axis_of(const Vel& v,
        Axis a) -> int {
  return a == Axis::X ? v.x : (a == Axis::Y ? v.y : v.z);
}

auto
axis_len(const int64_3d& v,
         Axis a) -> int64_t {
  return a == Axis::X ? v.x : (a == Axis::Y ? v.y : v.z);
}

auto
split(int64_t n,
      int parts,
      int idx) -> std::pair<int64_t,
                            int64_t> {  // (size, offset)
  const int64_t base = n / parts;
  const int64_t rem = n % parts;
  return {base + (idx < rem ? 1 : 0), idx * base + std::min<int64_t>(idx, rem)};
}

struct Rank {
  int32_3d coords{};
  int64_3d offset{};  // global index of local owned cell (0,0,0)
  AxisGeometry g{};
  int64_t comp_stride{};
  std::vector<uint64_t> buf;
  std::vector<uint64_t> stage;
};

enum class Semantic { Standard, Lattice, LatticeDistributed };

struct World {
  ProcessGrid pg = ProcessGrid::custom(1, 1, 1, 1);
  BoundaryModeDistributed bm{BoundaryModeDistributed::Open};
  int64_3d global{};
  int64_3d halo{};
  int comps{1};
  Semantic semantic{Semantic::Standard};
  std::vector<Vel> vel;
  std::vector<Rank> ranks;

  [[nodiscard]] auto
  active_send(int c,
              Axis axis,
              AxisSide side) const -> bool {
    if (semantic == Semantic::Standard) {
      return true;
    }
    const int v = axis_of(vel[static_cast<size_t>(c)], axis);
    if (semantic == Semantic::LatticeDistributed) {
      return v != 0;
    }
    return side == AxisSide::Upper ? v > 0 : v < 0;
  }

  [[nodiscard]] auto
  active_recv(int c,
              Axis axis,
              AxisSide side) const -> bool {
    if (semantic == Semantic::Standard) {
      return true;
    }
    const int v = axis_of(vel[static_cast<size_t>(c)], axis);
    if (semantic == Semantic::LatticeDistributed) {
      return v != 0;
    }
    return side == AxisSide::Upper ? v < 0 : v > 0;
  }
};

auto
encode(int64_t gx,
       int64_t gy,
       int64_t gz,
       int c) -> uint64_t {
  return ((static_cast<uint64_t>(c) * 1009U + static_cast<uint64_t>(gz)) * 997U +
          static_cast<uint64_t>(gy)) *
             991U +
         static_cast<uint64_t>(gx) + 1U;
}

// A stencil whose per-axis radius is exactly @p halo (axes with 0 get no point).
auto
stencil_from_halo(const int64_3d& halo) -> carbon::Stencil {
  std::vector<int32_3d> pts;
  const auto add = [&](int64_t r, int32_3d unit) {
    if (r > 0) {
      const auto k = static_cast<int32_t>(r);
      pts.push_back({unit.x * k, unit.y * k, unit.z * k});
      pts.push_back({-unit.x * k, -unit.y * k, -unit.z * k});
    }
  };
  add(halo.x, {1, 0, 0});
  add(halo.y, {0, 1, 0});
  add(halo.z, {0, 0, 1});
  return carbon::Stencil(std::move(pts), /*remove_center=*/true);
}

auto
make_world(const ProcessGrid& pg,
           BoundaryModeDistributed bm,
           int64_3d global,
           int64_3d halo,
           int comps,
           int lattice_q,
           bool lattice_distributed,
           int64_t pitch_pad) -> World {
  World w;
  w.pg = pg;
  w.bm = bm;
  w.global = global;
  w.halo = halo;
  w.semantic = lattice_q == 0
                   ? Semantic::Standard
                   : (lattice_distributed ? Semantic::LatticeDistributed : Semantic::Lattice);
  w.vel = lattice(lattice_q);
  w.comps = w.semantic == Semantic::Standard ? comps : static_cast<int>(w.vel.size());
  const int64_3d hh = dd::halo_extent(pg, bm, stencil_from_halo(halo));
  for (int32_t r = 0; r < pg.world_size(); ++r) {
    Rank rk;
    rk.coords = pg.coords_of(r);
    // Geometry comes from the production layout functions, so Part C exercises the real
    // decomposition (not a second copy of it).
    const dd::RankChunk chunk = dd::rank_chunk(global, pg, rk.coords, hh);
    rk.offset = {chunk.offset.x + hh.x, chunk.offset.y + hh.y, chunk.offset.z + hh.z};
    const auto sx = std::pair<int64_t, int64_t>{chunk.owned.x, rk.offset.x};
    const auto sy = std::pair<int64_t, int64_t>{chunk.owned.y, rk.offset.y};
    const auto sz = std::pair<int64_t, int64_t>{chunk.owned.z, rk.offset.z};
    rk.g.owned = chunk.owned;
    rk.g.halo = hh;
    rk.g.haloed = chunk.haloed;
    rk.g.pitch = rk.g.haloed.x * rk.g.haloed.y + pitch_pad;
    rk.comp_stride = rk.g.pitch * rk.g.haloed.z;
    rk.buf.assign(static_cast<size_t>(rk.comp_stride) * static_cast<size_t>(w.comps), kSentinel);
    // Stand-in for user boundary-condition data living in exterior halo cells: a value unique to
    // (rank, cell) so any stray overwrite is visible. Cells that a correct exchange fills get
    // overwritten anyway; cells outside an open axis must keep it.
    for (size_t k = 0; k < rk.buf.size(); ++k) {
      rk.buf[k] = kMarker ^ (static_cast<uint64_t>(r + 1) << 40) ^ static_cast<uint64_t>(k);
    }
    for (int c = 0; c < w.comps; ++c) {
      for (int64_t z = 0; z < sz.first; ++z) {
        for (int64_t y = 0; y < sy.first; ++y) {
          for (int64_t x = 0; x < sx.first; ++x) {
            const size_t idx = static_cast<size_t>(c * rk.comp_stride + (z + hh.z) * rk.g.pitch +
                                                   (y + hh.y) * rk.g.haloed.x + (x + hh.x));
            rk.buf[idx] = encode(rk.offset.x + x, rk.offset.y + y, rk.offset.z + z, c);
          }
        }
      }
    }
    w.ranks.push_back(std::move(rk));
  }
  return w;
}

// ---- the existing run-based exchange, used as oracle (and for the Z pass) -------------------
auto
exchange_runs(World& w,
              Axis axis) -> void {
  // Snapshot sources first so every rank reads pre-pass data, like concurrent streams would.
  const std::vector<Rank> snap = w.ranks;
  for (size_t i = 0; i < w.ranks.size(); ++i) {
    Rank& dst = w.ranks[i];
    if (axis_len(dst.g.halo, axis) <= 0) {
      continue;
    }
    for (const AxisSide side : {AxisSide::Lower, AxisSide::Upper}) {
      const auto nb = hb::axis_neighbour(w.pg, dst.coords, axis, w.bm, side);
      if (!nb.present) {
        continue;
      }
      const Rank& src = snap[static_cast<size_t>(nb.rank)];
      const AxisSide src_side = side == AxisSide::Lower ? AxisSide::Upper : AxisSide::Lower;
      const auto dst_runs = hb::axis_halo_runs(dst.g, axis, side);
      const auto src_runs = hb::axis_owned_face_runs(src.g, axis, src_side);
      CHECK(dst_runs.size() == src_runs.size());
      for (int c = 0; c < w.comps; ++c) {
        if (!w.active_recv(c, axis, side)) {
          continue;
        }
        for (size_t r = 0; r < dst_runs.size(); ++r) {
          CHECK(dst_runs[r].count == src_runs[r].count);
          for (size_t k = 0; k < dst_runs[r].count; ++k) {
            dst.buf[static_cast<size_t>(c * dst.comp_stride) + dst_runs[r].offset + k] =
                src.buf[static_cast<size_t>(c * src.comp_stride) + src_runs[r].offset + k];
          }
        }
      }
    }
  }
}

struct PassStats {
  size_t msgs{};
  size_t kernels{};
};

// ---- the packed exchange, driven by the real plan builder + the real per-thread code --------
auto
exchange_packed(World& w,
                Axis axis,
                std::vector<size_t>& stage_cursor) -> PassStats {
  PassStats stats;
  std::vector<hp::AxisPackPlan> plans(w.ranks.size());
  for (size_t i = 0; i < w.ranks.size(); ++i) {
    Rank& rk = w.ranks[i];
    plans[i] = hp::build_axis_pack_plan(
        w.pg, w.bm, rk.coords, static_cast<int32_t>(i), rk.g, axis, w.comps, rk.comp_stride,
        [&](int c, Axis a, AxisSide s) {
          return w.active_send(c, a, s);
        },
        [&](int c, Axis a, AxisSide s) {
          return w.active_recv(c, a, s);
        },
        stage_cursor[i]);
    hp::sort_msgs(plans[i].msgs);
    stage_cursor[i] += plans[i].stage_elems;
    rk.stage.resize(std::max(rk.stage.size(), stage_cursor[i]), kSentinel);
  }

  // 1) gather everywhere (before any message is delivered or scattered).
  for (size_t i = 0; i < w.ranks.size(); ++i) {
    for (const hp::PackStep& s : plans[i].gathers) {
      hp::host_gather<uint64_t>(w.ranks[i].buf.data(), w.ranks[i].stage.data() + s.stage_offset,
                                s.args);
      ++stats.kernels;
    }
  }

  // 2) deliver messages the way per-peer-ordered NCCL send/recv would.
  std::map<std::pair<int, int>, std::vector<const hp::PackMsg*>> sends;  // (src,dst)
  std::map<std::pair<int, int>, std::vector<const hp::PackMsg*>> recvs;  // (src,dst)
  for (size_t i = 0; i < w.ranks.size(); ++i) {
    for (const hp::PackMsg& m : plans[i].msgs) {
      ++stats.msgs;
      if (m.op == hp::MsgOp::Send) {
        sends[{static_cast<int>(i), m.peer}].push_back(&m);
      } else {
        recvs[{m.peer, static_cast<int>(i)}].push_back(&m);
      }
    }
  }
  CHECK(sends.size() == recvs.size());
  for (const auto& [pair, slist] : sends) {
    const auto it = recvs.find(pair);
    CHECK(it != recvs.end());
    const auto& rlist = it->second;
    CHECK(slist.size() == rlist.size());
    for (size_t k = 0; k < slist.size(); ++k) {
      CHECK(slist[k]->count == rlist[k]->count);  // same position => same size AND same key
      CHECK(slist[k]->key == rlist[k]->key);
      const Rank& s = w.ranks[static_cast<size_t>(pair.first)];
      Rank& d = w.ranks[static_cast<size_t>(pair.second)];
      std::copy_n(s.stage.begin() + static_cast<std::ptrdiff_t>(slist[k]->first),
                  static_cast<std::ptrdiff_t>(slist[k]->count),
                  d.stage.begin() + static_cast<std::ptrdiff_t>(rlist[k]->first));
    }
  }

  // 3) scatter everywhere.
  for (size_t i = 0; i < w.ranks.size(); ++i) {
    for (const hp::PackStep& s : plans[i].scatters) {
      hp::host_scatter<uint64_t>(w.ranks[i].stage.data() + s.stage_offset, w.ranks[i].buf.data(),
                                 s.args);
      ++stats.kernels;
    }
  }
  return stats;
}

// ---- ground truth -------------------------------------------------------------------------
// Returns true and the wrapped global coordinate if the cell exists in the global domain.
auto
wrap_global(const World& w,
            int64_t gx,
            int64_t gy,
            int64_t gz,
            std::array<int64_t,
                       3>& out) -> bool {
  const std::array<int64_t, 3> g{gx, gy, gz};
  const std::array<int64_t, 3> n{w.global.x, w.global.y, w.global.z};
  const std::array<Axis, 3> ax{Axis::X, Axis::Y, Axis::Z};
  for (size_t a = 0; a < 3; ++a) {
    int64_t v = g[a];
    if (v < 0 || v >= n[a]) {
      if (!carbon::is_periodic(w.bm, ax[a])) {
        return false;
      }
      v = ((v % n[a]) + n[a]) % n[a];
    }
    out[a] = v;
  }
  return true;
}

// Is population c at local cell (lx,ly,lz) [buffer coords] ever read by an owned cell?
auto
needed(const World& w,
       const Rank& rk,
       int c,
       int64_t lx,
       int64_t ly,
       int64_t lz) -> bool {
  if (w.semantic == Semantic::Standard) {
    return true;
  }
  const Vel& v = w.vel[static_cast<size_t>(c)];
  const int64_t px = lx + v.x;
  const int64_t py = ly + v.y;
  const int64_t pz = lz + v.z;
  return px >= rk.g.halo.x && px < rk.g.halo.x + rk.g.owned.x && py >= rk.g.halo.y &&
         py < rk.g.halo.y + rk.g.owned.y && pz >= rk.g.halo.z && pz < rk.g.halo.z + rk.g.owned.z;
}

// Verify one world against ground truth; optionally against another (old-path) world on every
// cell that matters. Returns the number of halo cells that were checked against ground truth.
auto
verify(const World& w,
       const World* other,
       bool check_exterior = true,
       bool z_packed = false) -> size_t {
  size_t checked = 0;
  // A run-based Z pass copies whole planes (including exterior x/y halos), so exterior cells may
  // only be demanded intact when no such pass exists: no Z halo at all, or Z is packed.
  bool no_z_pass = z_packed;
  if (!no_z_pass) {
    no_z_pass = true;
    for (const Rank& r : w.ranks) {
      no_z_pass = no_z_pass && r.g.halo.z == 0;
    }
  }
  for (size_t i = 0; i < w.ranks.size(); ++i) {
    const Rank& rk = w.ranks[i];
    for (int c = 0; c < w.comps; ++c) {
      for (int64_t z = 0; z < rk.g.haloed.z; ++z) {
        for (int64_t y = 0; y < rk.g.haloed.y; ++y) {
          for (int64_t x = 0; x < rk.g.haloed.x; ++x) {
            const size_t idx =
                static_cast<size_t>(c * rk.comp_stride + z * rk.g.pitch + y * rk.g.haloed.x + x);
            const uint64_t got = rk.buf[idx];
            const bool owned = x >= rk.g.halo.x && x < rk.g.halo.x + rk.g.owned.x &&
                               y >= rk.g.halo.y && y < rk.g.halo.y + rk.g.owned.y &&
                               z >= rk.g.halo.z && z < rk.g.halo.z + rk.g.owned.z;
            std::array<int64_t, 3> gw{};
            const bool valid =
                wrap_global(w, rk.offset.x + x - rk.g.halo.x, rk.offset.y + y - rk.g.halo.y,
                            rk.offset.z + z - rk.g.halo.z, gw);
            if (owned) {
              CHECK(got == encode(gw[0], gw[1], gw[2], c));
              continue;
            }
            if (!valid) {
              // Outside an open axis. With no Z pass, nothing may touch these cells: this is what
              // the exterior-x clipping in the Y pass (and the owned-only X extents) guarantee.
              if (no_z_pass && check_exterior) {
                CHECK(got == (kMarker ^ (static_cast<uint64_t>(i + 1) << 40) ^
                              static_cast<uint64_t>(idx)));
              }
              continue;
            }
            if (!needed(w, rk, c, x, y, z)) {
              continue;  // lattice: this population is never read at this halo cell
            }
            CHECK(got == encode(gw[0], gw[1], gw[2], c));
            if (other != nullptr) {
              CHECK(other->ranks[i].buf[idx] == got);  // packed == run-based, bit for bit
            }
            ++checked;
          }
        }
      }
    }
  }
  return checked;
}

// For LatticeDistributed, face-interior halo cells must carry every component whose velocity is
// nonzero on that face, and must leave zero-normal components untouched. Corners are skipped:
// several face passes may write them, so their source is not unique.
auto
verify_lattice_distributed_faces(const World& w) -> size_t {
  size_t checked = 0;
  for (size_t i = 0; i < w.ranks.size(); ++i) {
    const Rank& rk = w.ranks[i];
    for (int c = 0; c < w.comps; ++c) {
      for (int64_t z = 0; z < rk.g.haloed.z; ++z) {
        for (int64_t y = 0; y < rk.g.haloed.y; ++y) {
          for (int64_t x = 0; x < rk.g.haloed.x; ++x) {
            const bool owned = x >= rk.g.halo.x && x < rk.g.halo.x + rk.g.owned.x &&
                               y >= rk.g.halo.y && y < rk.g.halo.y + rk.g.owned.y &&
                               z >= rk.g.halo.z && z < rk.g.halo.z + rk.g.owned.z;
            if (owned) {
              continue;
            }
            const bool halo_x = x < rk.g.halo.x || x >= rk.g.halo.x + rk.g.owned.x;
            const bool halo_y = y < rk.g.halo.y || y >= rk.g.halo.y + rk.g.owned.y;
            const bool halo_z = z < rk.g.halo.z || z >= rk.g.halo.z + rk.g.owned.z;
            if (static_cast<int>(halo_x) + static_cast<int>(halo_y) + static_cast<int>(halo_z) != 1) {
              continue;
            }
            const Axis axis = halo_x ? Axis::X : (halo_y ? Axis::Y : Axis::Z);
            const size_t idx = static_cast<size_t>(c * rk.comp_stride + z * rk.g.pitch +
                                                   y * rk.g.haloed.x + x);
            const uint64_t marker =
                kMarker ^ (static_cast<uint64_t>(i + 1) << 40) ^ static_cast<uint64_t>(idx);
            std::array<int64_t, 3> gw{};
            const bool valid =
                wrap_global(w, rk.offset.x + x - rk.g.halo.x, rk.offset.y + y - rk.g.halo.y,
                            rk.offset.z + z - rk.g.halo.z, gw);
            const uint64_t got = rk.buf[idx];
            if (!valid) {
              CHECK(got == marker);
            } else if (axis_of(w.vel[static_cast<size_t>(c)], axis) != 0) {
              CHECK(got == encode(gw[0], gw[1], gw[2], c));
            } else {
              CHECK(got == marker);
            }
            ++checked;
          }
        }
      }
    }
  }
  return checked;
}

// ---- Part A: kernel index math in isolation, all element widths ---------------------------
struct Wide16 {
  uint64_t a, b;
  bool
  operator==(const Wide16& o) const {
    return a == o.a && b == o.b;
  }
};

template <typename W>
auto
make_elem(uint64_t v) -> W {
  if constexpr (sizeof(W) == 16) {
    return W{v, ~v};
  } else {
    return static_cast<W>(v);
  }
}

template <typename W>
auto
part_a_roundtrip() -> void {
  g_ctx = std::string("part A width=") + std::to_string(sizeof(W));
  AxisGeometry g;
  g.owned = {5, 6, 7};
  g.halo = {2, 1, 2};
  g.haloed = {9, 8, 11};
  g.pitch = g.haloed.x * g.haloed.y + 3;  // padded pitch: must not be assumed == x*y
  const int comps = 70;                   // ~47 listed comps > kMaxPackComps: forces chunking
  const int64_t cs = g.pitch * g.haloed.z;
  std::vector<W> field(static_cast<size_t>(cs) * comps);
  for (size_t i = 0; i < field.size(); ++i) {
    field[i] = make_elem<W>(i * 2654435761ULL + 17U);
  }
  for (const Axis axis : {Axis::X, Axis::Y}) {
    for (const AxisSide side : {AxisSide::Lower, AxisSide::Upper}) {
      std::vector<int> cl;
      for (int c = 0; c < comps; ++c) {
        if (c % 3 != 1) {
          cl.push_back(c);
        }
      }
      const hp::PackBox box = hp::pack_face_box(g, axis, side, axis == Axis::Y ? 1 : 0,
                                                axis == Axis::Y ? g.haloed.x - 2 : 0);
      std::vector<hp::PackStep> steps;
      hp::detail_plan::append_steps(steps, box, cl, g, cs, 0);
      CHECK(cl.size() > hp::kMaxPackComps);  // the case must really exercise chunking
      CHECK(steps.size() == (cl.size() + hp::kMaxPackComps - 1) / hp::kMaxPackComps);
      CHECK(steps.size() >= 2U);
      std::vector<W> stage(cl.size() * box.volume(), make_elem<W>(0));
      for (const auto& s : steps) {
        hp::host_gather<W>(field.data(), stage.data() + s.stage_offset, s.args);
      }
      // Against an explicit triple loop.
      size_t n = 0;
      for (const int c : cl) {
        for (int64_t z = 0; z < box.ext_z; ++z) {
          for (int64_t y = 0; y < box.ext_y; ++y) {
            for (int64_t x = 0; x < box.ext_x; ++x) {
              const W expect = field[static_cast<size_t>(
                  c * cs + (box.lo_z + z) * g.pitch + (box.lo_y + y) * g.haloed.x + box.lo_x + x)];
              CHECK(stage[n] == expect);
              ++n;
            }
          }
        }
      }
      CHECK(n == stage.size());
      // Scatter into a clean field: writes exactly the box cells of exactly the listed comps.
      std::vector<W> out(field.size(), make_elem<W>(0));
      for (const auto& s : steps) {
        hp::host_scatter<W>(stage.data() + s.stage_offset, out.data(), s.args);
      }
      size_t written = 0;
      for (size_t i = 0; i < out.size(); ++i) {
        if (!(out[i] == make_elem<W>(0))) {
          CHECK(out[i] == field[i]);
          ++written;
        }
      }
      CHECK(written <= stage.size());  // (<: a source element may legitimately equal 0)
    }
  }
}

// ---- Part A2: launch-shape corner cases (rows > one block, z beyond the 65535 grid cap) ------
auto
part_a_launch_shapes() -> void {
  struct Shape {
    int64_t ox, oy, oz;
    int comps;
  };
  // X face: row = owned.y;  Y face: row = haloed.x.  The last shape needs ext_z > 65535 so that
  // grid_y is capped and the grid-stride loop must wrap.
  const Shape shapes[] = {{4, 300, 3, 2}, {300, 4, 5, 3}, {2, 3, 70000, 2}, {37, 41, 9, 1}};
  for (const Shape& sh : shapes) {
    g_ctx = "part A2 shape " + std::to_string(sh.ox) + "x" + std::to_string(sh.oy) + "x" +
            std::to_string(sh.oz);
    AxisGeometry g;
    g.owned = {sh.ox, sh.oy, sh.oz};
    g.halo = {1, 1, 1};
    g.haloed = {sh.ox + 2, sh.oy + 2, sh.oz + 2};
    g.pitch = g.haloed.x * g.haloed.y;
    const int64_t cs = g.pitch * g.haloed.z;
    std::vector<uint32_t> field(static_cast<size_t>(cs) * static_cast<size_t>(sh.comps));
    for (size_t i = 0; i < field.size(); ++i) {
      field[i] = static_cast<uint32_t>(i * 2654435761ULL + 5U);
    }
    std::vector<int> cl;
    for (int c = 0; c < sh.comps; ++c) {
      cl.push_back(c);
    }
    for (const Axis axis : {Axis::X, Axis::Y}) {
      const hp::PackBox box = hp::pack_face_box(g, axis, AxisSide::Upper, 0, g.haloed.x);
      std::vector<hp::PackStep> steps;
      hp::detail_plan::append_steps(steps, box, cl, g, cs, 0);
      CHECK(steps.size() == 1U);
      const hp::PackLaunchShape ls = hp::pack_launch_shape(steps[0].args);
      if (axis == Axis::X && sh.oy == 300) {
        CHECK(ls.grid_x == 2U);  // 300 rows over 256-thread blocks
      }
      if (sh.oz == 70000) {
        CHECK(ls.grid_y == 65535U);  // capped: the grid-stride loop must cover the rest
      }
      std::vector<uint32_t> stage(cl.size() * box.volume(), 0U);
      hp::host_gather<uint32_t>(field.data(), stage.data(), steps[0].args);
      size_t n = 0;
      for (const int c : cl) {
        for (int64_t z = 0; z < box.ext_z; ++z) {
          for (int64_t y = 0; y < box.ext_y; ++y) {
            for (int64_t x = 0; x < box.ext_x; ++x) {
              CHECK(stage[n] ==
                    field[static_cast<size_t>(c * cs + (box.lo_z + z) * g.pitch +
                                              (box.lo_y + y) * g.haloed.x + box.lo_x + x)]);
              ++n;
            }
          }
        }
      }
      CHECK(n == stage.size());
    }
  }
}

// ---- Part B: whole-world scenarios --------------------------------------------------------
// ------------------------------------------------------------------------------------------------
// Part B: slab / pencil / brick geometry.
// ------------------------------------------------------------------------------------------------

struct GeomCase {
  const char* name;
  int px, py, pz;
  BoundaryModeDistributed bm;
  int64_t gx, gy, gz;
  std::vector<int32_3d> pts;  // stencil points (centre excluded)
  int64_3d want_halo;         // hand-written expectation (per side, uniform across ranks)
};

auto
six() -> std::vector<int32_3d> {
  return {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
}

auto
eq3(const int64_3d& a,
    const int64_3d& b) -> bool {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

auto
wrap_axis(int64_t v,
          int64_t n,
          bool periodic,
          int64_t& out) -> bool {
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

auto
check_geometry(const GeomCase& c,
               size_t& cells,
               size_t& neighbour_reads) -> void {
  g_ctx = std::string("part B ") + c.name;
  const ProcessGrid pg = ProcessGrid::custom(c.px * c.py * c.pz, c.px, c.py, c.pz);
  const carbon::Stencil stencil(c.pts, /*remove_center=*/true);
  const int64_3d global{c.gx, c.gy, c.gz};
  const BoundaryModeDistributed bm = c.bm;

  const int64_3d halo = dd::halo_extent(pg, bm, stencil);
  CHECK(eq3(halo, c.want_halo));

  const bool legacy_slab = dd::is_legacy_slab(pg, bm);
  CHECK(legacy_slab == (c.px == 1 && c.py == 1 && !carbon::is_periodic(bm, Axis::X) &&
                        !carbon::is_periodic(bm, Axis::Y)));

  std::vector<int> cover(static_cast<size_t>(c.gx * c.gy * c.gz), 0);
  const auto cover_at = [&](int64_t x, int64_t y, int64_t z) -> int& {
    return cover[static_cast<size_t>((z * c.gy + y) * c.gx + x)];
  };

  for (int32_t r = 0; r < pg.world_size(); ++r) {
    const int32_3d co = pg.coords_of(r);
    const dd::RankChunk ch = dd::rank_chunk(global, pg, co, halo);

    // Independent expectation of the split (the documented "first n % P get one extra layer").
    const auto sx = split(c.gx, c.px, co.x);
    const auto sy = split(c.gy, c.py, co.y);
    const auto sz = split(c.gz, c.pz, co.z);
    CHECK(eq3(ch.owned, {sx.first, sy.first, sz.first}));
    CHECK(eq3(ch.offset, {sx.second - halo.x, sy.second - halo.y, sz.second - halo.z}));
    CHECK(eq3(ch.halo, halo));
    CHECK(eq3(ch.haloed,
              {ch.owned.x + 2 * halo.x, ch.owned.y + 2 * halo.y, ch.owned.z + 2 * halo.z}));
    CHECK(ch.xy_pitch == ch.haloed.x * ch.haloed.y);

    const dd::Span span(ch.haloed, global, ch.xy_pitch, ch.offset, ch.halo, bm);
    CHECK(eq3(span.logic_extent(), ch.owned));
    CHECK(eq3(span.haloed_extent(), ch.haloed));
    CHECK(eq3(span.halo(), halo));
    CHECK(eq3(span.offset(), ch.offset));
    CHECK(eq3(span.global_extent(), global));
    CHECK(span.xy_slab_elements() == ch.xy_pitch);
    CHECK(span.linear_storage_elements() == ch.xy_pitch * ch.haloed.z);

    // Standard view: every owned cell once, mapped to the right global coordinate.
    const int64_3d own0{sx.second, sy.second, sz.second};  // global coordinate of owned cell 0
    CHECK(eq3(span.iter_extent(carbon::DataView::Standard), ch.owned));
    for (int64_t iz = 0; iz < ch.owned.z; ++iz) {
      for (int64_t iy = 0; iy < ch.owned.y; ++iy) {
        for (int64_t ix = 0; ix < ch.owned.x; ++ix) {
          const auto v = span.set_and_validate(carbon::DataView::Standard, ix, iy, iz);
          CHECK(v.valid);
          CHECK(eq3(v.idx.cell(), {halo.x + ix, halo.y + iy, halo.z + iz}));
          const int64_3d g = span.global_location(v.idx);
          CHECK(eq3(g, {own0.x + ix, own0.y + iy, own0.z + iz}));
          cover_at(g.x, g.y, g.z) += 1;
          ++cells;
        }
      }
    }
    CHECK(!span.set_and_validate(carbon::DataView::Standard, ch.owned.x, 0, 0).valid);

    // Internal + Boundary must tile the owned cells exactly once. (A legacy Z slab thinner than
    // 2 * halo.z has overlapping Boundary bands by construction; that degenerate shape is skipped.)
    if (!(legacy_slab && ch.owned.z < 2 * halo.z)) {
      std::vector<int> mark(static_cast<size_t>(ch.owned.x * ch.owned.y * ch.owned.z), 0);
      for (const carbon::DataView view : {carbon::DataView::Internal, carbon::DataView::Boundary}) {
        const int64_3d e = span.iter_extent(view);
        if (view == carbon::DataView::Boundary && !legacy_slab) {
          // Compact shell: the launch box holds the shell cells plus less than one 128-wide row
          // of padding (a full-owned-extent launch would fail this by a wide margin), and the
          // padded tail is exactly the invalid threads.
          const auto pos = [](int64_t v) {
            return v > 0 ? v : 0;
          };
          const int64_t owned_cells = ch.owned.x * ch.owned.y * ch.owned.z;
          const int64_t internal_cells = pos(ch.owned.x - 2 * halo.x) *
                                         pos(ch.owned.y - 2 * halo.y) *
                                         pos(ch.owned.z - 2 * halo.z);
          const int64_t shell = owned_cells - internal_cells;
          CHECK(e.z == 1 || shell == 0);
          CHECK(e.x * e.y * e.z >= shell);
          CHECK(e.x * e.y * e.z < shell + 128);
        }
        int64_t valid_threads = 0;
        for (int64_t iz = 0; iz < e.z; ++iz) {
          for (int64_t iy = 0; iy < e.y; ++iy) {
            for (int64_t ix = 0; ix < e.x; ++ix) {
              const auto v = span.set_and_validate(view, ix, iy, iz);
              if (!v.valid) {
                continue;
              }
              ++valid_threads;
              const int64_3d cell = v.idx.cell();
              const int64_t lx = cell.x - halo.x;
              const int64_t ly = cell.y - halo.y;
              const int64_t lz = cell.z - halo.z;
              CHECK(lx >= 0 && lx < ch.owned.x && ly >= 0 && ly < ch.owned.y && lz >= 0 &&
                    lz < ch.owned.z);
              mark[static_cast<size_t>((lz * ch.owned.y + ly) * ch.owned.x + lx)] += 1;
            }
          }
        }
        if (view == carbon::DataView::Boundary && !legacy_slab) {
          // Every valid Boundary thread is a distinct shell cell; the rest of the box is padding.
          const auto pos = [](int64_t v) {
            return v > 0 ? v : 0;
          };
          CHECK(valid_threads == ch.owned.x * ch.owned.y * ch.owned.z -
                                     pos(ch.owned.x - 2 * halo.x) * pos(ch.owned.y - 2 * halo.y) *
                                         pos(ch.owned.z - 2 * halo.z));
        }
      }
      for (const int m : mark) {
        CHECK(m == 1);
      }
    }

    // Neighbour resolution for every stencil point of every owned cell.
    for (int64_t iz = 0; iz < ch.owned.z; ++iz) {
      for (int64_t iy = 0; iy < ch.owned.y; ++iy) {
        for (int64_t ix = 0; ix < ch.owned.x; ++ix) {
          const dd::Idx idx = span.make_idx(halo.x + ix, halo.y + iy, halo.z + iz);
          const int64_3d g{own0.x + ix, own0.y + iy, own0.z + iz};
          for (const int32_3d& p : c.pts) {
            const dd::NeighbourIdx nb =
                dd::neighbour_idx(span, idx,
                                  int8_3d{static_cast<int8_t>(p.x), static_cast<int8_t>(p.y),
                                          static_cast<int8_t>(p.z)});
            int64_t wx = 0;
            int64_t wy = 0;
            int64_t wz = 0;
            const bool want = wrap_axis(g.x + p.x, c.gx, carbon::is_periodic(bm, Axis::X), wx) &&
                              wrap_axis(g.y + p.y, c.gy, carbon::is_periodic(bm, Axis::Y), wy) &&
                              wrap_axis(g.z + p.z, c.gz, carbon::is_periodic(bm, Axis::Z), wz);
            CHECK(nb.valid == want);
            ++neighbour_reads;
            if (!want) {
              continue;
            }
            const int64_3d cell = nb.idx.cell();
            CHECK(cell.x >= 0 && cell.x < ch.haloed.x && cell.y >= 0 && cell.y < ch.haloed.y &&
                  cell.z >= 0 && cell.z < ch.haloed.z);
            // The addressed cell is the (possibly halo) copy of the wrapped global cell.
            const int64_3d ng = span.global_location(nb.idx);
            int64_t ax = 0;
            int64_t ay = 0;
            int64_t az = 0;
            CHECK(wrap_axis(ng.x, c.gx, true, ax) && wrap_axis(ng.y, c.gy, true, ay) &&
                  wrap_axis(ng.z, c.gz, true, az));
            CHECK(ax == wx && ay == wy && az == wz);
          }
        }
      }
    }
  }

  for (const int m : cover) {
    CHECK(m == 1);  // owned boxes tile the global domain exactly once
  }
}

auto
part_b_geometry(size_t& cells,
                size_t& neighbour_reads) -> size_t {
  constexpr auto kO = BoundaryModeDistributed::Open;
  constexpr auto kX = BoundaryModeDistributed::PeriodicX;
  constexpr auto kY = BoundaryModeDistributed::PeriodicY;
  constexpr auto kZ = BoundaryModeDistributed::PeriodicZ;
  const std::vector<int32_3d> aniso = {{2, 0, 0},  {-2, 0, 0}, {0, 1, 0},
                                       {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
  const std::vector<int32_3d> r2 = {{2, 0, 0},  {-2, 0, 0}, {0, 1, 0},
                                    {0, -1, 0}, {0, 0, 2},  {0, 0, -2}};
  std::vector<int32_3d> d3q19;
  for (int z = -1; z <= 1; ++z) {
    for (int y = -1; y <= 1; ++y) {
      for (int x = -1; x <= 1; ++x) {
        const int nz = (x != 0) + (y != 0) + (z != 0);
        if (nz >= 1 && nz <= 2) {
          d3q19.push_back({x, y, z});
        }
      }
    }
  }
  const std::vector<GeomCase> cases = {
      // Slab (Px == Py == 1): historical Z halo, X/Y halos only when periodic.
      {"slab/single-open", 1, 1, 1, kO, 6, 5, 7, six(), {0, 0, 1}},
      {"slab/z3-open", 1, 1, 3, kO, 6, 5, 9, six(), {0, 0, 1}},
      {"slab/z3-radius2", 1, 1, 3, kO, 6, 5, 12, r2, {0, 0, 2}},
      {"slab/z2-periodicZ", 1, 1, 2, kZ, 6, 5, 8, six(), {0, 0, 1}},
      {"slab/z2-periodicX", 1, 1, 2, kX, 6, 5, 8, six(), {1, 0, 1}},
      {"slab/single-periodicXYZ", 1, 1, 1, kX | kY | kZ, 5, 5, 5, six(), {1, 1, 1}},
      // Pencil: two split axes.
      {"pencil/yz-open", 1, 2, 2, kO, 6, 7, 8, six(), {0, 1, 1}},
      {"pencil/yz-periodicZ", 1, 2, 2, kZ, 6, 7, 8, six(), {0, 1, 1}},
      {"pencil/yz-periodicX", 1, 2, 2, kX, 6, 7, 8, six(), {1, 1, 1}},
      {"pencil/yz-d3q19", 1, 2, 2, kO, 6, 7, 8, d3q19, {0, 1, 1}},
      // Brick: three split axes, uneven and thin chunks.
      {"brick/222-open", 2, 2, 2, kO, 7, 6, 5, six(), {1, 1, 1}},
      {"brick/322-uneven", 3, 2, 2, kO, 8, 5, 6, six(), {1, 1, 1}},
      {"brick/222-thin-periodicXYZ", 2, 2, 2, kX | kY | kZ, 3, 3, 3, six(), {1, 1, 1}},
      {"brick/221-aniso", 2, 2, 1, kO, 8, 6, 5, aniso, {2, 1, 0}},
      {"brick/221-aniso-periodicZ", 2, 2, 1, kZ, 8, 6, 5, aniso, {2, 1, 1}},
      {"brick/212-d3q19-periodicXZ", 2, 1, 2, kX | kZ, 6, 5, 6, d3q19, {1, 0, 1}},
  };
  for (const GeomCase& c : cases) {
    check_geometry(c, cells, neighbour_reads);
  }
  return cases.size();
}

auto
part_b_global_validation() -> size_t {
  g_ctx = "part B global validation";
  const carbon::Stencil stencil(six(), /*remove_center=*/true);
  const auto throws_invalid_argument = [](auto&& fn) -> bool {
    try {
      fn();
    } catch (const std::invalid_argument&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  };

  size_t cases = 0;
  {
    // z = 7 over 4 slabs -> [2,2,2,1]; the last rank is too thin for halo_z = 1. Every rank
    // must reject the global decomposition, not just the rank that owns the thin chunk.
    const ProcessGrid pg = ProcessGrid::custom(4, 1, 1, 4);
    const int64_3d global{9, 8, 7};
    const int64_3d halo = dd::halo_extent(pg, BoundaryModeDistributed::Open, stencil);
    CHECK(dd::is_legacy_slab(pg, BoundaryModeDistributed::Open));
    CHECK(throws_invalid_argument([&] {
      dd::validate_global_halo_fit(global, pg, halo, /*legacy_slab=*/true, stencil);
    }));
    ++cases;
  }
  {
    // x = 1 over 2 bricks -> [1,0]; a zero-owned chunk cannot host the X halo. Again the
    // decision is global, so ranks with a non-thin chunk throw as well.
    const ProcessGrid pg = ProcessGrid::custom(8, 2, 2, 2);
    const int64_3d global{1, 5, 7};
    const int64_3d halo = dd::halo_extent(pg, BoundaryModeDistributed::Open, stencil);
    CHECK(!dd::is_legacy_slab(pg, BoundaryModeDistributed::Open));
    CHECK(throws_invalid_argument([&] {
      dd::validate_global_halo_fit(global, pg, halo, /*legacy_slab=*/false, stencil);
    }));
    ++cases;
  }
  {
    const ProcessGrid pg = ProcessGrid::custom(4, 1, 1, 4);
    const int64_3d global{9, 8, 11};
    const int64_3d halo = dd::halo_extent(pg, BoundaryModeDistributed::Open, stencil);
    CHECK(!throws_invalid_argument([&] {
      dd::validate_global_halo_fit(global, pg, halo, /*legacy_slab=*/true, stencil);
    }));
    ++cases;
  }
  {
    const ProcessGrid pg = ProcessGrid::custom(8, 2, 2, 2);
    const int64_3d global{9, 8, 11};
    const int64_3d halo = dd::halo_extent(pg, BoundaryModeDistributed::Open, stencil);
    CHECK(!throws_invalid_argument([&] {
      dd::validate_global_halo_fit(global, pg, halo, /*legacy_slab=*/false, stencil);
    }));
    ++cases;
  }
  return cases;
}

// ------------------------------------------------------------------------------------------------
// Part D: VTK shard I/O for DenseDistributed.
// ------------------------------------------------------------------------------------------------

namespace vtk = carbon::io::detail;

// Injective for extents < 16 and up to 3 components, and it fits in int16.
auto
vtk_value(int64_t x,
          int64_t y,
          int64_t z,
          int c) -> int {
  return static_cast<int>(x + 16 * y + 256 * z + 4096 * c);
}

// Independent big-endian decoder (does not use the library's swap code).
template <typename T>
auto
decode_be(const unsigned char* p) -> T {
  unsigned char b[sizeof(T)];
  for (size_t i = 0; i < sizeof(T); ++i) {
    b[i] = p[sizeof(T) - 1 - i];  // BE on the wire -> LE host (this test runs on LE hosts)
  }
  T v;
  std::memcpy(&v, b, sizeof(T));
  return v;
}

struct VtkCase {
  const char* name;
  int px, py, pz;
  BoundaryModeDistributed bm;
  int64_t gx, gy, gz;
};

template <typename T>
auto
vtk_roundtrip_case(const VtkCase& c,
                   int comps,
                   std::vector<int>& cover,
                   size_t& shards,
                   size_t& empty_shards,
                   size_t& points) -> void {
  g_ctx = std::string("part D ") + c.name + " T" + std::to_string(sizeof(T)) + " c" +
          std::to_string(comps);
  static_assert(std::endian::native == std::endian::little, "decoder assumes a little-endian host");
  const ProcessGrid pg = ProcessGrid::custom(c.px * c.py * c.pz, c.px, c.py, c.pz);
  const carbon::Stencil stencil(
      std::vector<int32_3d>{{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);
  const int64_3d global{c.gx, c.gy, c.gz};
  const int64_3d halo = dd::halo_extent(pg, c.bm, stencil);
  const char* tag =
      sizeof(T) == 2 ? "i16\0\0\0\0" : (sizeof(T) == 4 ? "i32\0\0\0\0" : "f64\0\0\0\0");
  const char* kw = sizeof(T) == 2 ? "short" : (sizeof(T) == 4 ? "int" : "double");

  for (int32_t r = 0; r < pg.world_size(); ++r) {
    const dd::RankChunk ch = dd::rank_chunk(global, pg, pg.coords_of(r), halo);
    const int64_t per_comp = ch.xy_pitch * ch.haloed.z;
    const T sentinel = static_cast<T>(-777);
    std::vector<T> buf(static_cast<size_t>(per_comp) * static_cast<size_t>(comps), sentinel);
    const int64_3d own0{ch.offset.x + halo.x, ch.offset.y + halo.y, ch.offset.z + halo.z};
    for (int64_t z = 0; z < ch.owned.z; ++z) {
      for (int64_t y = 0; y < ch.owned.y; ++y) {
        for (int64_t x = 0; x < ch.owned.x; ++x) {
          for (int k = 0; k < comps; ++k) {
            const int64_t off =
                (halo.z + z) * ch.xy_pitch + (halo.y + y) * ch.haloed.x + halo.x + x;
            buf[static_cast<size_t>(k * per_comp + off)] =
                static_cast<T>(vtk_value(own0.x + x, own0.y + y, own0.z + z, k));
          }
        }
      }
    }

    vtk::VtkBox box;
    box.origin[0] = own0.x;
    box.origin[1] = own0.y;
    box.origin[2] = own0.z;
    const bool owns = ch.owned.x > 0 && ch.owned.y > 0 && ch.owned.z > 0;
    if (owns) {
      box.dims[0] = ch.owned.x;
      box.dims[1] = ch.owned.y;
      box.dims[2] = ch.owned.z;
    }
    std::vector<vtk::VtkBoxPiece> pieces;
    if (owns) {
      vtk::VtkBoxPiece p;
      p.start = buf.data() + halo.z * ch.xy_pitch + halo.y * ch.haloed.x + halo.x;
      p.nx = ch.owned.x;
      p.ny = ch.owned.y;
      p.nz = ch.owned.z;
      p.row_stride = ch.haloed.x;
      p.plane_pitch = ch.xy_pitch;
      p.comp_stride = per_comp;
      pieces.push_back(p);
    }

    std::ostringstream os;
    vtk::write_vtk_box_header(os, box);
    vtk::write_vtk_box_field(os, "my field", tag, sizeof(T), comps, box, pieces);
    const std::string bytes = os.str();
    ++shards;

    // --- independent parse of the text header -------------------------------------------------
    std::istringstream hs(bytes);
    std::string line;
    int64_t dims[3] = {-1, -1, -1};
    int64_t org[3] = {-1, -1, -1};
    int64_t npts = -1;
    std::string scal_name;
    std::string scal_type;
    int scal_comps = -1;
    size_t payload_at = 0;
    while (std::getline(hs, line)) {
      std::istringstream ls(line);
      std::string head;
      ls >> head;
      if (head == "DIMENSIONS") {
        ls >> dims[0] >> dims[1] >> dims[2];
      } else if (head == "ORIGIN") {
        ls >> org[0] >> org[1] >> org[2];
      } else if (head == "POINT_DATA") {
        ls >> npts;
      } else if (head == "SCALARS") {
        ls >> scal_name >> scal_type >> scal_comps;
      } else if (head == "LOOKUP_TABLE") {
        payload_at = static_cast<size_t>(hs.tellg());
        break;
      }
    }
    CHECK(scal_name == "my_field");  // whitespace in the label is made a single token
    CHECK(scal_type == kw);
    CHECK(scal_comps == comps);
    if (!owns) {
      // Placeholder shard: valid header, no payload, flagged as empty in the title.
      ++empty_shards;
      CHECK(dims[0] == 0 && dims[1] == 0 && dims[2] == 0 && npts == 0);
      CHECK(bytes.size() == payload_at + 1U);  // just the newline that ends the SCALARS block
      CHECK(bytes.find("empty") != std::string::npos);
    } else {
      CHECK(dims[0] == ch.owned.x && dims[1] == ch.owned.y && dims[2] == ch.owned.z);
      CHECK(org[0] == own0.x && org[1] == own0.y && org[2] == own0.z);
      CHECK(npts == ch.owned.x * ch.owned.y * ch.owned.z);
      const size_t payload_bytes =
          static_cast<size_t>(npts) * static_cast<size_t>(comps) * sizeof(T);
      CHECK(bytes.size() == payload_at + payload_bytes + 1U);
      const auto* raw = reinterpret_cast<const unsigned char*>(bytes.data()) + payload_at;
      for (int64_t z = 0; z < dims[2]; ++z) {
        for (int64_t y = 0; y < dims[1]; ++y) {
          for (int64_t x = 0; x < dims[0]; ++x) {
            const int64_t pt = (z * dims[1] + y) * dims[0] + x;
            for (int k = 0; k < comps; ++k) {
              const T got =
                  decode_be<T>(raw + (static_cast<size_t>(pt) * static_cast<size_t>(comps) +
                                      static_cast<size_t>(k)) *
                                         sizeof(T));
              CHECK(got == static_cast<T>(vtk_value(org[0] + x, org[1] + y, org[2] + z, k)));
            }
          }
        }
        // tiling bookkeeping (uses only what the header says)
      }
      for (int64_t z = 0; z < dims[2]; ++z) {
        for (int64_t y = 0; y < dims[1]; ++y) {
          for (int64_t x = 0; x < dims[0]; ++x) {
            cover[static_cast<size_t>(((org[2] + z) * c.gy + (org[1] + y)) * c.gx + org[0] + x)] +=
                1;
          }
        }
      }
      points += static_cast<size_t>(npts);
    }

    // --- library round trip: owned cells restored, halo cells untouched ------------------------
    std::istringstream in(bytes);
    std::vector<vtk::VtkBoxFieldRecord> recs;
    vtk::parse_vtk_box_header(in, "mem", box, recs);
    CHECK(recs.size() == 1U);
    CHECK(recs[0].components == comps && recs[0].vtk_dtype == kw);
    const T sentinel2 = static_cast<T>(-555);
    std::vector<T> back(buf.size(), sentinel2);
    std::vector<vtk::VtkBoxPieceMut> mut;
    if (owns) {
      vtk::VtkBoxPieceMut m;
      m.start = back.data() + halo.z * ch.xy_pitch + halo.y * ch.haloed.x + halo.x;
      m.nx = ch.owned.x;
      m.ny = ch.owned.y;
      m.nz = ch.owned.z;
      m.row_stride = ch.haloed.x;
      m.plane_pitch = ch.xy_pitch;
      m.comp_stride = per_comp;
      mut.push_back(m);
    }
    in.clear();
    in.seekg(recs[0].payload_offset);
    vtk::read_vtk_box_field_payload(in, sizeof(T), comps, box, mut);
    for (int64_t z = 0; z < ch.haloed.z; ++z) {
      for (int64_t y = 0; y < ch.haloed.y; ++y) {
        for (int64_t x = 0; x < ch.haloed.x; ++x) {
          const bool in_owned = x >= halo.x && x < halo.x + ch.owned.x && y >= halo.y &&
                                y < halo.y + ch.owned.y && z >= halo.z && z < halo.z + ch.owned.z;
          for (int k = 0; k < comps; ++k) {
            const size_t i =
                static_cast<size_t>(k * per_comp + z * ch.xy_pitch + y * ch.haloed.x + x);
            CHECK(in_owned ? back[i] == buf[i] : back[i] == sentinel2);
          }
        }
      }
    }

    // --- a shard from a different box must be rejected -----------------------------------------
    if (owns) {
      vtk::VtkBox other = box;
      other.origin[0] += 1;
      std::istringstream in2(bytes);
      std::vector<vtk::VtkBoxFieldRecord> recs2;
      bool threw = false;
      carbon::log::set_level(spdlog::level::off);  // the library logs the (expected) rejection
      try {
        vtk::parse_vtk_box_header(in2, "mem", other, recs2);
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      carbon::log::set_level(spdlog::level::info);
      CHECK(threw);
    }
  }
}

auto
part_d_vtk(size_t& shards,
           size_t& empty_shards,
           size_t& points) -> size_t {
  constexpr auto kO = BoundaryModeDistributed::Open;
  constexpr auto kX = BoundaryModeDistributed::PeriodicX;
  constexpr auto kZ = BoundaryModeDistributed::PeriodicZ;
  const std::vector<VtkCase> cases = {
      {"slab", 1, 1, 4, kO, 6, 5, 9},
      {"slab-periodicZ", 1, 1, 3, kZ, 5, 4, 7},
      {"pencil", 1, 2, 2, kO, 6, 7, 8},
      {"pencil-periodicX", 1, 2, 2, kX, 5, 6, 4},
      {"brick-uneven", 3, 2, 2, kO, 8, 5, 6},
      {"brick-thin", 2, 2, 2, kX | kZ, 3, 3, 3},
      // 2 layers over 4 z-parts and 1 over 2 y-parts: half of the ranks own no cells.
      {"brick-empty-ranks", 2, 2, 4, kO, 3, 1, 2},
  };
  for (const VtkCase& c : cases) {
    std::vector<int> cover(static_cast<size_t>(c.gx * c.gy * c.gz), 0);
    vtk_roundtrip_case<int16_t>(c, 2, cover, shards, empty_shards, points);
    for (const int m : cover) {
      CHECK(m == 1);  // shard boxes tile the global domain exactly once
    }
    // Other element sizes / component counts (tiling already shown above; reuse a scratch map).
    std::vector<int> scratch32(cover.size(), 0);
    vtk_roundtrip_case<int32_t>(c, 3, scratch32, shards, empty_shards, points);
    std::vector<int> scratch64(cover.size(), 0);
    vtk_roundtrip_case<double>(c, 1, scratch64, shards, empty_shards, points);
  }
  CHECK(empty_shards > 0U);  // the placeholder path must actually be exercised
  return cases.size();
}

struct Scenario {
  const char* name;
  int px, py, pz;
  BoundaryModeDistributed bm;
  int64_t gx, gy, gz;
  int64_t hx, hy, hz;
  int comps;
  int lattice_q;
  int64_t pad;
  bool lattice_distributed = false;
};

constexpr BoundaryModeDistributed kOpen = BoundaryModeDistributed::Open;
constexpr BoundaryModeDistributed kPX = BoundaryModeDistributed::PeriodicX;
constexpr BoundaryModeDistributed kPY = BoundaryModeDistributed::PeriodicY;
constexpr BoundaryModeDistributed kPZ = BoundaryModeDistributed::PeriodicZ;

auto
bm_or(BoundaryModeDistributed a,
      BoundaryModeDistributed b) -> BoundaryModeDistributed {
  return static_cast<BoundaryModeDistributed>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}

auto
run_scenario(const Scenario& s,
             size_t& total_checked,
             size_t& total_msgs,
             size_t& z_msgs) -> void {
  g_ctx = std::string(s.name) + " grid=" + std::to_string(s.px) + "x" + std::to_string(s.py) + "x" +
          std::to_string(s.pz) + " bm=" + std::to_string(static_cast<int>(s.bm)) +
          " lat=" + std::to_string(s.lattice_q) +
          " semantic=" + (s.lattice_distributed ? "lattice-distributed" : "lattice-or-standard") +
          " pad=" + std::to_string(s.pad);
  const ProcessGrid pg = ProcessGrid::custom(s.px * s.py * s.pz, s.px, s.py, s.pz);
  World packed = make_world(pg, s.bm, {s.gx, s.gy, s.gz}, {s.hx, s.hy, s.hz}, s.comps,
                            s.lattice_q, s.lattice_distributed, s.pad);
  World oracle = packed;  // identical start state

  // Old path: run-based X, Y, Z.
  for (const Axis a : {Axis::X, Axis::Y, Axis::Z}) {
    exchange_runs(oracle, a);
  }

  // New path: packed X, Y; Z is packed iff it carries several components (Halo's rule), else it
  // stays on the run-based path (a single contiguous run per side).
  const bool pack_z = packed.comps > 1;
  std::vector<size_t> cursor(packed.ranks.size(), 0);
  for (const Axis a : {Axis::X, Axis::Y}) {
    const PassStats st = exchange_packed(packed, a, cursor);
    total_msgs += st.msgs;
  }
  if (pack_z) {
    const PassStats st = exchange_packed(packed, Axis::Z, cursor);
    total_msgs += st.msgs;
    z_msgs += st.msgs;
  } else {
    exchange_runs(packed, Axis::Z);
  }

  total_checked +=
      verify(oracle, nullptr, /*check_exterior=*/
             false);  // oracle: right on needed cells (it clobbers exterior halos, by design)...
  total_checked +=
      verify(packed, &oracle, true, pack_z);  // ...and the packed result must match it bit for bit
  if (packed.semantic == Semantic::LatticeDistributed) {
    const size_t face_checked = verify_lattice_distributed_faces(oracle) +
                                verify_lattice_distributed_faces(packed);
    CHECK(face_checked > 0U);
    total_checked += face_checked;
  }
}

}  // namespace

int
main() {
  part_a_roundtrip<uint8_t>();
  part_a_roundtrip<uint16_t>();
  part_a_roundtrip<uint32_t>();
  part_a_roundtrip<uint64_t>();
  part_a_roundtrip<Wide16>();
  part_a_launch_shapes();

  size_t geom_cells = 0;
  size_t geom_neighbours = 0;
  const size_t geom_cases = part_b_geometry(geom_cells, geom_neighbours);
  const size_t validation_cases = part_b_global_validation();
  std::printf("geometry cases=%zu  owned cells checked=%zu  neighbour reads checked=%zu  "
              "global-validation cases=%zu\n",
              geom_cases, geom_cells, geom_neighbours, validation_cases);

  size_t vtk_shards = 0;
  size_t vtk_empty = 0;
  size_t vtk_points = 0;
  const size_t vtk_cases = part_d_vtk(vtk_shards, vtk_empty, vtk_points);
  std::printf("vtk cases=%zu  shards=%zu (placeholders=%zu)  points verified=%zu\n", vtk_cases,
              vtk_shards, vtk_empty, vtk_points);

  std::vector<Scenario> sc;
  const BoundaryModeDistributed modes[] = {
      kOpen, kPX, kPY, kPZ, bm_or(kPX, kPZ), bm_or(bm_or(kPX, kPY), kPZ)};
  struct G {
    const char* n;
    int x, y, z;
  };
  const G grids[] = {{"single", 1, 1, 1}, {"x2", 2, 1, 1},           {"x3", 3, 1, 1},
                     {"y2", 1, 2, 1},     {"slab4", 1, 1, 4},        {"pencil", 1, 2, 2},
                     {"brick", 2, 2, 2},  {"brick-uneven", 2, 3, 2}, {"brick3", 3, 2, 2}};
  for (const BoundaryModeDistributed bm : modes) {
    for (const G& g : grids) {
      // Standard semantics, 3 components, halo (1,1,1), padded and unpadded pitch.
      sc.push_back({"std-c1-runsZ", g.x, g.y, g.z, bm, 11, 9, 10, 1, 1, 1, 1, 0, 0});
      sc.push_back({"std-h111", g.x, g.y, g.z, bm, 11, 9, 10, 1, 1, 1, 3, 0, 0});
      sc.push_back({"std-h111-pad", g.x, g.y, g.z, bm, 11, 9, 10, 1, 1, 1, 3, 0, 7});
      // Wider, anisotropic halo (radius 2 on X and Z).
      sc.push_back({"std-h212", g.x, g.y, g.z, bm, 13, 9, 14, 2, 1, 2, 2, 0, 3});
      // Lattice direction filtering (per-axis send/recv component lists).
      sc.push_back({"d3q19", g.x, g.y, g.z, bm, 11, 9, 10, 1, 1, 1, 0, 19, 0});
      sc.push_back({"d3q27", g.x, g.y, g.z, bm, 11, 9, 10, 1, 1, 1, 0, 27, 5});
      sc.push_back({"d3q19-dist", g.x, g.y, g.z, bm, 11, 9, 10, 1, 1, 1, 0, 19, 0, true});
      sc.push_back({"d3q27-dist-pad", g.x, g.y, g.z, bm, 11, 9, 10, 1, 1, 1, 0, 27, 5, true});
    }
  }

  size_t checked = 0;
  size_t msgs = 0;
  size_t z_msgs = 0;
  for (const Scenario& s : sc) {
    run_scenario(s, checked, msgs, z_msgs);
  }
  std::printf("scenarios=%zu  halo-cells-verified=%zu  packed messages=%zu (of which Z=%zu)\n",
              sc.size(), checked, msgs, z_msgs);
  CHECK(z_msgs > 0U);  // packed Z must actually be exercised

  // Transfer-count comparison for a realistic block (printed as evidence, asserted as a bound).
  g_ctx = "transfer-count comparison";
  AxisGeometry g;
  g.owned = {256, 256, 256};
  g.halo = {1, 1, 1};
  g.haloed = {258, 258, 258};
  g.pitch = 258 * 258;
  const size_t old_x = hb::axis_halo_runs(g, Axis::X, AxisSide::Lower).size();
  const size_t old_y = hb::axis_halo_runs(g, Axis::Y, AxisSide::Lower).size();
  const size_t old_z = hb::axis_halo_runs(g, Axis::Z, AxisSide::Lower).size();
  std::printf(
      "256^3 local block, halo 1, 1 component, per side: run-based transfers  X=%zu Y=%zu Z=%zu"
      "  |  packed X=1 Y=1 Z=%zu\n",
      old_x, old_y, old_z, old_z);
  CHECK(old_x == 258U * 258U);
  CHECK(old_y == 258U);

  std::puts("OK");
  return 0;
}
