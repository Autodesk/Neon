// Host-side tests for spatially varying boundary data on carbon::DenseDistributed
// (HostBoundaryStore, BoundaryView::resolve / dirichlet_at, Partition::neighbour_bc with a view).
//
// No GPU and no MPI: every rank of a process grid is emulated on the host. Each rank gets the Span
// the grid would build, a HostBoundaryStore filled from a deterministic function of global position,
// and a host buffer filled the way a completed halo exchange leaves it (in-domain cells, including
// halo cells that alias a neighbour's owned cells, hold the analytic field; halo cells outside the
// domain hold a sentinel that must never be read).
//
// Parts:
//   A  Equivalence guard: constant data reproduces the per-face constants path
//      (Partition::neighbour_bc without a view) for every offset, layout and mode.
//   B  Oracle: a Dirichlet rectangle and a Dirichlet disc on otherwise Neumann faces, faces with no
//      condition, and a two-component field with a different kind per component, compared with an
//      independent per-cell reference. Offsets include diagonals, so the condition must come from the
//      face cell under the target, across rank boundaries and on unsplit axes.
//   C  Geometry: stored ranges per rank, empty ranks, coverage of every global face cell, and
//      deterministic agreement of cells stored on several ranks.
//   D  Pinning helper and validation.

#include <carbon/boundary_condition_distributed.h>
#include <carbon/boundary_mode_distributed.h>
#include <carbon/detail/dense_distributed/boundary_store.h>
#include <carbon/detail/dense_distributed/boundary_view.h>
#include <carbon/detail/dense_distributed/layout.h>
#include <carbon/detail/dense_distributed/partition.h>
#include <carbon/detail/dense_distributed/span.h>
#include <carbon/process_grid.h>
#include <carbon/stencil.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

namespace dd = carbon::detail::dense_distributed;
using carbon::Axis;
using carbon::BcKind;
using carbon::BoundaryConditions;
using carbon::BoundaryModeDistributed;
using carbon::Face;
using carbon::FaceBc;
using carbon::ProcessGrid;
using carbon::Stencil;
using dd::Idx;
using dd::Partition;
using dd::Span;

int g_checks = 0;

#define CHECK(cond, ctx)                                                        \
  do {                                                                          \
    ++g_checks;                                                                 \
    if (!(cond)) {                                                              \
      std::fprintf(stderr, "FAIL %s:%d  %s  [%s]\n", __FILE__, __LINE__, #cond, \
                   std::string(ctx).c_str());                                   \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

constexpr double kSentinel = 1.0e30;

[[nodiscard]] auto
field(int64_t x,
      int64_t y,
      int64_t z,
      int comp) -> double {
  const double fx = static_cast<double>(x);
  const double fy = static_cast<double>(y);
  const double fz = static_cast<double>(z);
  return 1.0 + 0.37 * fx + 0.11 * fy + 0.053 * fz + 0.013 * fx * fy + 0.007 * fy * fz -
         0.003 * fx * fz + 0.0011 * fx * fy * fz + 100.0 * comp;
}

[[nodiscard]] auto
cross_stencil(int radius) -> Stencil {
  std::vector<carbon::int32_3d> pts;
  for (int r = 1; r <= radius; ++r) {
    pts.push_back({r, 0, 0});
    pts.push_back({-r, 0, 0});
    pts.push_back({0, r, 0});
    pts.push_back({0, -r, 0});
    pts.push_back({0, 0, r});
    pts.push_back({0, 0, -r});
  }
  return Stencil(pts);
}

[[nodiscard]] auto
mod_pos(int64_t v,
        int64_t n) -> int64_t {
  const int64_t r = v % n;
  return r < 0 ? r + n : r;
}

struct Layout {
  std::string name;
  ProcessGrid pg;
};

[[nodiscard]] auto
layouts() -> std::vector<Layout> {
  return {
      {"single(1,1,1)", ProcessGrid::custom(1, 1, 1, 1)},
      {"slab(1,1,2)", ProcessGrid::custom(2, 1, 1, 2)},
      {"slab(1,1,3)", ProcessGrid::custom(3, 1, 1, 3)},
      {"pencil(1,2,2)", ProcessGrid::custom(4, 1, 2, 2)},
      {"pencil(2,1,2)", ProcessGrid::custom(4, 2, 1, 2)},
      {"brick(2,2,2)", ProcessGrid::custom(8, 2, 2, 2)},
  };
}

[[nodiscard]] auto
mode_name(BoundaryModeDistributed m) -> std::string {
  std::string s;
  s += carbon::is_periodic(m, Axis::X) ? "X" : "";
  s += carbon::is_periodic(m, Axis::Y) ? "Y" : "";
  s += carbon::is_periodic(m, Axis::Z) ? "Z" : "";
  return s.empty() ? "open" : "periodic:" + s;
}

[[nodiscard]] auto
modes() -> std::vector<BoundaryModeDistributed> {
  using M = BoundaryModeDistributed;
  return {M::Open, M::PeriodicZ, M::PeriodicX, M::PeriodicX | M::PeriodicY};
}

// One emulated rank.
struct RankView {
  Span span;
  std::vector<double> buf;  // components * linear_storage_elements
  carbon::int64_3d owned{};
  carbon::int64_3d halo{};
  dd::HostBoundaryStore store;
};

template <typename Fn>
[[nodiscard]] auto
build_rank(const carbon::int64_3d& global,
           const ProcessGrid& pg,
           carbon::int32_3d coords,
           BoundaryModeDistributed mode,
           const Stencil& stencil,
           float spacing,
           int comps,
           const BoundaryConditions& span_bc,
           Fn&& cond) -> RankView {
  const carbon::int64_3d halo = dd::halo_extent(pg, mode, stencil);
  const dd::RankChunk chunk = dd::rank_chunk(global, pg, coords, halo);
  RankView rv;
  rv.span = Span(chunk.haloed, global, chunk.xy_pitch, chunk.offset, chunk.halo, mode,
                 carbon::float32_3d{}, spacing, span_bc);
  rv.owned = chunk.owned;
  rv.halo = chunk.halo;
  rv.store = dd::HostBoundaryStore(rv.span, stencil.radius3(), comps);
  rv.store.fill(global, cond);
  const size_t stride = static_cast<size_t>(rv.span.linear_storage_elements());
  rv.buf.assign(stride * static_cast<size_t>(comps), kSentinel);
  const int64_t gdim[3] = {global.x, global.y, global.z};
  const Axis axes[3] = {Axis::X, Axis::Y, Axis::Z};
  for (int c = 0; c < comps; ++c) {
    for (int64_t bz = 0; bz < chunk.haloed.z; ++bz) {
      for (int64_t by = 0; by < chunk.haloed.y; ++by) {
        for (int64_t bx = 0; bx < chunk.haloed.x; ++bx) {
          const int64_t g[3] = {bx + chunk.offset.x, by + chunk.offset.y, bz + chunk.offset.z};
          int64_t w[3];
          bool fillable = true;
          for (int a = 0; a < 3; ++a) {
            const bool out = g[a] < 0 || g[a] >= gdim[a];
            if (!out) {
              w[a] = g[a];
            } else if (carbon::is_periodic(mode, axes[a])) {
              w[a] = mod_pos(g[a], gdim[a]);
            } else {
              fillable = false;
            }
          }
          if (fillable) {
            rv.buf[static_cast<size_t>(rv.span.linear_offset(Idx(bx, by, bz), c))] =
                field(w[0], w[1], w[2], c);
          }
        }
      }
    }
  }
  return rv;
}

template <typename Fn>
void
for_each_rank_cell_offset(const RankView& rv,
                          const Stencil& stencil,
                          Fn&& fn) {
  const carbon::int32_3d r3 = stencil.radius3();
  for (int64_t iz = 0; iz < rv.owned.z; ++iz) {
    for (int64_t iy = 0; iy < rv.owned.y; ++iy) {
      for (int64_t ix = 0; ix < rv.owned.x; ++ix) {
        const Idx idx(rv.halo.x + ix, rv.halo.y + iy, rv.halo.z + iz);
        for (int dz = -r3.z; dz <= r3.z; ++dz) {
          for (int dy = -r3.y; dy <= r3.y; ++dy) {
            for (int dx = -r3.x; dx <= r3.x; ++dx) {
              fn(idx, carbon::int8_3d{static_cast<int8_t>(dx), static_cast<int8_t>(dy),
                                      static_cast<int8_t>(dz)});
            }
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Part A: constant data equals the per-face constants path.
// ---------------------------------------------------------------------------------------------
void
part_a_equivalence() {
  const carbon::int64_3d global{9, 8, 7};
  using carbon::dirichlet;
  using carbon::neumann;
  const BoundaryConditions all = BoundaryConditions::none()
                                     .with(Face::XLow, neumann(0.0))
                                     .with(Face::XHigh, neumann(0.75))
                                     .with(Face::YLow, dirichlet(-1.25))
                                     .with(Face::YHigh, neumann(0.0))
                                     .with(Face::ZLow, dirichlet(2.0))
                                     .with(Face::ZHigh, neumann(-0.5));
  const BoundaryConditions one = BoundaryConditions::none().with(Face::ZHigh, dirichlet(1.0));
  for (const BoundaryConditions& bc : {all, one}) {
    for (int radius = 1; radius <= 2; ++radius) {
      const Stencil stencil = cross_stencil(radius);
      for (const BoundaryModeDistributed mode : modes()) {
        // Conditions on a periodic axis are rejected by the grid; keep them off the spec here.
        BoundaryConditions spec = bc;
        const Axis axes[3] = {Axis::X, Axis::Y, Axis::Z};
        for (int f = 0; f < carbon::kFaceCount; ++f) {
          if (carbon::is_periodic(mode, axes[f / 2])) {
            spec = spec.with(static_cast<Face>(f), FaceBc{});
          }
        }
        for (const Layout& lay : layouts()) {
          for (int rank = 0; rank < lay.pg.world_size(); ++rank) {
            const std::string ctx = "A r=" + std::to_string(radius) + " " + mode_name(mode) + " " +
                                    lay.name + " rank=" + std::to_string(rank);
            const RankView rv = build_rank(global, lay.pg, lay.pg.coords_of(rank), mode, stencil,
                                           0.5F, 1, spec, [&](Face f, carbon::int64_3d, int) {
                                             return spec.at(f);
                                           });
            const Partition<double, 1> part(const_cast<double*>(rv.buf.data()));
            const dd::BoundaryView view = rv.store.host_view();
            for_each_rank_cell_offset(rv, stencil, [&](const Idx& idx, carbon::int8_3d off) {
              const auto a = part.neighbour_bc(rv.span, idx, off, 0);
              const auto b = part.neighbour_bc(rv.span, view, idx, off, 0);
              CHECK(a.valid == b.valid, ctx);
              if (a.valid) {
                CHECK(a.value == b.value, ctx);
              }
            });
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Part B: varying data against an independent oracle.
// ---------------------------------------------------------------------------------------------

// The spatially varying conditions under test, a deterministic function of global position.
// comps == 1: ZHigh = Dirichlet rectangle in Neumann; XLow = Dirichlet disc in Neumann; YHigh has no
// condition elsewhere than a Neumann strip; other faces None.
// comps == 2: ZLow carries Dirichlet for component 0 and Neumann for component 1.
[[nodiscard]] auto
cond_patches(Face f,
             int64_t t0,
             int64_t t1,
             int comp) -> FaceBc {
  using carbon::dirichlet;
  using carbon::neumann;
  switch (f) {
    case Face::ZHigh:  // tangential (x, y)
      if (t0 >= 3 && t0 < 6 && t1 >= 2 && t1 < 5) {
        return dirichlet(2.0 + 0.1 * static_cast<double>(t0) - 0.05 * static_cast<double>(t1));
      }
      return neumann(0.5);
    case Face::XLow: {  // tangential (y, z)
      const double dy = static_cast<double>(t0) - 3.5;
      const double dz = static_cast<double>(t1) - 3.0;
      if (dy * dy + dz * dz <= 2.6 * 2.6) {
        return dirichlet(-1.0 + 0.2 * static_cast<double>(t1));
      }
      return neumann(0.0);
    }
    case Face::YHigh:  // tangential (x, z)
      if (t1 >= 2 && t1 < 4) {
        return neumann(-0.75);
      }
      return {};
    case Face::ZLow:  // tangential (x, y): per-component kind
      return comp == 0 ? dirichlet(0.25 * static_cast<double>(t0)) : neumann(1.5);
    default:
      return {};
  }
}

[[nodiscard]] auto
cond_for(int comps,
         Face f,
         const carbon::int64_3d& g,
         int comp) -> FaceBc {
  const int a = static_cast<int>(f) / 2;
  const int64_t gg[3] = {g.x, g.y, g.z};
  const int64_t t0 = gg[dd::tangential_axis(a, 0)];
  const int64_t t1 = gg[dd::tangential_axis(a, 1)];
  if (comps == 1) {
    return f == Face::ZLow ? FaceBc{} : cond_patches(f, t0, t1, comp);
  }
  return f == Face::ZLow ? cond_patches(f, t0, t1, comp) : FaceBc{};
}

struct Expected {
  bool valid{false};
  double value{0.0};
};

[[nodiscard]] auto
oracle(const carbon::int64_3d& global,
       BoundaryModeDistributed mode,
       int comps,
       float spacing,
       const int64_t cell_global[3],
       carbon::int8_3d off,
       int comp) -> Expected {
  const int64_t gdim[3] = {global.x, global.y, global.z};
  const Axis axes[3] = {Axis::X, Axis::Y, Axis::Z};
  const int64_t o[3] = {off.x, off.y, off.z};
  int64_t t[3];
  int out_axes = 0;
  int bc_axis = -1;
  for (int a = 0; a < 3; ++a) {
    t[a] = cell_global[a] + o[a];
    if (t[a] >= 0 && t[a] < gdim[a]) {
      continue;
    }
    if (carbon::is_periodic(mode, axes[a])) {
      t[a] = mod_pos(t[a], gdim[a]);
      continue;
    }
    ++out_axes;
    bc_axis = a;
  }
  if (out_axes == 0) {
    return {true, field(t[0], t[1], t[2], comp)};
  }
  if (out_axes > 1) {
    return {};
  }
  const bool is_low = t[bc_axis] < 0;
  const int64_t k = is_low ? -t[bc_axis] : t[bc_axis] - (gdim[bc_axis] - 1);
  int64_t m[3] = {t[0], t[1], t[2]};
  m[bc_axis] = is_low ? k : (gdim[bc_axis] - 1) - k;
  if (m[bc_axis] < 0 || m[bc_axis] >= gdim[bc_axis]) {
    return {};
  }
  // The condition is the one at the face cell under the target: face axis at 0 / extent-1, the
  // tangential coordinates of the target.
  int64_t fc[3] = {t[0], t[1], t[2]};
  fc[bc_axis] = is_low ? 0 : gdim[bc_axis] - 1;
  const Face f = static_cast<Face>(2 * bc_axis + (is_low ? 0 : 1));
  const FaceBc bc = cond_for(comps, f, carbon::int64_3d{fc[0], fc[1], fc[2]}, comp);
  if (bc.kind == BcKind::None) {
    return {};
  }
  const double u_m = field(m[0], m[1], m[2], comp);
  if (bc.kind == BcKind::Dirichlet) {
    return {true, 2.0 * bc.value - u_m};
  }
  return {true, u_m + 2.0 * static_cast<double>(k) * static_cast<double>(spacing) * bc.value};
}

void
run_oracle_case(int comps,
                int radius,
                BoundaryModeDistributed mode) {
  const carbon::int64_3d global{9, 8, 7};
  const Stencil stencil = cross_stencil(radius);
  const float spacing = 0.5F;
  for (const Layout& lay : layouts()) {
    for (int rank = 0; rank < lay.pg.world_size(); ++rank) {
      const std::string ctx = "B comps=" + std::to_string(comps) + " r=" + std::to_string(radius) +
                              " " + mode_name(mode) + " " + lay.name +
                              " rank=" + std::to_string(rank);
      const RankView rv =
          build_rank(global, lay.pg, lay.pg.coords_of(rank), mode, stencil, spacing, comps,
                     BoundaryConditions::none(), [&](Face f, carbon::int64_3d g, int c) {
                       return cond_for(comps, f, g, c);
                     });
      const dd::BoundaryView view = rv.store.host_view();
      const carbon::int64_3d origin = rv.span.offset();
      const auto run = [&](const auto& part) {
        for_each_rank_cell_offset(rv, stencil, [&](const Idx& idx, carbon::int8_3d off) {
          const carbon::int64_3d c = idx.cell();
          const int64_t cg[3] = {c.x + origin.x, c.y + origin.y, c.z + origin.z};
          for (int comp = 0; comp < comps; ++comp) {
            const Expected e = oracle(global, mode, comps, spacing, cg, off, comp);
            const auto got = part.neighbour_bc(rv.span, view, idx, off, comp);
            CHECK(got.valid == e.valid, ctx);
            if (e.valid) {
              CHECK(std::fabs(got.value - e.value) <= 1e-12 * (1.0 + std::fabs(e.value)), ctx);
              CHECK(std::fabs(got.value) < 1.0e20, ctx + " (sentinel consumed)");
            }
          }
        });
      };
      if (comps == 1) {
        run(Partition<double, 1>(const_cast<double*>(rv.buf.data())));
      } else {
        run(Partition<double, 2>(const_cast<double*>(rv.buf.data())));
      }
    }
  }
}

void
part_b_oracle() {
  using M = BoundaryModeDistributed;
  for (int radius = 1; radius <= 2; ++radius) {
    for (const M mode : modes()) {
      run_oracle_case(1, radius, mode);
      run_oracle_case(2, radius, mode);
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Part C: geometry, coverage, deterministic agreement.
// ---------------------------------------------------------------------------------------------
void
part_c_geometry() {
  using M = BoundaryModeDistributed;
  struct Case {
    carbon::int64_3d global;
    M mode;
  };
  const std::vector<Case> cases = {{{9, 8, 7}, M::Open},
                                   {{9, 8, 7}, M::PeriodicZ},
                                   {{6, 6, 2}, M::Open},
                                   {{5, 7, 3}, M::PeriodicX}};
  for (int radius = 1; radius <= 2; ++radius) {
    const Stencil stencil = cross_stencil(radius);
    const carbon::int32_3d r3 = stencil.radius3();
    for (const Case& cs : cases) {
      if (std::min({cs.global.x, cs.global.y, cs.global.z}) <= radius) {
        continue;  // a condition on such an axis is rejected (checked in part D)
      }
      for (const Layout& lay : layouts()) {
        const std::string base = "C r=" + std::to_string(radius) + " " + mode_name(cs.mode) + " " +
                                 lay.name + " global=" + std::to_string(cs.global.x) + "x" +
                                 std::to_string(cs.global.y) + "x" + std::to_string(cs.global.z);
        const carbon::int64_3d halo = dd::halo_extent(lay.pg, cs.mode, stencil);
        const int64_t gdim[3] = {cs.global.x, cs.global.y, cs.global.z};
        const int64_t rad[3] = {r3.x, r3.y, r3.z};
        const Axis axes[3] = {Axis::X, Axis::Y, Axis::Z};
        // covered[face] = set of (t0, t1) stored by at least one rank; value map checks agreement.
        std::map<std::tuple<int, int64_t, int64_t, int>, std::pair<uint8_t, double>> seen;
        std::map<std::tuple<int, int64_t, int64_t>, bool> covered;
        for (int rank = 0; rank < lay.pg.world_size(); ++rank) {
          const carbon::int32_3d co = lay.pg.coords_of(rank);
          const dd::RankChunk chunk = dd::rank_chunk(cs.global, lay.pg, co, halo);
          // independent owned range from the split sizes
          const int32_t cc[3] = {co.x, co.y, co.z};
          const int32_t dims[3] = {lay.pg.dims().x, lay.pg.dims().y, lay.pg.dims().z};
          int64_t lo[3], hi[3];
          for (int a = 0; a < 3; ++a) {
            const auto counts = dd::split_extent(gdim[a], static_cast<size_t>(dims[a]));
            lo[a] = 0;
            for (int i = 0; i < cc[a]; ++i) {
              lo[a] += static_cast<int64_t>(counts[static_cast<size_t>(i)]);
            }
            hi[a] = lo[a] + static_cast<int64_t>(counts[static_cast<size_t>(cc[a])]);
          }
          const bool has_cells = hi[0] > lo[0] && hi[1] > lo[1] && hi[2] > lo[2];
          const Span span(chunk.haloed, cs.global, chunk.xy_pitch, chunk.offset, chunk.halo,
                          cs.mode);
          dd::HostBoundaryStore store(span, r3, 1);
          store.fill(cs.global, [&](Face f, carbon::int64_3d g, int c) {
            return FaceBc{BcKind::Dirichlet, static_cast<double>(g.x * 10000 + g.y * 100 + g.z) +
                                                 static_cast<double>(static_cast<int>(f)) * 0.001 +
                                                 c};
          });
          for (int f = 0; f < carbon::kFaceCount; ++f) {
            const int a = f / 2;
            const bool high = (f % 2) == 1;
            const dd::FaceRange& r = store.range(static_cast<Face>(f));
            const bool touches = has_cells && !carbon::is_periodic(cs.mode, axes[a]) &&
                                 (high ? hi[a] == gdim[a] : lo[a] == 0);
            if (!touches) {
              CHECK(r.cells() == 0,
                    base + " rank=" + std::to_string(rank) + " untouched face stores 0");
              continue;
            }
            for (int w = 0; w < 2; ++w) {
              const int t = dd::tangential_axis(a, w);
              int64_t elo = std::max<int64_t>(0, lo[t] - rad[t]);
              int64_t ehi = std::min<int64_t>(gdim[t], hi[t] + rad[t]);
              if (carbon::is_periodic(cs.mode, axes[t])) {
                elo = 0;
                ehi = gdim[t];
              }
              CHECK(r.lo[w] == elo && r.hi[w] == ehi, base + " stored range");
            }
            const dd::BoundaryView view = store.host_view();
            for (int64_t j = r.lo[1]; j < r.hi[1]; ++j) {
              for (int64_t i = r.lo[0]; i < r.hi[0]; ++i) {
                covered[{f, i, j}] = true;
                const FaceBc bc = view.faces[f].at(i, j, 0);
                CHECK(bc.kind == BcKind::Dirichlet, base + " stored kind");
                const auto key = std::make_tuple(f, i, j, 0);
                const auto it = seen.find(key);
                if (it == seen.end()) {
                  seen[key] = {static_cast<uint8_t>(bc.kind), bc.value};
                } else {
                  CHECK(it->second.second == bc.value, base + " ranks disagree on a shared cell");
                }
              }
            }
          }
        }
        // every global face cell on a non-periodic axis is stored by at least one rank
        for (int f = 0; f < carbon::kFaceCount; ++f) {
          const int a = f / 2;
          if (carbon::is_periodic(cs.mode, axes[a])) {
            continue;
          }
          const int t0 = dd::tangential_axis(a, 0);
          const int t1 = dd::tangential_axis(a, 1);
          for (int64_t j = 0; j < gdim[t1]; ++j) {
            for (int64_t i = 0; i < gdim[t0]; ++i) {
              CHECK(covered.count({f, i, j}) == 1, base + " face cell not stored by any rank");
            }
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Part D: pinning helper and validation.
// ---------------------------------------------------------------------------------------------
void
part_d_pinning_and_validation() {
  const carbon::int64_3d global{6, 6, 6};
  const Stencil stencil = cross_stencil(1);
  const ProcessGrid pg = ProcessGrid::custom(1, 1, 1, 1);
  const auto make = [&](auto&& cond) {
    return build_rank(global, pg, {0, 0, 0}, BoundaryModeDistributed::Open, stencil, 1.0F, 1,
                      BoundaryConditions::none(), cond);
  };
  // XLow Dirichlet 1 (an electrode patch only for y < 3), ZHigh Dirichlet 5, rest Neumann.
  const RankView rv = make([](Face f, carbon::int64_3d g, int) -> FaceBc {
    if (f == Face::XLow) {
      return g.y < 3 ? carbon::dirichlet(1.0) : carbon::neumann(0.0);
    }
    if (f == Face::ZHigh) {
      return carbon::dirichlet(5.0);
    }
    return carbon::neumann(0.0);
  });
  const dd::BoundaryView view = rv.store.host_view();
  const auto at = [&](int64_t x, int64_t y, int64_t z, double& v) {
    return view.dirichlet_at(rv.span, Idx(x + rv.halo.x, y + rv.halo.y, z + rv.halo.z), 0, v);
  };
  double v = 0.0;
  CHECK(at(0, 1, 2, v) && v == 1.0, "D XLow electrode cell is Dirichlet 1");
  CHECK(!at(0, 4, 2, v), "D XLow insulating cell is not Dirichlet");
  CHECK(at(3, 3, 5, v) && v == 5.0, "D ZHigh cell is Dirichlet 5");
  CHECK(at(0, 1, 5, v) && v == 1.0, "D edge cell takes the first Dirichlet face in order (XLow)");
  CHECK(at(0, 4, 5, v) && v == 5.0,
        "D edge cell falls through to ZHigh when XLow is Neumann there");
  CHECK(!at(3, 3, 3, v), "D interior cell is not on any face");

  const auto throws = [&](auto&& cond, const Stencil& st, const carbon::int64_3d& g) {
    try {
      (void)build_rank(g, pg, {0, 0, 0}, BoundaryModeDistributed::Open, st, 1.0F, 1,
                       BoundaryConditions::none(), cond);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  CHECK(throws(
            [](Face, carbon::int64_3d, int) {
              return carbon::dirichlet(std::numeric_limits<double>::quiet_NaN());
            },
            stencil, global),
        "D non-finite value must throw");
  CHECK(throws(
            [](Face f, carbon::int64_3d, int) {
              return f == Face::ZLow ? carbon::neumann() : FaceBc{};
            },
            stencil, carbon::int64_3d{6, 6, 1}),
        "D extent equal to the radius must throw");
  CHECK(!throws(
            [](Face, carbon::int64_3d, int) {
              return FaceBc{};
            },
            stencil, carbon::int64_3d{6, 6, 1}),
        "D no conditions never throws");
}

}  // namespace

auto
main() -> int {
  part_a_equivalence();
  std::printf("part A (constant data == per-face constants): ok\n");
  part_b_oracle();
  std::printf("part B (patches, per-component, diagonal reads vs oracle): ok\n");
  part_c_geometry();
  std::printf("part C (geometry, coverage, agreement): ok\n");
  part_d_pinning_and_validation();
  std::printf("part D (pinning helper, validation): ok\n");
  std::printf("test_boundary_data_host: %d checks passed\n", g_checks);
  return 0;
}
