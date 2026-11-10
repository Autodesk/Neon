// Host-side tests for per-face boundary conditions on carbon::DenseDistributed
// (BoundaryConditions, Span::boundary_conditions, neighbour_idx_bc, Partition::neighbour_bc).
//
// No GPU and no MPI: every rank of a process grid is emulated on the host. Each rank gets the Span
// the grid would build (halo_extent + rank_chunk) and a host buffer filled the way a completed halo
// exchange leaves it: in-domain cells (including halo cells that alias a neighbour rank's owned
// cells) hold the analytic global field, periodic halo cells hold the wrapped value, and halo cells
// outside the domain on an open face hold a sentinel that must never be read.
//
// Parts:
//   A  Regression guard: with no boundary conditions neighbour_bc == neighbour (value and valid)
//      for every offset, every layout, every boundary mode.
//   B  Oracle: with conditions, neighbour_bc matches an independent per-cell reference (restated
//      from the maths in boundary_condition_distributed.h) for every owned cell and every offset
//      within the stencil radius, on slab / pencil / brick layouts, radius 1 and 2, including
//      mixed periodic + condition axes. Also checks the sentinel is never consumed.
//   C  Analytic: a linear field is reproduced exactly by the Neumann (matching outward derivative)
//      and Dirichlet (odd about the boundary value) virtual values, on low and high faces.
//   D  Documented limits: edge / corner reads and faces without a condition stay invalid.
//   E  Types: BoundaryConditions builder / accessors, Span stays trivially copyable.

#include <carbon/boundary_condition_distributed.h>
#include <carbon/boundary_mode_distributed.h>
#include <carbon/detail/dense_distributed/layout.h>
#include <carbon/detail/dense_distributed/partition.h>
#include <carbon/detail/dense_distributed/span.h>
#include <carbon/process_grid.h>
#include <carbon/stencil.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using carbon::Axis;
using carbon::BcKind;
using carbon::BoundaryConditions;
using carbon::BoundaryModeDistributed;
using carbon::DecompositionMode;
using carbon::Face;
using carbon::ProcessGrid;
using carbon::Stencil;
using carbon::detail::dense_distributed::Idx;
using carbon::detail::dense_distributed::Partition;
using carbon::detail::dense_distributed::Span;

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

// Analytic global field (non-trivial in all three axes so mirrored values are distinguishable).
[[nodiscard]] auto
field(int64_t x,
      int64_t y,
      int64_t z) -> double {
  const double fx = static_cast<double>(x);
  const double fy = static_cast<double>(y);
  const double fz = static_cast<double>(z);
  return 1.0 + 0.37 * fx + 0.11 * fy + 0.053 * fz + 0.013 * fx * fy + 0.007 * fy * fz -
         0.003 * fx * fz + 0.0011 * fx * fy * fz;
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

// One emulated rank: its Span and a filled host buffer (single component).
struct RankView {
  Span span;
  std::vector<double> buf;
  carbon::int64_3d owned{};
  carbon::int64_3d halo{};
};

[[nodiscard]] auto
build_rank(const carbon::int64_3d& global,
           const ProcessGrid& pg,
           carbon::int32_3d coords,
           BoundaryModeDistributed mode,
           const Stencil& stencil,
           float spacing,
           const BoundaryConditions& bc) -> RankView {
  namespace dd = carbon::detail::dense_distributed;
  const carbon::int64_3d halo = dd::halo_extent(pg, mode, stencil);
  const dd::RankChunk chunk = dd::rank_chunk(global, pg, coords, halo);
  RankView rv;
  rv.span = Span(chunk.haloed, global, chunk.xy_pitch, chunk.offset, chunk.halo, mode,
                 carbon::float32_3d{}, spacing, bc);
  rv.owned = chunk.owned;
  rv.halo = chunk.halo;
  rv.buf.assign(static_cast<size_t>(rv.span.linear_storage_elements()), kSentinel);
  const int64_t gdim[3] = {global.x, global.y, global.z};
  const Axis axes[3] = {Axis::X, Axis::Y, Axis::Z};
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
          rv.buf[static_cast<size_t>(rv.span.linear_offset(Idx(bx, by, bz), 0))] =
              field(w[0], w[1], w[2]);
        }
      }
    }
  }
  return rv;
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
  if (carbon::is_periodic(m, Axis::X)) {
    s += "X";
  }
  if (carbon::is_periodic(m, Axis::Y)) {
    s += "Y";
  }
  if (carbon::is_periodic(m, Axis::Z)) {
    s += "Z";
  }
  return s.empty() ? "open" : "periodic:" + s;
}

[[nodiscard]] auto
modes() -> std::vector<BoundaryModeDistributed> {
  using M = BoundaryModeDistributed;
  return {M::Open, M::PeriodicZ, M::PeriodicX, M::PeriodicX | M::PeriodicY,
          M::PeriodicX | M::PeriodicY | M::PeriodicZ};
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
// Part A: no conditions => neighbour_bc is neighbour.
// ---------------------------------------------------------------------------------------------
void
part_a_regression_guard() {
  const carbon::int64_3d global{9, 8, 7};
  for (int radius = 1; radius <= 2; ++radius) {
    const Stencil stencil = cross_stencil(radius);
    for (const BoundaryModeDistributed mode : modes()) {
      for (const Layout& lay : layouts()) {
        for (int rank = 0; rank < lay.pg.world_size(); ++rank) {
          const std::string ctx = "A r=" + std::to_string(radius) + " " + mode_name(mode) + " " +
                                  lay.name + " rank=" + std::to_string(rank);
          const RankView rv = build_rank(global, lay.pg, lay.pg.coords_of(rank), mode, stencil,
                                         0.5F, BoundaryConditions::none());
          const Partition<double, 1> part(const_cast<double*>(rv.buf.data()));
          for_each_rank_cell_offset(rv, stencil, [&](const Idx& idx, carbon::int8_3d off) {
            const auto a = part.neighbour(rv.span, idx, off, 0);
            const auto b = part.neighbour_bc(rv.span, idx, off, 0);
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

// ---------------------------------------------------------------------------------------------
// Part B: with conditions, neighbour_bc matches an independent oracle.
// ---------------------------------------------------------------------------------------------
struct Expected {
  bool valid{false};
  double value{0.0};
};

[[nodiscard]] auto
oracle(const carbon::int64_3d& global,
       BoundaryModeDistributed mode,
       const BoundaryConditions& bc,
       float spacing,
       const int64_t cell_global[3],
       carbon::int8_3d off) -> Expected {
  const int64_t gdim[3] = {global.x, global.y, global.z};
  const Axis axes[3] = {Axis::X, Axis::Y, Axis::Z};
  const Face low[3] = {Face::XLow, Face::YLow, Face::ZLow};
  const Face high[3] = {Face::XHigh, Face::YHigh, Face::ZHigh};
  const int64_t o[3] = {off.x, off.y, off.z};
  int64_t t[3];
  int bc_axis = -1;
  int bc_count = 0;
  for (int a = 0; a < 3; ++a) {
    t[a] = cell_global[a] + o[a];
    const bool out = t[a] < 0 || t[a] >= gdim[a];
    if (!out) {
      continue;
    }
    if (carbon::is_periodic(mode, axes[a])) {
      t[a] = mod_pos(t[a], gdim[a]);
      continue;
    }
    const Face f = t[a] < 0 ? low[a] : high[a];
    if (bc.at(f).kind == BcKind::None) {
      return {};
    }
    bc_axis = a;
    ++bc_count;
  }
  if (bc_count == 0) {
    return {true, field(t[0], t[1], t[2])};
  }
  if (bc_count > 1) {
    return {};
  }
  const bool is_low = t[bc_axis] < 0;
  const int64_t k = is_low ? -t[bc_axis] : t[bc_axis] - (gdim[bc_axis] - 1);
  int64_t m[3] = {t[0], t[1], t[2]};
  m[bc_axis] = is_low ? k : (gdim[bc_axis] - 1) - k;
  const double u_m = field(m[0], m[1], m[2]);
  const carbon::FaceBc fbc = bc.at(is_low ? low[bc_axis] : high[bc_axis]);
  const double h = static_cast<double>(spacing);
  if (fbc.kind == BcKind::Dirichlet) {
    return {true, 2.0 * fbc.value - u_m};
  }
  return {true, u_m + 2.0 * static_cast<double>(k) * h * fbc.value};
}

void
run_oracle_case(const std::string& label,
                const carbon::int64_3d& global,
                int radius,
                BoundaryModeDistributed mode,
                const BoundaryConditions& bc) {
  const Stencil stencil = cross_stencil(radius);
  const float spacing = 0.5F;
  for (const Layout& lay : layouts()) {
    for (int rank = 0; rank < lay.pg.world_size(); ++rank) {
      const std::string ctx = "B " + label + " r=" + std::to_string(radius) + " " +
                              mode_name(mode) + " " + lay.name + " rank=" + std::to_string(rank);
      const RankView rv =
          build_rank(global, lay.pg, lay.pg.coords_of(rank), mode, stencil, spacing, bc);
      const Partition<double, 1> part(const_cast<double*>(rv.buf.data()));
      const carbon::int64_3d origin = rv.span.offset();
      for_each_rank_cell_offset(rv, stencil, [&](const Idx& idx, carbon::int8_3d off) {
        const carbon::int64_3d c = idx.cell();
        const int64_t cg[3] = {c.x + origin.x, c.y + origin.y, c.z + origin.z};
        const Expected e = oracle(global, mode, bc, spacing, cg, off);
        const auto got = part.neighbour_bc(rv.span, idx, off, 0);
        CHECK(got.valid == e.valid, ctx);
        if (e.valid) {
          CHECK(std::fabs(got.value - e.value) <= 1e-12 * (1.0 + std::fabs(e.value)), ctx);
          CHECK(std::fabs(got.value) < 1.0e20, ctx + " (sentinel consumed)");
        }
      });
    }
  }
}

void
part_b_oracle() {
  const carbon::int64_3d global{9, 8, 7};
  using M = BoundaryModeDistributed;
  using carbon::dirichlet;
  using carbon::neumann;
  const BoundaryConditions all_neumann = BoundaryConditions::none()
                                             .with(Face::XLow, neumann(0.0))
                                             .with(Face::XHigh, neumann(0.75))
                                             .with(Face::YLow, neumann(-1.25))
                                             .with(Face::YHigh, neumann(0.0))
                                             .with(Face::ZLow, neumann(2.0))
                                             .with(Face::ZHigh, neumann(-0.5));
  const BoundaryConditions mixed = BoundaryConditions::none()
                                       .with(Face::XLow, neumann(0.0))
                                       .with(Face::XHigh, neumann(0.0))
                                       .with(Face::YLow, neumann(0.0))
                                       .with(Face::YHigh, neumann(0.0))
                                       .with(Face::ZLow, dirichlet(0.0))
                                       .with(Face::ZHigh, dirichlet(-3.5));
  const BoundaryConditions one_face = BoundaryConditions::none().with(Face::ZHigh, dirichlet(1.0));
  // The octant setup used by the magnetostatics example: symmetry planes and pole face.
  const BoundaryConditions octant = BoundaryConditions::none()
                                        .with(Face::XLow, neumann(0.0))
                                        .with(Face::YLow, neumann(0.0))
                                        .with(Face::ZLow, dirichlet(0.0))
                                        .with(Face::ZHigh, dirichlet(-2.0))
                                        .with(Face::XHigh, neumann(0.0))
                                        .with(Face::YHigh, neumann(0.0));
  for (int radius = 1; radius <= 2; ++radius) {
    run_oracle_case("all_neumann", global, radius, M::Open, all_neumann);
    run_oracle_case("mixed", global, radius, M::Open, mixed);
    run_oracle_case("one_face", global, radius, M::Open, one_face);
    run_oracle_case("octant", global, radius, M::Open, octant);
    // Mixed periodic + conditions: periodic axes carry no condition.
    run_oracle_case("periodicZ_bcXY", global, radius, M::PeriodicZ,
                    BoundaryConditions::none()
                        .with(Face::XLow, neumann(0.0))
                        .with(Face::XHigh, dirichlet(1.5))
                        .with(Face::YLow, neumann(0.5)));
    run_oracle_case("periodicX_bcZ", global, radius, M::PeriodicX,
                    BoundaryConditions::none()
                        .with(Face::ZLow, dirichlet(0.25))
                        .with(Face::ZHigh, neumann(-1.0))
                        .with(Face::YHigh, neumann(0.0)));
    run_oracle_case("periodicXY_bcZ", global, radius, M::PeriodicX | M::PeriodicY,
                    BoundaryConditions::none()
                        .with(Face::ZLow, neumann(0.0))
                        .with(Face::ZHigh, dirichlet(2.0)));
  }
}

// ---------------------------------------------------------------------------------------------
// Part C: a linear field is reproduced exactly (sign convention check).
// ---------------------------------------------------------------------------------------------
void
part_c_analytic() {
  // u(i,j,k) = b + ax*x*h + ay*y*h + az*z*h on a single rank, so physical derivatives are ax, ay, az.
  const carbon::int64_3d global{8, 8, 8};
  const float h = 0.5F;
  const double hd = static_cast<double>(h);
  const double b = 3.0;
  const double a[3] = {0.8, -1.3, 2.1};
  const auto u = [&](int64_t x, int64_t y, int64_t z) {
    return b + a[0] * static_cast<double>(x) * hd + a[1] * static_cast<double>(y) * hd +
           a[2] * static_cast<double>(z) * hd;
  };
  const Stencil stencil = cross_stencil(2);
  namespace dd = carbon::detail::dense_distributed;
  const ProcessGrid pg = ProcessGrid::custom(1, 1, 1, 1);
  const carbon::int64_3d halo = dd::halo_extent(pg, BoundaryModeDistributed::Open, stencil);
  const dd::RankChunk chunk = dd::rank_chunk(global, pg, {0, 0, 0}, halo);

  // Outward derivative on a low face is -a, on a high face +a. Dirichlet value is u at the face.
  const Face low[3] = {Face::XLow, Face::YLow, Face::ZLow};
  const Face high[3] = {Face::XHigh, Face::YHigh, Face::ZHigh};
  BoundaryConditions neu = BoundaryConditions::none();
  BoundaryConditions dir = BoundaryConditions::none();
  const int64_t last = 7;
  for (int ax = 0; ax < 3; ++ax) {
    neu = neu.with(low[ax], carbon::neumann(-a[ax])).with(high[ax], carbon::neumann(a[ax]));
    int64_t lo_c[3] = {3, 3, 3};
    int64_t hi_c[3] = {3, 3, 3};
    lo_c[ax] = 0;
    hi_c[ax] = last;
    dir = dir.with(low[ax], carbon::dirichlet(u(lo_c[0], lo_c[1], lo_c[2])))
              .with(high[ax], carbon::dirichlet(u(hi_c[0], hi_c[1], hi_c[2])));
  }
  for (const BoundaryConditions* bc : {&neu, &dir}) {
    Span span(chunk.haloed, global, chunk.xy_pitch, chunk.offset, chunk.halo,
              BoundaryModeDistributed::Open, carbon::float32_3d{}, h, *bc);
    std::vector<double> buf(static_cast<size_t>(span.linear_storage_elements()), kSentinel);
    for (int64_t z = 0; z < 8; ++z) {
      for (int64_t y = 0; y < 8; ++y) {
        for (int64_t x = 0; x < 8; ++x) {
          buf[static_cast<size_t>(span.linear_offset(Idx(x + halo.x, y + halo.y, z + halo.z), 0))] =
              u(x, y, z);
        }
      }
    }
    const Partition<double, 1> part(buf.data());
    const std::string ctx = (bc == &neu) ? "C neumann" : "C dirichlet";
    for (int ax = 0; ax < 3; ++ax) {
      for (int k = 1; k <= 2; ++k) {
        for (int side = 0; side < 2; ++side) {
          // Transverse coordinates (3,3,3) match where the constant Dirichlet value g was sampled.
          int64_t c[3] = {3, 3, 3};
          c[ax] = side == 0 ? 0 : last;
          int8_t d[3] = {0, 0, 0};
          d[ax] = static_cast<int8_t>(side == 0 ? -k : k);
          int64_t t[3] = {c[0] + d[0], c[1] + d[1], c[2] + d[2]};
          const auto got = part.neighbour_bc(span, Idx(c[0] + halo.x, c[1] + halo.y, c[2] + halo.z),
                                             carbon::int8_3d{d[0], d[1], d[2]}, 0);
          CHECK(got.valid, ctx);
          // The extrapolated linear field at the outside location.
          CHECK(std::fabs(got.value - u(t[0], t[1], t[2])) < 1e-12, ctx);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Part D: documented limits.
// ---------------------------------------------------------------------------------------------
void
part_d_limits() {
  const carbon::int64_3d global{6, 6, 6};
  const Stencil stencil = cross_stencil(1);
  const ProcessGrid pg = ProcessGrid::custom(1, 1, 1, 1);
  const BoundaryConditions bc = BoundaryConditions::none()
                                    .with(Face::XLow, carbon::neumann())
                                    .with(Face::ZHigh, carbon::dirichlet(1.0));
  const RankView rv =
      build_rank(global, pg, {0, 0, 0}, BoundaryModeDistributed::Open, stencil, 1.0F, bc);
  const Partition<double, 1> part(const_cast<double*>(rv.buf.data()));
  const auto at = [&](int64_t x, int64_t y, int64_t z, int8_t dx, int8_t dy, int8_t dz) {
    return part.neighbour_bc(rv.span, Idx(x + rv.halo.x, y + rv.halo.y, z + rv.halo.z),
                             carbon::int8_3d{dx, dy, dz}, 0);
  };
  // Single BC axis: valid.
  CHECK(at(0, 2, 2, -1, 0, 0).valid, "D xlow face read");
  CHECK(at(2, 2, 5, 0, 0, 1).valid, "D zhigh face read");
  // Edge (XLow x ZHigh): two BC axes at once.
  CHECK(!at(0, 2, 5, -1, 0, 1).valid, "D edge read is invalid");
  // Face without a condition stays invalid (historical open-face behaviour).
  CHECK(!at(2, 0, 2, 0, -1, 0).valid, "D ylow has no condition");
  CHECK(!at(5, 2, 2, 1, 0, 0).valid, "D xhigh has no condition");
  // In-domain reads unaffected.
  CHECK(at(2, 2, 2, 1, 0, 0).valid, "D interior read");
}

// ---------------------------------------------------------------------------------------------
// Part E: types.
// ---------------------------------------------------------------------------------------------
void
part_e_types() {
  static_assert(std::is_trivially_copyable_v<BoundaryConditions>);
  static_assert(std::is_trivially_copyable_v<Span>);
  static_assert(!BoundaryConditions::none().any());
  constexpr BoundaryConditions bc = BoundaryConditions::none()
                                        .with(Face::XHigh, carbon::neumann(2.0))
                                        .with(Face::ZLow, carbon::dirichlet(-1.0));
  static_assert(bc.any());
  static_assert(bc.at(Face::XHigh).kind == BcKind::Neumann && bc.at(Face::XHigh).value == 2.0);
  static_assert(bc.at(Face::ZLow).kind == BcKind::Dirichlet && bc.at(Face::ZLow).value == -1.0);
  static_assert(bc.at(Face::YLow).kind == BcKind::None);
  // A default Span carries no conditions; the Z-only overload forwards them.
  const Span def;
  CHECK(!def.boundary_conditions().any(), "E default span has no conditions");
  const Span z(carbon::int64_3d{4, 4, 6}, carbon::int64_3d{4, 4, 4}, 16, int64_t{0}, int64_t{1},
               BoundaryModeDistributed::Open, carbon::float32_3d{}, 1.0F, bc);
  CHECK(z.boundary_conditions().at(Face::ZLow).kind == BcKind::Dirichlet, "E z-only ctor forwards");
}

}  // namespace

auto
main() -> int {
  part_a_regression_guard();
  std::printf("part A (no conditions == neighbour): ok\n");
  part_b_oracle();
  std::printf("part B (oracle, layouts x radius x modes): ok\n");
  part_c_analytic();
  std::printf("part C (linear field exact): ok\n");
  part_d_limits();
  std::printf("part D (documented limits): ok\n");
  part_e_types();
  std::printf("part E (types): ok\n");
  std::printf("test_dense_distributed_bc_host: %d checks passed\n", g_checks);
  return 0;
}
