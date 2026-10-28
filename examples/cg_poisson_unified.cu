// Unified Conjugate-Gradient 3D Poisson solver across every carbon grid +
// every transport. Companion to @c jacobi_unified.cu : same Skeleton-only main
// loop, same per-cell device lambdas reused verbatim across:
//
//   - Grids:     carbon::Dense
//                carbon::DenseDisg
//                carbon::Block<DefaultStaticBlock>
//   - Backends:  CUDA  (default, nvcc)
//                SYCL  (AdaptiveCpp -DCARBON_BACKEND=Sycl)
//   - Transports: single-GPU, multi-GPU same node, cross-rank NCCL/RCCL/oneCCL
//                 (the latter with -DCARBON_ENABLE_MPI=ON + mpiexec)
//
// PDE
// ---
//   -Laplacian(u) = 0     on a 3D Cartesian box (Dense / DenseDisg) or on the
//                         retained sphere of the @c Block predicate.
//
//   Boundary conditions are encoded topologically:
//     - cells whose @c (0,0,-1) neighbour is invalid (south cap / lower z-face)
//       are Dirichlet at @c bd_min ;
//     - cells whose @c (0,0,+1) neighbour is invalid (north cap / upper z-face)
//       are Dirichlet at @c bd_max ;
//     - cells with a missing x/y face neighbour drop that term from the
//       Laplacian sum (the same implicit Neumann handling Neon uses).
//
//   On Dense / DenseDisg with @c BoundaryMode::Open the south + north faces
//   collapse to the global @c z=0 and @c z=Nz-1 planes, so the analytical
//   solution is the linear-z ramp
//       @c u(x, y, z) = bd_min + (bd_max - bd_min) * z / (Nz - 1) .
//   On Block with the bundled sphere predicate the "south cap" is the lower
//   hemisphere shell and the "north cap" is the upper hemisphere shell, so the
//   analytical solution is no longer linear-z; the example still reports the
//   final residual norm and the RMS deviation against the linear-z reference
//   purely as a sanity-of-convergence check (smaller after CG than before,
//   *not* a correctness claim).
//
// Discretization (matches Neon's @c LaplacianMatVec):
//   free cell:      @c (A x)[c] = (n_valid * x[c] - sum_valid_nghs(x)) / h^2
//   Dirichlet cell: @c (A x)[c] = x[c]                    (identity row)
//   Solver:         standard CG with Dirichlet BCs folded into the iterate
//                   ( @c b == 0 , @c r = (Free) ? b : x , then @c r -= A x ).
//
// What Skeleton buys
// ------------------
// The three CG phases ("init", "iter_pre" = update_p + matvec + dot pAp,
// "iter_post" = update_x_r + dot rTr) are each a @c carbon::Skeleton . The
// @c loader.read_stencil(in) annotation on @c matvec triggers an auto-inserted
// halo for the stencil read on every cycle, so the same source line exchanges:
//   - nothing on single-GPU,
//   - cross-GPU intra-process via @c Transfer<T> on multi-GPU same node,
//   - cross-rank via NCCL / RCCL / oneCCL when @c sys.distributed() is set.
//
// Overlap mode
// ------------
// Same four modes as @c jacobi_unified.cu , selected via @c --overlap :
//   - @c none            (default): halos run serially before each stencil step.
//   - @c standard        : Internal+Boundary split, halo concurrent with Internal.
//   - @c extended        : Standard plus one step of propagation.
//   - @c two-way-extended: Extended in both directions.
//
// CLI
// ---
//   --grid {dense|disg|block}    Grid family (default: dense)
//   --nx N --ny N --nz N         Grid extents (default 16 16 32; nz >= 2)
//   --max_iter N                 Hard cap on CG iterations (default 4 * nz)
//   --tol_rel R                  Relative tolerance ||r||/||r0|| (default 1e-10)
//   --tol_abs R                  Absolute tolerance ||r||       (default 1e-12)
//   --bd_min R --bd_max R        Dirichlet z-face values (default +1, -1)
//   --report_every N             Log every N iterations (default 25)
//   --overlap {none|standard|extended|two-way-extended}
//                                Skeleton::Options::overlap_mode for the
//                                in-iteration pipelines (default: none).
//   --dot                        Dump the inferred main-loop DAGs as GraphViz
//                                DOT on stdout (one block per Skeleton).
//   --dot-out prefix             Write GraphViz DOT + launch-order listings
//                                for init / iter_pre / iter_post as
//                                prefix_<tag>.dot and prefix_<tag>.txt.
//   --dot-matrix                 With --dot-out, also write launch-matrix and
//                                kernel-list DOT
//                                (prefix_<tag>_matrix.dot / _kernels.dot).
//   --help                       Print this help and exit.
//
// Build + run
// -----------
//   ./carbon_example_cg_poisson_unified                  # default: dense, 1 GPU
//   ./carbon_example_cg_poisson_unified --grid disg --nx 32 --ny 32 --nz 64
//   ./carbon_example_cg_poisson_unified --grid block --nx 32 --ny 32 --nz 32
//   ./carbon_example_cg_poisson_unified --grid dense --overlap standard
//   mpiexec -n 2 ./carbon_example_cg_poisson_unified --grid dense --nz 64
//
// Block + MPI cross-rank stencil parity is covered by
// tests/test_block_dist_stencil_parity.cu (see jacobi_unified.cu header).

#include <carbon/block.h>
#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/dense_disg.h>
#include <carbon/detail/backend/select.h>
#include <carbon/devices.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/reduce.h>
#include <carbon/skeleton.h>
#include <carbon/stencil.h>
#include <carbon/system.h>

#ifdef CARBON_HAVE_MPI_NCCL
#include <carbon/distributed.h>
#endif

#include <clipp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using SB = carbon::detail::block::DefaultStaticBlock;

struct Options {
  std::string grid_kind = "dense";
  int64_t nx = 16;
  int64_t ny = 16;
  int64_t nz = 32;
  size_t max_iter = 0;  // 0 = auto -> 4 * nz
  double tol_rel = 1e-10;
  double tol_abs = 1e-12;
  double bd_min = 1.0;
  double bd_max = -1.0;
  int report_every = 25;
  std::string overlap_raw;
  bool dump_dot = false;
  std::string dot_out_prefix;
  bool dot_matrix = false;
  bool show_help = false;
};

// Build a clipp parser bound directly to @p opts. Mirrors the structure of the
// pre-unified single-GPU example.
[[nodiscard]] auto
build_cli(Options& opts) -> clipp::group {
  using clipp::integer;
  using clipp::number;
  using clipp::option;
  using clipp::value;

  return clipp::group(
      (option("-h", "--help").set(opts.show_help) % "Show this help message and exit"),
      (option("--grid") & value("kind", opts.grid_kind)) %
          "Grid family: dense | disg | block (default: dense)",
      (option("--nx") & integer("nx", opts.nx)) % "Grid x extent (default 16)",
      (option("--ny") & integer("ny", opts.ny)) % "Grid y extent (default 16)",
      (option("--nz") & integer("nz", opts.nz)) % "Grid z extent (default 32, must be >= 2)",
      (option("--max_iter") & integer("max_iter", opts.max_iter)) %
          "Maximum CG iterations (default 4 * nz)",
      (option("--tol_rel") & number("tol_rel", opts.tol_rel)) %
          "Relative tolerance ||r||/||r0|| (default 1e-10)",
      (option("--tol_abs") & number("tol_abs", opts.tol_abs)) %
          "Absolute tolerance ||r|| (default 1e-12)",
      (option("--bd_min") & number("bd_min", opts.bd_min)) %
          "Dirichlet value at z=0 face / lower cap (default +1.0)",
      (option("--bd_max") & number("bd_max", opts.bd_max)) %
          "Dirichlet value at z=Nz-1 face / upper cap (default -1.0)",
      (option("--report_every") & integer("n", opts.report_every)) %
          "Log every N iterations (default 25)",
      (option("--overlap") & value("mode", opts.overlap_raw)) %
          "Skeleton overlap mode: none | standard | extended | two-way-extended (default: none)",
      (option("--dot").set(opts.dump_dot) %
       "Dump inferred main-loop Skeleton DAGs as GraphViz DOT on stdout"),
      (option("--dot-out") & value("prefix", opts.dot_out_prefix)) %
          "Write DOT + launch-order files as prefix_<tag>.dot / .txt "
          "(tag = init | iter_pre | iter_post)",
      (option("--dot-matrix").set(opts.dot_matrix) %
       "With --dot-out, also write launch-matrix and kernel-list DOT files"));
}

[[nodiscard]] auto
parse_cli(int argc,
          char** argv,
          Options& opts) -> bool {
  auto cli = build_cli(opts);
  if (!clipp::parse(argc, argv, cli)) {
    auto fmt = clipp::doc_formatting{}.doc_column(31);
    std::cout << clipp::make_man_page(cli, argv[0], fmt) << '\n';
    return false;
  }
  if (opts.show_help) {
    auto fmt = clipp::doc_formatting{}.doc_column(31);
    std::cout << clipp::make_man_page(cli, argv[0], fmt) << '\n';
    return false;
  }
  if (opts.nx <= 0 || opts.ny <= 0 || opts.nz < 2) {
    std::cerr << "cg_poisson_unified: nx, ny must be > 0 and nz must be >= 2\n";
    return false;
  }
  if (opts.max_iter == 0U) {
    opts.max_iter = static_cast<size_t>(opts.nz) * 4U;
  }
  if (opts.report_every <= 0) {
    opts.report_every = 1;
  }
  return true;
}

[[nodiscard]] auto
parse_overlap_mode(std::string_view s) -> std::optional<carbon::OverlapMode> {
  if (s == "none" || s.empty()) {
    return carbon::OverlapMode::None;
  }
  if (s == "standard") {
    return carbon::OverlapMode::Standard;
  }
  if (s == "extended") {
    return carbon::OverlapMode::Extended;
  }
  if (s == "two-way-extended" || s == "two_way_extended" || s == "twoway") {
    return carbon::OverlapMode::TwoWayExtended;
  }
  return std::nullopt;
}

[[nodiscard]] auto
overlap_mode_name(carbon::OverlapMode m) -> const char* {
  switch (m) {
    case carbon::OverlapMode::None:
      return "none";
    case carbon::OverlapMode::Standard:
      return "standard";
    case carbon::OverlapMode::Extended:
      return "extended";
    case carbon::OverlapMode::TwoWayExtended:
      return "two-way-extended";
  }
  return "?";
}

// Sphere predicate for Block, mirroring jacobi_unified.cu. Retained cells form
// an inscribed sphere of radius 0.42 * max(nx, ny, nz). The CG solver then
// sees Dirichlet on the lower / upper polar caps and Neumann on the
// equatorial belt.
[[nodiscard]] auto
sphere_predicate(int64_t nmax,
                 const carbon::int64_3d& gc) -> bool {
  const double cx = (static_cast<double>(nmax) - 1.0) * 0.5;
  const double cy = (static_cast<double>(nmax) - 1.0) * 0.5;
  const double cz = (static_cast<double>(nmax) - 1.0) * 0.5;
  const double r = static_cast<double>(nmax) * 0.42;
  const double dx = static_cast<double>(gc.x) - cx;
  const double dy = static_cast<double>(gc.y) - cy;
  const double dz = static_cast<double>(gc.z) - cz;
  return (dx * dx + dy * dy + dz * dz) <= (r * r);
}

// ---- Kernel builders (grid-agnostic via Kernel::compute / Kernel::reduce) ----

// Initialise @c u with the BC values:
//   south face (!zm.valid) -> bd_min
//   north face (!zp.valid) -> bd_max
//   everything else        -> 0
// Topology probes through @c loader.write(u) -> @c partition.neighbour are
// safe even before any write: @c .valid is determined by Span+offset only, not
// by the buffer's contents.
template <typename Grid>
[[nodiscard]] auto
build_init_u(const Grid& grid,
             typename Grid::template Field<double,
                                           1>& u,
             double bd_min,
             double bd_max,
             std::string name) -> carbon::Kernel {
  using Span = typename Grid::Span;
  using Idx = typename Grid::Idx;
  return carbon::Kernel::compute(
      grid,
      [&u, bd_min, bd_max](carbon::KernelLoader& loader) {
        auto u_p = loader.write(u);
        return [u_p, bd_min, bd_max] CARBON_HOST_DEVICE(Span s, Idx idx) -> void {
          const auto zm = u_p.neighbour(s, idx, carbon::int8_3d{0, 0, -1}, 0);
          if (!zm.valid) {
            u_p.write(s, idx, 0, bd_min);
            return;
          }
          const auto zp = u_p.neighbour(s, idx, carbon::int8_3d{0, 0, 1}, 0);
          if (!zp.valid) {
            u_p.write(s, idx, 0, bd_max);
            return;
          }
          u_p.write(s, idx, 0, 0.0);
        };
      },
      std::move(name));
}

// Set @p field to @p value on every owned (and active, for Block) cell.
template <typename Grid,
          typename T>
[[nodiscard]] auto
build_set_const(const Grid& grid,
                typename Grid::template Field<T,
                                              1>& field,
                T value,
                std::string name) -> carbon::Kernel {
  using Span = typename Grid::Span;
  using Idx = typename Grid::Idx;
  return carbon::Kernel::compute(
      grid,
      [&field, value](carbon::KernelLoader& loader) {
        auto out = loader.write(field);
        return [out, value] CARBON_HOST_DEVICE(Span s, Idx idx) -> void {
          out.write(s, idx, 0, value);
        };
      },
      std::move(name));
}

// Discrete Laplacian: out = A * in. Halo on @c in auto-inserted by Skeleton
// via @c read_stencil. Topological BC handling (south / north face = identity
// row) lets the same kernel work on Dense, DenseDisg, and Block without an
// explicit boundary-mask field.
template <typename Grid>
[[nodiscard]] auto
build_matvec(const Grid& grid,
             typename Grid::template Field<double,
                                           1>& in,
             typename Grid::template Field<double,
                                           1>& out,
             double invh2,
             std::string name) -> carbon::Kernel {
  using Span = typename Grid::Span;
  using Idx = typename Grid::Idx;
  return carbon::Kernel::compute(
      grid,
      [&in, &out, invh2](carbon::KernelLoader& loader) {
        const auto in_p = loader.read_stencil(in);
        auto out_p = loader.write(out);
        return [in_p, out_p, invh2] CARBON_HOST_DEVICE(Span s, Idx idx) -> void {
          const double center = in_p.read(s, idx, 0);
          const auto zm = in_p.neighbour(s, idx, carbon::int8_3d{0, 0, -1}, 0);
          const auto zp = in_p.neighbour(s, idx, carbon::int8_3d{0, 0, 1}, 0);
          if (!zm.valid || !zp.valid) {
            out_p.write(s, idx, 0, center);
            return;
          }
          const auto xp = in_p.neighbour(s, idx, carbon::int8_3d{1, 0, 0}, 0);
          const auto xm = in_p.neighbour(s, idx, carbon::int8_3d{-1, 0, 0}, 0);
          const auto yp = in_p.neighbour(s, idx, carbon::int8_3d{0, 1, 0}, 0);
          const auto ym = in_p.neighbour(s, idx, carbon::int8_3d{0, -1, 0}, 0);
          double sum = 0.0;
          int n_valid = 0;
          if (xp.valid) {
            sum += xp.value;
            ++n_valid;
          }
          if (xm.valid) {
            sum += xm.value;
            ++n_valid;
          }
          if (yp.valid) {
            sum += yp.value;
            ++n_valid;
          }
          if (ym.valid) {
            sum += ym.value;
            ++n_valid;
          }
          sum += zp.value;
          ++n_valid;
          sum += zm.value;
          ++n_valid;
          out_p.write(s, idx, 0, (static_cast<double>(n_valid) * center - sum) * invh2);
        };
      },
      std::move(name));
}

// r := (Dirichlet) ? u : b   — Dirichlet detected topologically.
template <typename Grid>
[[nodiscard]] auto
build_init_r(const Grid& grid,
             typename Grid::template Field<double,
                                           1>& r,
             typename Grid::template Field<double,
                                           1>& u,
             typename Grid::template Field<double,
                                           1>& b,
             std::string name) -> carbon::Kernel {
  using Span = typename Grid::Span;
  using Idx = typename Grid::Idx;
  return carbon::Kernel::compute(
      grid,
      [&r, &u, &b](carbon::KernelLoader& loader) {
        auto r_p = loader.write(r);
        const auto u_p = loader.read(u);
        const auto b_p = loader.read(b);
        return [r_p, u_p, b_p] CARBON_HOST_DEVICE(Span s, Idx idx) -> void {
          const auto zm = u_p.neighbour(s, idx, carbon::int8_3d{0, 0, -1}, 0);
          const auto zp = u_p.neighbour(s, idx, carbon::int8_3d{0, 0, 1}, 0);
          const bool is_dirichlet = (!zm.valid || !zp.valid);
          const double v = is_dirichlet ? u_p.read(s, idx, 0) : b_p.read(s, idx, 0);
          r_p.write(s, idx, 0, v);
        };
      },
      std::move(name));
}

// r := r - s.
template <typename Grid>
[[nodiscard]] auto
build_axpy_minus(const Grid& grid,
                 typename Grid::template Field<double,
                                               1>& r,
                 typename Grid::template Field<double,
                                               1>& s_field,
                 std::string name) -> carbon::Kernel {
  using Span = typename Grid::Span;
  using Idx = typename Grid::Idx;
  return carbon::Kernel::compute(
      grid,
      [&r, &s_field](carbon::KernelLoader& loader) {
        auto r_p = loader.read_write(r);
        const auto s_p = loader.read(s_field);
        return [r_p, s_p] CARBON_HOST_DEVICE(Span s, Idx idx) -> void {
          r_p.write(s, idx, 0, r_p.read(s, idx, 0) - s_p.read(s, idx, 0));
        };
      },
      std::move(name));
}

// p := r (one-shot, right before the iteration loop).
template <typename Grid>
[[nodiscard]] auto
build_copy_p_from_r(const Grid& grid,
                    typename Grid::template Field<double,
                                                  1>& p,
                    typename Grid::template Field<double,
                                                  1>& r,
                    std::string name) -> carbon::Kernel {
  using Span = typename Grid::Span;
  using Idx = typename Grid::Idx;
  return carbon::Kernel::compute(
      grid,
      [&p, &r](carbon::KernelLoader& loader) {
        auto p_p = loader.write(p);
        const auto r_p = loader.read(r);
        return [p_p, r_p] CARBON_HOST_DEVICE(Span s, Idx idx) -> void {
          p_p.write(s, idx, 0, r_p.read(s, idx, 0));
        };
      },
      std::move(name));
}

// p := r + beta * p ; @c beta_host captured @b by reference so the device
// closure picks up the latest host value at every launch.
template <typename Grid>
[[nodiscard]] auto
build_update_p(const Grid& grid,
               typename Grid::template Field<double,
                                             1>& p,
               typename Grid::template Field<double,
                                             1>& r,
               double& beta_host,
               std::string name) -> carbon::Kernel {
  using Span = typename Grid::Span;
  using Idx = typename Grid::Idx;
  return carbon::Kernel::compute(
      grid,
      [&p, &r, &beta_host](carbon::KernelLoader& loader) {
        auto p_p = loader.read_write(p);
        const auto r_p = loader.read(r);
        const double beta = beta_host;
        return [p_p, r_p, beta] CARBON_HOST_DEVICE(Span s, Idx idx) -> void {
          p_p.write(s, idx, 0, r_p.read(s, idx, 0) + beta * p_p.read(s, idx, 0));
        };
      },
      std::move(name));
}

// x := x + alpha * p ; r := r - alpha * s ; same alpha-by-reference trick.
template <typename Grid>
[[nodiscard]] auto
build_update_x_r(const Grid& grid,
                 typename Grid::template Field<double,
                                               1>& u,
                 typename Grid::template Field<double,
                                               1>& r,
                 typename Grid::template Field<double,
                                               1>& p,
                 typename Grid::template Field<double,
                                               1>& s_field,
                 double& alpha_host,
                 std::string name) -> carbon::Kernel {
  using Span = typename Grid::Span;
  using Idx = typename Grid::Idx;
  return carbon::Kernel::compute(
      grid,
      [&u, &r, &p, &s_field, &alpha_host](carbon::KernelLoader& loader) {
        auto x_p = loader.read_write(u);
        auto r_p = loader.read_write(r);
        const auto p_p = loader.read(p);
        const auto s_p = loader.read(s_field);
        const double alpha = alpha_host;
        return [x_p, r_p, p_p, s_p, alpha] CARBON_HOST_DEVICE(Span s, Idx idx) -> void {
          x_p.write(s, idx, 0, x_p.read(s, idx, 0) + alpha * p_p.read(s, idx, 0));
          r_p.write(s, idx, 0, r_p.read(s, idx, 0) - alpha * s_p.read(s, idx, 0));
        };
      },
      std::move(name));
}

// <r, r>
template <typename Grid>
[[nodiscard]] auto
build_reduce_rTr(const Grid& grid,
                 typename Grid::template Field<double,
                                               1>& r,
                 carbon::ReductionResult<double>& dst,
                 std::string name) -> carbon::Kernel {
  using Span = typename Grid::Span;
  using Idx = typename Grid::Idx;
  return carbon::Kernel::reduce<double>(
      grid,
      [&r](carbon::KernelLoader& loader) {
        const auto r_p = loader.read(r);
        return [r_p] CARBON_HOST_DEVICE(Span s, Idx idx) -> double {
          const double v = r_p.read(s, idx, 0);
          return v * v;
        };
      },
      dst, carbon::reduce::Sum<double>{}, std::move(name), /*shared_mem=*/0U);
}

// <p, s>
template <typename Grid>
[[nodiscard]] auto
build_reduce_pAp(const Grid& grid,
                 typename Grid::template Field<double,
                                               1>& p,
                 typename Grid::template Field<double,
                                               1>& s_field,
                 carbon::ReductionResult<double>& dst,
                 std::string name) -> carbon::Kernel {
  using Span = typename Grid::Span;
  using Idx = typename Grid::Idx;
  return carbon::Kernel::reduce<double>(
      grid,
      [&p, &s_field](carbon::KernelLoader& loader) {
        const auto p_p = loader.read(p);
        const auto s_p = loader.read(s_field);
        return [p_p, s_p] CARBON_HOST_DEVICE(Span s, Idx idx) -> double {
          return p_p.read(s, idx, 0) * s_p.read(s, idx, 0);
        };
      },
      dst, carbon::reduce::Sum<double>{}, std::move(name), /*shared_mem=*/0U);
}

// ---- Solver ----

template <typename Grid>
auto
run_cg(Grid& grid,
       carbon::System& sys,
       const Options& opts,
       std::string_view grid_tag,
       int world_rank) -> int {
  using Field = typename Grid::template Field<double, 1>;

  Field u = grid.template newField<double, 1>(carbon::MemorySide::Device, "u");
  Field b = grid.template newField<double, 1>(carbon::MemorySide::Device, "b");
  Field p = grid.template newField<double, 1>(carbon::MemorySide::Device, "p");
  Field r = grid.template newField<double, 1>(carbon::MemorySide::Device, "r");
  Field s_field = grid.template newField<double, 1>(carbon::MemorySide::Device, "s");

  // h = 1, so invh2 = 1. Matches the upstream Neon LaplacianMatVec scaling.
  constexpr double invh2 = 1.0;

  double alpha_host = 0.0;
  double beta_host = 0.0;

  carbon::ReductionResult<double> delta_new(sys);
  carbon::ReductionResult<double> pAp(sys);

  carbon::Stream stream = sys.stream();

  // Initialisation skeleton:
  //   zero b/p/r/s ; init u with topological BCs ; r := (Dir) ? u : b ;
  //   matvec(u -> s) ; r -= s ; reduce <r,r> ; p := r .
  // The matvec inserts an auto-halo on u because @c init_u writes u and
  // @c matvec read_stencil-s it. Single-GPU runs collapse the halo to a no-op.
  carbon::Skeleton init(sys);
  init.add(build_set_const<Grid, double>(grid, b, 0.0, "set_b=0"));
  init.add(build_set_const<Grid, double>(grid, p, 0.0, "set_p=0"));
  init.add(build_set_const<Grid, double>(grid, r, 0.0, "set_r=0"));
  init.add(build_set_const<Grid, double>(grid, s_field, 0.0, "set_s=0"));
  init.add(build_init_u(grid, u, opts.bd_min, opts.bd_max, "init_u"));
  init.add(build_init_r(grid, r, u, b, "init_r"));
  init.add(build_matvec(grid, u, s_field, invh2, "s:=Ax"));
  init.add(build_axpy_minus(grid, r, s_field, "r-=s"));
  init.add(build_reduce_rTr(grid, r, delta_new, "delta:=<r,r>"));
  init.add(build_copy_p_from_r(grid, p, r, "p:=r"));
  init.compile();

  // iter_pre  : update_p (uses beta_host) ; matvec(p -> s) ; reduce <p,s> .
  // iter_post : update_x_r (uses alpha_host) ; reduce <r,r> .
  const auto overlap_opt = parse_overlap_mode(opts.overlap_raw);
  if (!overlap_opt) {
    CARBON_LOG_ERROR(
        "cg_poisson_unified: unknown --overlap '{}' (expected "
        "none|standard|extended|two-way-extended)",
        opts.overlap_raw);
    return EXIT_FAILURE;
  }
  const carbon::OverlapMode overlap_mode = *overlap_opt;

  carbon::Skeleton iter_pre(sys, carbon::Skeleton::Options{overlap_mode});
  iter_pre.add(build_update_p(grid, p, r, beta_host, "p:=r+beta*p"));
  iter_pre.add(build_matvec(grid, p, s_field, invh2, "s:=Ap"));
  iter_pre.add(build_reduce_pAp(grid, p, s_field, pAp, "pAp:=<p,s>"));
  iter_pre.compile();

  carbon::Skeleton iter_post(sys, carbon::Skeleton::Options{overlap_mode});
  iter_post.add(build_update_x_r(grid, u, r, p, s_field, alpha_host, "x+=ap;r-=as"));
  iter_post.add(build_reduce_rTr(grid, r, delta_new, "delta:=<r,r>"));
  iter_post.compile();

  if (world_rank == 0) {
    CARBON_LOG_INFO(
        "cg_poisson_unified[{}]: init.kernels={}/{} iter_pre.kernels={}/{} iter_post.kernels={}/{} "
        "overlap={}",
        grid_tag, init.size(), init.effective_kernels().size(), iter_pre.size(),
        iter_pre.effective_kernels().size(), iter_post.size(), iter_post.effective_kernels().size(),
        overlap_mode_name(overlap_mode));
    if (opts.dump_dot) {
      CARBON_LOG_INFO("cg_poisson_unified[{}]: --- init DAG ---\n{}", grid_tag,
                      init.export_dot("cg_init"));
      CARBON_LOG_INFO("cg_poisson_unified[{}]: --- iter_pre DAG ---\n{}", grid_tag,
                      iter_pre.export_dot("cg_iter_pre"));
      CARBON_LOG_INFO("cg_poisson_unified[{}]: --- iter_post DAG ---\n{}", grid_tag,
                      iter_post.export_dot("cg_iter_post"));
    }
    if (!opts.dot_out_prefix.empty()) {
      const auto export_one = [&](carbon::Skeleton& sk, const char* tag) {
        const std::string base = opts.dot_out_prefix + "_" + tag;
        if (!sk.export_dot_file(base + ".dot", std::string{"cg_"} + tag)) {
          CARBON_LOG_ERROR("cg_poisson_unified: failed to write {}.dot", base);
        }
        if (!sk.export_launch_order_file(base + ".txt")) {
          CARBON_LOG_ERROR("cg_poisson_unified: failed to write {}.txt", base);
        }
        if (opts.dot_matrix) {
          if (!sk.export_launch_matrix_dot_file(base + "_matrix.dot", std::string{"cg_"} + tag)) {
            CARBON_LOG_ERROR("cg_poisson_unified: failed to write {}_matrix.dot", base);
          }
          if (!sk.export_kernel_list_dot_file(base + "_kernels.dot", std::string{"cg_"} + tag)) {
            CARBON_LOG_ERROR("cg_poisson_unified: failed to write {}_kernels.dot", base);
          }
        }
      };
      export_one(init, "init");
      export_one(iter_pre, "iter_pre");
      export_one(iter_post, "iter_post");
      CARBON_LOG_INFO("cg_poisson_unified[{}]: wrote Skeleton exports under prefix '{}'", grid_tag,
                      opts.dot_out_prefix);
    }
  }

  init.run();
  init.wait();
  delta_new.sync(stream);
  stream.wait();

  const double delta_init = delta_new.host_value();
  const double r0_norm = std::sqrt(std::max(0.0, delta_init));
  const double abs_target = std::max(opts.tol_abs, opts.tol_rel * r0_norm);
  if (world_rank == 0) {
    CARBON_LOG_INFO("cg_poisson_unified[{}]: ||r_0|| = {:.6e}, target ||r|| = {:.6e}", grid_tag,
                    r0_norm, abs_target);
  }

  double delta_new_h = delta_init;
  double delta_old_h = 0.0;
  size_t iter = 0;
  bool converged = (r0_norm <= abs_target);

  const auto t0 = std::chrono::steady_clock::now();
  for (; iter < opts.max_iter && !converged; ++iter) {
    beta_host = (delta_old_h > 0.0) ? (delta_new_h / delta_old_h) : 0.0;

    iter_pre.run();
    iter_pre.wait();
    pAp.sync(stream);
    stream.wait();
    const double pAp_h = pAp.host_value();
    if (!(pAp_h > 0.0)) {
      if (world_rank == 0) {
        CARBON_LOG_ERROR(
            "cg_poisson_unified[{}]: numerical issue at iter {}: <p,Ap>={:.6e} (not > 0)", grid_tag,
            iter, pAp_h);
      }
      return EXIT_FAILURE;
    }
    alpha_host = delta_new_h / pAp_h;

    iter_post.run();
    iter_post.wait();
    delta_new.sync(stream);
    stream.wait();

    delta_old_h = delta_new_h;
    delta_new_h = delta_new.host_value();
    const double r_norm = std::sqrt(std::max(0.0, delta_new_h));

    if (world_rank == 0 &&
        ((iter + 1) % static_cast<size_t>(opts.report_every) == 0U || r_norm <= abs_target)) {
      CARBON_LOG_INFO(
          "cg_poisson_unified[{}]: iter {:5d}  ||r||={:.6e}  alpha={:+.3e}  beta={:+.3e}", grid_tag,
          iter + 1, r_norm, alpha_host, beta_host);
    }
    if (r_norm <= abs_target) {
      converged = true;
    }
  }
  const auto t1 = std::chrono::steady_clock::now();
  const double solve_s = std::chrono::duration<double>(t1 - t0).count();

  const double final_r_norm = std::sqrt(std::max(0.0, delta_new_h));
  if (world_rank == 0) {
    const double ms_per_iter = (iter > 0U) ? (1.0e3 * solve_s / static_cast<double>(iter)) : 0.0;
    CARBON_LOG_INFO(
        "cg_poisson_unified[{}]: TIMING overlap={} iters={} solve_s={:.6f} ms_per_iter={:.4f}",
        grid_tag, overlap_mode_name(overlap_mode), iter, solve_s, ms_per_iter);
    if (converged) {
      CARBON_LOG_INFO("cg_poisson_unified[{}]: CONVERGED at iter {}  ||r||={:.6e}", grid_tag, iter,
                      final_r_norm);
    } else {
      CARBON_LOG_INFO(
          "cg_poisson_unified[{}]: STOPPED at iter {}  ||r||={:.6e}  (tolerance not met)", grid_tag,
          iter, final_r_norm);
    }
  }
  return (final_r_norm <= abs_target) ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace

auto
main(int argc,
     char** argv) -> int {
  carbon::init();

  Options opts;
  if (!parse_cli(argc, argv, opts)) {
    return opts.show_help ? EXIT_SUCCESS : EXIT_FAILURE;
  }

#ifdef CARBON_HAVE_MPI_NCCL
  auto distributed = std::make_shared<const carbon::Distributed>();
  const int world_size = distributed->world_size();
  const int world_rank = distributed->world_rank();
  carbon::System sys =
      (world_size > 1) ? carbon::System(distributed) : carbon::System(carbon::get_all_devices());
  const bool is_distributed = (world_size > 1);
#else
  carbon::System sys(carbon::get_all_devices());
  const int world_size = 1;
  const int world_rank = 0;
  const bool is_distributed = false;
#endif

  if (sys.gpu_group().size() == 0U) {
    CARBON_LOG_ERROR("cg_poisson_unified: no devices visible");
    return EXIT_FAILURE;
  }

  if (world_rank == 0) {
    CARBON_LOG_INFO(
        "cg_poisson_unified: grid={} nx={} ny={} nz={} max_iter={} tol_rel={:.1e} tol_abs={:.1e} "
        "bd_min={:+.3f} bd_max={:+.3f} backend={} gpus_local={} world_size={} distributed={}",
        opts.grid_kind, opts.nx, opts.ny, opts.nz, opts.max_iter, opts.tol_rel, opts.tol_abs,
        opts.bd_min, opts.bd_max,
#if CARBON_BACKEND_IS_SYCL
        "SYCL",
#else
        "CUDA",
#endif
        sys.gpu_group().size(), world_size, is_distributed);
  }

  const carbon::int64_3d global{opts.nx, opts.ny, opts.nz};

  // 6-point face stencil -> halo_z = 1 .
  carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);

  try {
    if (opts.grid_kind == "dense") {
      carbon::Dense dense(sys, global, stencil);
      return run_cg(dense, sys, opts, "dense", world_rank);
    }
    if (opts.grid_kind == "disg") {
      carbon::DenseDisg disg(sys, global, /*halo_z=*/1);
      return run_cg(disg, sys, opts, "disg", world_rank);
    }
    if (opts.grid_kind == "block") {
      const int64_t nmax = std::max({opts.nx, opts.ny, opts.nz});
      const auto predicate = [nmax](const carbon::int64_3d& gc) -> bool {
        return sphere_predicate(nmax, gc);
      };
      using BlockGrid = carbon::Block<SB>;
      BlockGrid block(sys, global, predicate, "cg_poisson_unified_block");
      return run_cg(block, sys, opts, "block", world_rank);
    }
    CARBON_LOG_ERROR("cg_poisson_unified: unknown --grid '{}' (expected dense|disg|block)",
                     opts.grid_kind);
    return EXIT_FAILURE;
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("cg_poisson_unified: failed: {}", e.what());
    return EXIT_FAILURE;
  }
}
