// Carbon LBM lid-driven cavity benchmark — port of
// Neon/benchmarks/lbm-lid-driven-cavity-flow on top of carbon::Skeleton.
//
// Models a unit-cube cavity with bounce-back walls on the -x/+x, -y/+y, -z
// faces and a moving lid (u = (ulb, 0, 0)) on the +y face. Each iteration
// is wrapped in a 1-kernel carbon::Skeleton (the halo on the input
// population is auto-inserted by Skeleton::compile because the iteration
// kernel calls loader.read_stencil(fIn)). Two skeletons (one per
// ping-pong direction) are precompiled before the timing loop so the hot
// loop is a tight `sk.run(); sk.wait();` pair with no allocation /
// no recompilation.
//
// Lattice + collision are runtime-selectable via CLI:
//   --lattice {q19,q27}   : D3Q19 (default) or D3Q27.
//   --collision {bgk,kbc} : BGK SRT (default) or KBC entropic stabilisation
//                           (Karlin-Bösch-Chikatamarla, 2014). KBC requires
//                           D3Q27 — the moment matrix and shear-decomposition
//                           table needed by the entropic step are only
//                           defined for the full 27-velocity lattice.
//
// Headline metric is MLUPS — Million Lattice Updates Per Second — computed
// over the post-warmup iterations only:
//
//     MLUPS = (N^3 * iters_timed) / (1e6 * elapsed_seconds)
//
// CLI (clipp; same one-shot man page as examples/cg_poisson_unified.cu).

#include "src/cell_type.h"
#include "src/d3q19.h"
#include "src/d3q27.h"
#include "src/lbm_kernels.h"

#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/dense_disg.h>
// CARBON_BACKEND_NAME is "cuda" or "sycl" depending on which backend the
// library was configured with. Stamped into the JSON report's system block
// so a sweep that drives both backends from the same scripts/ directory
// can colour-code them without resorting to filename heuristics.
#include <carbon/detail/backend/select.h>
#ifdef CARBON_HAVE_MPI_NCCL
#include <carbon/distributed.h>
#endif
#include <carbon/integer3d.h>
#include <carbon/io.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/reduce.h>
#include <carbon/skeleton.h>
#include <carbon/stencil.h>

#include <clipp.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using carbon::bench::lbm::BgkCollision;
using carbon::bench::lbm::Classification;
using carbon::bench::lbm::D3Q19;
using carbon::bench::lbm::D3Q27;
using carbon::bench::lbm::encode_cell;
using carbon::bench::lbm::KbcCollision;
using carbon::bench::lbm::LbmParameters;
using carbon::bench::lbm::make_compute_wall_mask_kernel;
using carbon::bench::lbm::make_iteration_kernel;
using carbon::bench::lbm::make_iteration_kernel_d3q19_bgk_fast;
using carbon::bench::lbm::make_macroscopic_kernel;

struct Options {
  int64_t n = 128;
  int64_t iters = 1000;
  int64_t warmup = 100;
  double re = 100.0;
  double ulb = 0.05;
  bool ulb_set = false;
  int64_t report_every = 100;
  bool no_warmup_skip = false;
  std::string lattice = "q19";
  std::string collision = "bgk";
  // Grid family the benchmark runs on. "dense" is carbon::Dense (aggregated
  // Cartesian z-slabs); "disg" is carbon::DenseDisg (the disaggregated,
  // five-region-per-slab lattice-Boltzmann layout). Both run the identical hot
  // kernel (the pull-stream + half-way bounce-back + symmetric BGK body) — the
  // only difference is the per-slab storage layout and the auto-inserted halo
  // family (Halo vs HaloDisg). Selecting "disg" is the fair apples-to-apples
  // measurement of the disaggregated layout on a real LBM workload.
  std::string grid = "dense";
  std::string report_path = "";
  std::string label = "";
  std::string vtk_path = "";
  // Skeleton OCC mode applied to both ping-pong skeletons. "none" (default,
  // single-stream Standard launch) keeps the existing baseline behaviour;
  // "standard" enables the carbon::Skeleton OCC rewrite that splits the
  // stencil node into Internal + Boundary halves and overlaps the halo with
  // the Internal launch on a second stream — the default-OFF state of
  // OverlapMode is the dominant reason multi-GPU regresses below 1-GPU on
  // PCIe-only hosts (the halo serialises behind every iteration).
  // "extended" / "twoway-extended" are the more aggressive OCC modes; in
  // the LBM iteration there is only one stencil node so they degenerate to
  // "standard" but are exposed here for symmetry with carbon::OverlapMode.
  std::string overlap = "none";
  std::string export_dot_path = "";
  std::string export_dot_png_path = "";
  // Opt in to carbon::Skeleton CUDA-graph capture + replay of each ping-pong
  // skeleton. Engaged only for single-GPU / single-stream schedules (the
  // OverlapMode::None single-GPU case); a no-op fallback otherwise.
  bool cuda_graph = false;
  // After the timed loop, Sum-reduce the whole final population field and print
  // CHECKSUM=<value>. Because sum_q pop_q == rho per cell, this is the total
  // mass in the domain — a single scalar that is invariant across grid families
  // when the streaming/collision is numerically identical. Used to verify that
  // --grid dense and --grid disg produce bit-comparable physics (they run the
  // same hot kernel; only the storage layout + halo family differ).
  bool checksum = false;
  bool show_help = false;
};

[[nodiscard]] auto
build_cli(Options& opts) -> clipp::group {
  using clipp::integer;
  using clipp::number;
  using clipp::option;
  using clipp::value;
  return clipp::group(
      (option("-h", "--help").set(opts.show_help) % "Show this help message and exit"),
      (option("--n") & integer("n", opts.n)) % "Cavity edge length in cells (default 128)",
      (option("--iters") & integer("iters", opts.iters)) %
          "Total iterations including warmup (default 1000)",
      (option("--warmup") & integer("warmup", opts.warmup)) %
          "Warmup iterations excluded from timing (default 100)",
      (option("--re") & number("re", opts.re)) % "Reynolds number (default 1e2)",
      (option("--ulb") & number("ulb", opts.ulb).set(opts.ulb_set)) %
          "Override lid velocity in lattice units (default 0.05)",
      (option("--lattice") & value("q19|q27", opts.lattice)) %
          "Lattice stencil: q19 (D3Q19, default) or q27 (D3Q27)",
      (option("--collision") & value("bgk|kbc", opts.collision)) %
          "Collision model: bgk (default) or kbc (entropic; D3Q27 only)",
      (option("--grid") & value("dense|disg", opts.grid)) %
          "Grid family: dense (carbon::Dense, default) or disg (carbon::DenseDisg, "
          "the disaggregated five-region LBM layout). Same hot kernel on both.",
      (option("--report-every") & integer("n", opts.report_every)) %
          "Log progress every N iterations (default 100; 0 silences)",
      (option("--report") & value("path", opts.report_path)) %
          "Write a structured JSON report to PATH on success "
          "(consumed by scripts/sweep_and_plot.py)",
      (option("--label") & value("string", opts.label)) %
          "Free-form label written into the JSON report under config.label "
          "(useful for tagging sweeps)",
      (option("--vtk-out") & value("path", opts.vtk_path)) %
          "After the timed loop, run the macroscopic kernel and dump rho + u "
          "as a VTK legacy STRUCTURED_POINTS file at PATH (carbon::io::write_vtk; "
          "appends .rankN under MPI). Off by default — adds ~one kernel + a "
          "GPU->host copy of size cells*16B.",
      (option("--overlap") & value("none|standard|extended|twoway-extended", opts.overlap)) %
          "Skeleton OCC mode (default 'none'). 'standard' splits the stencil "
          "node into Internal + Boundary so the halo overlaps with Internal on "
          "a second stream — improves multi-GPU significantly on PCIe-only "
          "hosts where halo otherwise blocks compute.",
      (option("--export-dot") & value("path", opts.export_dot_path)) %
          "Write the compiled Skeleton schedule graph (GraphViz .dot) for the "
          "AB ping-pong skeleton to PATH after compile.",
      (option("--export-dot-png") & value("path", opts.export_dot_png_path)) %
          "Render the schedule graph to PNG via GraphViz (requires dot on PATH). "
          "Also writes a sibling .dot file.",
      option("--cuda-graph").set(opts.cuda_graph) %
          "Capture + replay each ping-pong skeleton as a CUDA graph (single-GPU / "
          "single-stream only; transparently falls back to eager launch otherwise). "
          "Collapses per-iteration host launch overhead.",
      option("--checksum").set(opts.checksum) %
          "After the loop, Sum-reduce the final population field and print "
          "CHECKSUM=<total mass>. Grid-family invariant — used to verify "
          "--grid dense and --grid disg match numerically.",
      option("--no-warmup-skip").set(opts.no_warmup_skip) %
          "Include warmup iterations in the timing window (debug aid)");
}

[[nodiscard]] auto
parse_cli(int argc,
          char** argv,
          Options& opts) -> bool {
  auto cli = build_cli(opts);
  if (!clipp::parse(argc, argv, cli) || opts.show_help) {
    auto fmt = clipp::doc_formatting{}.doc_column(31);
    std::cout << clipp::make_man_page(cli, argv[0], fmt) << '\n';
    return opts.show_help;
  }
  if (opts.n < 4) {
    std::cerr << "lbm_lid_driven_cavity: --n must be >= 4 (got " << opts.n << ")\n";
    return false;
  }
  if (opts.iters <= 0) {
    std::cerr << "lbm_lid_driven_cavity: --iters must be > 0 (got " << opts.iters << ")\n";
    return false;
  }
  if (opts.warmup < 0 || opts.warmup >= opts.iters) {
    std::cerr << "lbm_lid_driven_cavity: --warmup must be in [0, iters) (got " << opts.warmup
              << " for iters=" << opts.iters << ")\n";
    return false;
  }
  if (opts.lattice != "q19" && opts.lattice != "q27") {
    std::cerr << "lbm_lid_driven_cavity: --lattice must be q19 or q27 (got '" << opts.lattice
              << "')\n";
    return false;
  }
  if (opts.collision != "bgk" && opts.collision != "kbc") {
    std::cerr << "lbm_lid_driven_cavity: --collision must be bgk or kbc (got '" << opts.collision
              << "')\n";
    return false;
  }
  if (opts.collision == "kbc" && opts.lattice != "q27") {
    std::cerr << "lbm_lid_driven_cavity: --collision kbc requires --lattice q27 "
                 "(KBC's moment basis is defined for D3Q27 only)\n";
    return false;
  }
  if (opts.grid != "dense" && opts.grid != "disg") {
    std::cerr << "lbm_lid_driven_cavity: --grid must be dense or disg (got '" << opts.grid
              << "')\n";
    return false;
  }
  if (opts.overlap != "none" && opts.overlap != "standard" && opts.overlap != "extended" &&
      opts.overlap != "twoway-extended") {
    std::cerr << "lbm_lid_driven_cavity: --overlap must be one of "
                 "{none, standard, extended, twoway-extended} (got '"
              << opts.overlap << "')\n";
    return false;
  }
  return true;
}

[[nodiscard]] auto
overlap_mode_from_string(std::string_view s) -> carbon::OverlapMode {
  if (s == "standard") {
    return carbon::OverlapMode::Standard;
  }
  if (s == "extended") {
    return carbon::OverlapMode::Extended;
  }
  if (s == "twoway-extended") {
    return carbon::OverlapMode::TwoWayExtended;
  }
  return carbon::OverlapMode::None;
}

// Tiny RFC-3339 / ISO-8601 timestamp helper used in the JSON report.
[[nodiscard]] auto
utc_timestamp_now() -> std::string {
  const std::time_t now = std::time(nullptr);
  std::tm tm_utc{};
#if defined(_WIN32)
  gmtime_s(&tm_utc, &now);
#else
  gmtime_r(&now, &tm_utc);
#endif
  std::ostringstream oss;
  oss << std::put_time(&tm_utc, "%Y-%m-%dT%H:%M:%SZ");
  return oss.str();
}

// Minimal JSON string escape (handles ", \, control chars). Hand-rolled
// because a full rapidjson dependency would be overkill for the ~12 fields
// we emit. Keep this function private to the TU.
[[nodiscard]] auto
json_escape(const std::string& s) -> std::string {
  std::string out;
  out.reserve(s.size() + 2);
  for (char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned int>(c));
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out;
}

// Equilibrium population for component q at zero velocity and rho = 1: w_q.
// The lid layer also seeds populations with the bounce-back lid-momentum
// correction (-6 * w_q * ulb * (c_q . ulid)) so the very first iteration
// sees a reasonable approximation of the steady state instead of having to
// spin up from rest. Templated on Lattice so the same init logic feeds
// either D3Q19 or D3Q27.
template <typename Lattice>
[[nodiscard]] auto
init_population_value(const carbon::int64_3d& gc,
                      const carbon::int64_3d& global,
                      int q,
                      double ulb) -> float {
  const float w = Lattice::w(q);
  const bool on_boundary = (gc.x == 0 || gc.x == global.x - 1 || gc.y == 0 ||
                            gc.y == global.y - 1 || gc.z == 0 || gc.z == global.z - 1);
  if (!on_boundary) {
    return w;
  }
  if (gc.y == global.y - 1) {
    // Moving lid. Mirrors Neon's init: -6 * w_q * ulb * (c_q . ulid),
    // where ulid = (1, 0, 0). c_q.x carries the sign.
    const auto cq = Lattice::c(q);
    const float dot = static_cast<float>(cq.x) * 1.0F;
    return static_cast<float>(-6.0F * w * static_cast<float>(ulb) * dot);
  }
  return 0.0F;
}

[[nodiscard]] auto
init_cell_value(const carbon::int64_3d& gc,
                const carbon::int64_3d& global) -> int32_t {
  const bool on_boundary = (gc.x == 0 || gc.x == global.x - 1 || gc.y == 0 ||
                            gc.y == global.y - 1 || gc.z == 0 || gc.z == global.z - 1);
  if (!on_boundary) {
    return encode_cell(Classification::Bulk);
  }
  if (gc.y == global.y - 1) {
    return encode_cell(Classification::MovingWall);
  }
  return encode_cell(Classification::BounceBack);
}

// Build the carbon::Stencil for a given lattice — radius-1 stencil
// covering every non-center direction. Used to seed Span::halo_z so multi-
// GPU runs allocate the right z-halo on each slab. Both D3Q19 and D3Q27 have
// |c_q|_inf == 1 so radius is 1 either way; what changes is the count
// (18 for D3Q19, 26 for D3Q27).
template <typename Lattice>
[[nodiscard]] auto
make_lbm_stencil() -> carbon::Stencil {
  std::vector<carbon::int32_3d> stencil_points;
  stencil_points.reserve(Lattice::Q - 1);
  for (int q = 0; q < Lattice::Q; ++q) {
    if (q == Lattice::kCenter) {
      continue;
    }
    const auto cq = Lattice::c(q);
    stencil_points.push_back({cq.x, cq.y, cq.z});
  }
  return carbon::Stencil(stencil_points, /*remove_center=*/true);
}

// One-shot benchmark body, instantiated once per (lattice, collision) pair
// from main(). The dispatch is: parse opts, switch on the (lattice,
// collision) combo, instantiate this template, run it, return its exit
// code. This keeps all the instantiation cost on configurations the user
// actually exercises (the dispatch only calls one of the three
// instantiations for a given run).
template <typename Grid,
          typename Lattice,
          typename Collision>
[[nodiscard]] auto
run_benchmark(const Options& opts,
              const std::string& lattice_name,
              const std::string& collision_name) -> int {
  static_assert(std::is_same_v<Grid, carbon::Dense> || std::is_same_v<Grid, carbon::DenseDisg>,
                "LBM benchmark supports carbon::Dense and carbon::DenseDisg only.");
#ifdef CARBON_HAVE_MPI_NCCL
  // Auto-detect MPI launches: the `Distributed` ctor invokes `MPI_Init` (a
  // no-op when already initialised) and bootstraps an `ncclComm_t` over every
  // visible rank. Under a single-process launch (no `mpiexec` wrapping)
  // `world_size() == 1` and we fall through to the local-`get_all_devices()`
  // path so the existing single-host behaviour is byte-for-byte preserved.
  // Under `mpiexec -n R` the constructor binds rank `r` to its local GPU via
  // `pin_local_device` and `System(distributed)` carries `gpu_group.size() ==
  // 1` (carbon's strict one-rank-per-GPU model).
  auto distributed = std::make_shared<const carbon::Distributed>();
  const bool use_mpi = distributed->world_size() > 1;
  carbon::System sys =
      use_mpi ? carbon::System(distributed) : carbon::System(carbon::get_all_devices());
#else
  carbon::System sys(carbon::get_all_devices());
#endif
  const size_t n_gpus = sys.gpu_group().size();

  const carbon::Stencil stencil = make_lbm_stencil<Lattice>();
  const carbon::int64_3d global{opts.n, opts.n, opts.n};
  // Construct the selected grid family. Both give a radius-1 z-halo (D3Q19 /
  // D3Q27 have |c|_inf == 1); Dense derives halo_z from the stencil, DenseDisg
  // takes it directly (its five-region layout pins ghost/boundary radius to 1).
  // Only one branch is instantiated per Grid, so the IIFE's return type is the
  // concrete grid type for this instantiation.
  Grid grid = [&]() -> Grid {
    if constexpr (std::is_same_v<Grid, carbon::Dense>) {
      return carbon::Dense(sys, global, stencil);
    } else {
      return carbon::DenseDisg(sys, global, /*halo_z=*/1);
    }
  }();

  const double ulb = opts.ulb;
  // Re = u_lb * L / nu_lb  =>  nu_lb = u_lb * L / Re. omega is the BGK
  // (or KBC, beta = omega/2) relaxation timescale tied to nu via
  // nu = (1/omega - 1/2) / 3.
  const double nu_lb = ulb * static_cast<double>(opts.n - 1) / opts.re;
  const double omega = 1.0 / (3.0 * nu_lb + 0.5);
  const LbmParameters params{static_cast<float>(ulb), static_cast<float>(omega)};

  CARBON_LOG_INFO(
      "lbm_lid_driven_cavity: lattice={} collision={} N={} iters={} warmup={} "
      "Re={:.3g} ulb={:.4g} omega={:.4g} GPUs={} Q={}",
      lattice_name, collision_name, opts.n, opts.iters, opts.warmup, opts.re, ulb, omega, n_gpus,
      Lattice::Q);

  // Two ping-pong populations + a cell-type field. HostDevice so we can
  // seed them from the CPU via fill_owned_host then upload once.
  auto pop_a = grid.template newField<float, Lattice::Q>(carbon::MemorySide::HostDevice, "pop_a");
  auto pop_b = grid.template newField<float, Lattice::Q>(carbon::MemorySide::HostDevice, "pop_b");
  auto cells = grid.template newField<int32_t, 1>(carbon::MemorySide::HostDevice, "cells");
  auto cells_scratch =
      grid.template newField<int32_t, 1>(carbon::MemorySide::HostDevice, "cells_scratch");

  auto pop_filler = [&global, ulb](carbon::int64_3d gc, int q) -> float {
    return init_population_value<Lattice>(gc, global, q, ulb);
  };
  pop_a.fill_owned_host(pop_filler);
  pop_b.fill_owned_host(pop_filler);
  cells.fill_owned_host([&global](carbon::int64_3d gc, int /*c*/) {
    return init_cell_value(gc, global);
  });
  cells_scratch.fill_owned_host([&global](carbon::int64_3d gc, int /*c*/) {
    return init_cell_value(gc, global);
  });

  carbon::Stream upload_stream = sys.stream();
  try {
    carbon::Transfer<float>(sys, pop_a.cpu_buffers(), pop_a.gpu_buffers())(upload_stream);
    carbon::Transfer<float>(sys, pop_b.cpu_buffers(), pop_b.gpu_buffers())(upload_stream);
    carbon::Transfer<int32_t>(sys, cells.cpu_buffers(), cells.gpu_buffers())(upload_stream);
    carbon::Transfer<int32_t>(sys, cells_scratch.cpu_buffers(),
                              cells_scratch.gpu_buffers())(upload_stream);
    upload_stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("lbm_lid_driven_cavity: upload failed: {}", e.what());
    return EXIT_FAILURE;
  }

  // Precompute wall_neighbour_bits once into cells (reads cells, writes
  // cells_scratch through the auto-halo + Stencil pattern; then a peer-copy
  // publishes the result back into the canonical "cells" field).
  {
    carbon::Skeleton precompute(sys);
    precompute.add(make_compute_wall_mask_kernel<Grid, Lattice>(grid, cells, cells_scratch));
    try {
      precompute.compile();
      precompute.run();
      precompute.wait();
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("lbm_lid_driven_cavity: wall-mask compute failed: {}", e.what());
      return EXIT_FAILURE;
    }
    try {
      carbon::Transfer<int32_t>(sys, cells_scratch.gpu_buffers(),
                                cells.gpu_buffers())(upload_stream);
      upload_stream.wait();
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("lbm_lid_driven_cavity: wall-mask publish failed: {}", e.what());
      return EXIT_FAILURE;
    }
  }

  // Pick the symmetry-unrolled D3Q19+BGK fast-path when the (lattice,
  // collision) combo allows it (factored macroscopic + GO/BK eqopp = eq +
  // rho * w * 6 * c.u, mirroring Neon's lbm-flow-over-sphere SOTA path);
  // otherwise fall through to the lattice-generic body that also covers
  // (D3Q27, BGK) and (D3Q27, KBC). Selected at compile time so the two
  // ping-pong skeletons each instantiate exactly one kernel.
  auto build_iter = [&](typename Grid::template Field<float, Lattice::Q>& fIn,
                        typename Grid::template Field<float, Lattice::Q>& fOut, std::string name) {
    if constexpr (std::is_same_v<Lattice, D3Q19> && std::is_same_v<Collision, BgkCollision>) {
      return make_iteration_kernel_d3q19_bgk_fast<Grid>(grid, fIn, fOut, cells, params,
                                                        std::move(name));
    } else {
      return make_iteration_kernel<Grid, Lattice, Collision>(grid, fIn, fOut, cells, params,
                                                             std::move(name));
    }
  };

  const carbon::Skeleton::Options sk_opts(overlap_mode_from_string(opts.overlap),
                                          /*strict_halo=*/false, /*max_streams=*/0U,
                                          /*log_dot_on_compile=*/false,
                                          /*use_cuda_graph=*/opts.cuda_graph);
  carbon::Skeleton sk_ab(sys, sk_opts);
  sk_ab.add(build_iter(pop_a, pop_b, "lbm_step_ab"));
  carbon::Skeleton sk_ba(sys, sk_opts);
  sk_ba.add(build_iter(pop_b, pop_a, "lbm_step_ba"));
  try {
    sk_ab.compile();
    sk_ba.compile();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("lbm_lid_driven_cavity: skeleton compile failed: {}", e.what());
    return EXIT_FAILURE;
  }
  CARBON_LOG_INFO(
      "lbm_lid_driven_cavity: skeletons compiled — sk_ab.effective_kernels={} "
      "sk_ba.effective_kernels={} (auto-halos included)",
      sk_ab.effective_kernels().size(), sk_ba.effective_kernels().size());

  if (!opts.export_dot_path.empty()) {
    if (!sk_ab.export_dot_file(opts.export_dot_path, "lbm_sk_ab")) {
      CARBON_LOG_ERROR("lbm_lid_driven_cavity: --export-dot failed for {}", opts.export_dot_path);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("lbm_lid_driven_cavity: wrote schedule DOT to {}", opts.export_dot_path);
  }
  if (!opts.export_dot_png_path.empty()) {
    if (!sk_ab.export_dot_png(opts.export_dot_png_path, "lbm_sk_ab")) {
      CARBON_LOG_ERROR(
          "lbm_lid_driven_cavity: --export-dot-png failed for {} (is graphviz dot on PATH?)",
          opts.export_dot_png_path);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("lbm_lid_driven_cavity: wrote schedule PNG to {}", opts.export_dot_png_path);
  }

  using Clock = std::chrono::steady_clock;
  using Seconds = std::chrono::duration<double>;
  const int64_t timed_start = opts.no_warmup_skip ? 0 : opts.warmup;

  Clock::time_point timing_started{};
  Clock::time_point overall_started = Clock::now();
  bool timing_running = false;

  for (int64_t it = 0; it < opts.iters; ++it) {
    if (it == timed_start) {
      try {
        sk_ab.wait();
        sk_ba.wait();
      } catch (const std::exception& e) {
        CARBON_LOG_ERROR("lbm_lid_driven_cavity: pre-timing drain failed at iter {}: {}", it,
                         e.what());
        return EXIT_FAILURE;
      }
      timing_started = Clock::now();
      timing_running = true;
      if (!opts.no_warmup_skip) {
        CARBON_LOG_INFO("lbm_lid_driven_cavity: warmup complete ({} iters), starting timing window",
                        opts.warmup);
      }
    }
    carbon::Skeleton& sk = (it % 2 == 0) ? sk_ab : sk_ba;
    try {
      sk.run();
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("lbm_lid_driven_cavity: iter {} failed: {}", it, e.what());
      return EXIT_FAILURE;
    }
    if (opts.report_every > 0 && (it + 1) % opts.report_every == 0) {
      try {
        sk_ab.wait();
        sk_ba.wait();
      } catch (const std::exception& e) {
        CARBON_LOG_ERROR("lbm_lid_driven_cavity: progress sync failed at iter {}: {}", it + 1,
                         e.what());
        return EXIT_FAILURE;
      }
      const auto now = Clock::now();
      const double elapsed_overall = Seconds(now - overall_started).count();
      CARBON_LOG_INFO("lbm_lid_driven_cavity: iter {}/{} elapsed={:.3f}s", it + 1, opts.iters,
                      elapsed_overall);
    }
  }

  try {
    sk_ab.wait();
    sk_ba.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("lbm_lid_driven_cavity: final sync failed: {}", e.what());
    return EXIT_FAILURE;
  }
  const auto timing_ended = Clock::now();
  if (!timing_running) {
    CARBON_LOG_ERROR(
        "lbm_lid_driven_cavity: timing window never opened (timed_start={} but the loop "
        "never reached that iteration).",
        timed_start);
    return EXIT_FAILURE;
  }
  const double timing_seconds = Seconds(timing_ended - timing_started).count();
  const int64_t timed_iters = opts.iters - timed_start;
  const double cells_per_iter =
      static_cast<double>(opts.n) * static_cast<double>(opts.n) * static_cast<double>(opts.n);
  const double mlups =
      (cells_per_iter * static_cast<double>(timed_iters)) / (1.0e6 * timing_seconds);
  const double overall_seconds = Seconds(timing_ended - overall_started).count();

#ifdef CARBON_HAVE_MPI_NCCL
  // Under MPI all ranks measure the same `cells_per_iter` (= global N^3) and
  // run lock-step (halo exchanges synchronise them), so each rank's `mlups`
  // is the global throughput up to small per-rank timing noise. Print only on
  // rank 0 to keep the scrape-friendly headline single-shot. `world_size`
  // reports the actual rank count (= total GPU count under carbon's strict
  // one-rank-per-GPU model); `n_gpus` (the local `gpu_group()`) is always 1
  // under MPI so we substitute it in the per-GPU display.
  const int world_rank = use_mpi ? distributed->world_rank() : 0;
  const int world_size = use_mpi ? distributed->world_size() : static_cast<int>(n_gpus);
  if (world_rank == 0) {
    CARBON_LOG_INFO(
        "lbm_lid_driven_cavity: DONE  lattice={} collision={} timing_window={} iters in "
        "{:.3f}s  =>  {:.2f} MLUPS ({:.2f} MLUPS/GPU)  overall={:.3f}s  ranks={}",
        lattice_name, collision_name, timed_iters, timing_seconds, mlups,
        mlups / static_cast<double>(world_size), overall_seconds, world_size);
    std::cout << "MLUPS=" << mlups << " (cavity N=" << opts.n << " iters=" << timed_iters
              << " gpus=" << world_size << " lattice=" << lattice_name
              << " collision=" << collision_name << (use_mpi ? " transport=mpi+nccl" : "") << ")"
              << '\n';
  }
#else
  CARBON_LOG_INFO(
      "lbm_lid_driven_cavity: DONE  lattice={} collision={} timing_window={} iters in "
      "{:.3f}s  =>  {:.2f} MLUPS ({:.2f} MLUPS/GPU)  overall={:.3f}s",
      lattice_name, collision_name, timed_iters, timing_seconds, mlups,
      mlups / static_cast<double>(n_gpus), overall_seconds);
  std::cout << "MLUPS=" << mlups << " (cavity N=" << opts.n << " iters=" << timed_iters
            << " gpus=" << n_gpus << " lattice=" << lattice_name << " collision=" << collision_name
            << ")" << '\n';
#endif

  // Optional grid-family parity checksum: Sum-reduce the whole final population
  // field. sum_q pop_q == rho per cell, so this equals the total mass in the
  // domain — a single scalar that must agree across --grid dense and --grid
  // disg when the physics is numerically identical (they share the hot kernel;
  // only the storage layout + auto-halo family differ). Cross-rank allreduce
  // kicks in automatically under MPI via ReductionResult's stage-3 path.
  if (opts.checksum) {
    using Span = typename Grid::Span;
    using Idx = typename Grid::Idx;
    auto& latest_pop = (opts.iters % 2 == 1) ? pop_b : pop_a;
    carbon::ReductionResult<double> mass_result(sys);
    carbon::Skeleton checksum_sk(sys);
    checksum_sk.add(carbon::Kernel::reduce<double>(
        grid,
        [&latest_pop](carbon::KernelLoader& loader) {
          const auto fin = loader.read(latest_pop);
          return [fin] CARBON_HOST_DEVICE(Span s, Idx idx) -> double {
            double acc = 0.0;
#pragma unroll
            for (int q = 0; q < Lattice::Q; ++q) {
              acc += static_cast<double>(fin.read(s, idx, q));
            }
            return acc;
          };
        },
        mass_result, carbon::reduce::Sum<double>{}, "lbm_mass_checksum", /*shared_mem=*/0U));
    try {
      checksum_sk.compile();
      checksum_sk.run();
      checksum_sk.wait();
      mass_result.sync(sys.stream());
      sys.stream().wait();
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("lbm_lid_driven_cavity: checksum reduction failed: {}", e.what());
      return EXIT_FAILURE;
    }
    const double total_mass = mass_result.host_value();
    CARBON_LOG_INFO("lbm_lid_driven_cavity: CHECKSUM (total mass, grid={}) = {:.10g}", opts.grid,
                    total_mass);
    std::cout << "CHECKSUM=" << std::setprecision(12) << total_mass << " (grid=" << opts.grid
              << " N=" << opts.n << " iters=" << opts.iters << ")" << '\n';
  }

  // Optional VTK dump of the final macroscopic fields (rho, u). Off the hot
  // path: only triggered with --vtk-out, runs make_macroscopic_kernel against
  // whichever ping-pong buffer holds the latest populations, downloads the
  // host mirrors, and writes via carbon::io::write_vtk (legacy
  // STRUCTURED_POINTS BINARY; one .rankN shard per rank under MPI).
  if (!opts.vtk_path.empty()) {
    // carbon::io::write_vtk is specialised on carbon::Dense (the legacy
    // STRUCTURED_POINTS writer walks the aggregated dense row layout). The
    // disaggregated grid has no VTK writer today, so the dump is Dense-only;
    // the disg run still reports MLUPS, it just skips the optional VTK export.
    if constexpr (std::is_same_v<Grid, carbon::Dense>) {
      // The hot loop alternates sk_ab (reads pop_a, writes pop_b) on even iters
      // with sk_ba (reads pop_b, writes pop_a) on odd. After `iters` iterations
      // the freshest output buffer is therefore pop_b when iters is odd and
      // pop_a otherwise.
      auto& latest_pop = (opts.iters % 2 == 1) ? pop_b : pop_a;

      // The field names propagate into the VTK SCALARS labels, so name them
      // "rho" and "u" — that's what shows up in ParaView's Pipeline Browser
      // (rather than the legacy ordinal "field_0" / "field_1") and keeps the
      // dump self-describing.
      auto rho_field = grid.template newField<float, 1>(carbon::MemorySide::HostDevice, "rho");
      auto u_field = grid.template newField<float, 3>(carbon::MemorySide::HostDevice, "u");

      carbon::Skeleton macro_sk(sys);
      macro_sk.add(make_macroscopic_kernel<Grid, Lattice>(grid, latest_pop, cells, rho_field,
                                                          u_field, params));
      try {
        macro_sk.compile();
        macro_sk.run();
        macro_sk.wait();
      } catch (const std::exception& e) {
        CARBON_LOG_ERROR("lbm_lid_driven_cavity: macroscopic kernel failed: {}", e.what());
        return EXIT_FAILURE;
      }

      carbon::Stream dl_stream = sys.stream();
      try {
        carbon::Transfer<float>(sys, rho_field.gpu_buffers(), rho_field.cpu_buffers())(dl_stream);
        carbon::Transfer<float>(sys, u_field.gpu_buffers(), u_field.cpu_buffers())(dl_stream);
        dl_stream.wait();
      } catch (const std::exception& e) {
        CARBON_LOG_ERROR("lbm_lid_driven_cavity: VTK download failed: {}", e.what());
        return EXIT_FAILURE;
      }

      try {
        carbon::io::write_vtk(opts.vtk_path, grid, rho_field, u_field);
        // Drop a `.vtm` master alongside the per-rank shards so users can open
        // one file in ParaView instead of N. No-op in the single-process case.
        carbon::io::write_vtk_multiblock_index(opts.vtk_path, grid);
      } catch (const std::exception& e) {
        CARBON_LOG_ERROR("lbm_lid_driven_cavity: write_vtk('{}') failed: {}", opts.vtk_path,
                         e.what());
        return EXIT_FAILURE;
      }
      CARBON_LOG_INFO("lbm_lid_driven_cavity: wrote VTK (rho + u) to {} after {} iters",
                      opts.vtk_path, opts.iters);
    } else {
      CARBON_LOG_WARN(
          "lbm_lid_driven_cavity: --vtk-out is only supported for --grid dense "
          "(carbon::io::write_vtk has no DenseDisg overload); skipping VTK dump.");
    }
  }

  // Optional structured JSON report. Schema documented in
  // benchmarks/lbm_lid_driven_cavity/scripts/sweep_and_plot.py — bump the
  // schema string below and the parser there in lockstep when extending.
  // Effective bandwidth assumes a pull-stream + ping-pong layout: each cell
  // update reads Q populations from fIn and writes Q populations to fOut,
  // so the dominant traffic is 2 * Q * sizeof(float) bytes per update.
  // Only rank 0 writes the report: under MPI every rank runs this block, and
  // if they all open the same --report path the writes interleave and produce
  // corrupt JSON (e.g. a duplicated trailing brace). MLUPS is a global figure
  // already reduced across ranks, so rank 0's copy is authoritative.
#ifdef CARBON_HAVE_MPI_NCCL
  const int report_rank = use_mpi ? distributed->world_rank() : 0;
#else
  const int report_rank = 0;
#endif
  if (!opts.report_path.empty() && report_rank == 0) {
    const double bytes_per_update =
        2.0 * static_cast<double>(Lattice::Q) * static_cast<double>(sizeof(float));
    const double effective_bandwidth_gb_s = (mlups * 1.0e6 * bytes_per_update) / 1.0e9;
    std::ofstream out(opts.report_path);
    if (!out.is_open()) {
      CARBON_LOG_ERROR("lbm_lid_driven_cavity: failed to open --report path '{}'",
                       opts.report_path);
      return EXIT_FAILURE;
    }
    out << std::setprecision(10);
    out << "{\n";
    out << "  \"schema\": \"carbon-lbm-report-v1\",\n";
    out << "  \"benchmark\": \"lbm_lid_driven_cavity\",\n";
    out << "  \"timestamp_utc\": \"" << utc_timestamp_now() << "\",\n";
    out << "  \"config\": {\n";
    out << "    \"n\": " << opts.n << ",\n";
    out << "    \"iters\": " << opts.iters << ",\n";
    out << "    \"warmup\": " << opts.warmup << ",\n";
    out << "    \"lattice\": \"" << json_escape(lattice_name) << "\",\n";
    out << "    \"collision\": \"" << json_escape(collision_name) << "\",\n";
    out << "    \"grid\": \"" << json_escape(opts.grid) << "\",\n";
    out << "    \"q\": " << Lattice::Q << ",\n";
    out << "    \"re\": " << opts.re << ",\n";
    out << "    \"ulb\": " << ulb << ",\n";
    out << "    \"nu_lb\": " << nu_lb << ",\n";
    out << "    \"omega\": " << omega << ",\n";
    out << "    \"no_warmup_skip\": " << (opts.no_warmup_skip ? "true" : "false") << ",\n";
    out << "    \"overlap\": \"" << json_escape(opts.overlap) << "\",\n";
    out << "    \"cuda_graph\": " << (opts.cuda_graph ? "true" : "false") << ",\n";
    out << "    \"label\": \"" << json_escape(opts.label) << "\"\n";
    out << "  },\n";
    out << "  \"system\": {\n";
    out << "    \"backend\": \"" << CARBON_BACKEND_NAME << "\",\n";
    // `distributed` distinguishes the carbon transport path actually in use:
    //   - false : intra-process multi-GPU only (cudaMemcpyPeerAsync /
    //             SYCL queue-to-queue copy for halo).
    //   - true  : MPI/NCCL build, *and* >1 rank present at runtime — halo
    //             flows over NCCL between adjacent ranks (one rank per GPU
    //             under carbon's strict model). The single-rank case still
    //             reports false even on an MPI build, because no NCCL ever
    //             gets called.
#ifdef CARBON_HAVE_MPI_NCCL
    out << "    \"distributed\": " << (use_mpi ? "true" : "false") << ",\n";
#else
    out << "    \"distributed\": false,\n";
#endif
    out << "    \"gpu_count\": " << n_gpus << ",\n";
    out << "    \"gpus\": [\n";
    const auto& gpu_devices = sys.gpu_group().devices();
    for (size_t i = 0; i < gpu_devices.size(); ++i) {
      const auto& props = gpu_devices[i].properties();
      out << "      {\n";
      out << "        \"device_id\": " << gpu_devices[i].device_id() << ",\n";
      out << "        \"name\": \"" << json_escape(props.name) << "\",\n";
      out << "        \"compute_capability\": \"" << props.compute_major << "."
          << props.compute_minor << "\",\n";
      out << "        \"total_memory_bytes\": " << props.total_global_mem_bytes << ",\n";
      out << "        \"multiprocessor_count\": " << props.multiprocessor_count << "\n";
      out << "      }" << (i + 1 == gpu_devices.size() ? "" : ",") << "\n";
    }
    out << "    ]\n";
    out << "  },\n";
    out << "  \"results\": {\n";
    out << "    \"timed_iters\": " << timed_iters << ",\n";
    out << "    \"timing_seconds\": " << timing_seconds << ",\n";
    out << "    \"overall_seconds\": " << overall_seconds << ",\n";
    out << "    \"cells_per_iter\": " << static_cast<int64_t>(cells_per_iter) << ",\n";
    out << "    \"mlups\": " << mlups << ",\n";
    out << "    \"mlups_per_gpu\": " << (mlups / static_cast<double>(n_gpus)) << ",\n";
    out << "    \"effective_bandwidth_gb_s\": " << effective_bandwidth_gb_s << "\n";
    out << "  }\n";
    out << "}\n";
    out.close();
    if (out.fail()) {
      CARBON_LOG_ERROR("lbm_lid_driven_cavity: failed to write --report path '{}'",
                       opts.report_path);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("lbm_lid_driven_cavity: wrote JSON report to {}", opts.report_path);
  }
  return EXIT_SUCCESS;
}

}  // namespace

int
main(int argc,
     char** argv) {
  Options opts;
  if (!parse_cli(argc, argv, opts)) {
    return opts.show_help ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  carbon::init();
  if (carbon::device_count() <= 0) {
    CARBON_LOG_ERROR("lbm_lid_driven_cavity: no CUDA devices visible");
    return EXIT_FAILURE;
  }

  // Dispatch to the matching template instantiation. The runtime knobs
  // --grid {dense,disg} x --lattice {q19,q27} x --collision {bgk,kbc} select
  // one compile-time run_benchmark<Grid, Lattice, Collision>. Three valid
  // (lattice, collision) combos: (q19, bgk), (q27, bgk), (q27, kbc) — (q19,
  // kbc) is rejected upstream in parse_cli because KBC's moment basis is
  // D3Q27-specific. Each grid family runs the identical hot kernel.
  auto dispatch_grid = [&](auto grid_tag) -> int {
    using GridT = typename decltype(grid_tag)::type;
    if (opts.lattice == "q19" && opts.collision == "bgk") {
      return run_benchmark<GridT, D3Q19, BgkCollision>(opts, "q19", "bgk");
    }
    if (opts.lattice == "q27" && opts.collision == "bgk") {
      return run_benchmark<GridT, D3Q27, BgkCollision>(opts, "q27", "bgk");
    }
    if (opts.lattice == "q27" && opts.collision == "kbc") {
      return run_benchmark<GridT, D3Q27, KbcCollision>(opts, "q27", "kbc");
    }
    CARBON_LOG_ERROR(
        "lbm_lid_driven_cavity: unsupported (lattice='{}', collision='{}') combination — "
        "KBC requires D3Q27.",
        opts.lattice, opts.collision);
    return EXIT_FAILURE;
  };

  if (opts.grid == "disg") {
    return dispatch_grid(std::type_identity<carbon::DenseDisg>{});
  }
  return dispatch_grid(std::type_identity<carbon::Dense>{});
}
