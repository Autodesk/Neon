// Halo-exchange scalability benchmark: carbon::Dense vs carbon::DenseDisg, in
// both StencilSemantic::Standard and StencilSemantic::Lattice flavours.
//
// Standard semantic: every component plane of every adjacency pair's boundary is
// transferred. The aggregated layout dispatches one cudaMemcpyAsync per cardinality
// (2 * (n_gpus - 1) * Components entries); the disaggregated layout groups them
// into a single contiguous block per direction (2 * (n_gpus - 1) entries).
//
// Lattice semantic: only populations whose discrete velocity matches the transfer
// direction are sent. For D3Q19 that's 5/19 per direction; for D3Q27 9/27. On the
// disaggregated layout the active populations are run-length-fused into one
// transfer per maximal contiguous run -- "lattice-friendly" orderings (all c_q.z>0
// adjacent, same for c_q.z<0) collapse to two transfers per pair regardless of Q.
//
// Output: CSV (--csv) the bundled scripts/plot.py can chart. Each row records
// the (semantic, lattice) tuple alongside the (grid, N, C, GB/s) data.

#include <carbon/carbon.h>
#include <carbon/dense.h>
#include <carbon/dense_disg.h>
#include <carbon/halo.h>
#include <carbon/halo_disg.h>
#include <carbon/integer3d.h>
#include <carbon/stencil.h>
#include <carbon/system.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

enum class Semantic : int {
  Standard = 0,
  Lattice = 1
};
enum class LatticeKind : int {
  None = 0,
  D3Q19,
  D3Q19Grouped,
  D3Q27,
  D3Q27Grouped
};

struct RunArgs {
  int64_t cube_edge = 256;
  int n_gpus = -1;  // -1 => use all visible
  std::vector<int> components{1, 3, 9, 19, 27};
  int iters = 200;
  int warmup = 50;
  std::string csv_path{};
  Semantic semantic = Semantic::Standard;
  LatticeKind lattice = LatticeKind::None;
};

[[nodiscard]] auto
to_string(Semantic s) noexcept -> const char* {
  return s == Semantic::Standard ? "standard" : "lattice";
}

[[nodiscard]] auto
to_string(LatticeKind l) noexcept -> const char* {
  switch (l) {
    case LatticeKind::None:
      return "-";
    case LatticeKind::D3Q19:
      return "d3q19";
    case LatticeKind::D3Q19Grouped:
      return "d3q19_grouped";
    case LatticeKind::D3Q27:
      return "d3q27";
    case LatticeKind::D3Q27Grouped:
      return "d3q27_grouped";
  }
  return "?";
}

[[nodiscard]] auto
lattice_components(LatticeKind l) noexcept -> int {
  switch (l) {
    case LatticeKind::D3Q19:
    case LatticeKind::D3Q19Grouped:
      return 19;
    case LatticeKind::D3Q27:
    case LatticeKind::D3Q27Grouped:
      return 27;
    default:
      return 0;
  }
}

// Velocity tables. The "natural" orderings match Carbon's LBM benchmark headers
// (benchmarks/lbm_lid_driven_cavity/src/d3q19.h / d3q27.h); the "_grouped"
// variants are the same set of velocities permuted so c.z<0 entries come first,
// then c.z==0, then c.z>0 -- the layout that lets the disg lattice halo collapse
// to one fused transfer per direction per pair.
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
d3q27_natural() -> std::vector<carbon::int32_3d> {
  // 27-velocity ordering iterating (z, y, x) over {-1, 0, 1}.
  std::vector<carbon::int32_3d> out;
  out.reserve(27);
  for (int z = -1; z <= 1; ++z) {
    for (int y = -1; y <= 1; ++y) {
      for (int x = -1; x <= 1; ++x) {
        out.push_back({x, y, z});
      }
    }
  }
  return out;  // c.z<0 in [0,9), c.z==0 in [9,18), c.z>0 in [18,27).
}

[[nodiscard]] auto
d3q27_grouped() -> std::vector<carbon::int32_3d> {
  // The natural (z, y, x) ordering above is *already* fully grouped by c.z sign.
  // Provided as an alias for symmetry with the D3Q19 case.
  return d3q27_natural();
}

[[nodiscard]] auto
make_lattice_stencil(LatticeKind l) -> carbon::Stencil {
  switch (l) {
    case LatticeKind::D3Q19:
      return carbon::Stencil(d3q19_natural());
    case LatticeKind::D3Q19Grouped:
      return carbon::Stencil(d3q19_grouped());
    case LatticeKind::D3Q27:
      return carbon::Stencil(d3q27_natural());
    case LatticeKind::D3Q27Grouped:
      return carbon::Stencil(d3q27_grouped());
    default:
      return carbon::Stencil{};
  }
}

[[nodiscard]] auto
parse_int_csv_list(const std::string& s) -> std::vector<int> {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (!tok.empty()) {
      out.push_back(std::atoi(tok.c_str()));
    }
  }
  return out;
}

[[nodiscard]] auto
parse_args(int argc,
           char** argv) -> RunArgs {
  RunArgs a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        return {};
      }
      return argv[++i];
    };
    if (k == "--N") {
      a.cube_edge = std::atoll(next().c_str());
    } else if (k == "--gpus") {
      a.n_gpus = std::atoi(next().c_str());
    } else if (k == "--components") {
      a.components = parse_int_csv_list(next());
    } else if (k == "--iters") {
      a.iters = std::atoi(next().c_str());
    } else if (k == "--warmup") {
      a.warmup = std::atoi(next().c_str());
    } else if (k == "--csv") {
      a.csv_path = next();
    } else if (k == "--semantic") {
      const std::string v = next();
      if (v == "standard") {
        a.semantic = Semantic::Standard;
      } else if (v == "lattice") {
        a.semantic = Semantic::Lattice;
      } else {
        std::cerr << "Unknown --semantic value: " << v << " (expected 'standard' or 'lattice')\n";
        std::exit(1);
      }
    } else if (k == "--lattice") {
      const std::string v = next();
      if (v == "d3q19") {
        a.lattice = LatticeKind::D3Q19;
      } else if (v == "d3q19_grouped") {
        a.lattice = LatticeKind::D3Q19Grouped;
      } else if (v == "d3q27") {
        a.lattice = LatticeKind::D3Q27;
      } else if (v == "d3q27_grouped") {
        a.lattice = LatticeKind::D3Q27Grouped;
      } else {
        std::cerr << "Unknown --lattice value: " << v << "\n";
        std::exit(1);
      }
    } else if (k == "--help" || k == "-h") {
      std::cout << "Usage: carbon_benchmark_halo_scalability [options]\n"
                << "  --N <int>           Cube edge (default 256)\n"
                << "  --gpus <int>        Number of GPUs to use (default: all visible)\n"
                << "  --components <csv>  Comma-separated component counts (default 1,3,9,19,27;\n"
                << "                      ignored when --semantic lattice is set)\n"
                << "  --iters <int>       Timed iterations per config (default 200)\n"
                << "  --warmup <int>      Warmup iterations (default 50)\n"
                << "  --csv <path>        Append CSV rows to this file\n"
                << "  --semantic <s>      'standard' (default) or 'lattice'\n"
                << "  --lattice <kind>    Required when --semantic lattice. One of\n"
                << "                      d3q19, d3q19_grouped, d3q27, d3q27_grouped\n";
      std::exit(0);
    } else {
      std::cerr << "Unknown argument: " << k << "\n";
      std::exit(1);
    }
  }
  return a;
}

struct Result {
  std::string grid;
  std::string semantic;
  std::string lattice;
  int64_t cube_edge{};
  int components{};
  int n_gpus{};
  size_t pair_count{};
  int iters{};
  double wall_ms_total{};
  double wall_ms_per_iter{};
  double bytes_per_iter{};
  double gb_per_s{};
};

template <typename HaloT>
[[nodiscard]] auto
time_halo(const HaloT& halo,
          const carbon::Stream& stream,
          int warmup,
          int iters) -> double {
  for (int i = 0; i < warmup; ++i) {
    halo(stream);
  }
  stream.wait();

  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    halo(stream);
  }
  stream.wait();
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Bytes moved across all GPU<->GPU edges per single halo call. The standard halo
// always moves the full Components per face per direction. Lattice halos
// move only positive_z (up) + negative_z (down) populations per pair.
[[nodiscard]] auto
halo_bytes_per_iter(int64_t cube_edge,
                    int active_components_per_pair,
                    int n_gpus,
                    size_t element_size) noexcept -> double {
  const double face_xy = static_cast<double>(cube_edge) * static_cast<double>(cube_edge);
  // Per-pair payload (sum of both directions). Standard halo: 2 * Components;
  // lattice: positive_z + negative_z (each entry counted once, no factor of 2).
  return face_xy * static_cast<double>(active_components_per_pair) *
         static_cast<double>(element_size) * static_cast<double>(n_gpus - 1);
}

template <int Components>
auto
run_one_pair(const RunArgs& args,
             carbon::System& sys,
             std::vector<Result>& out) -> void {
  using T = float;
  const int64_t N = args.cube_edge;
  const carbon::int64_3d global{N, N, N};
  const int n_gpus = static_cast<int>(sys.gpu_group().size());
  const bool lattice = (args.semantic == Semantic::Lattice);
  const carbon::Stencil stencil = lattice ? make_lattice_stencil(args.lattice) : carbon::Stencil{};
  const carbon::StencilSemantic semantic =
      lattice ? carbon::StencilSemantic::Lattice : carbon::StencilSemantic::Standard;

  int active_per_pair = 2 * Components;  // Standard: 2 * C (both directions).
  if (lattice) {
    const auto profile = stencil.lattice_profile();
    active_per_pair = profile.positive_z + profile.negative_z;
  }

  // ---- carbon::Dense halo (aggregated) ----
  {
    // Aggregated grids need a non-trivial Stencil to allocate a halo. For
    // standard semantic we use a single-point z stencil (radius 1, halo_z = 1);
    // for lattice we use the actual lattice stencil so the radius matches.
    const carbon::Stencil halo_stencil =
        lattice ? stencil : carbon::Stencil(std::vector<carbon::int32_3d>{{0, 0, 1}});
    carbon::Dense dense(sys, global, halo_stencil);
    auto field = dense.template newField<T, Components>(carbon::MemorySide::Device);
    carbon::Halo<T, Components> halo(field, stencil, semantic);
    auto stream = sys.stream();
    const double total_ms = time_halo(halo, stream, args.warmup, args.iters);
    const double per_iter = total_ms / static_cast<double>(args.iters);
    const double bytes = halo_bytes_per_iter(N, active_per_pair, n_gpus, sizeof(T));
    const double gbs = (bytes / (per_iter * 1e-3)) / 1e9;
    Result r;
    r.grid = "dense";
    r.semantic = to_string(args.semantic);
    r.lattice = to_string(args.lattice);
    r.cube_edge = N;
    r.components = Components;
    r.n_gpus = n_gpus;
    r.pair_count = halo.pair_count();
    r.iters = args.iters;
    r.wall_ms_total = total_ms;
    r.wall_ms_per_iter = per_iter;
    r.bytes_per_iter = bytes;
    r.gb_per_s = gbs;
    out.push_back(r);
  }

  // ---- carbon::DenseDisg halo (disaggregated) ----
  {
    carbon::DenseDisg dense(sys, global, /*halo_z=*/1);
    auto field = dense.template newField<T, Components>(carbon::MemorySide::Device);
    carbon::HaloDisg<T, Components> halo(field, stencil, semantic);
    auto stream = sys.stream();
    const double total_ms = time_halo(halo, stream, args.warmup, args.iters);
    const double per_iter = total_ms / static_cast<double>(args.iters);
    const double bytes = halo_bytes_per_iter(N, active_per_pair, n_gpus, sizeof(T));
    const double gbs = (bytes / (per_iter * 1e-3)) / 1e9;
    Result r;
    r.grid = "disg";
    r.semantic = to_string(args.semantic);
    r.lattice = to_string(args.lattice);
    r.cube_edge = N;
    r.components = Components;
    r.n_gpus = n_gpus;
    r.pair_count = halo.pair_count();
    r.iters = args.iters;
    r.wall_ms_total = total_ms;
    r.wall_ms_per_iter = per_iter;
    r.bytes_per_iter = bytes;
    r.gb_per_s = gbs;
    out.push_back(r);
  }
}

auto
dispatch_components(const RunArgs& args,
                    carbon::System& sys,
                    int components,
                    std::vector<Result>& out) -> void {
  switch (components) {
    case 1:
      run_one_pair<1>(args, sys, out);
      return;
    case 2:
      run_one_pair<2>(args, sys, out);
      return;
    case 3:
      run_one_pair<3>(args, sys, out);
      return;
    case 5:
      run_one_pair<5>(args, sys, out);
      return;
    case 7:
      run_one_pair<7>(args, sys, out);
      return;
    case 9:
      run_one_pair<9>(args, sys, out);
      return;
    case 13:
      run_one_pair<13>(args, sys, out);
      return;
    case 19:
      run_one_pair<19>(args, sys, out);
      return;
    case 27:
      run_one_pair<27>(args, sys, out);
      return;
    default:
      std::cerr << "Skipping Components=" << components
                << " (not in compile-time switch; add a case in dispatch_components)\n";
  }
}

auto
write_csv_header(std::ofstream& f) -> void {
  f << "grid,semantic,lattice,cube_edge,components,n_gpus,pair_count,iters,wall_ms_total,"
       "wall_ms_per_iter,bytes_per_iter,gb_per_s\n";
}

auto
write_csv_row(std::ofstream& f,
              const Result& r) -> void {
  f << r.grid << ',' << r.semantic << ',' << r.lattice << ',' << r.cube_edge << ',' << r.components
    << ',' << r.n_gpus << ',' << r.pair_count << ',' << r.iters << ',' << r.wall_ms_total << ','
    << r.wall_ms_per_iter << ',' << r.bytes_per_iter << ',' << r.gb_per_s << '\n';
}

auto
print_table(const std::vector<Result>& rows) -> void {
  std::printf("%-6s  %-9s  %-14s  %-6s  %-3s  %-6s  %-5s  %-7s  %-12s  %-9s\n", "grid", "semantic",
              "lattice", "N", "C", "n_gpus", "pairs", "iters", "ms/iter", "GB/s");
  std::printf("%-6s  %-9s  %-14s  %-6s  %-3s  %-6s  %-5s  %-7s  %-12s  %-9s\n", "----", "---------",
              "--------------", "------", "---", "------", "-----", "-------", "------------",
              "---------");
  for (const auto& r : rows) {
    std::printf("%-6s  %-9s  %-14s  %-6lld  %-3d  %-6d  %-5zu  %-7d  %12.4f  %9.3f\n",
                r.grid.c_str(), r.semantic.c_str(), r.lattice.c_str(),
                static_cast<long long>(r.cube_edge), r.components, r.n_gpus, r.pair_count, r.iters,
                r.wall_ms_per_iter, r.gb_per_s);
  }
}

}  // namespace

int
main(int argc,
     char** argv) {
  carbon::init();
  const RunArgs args = parse_args(argc, argv);

  // Honour --gpus by trimming the device list to the requested prefix.
  auto devices = carbon::get_all_devices();
  if (args.n_gpus > 0 && static_cast<size_t>(args.n_gpus) < devices.size()) {
    devices.erase(devices.begin() + static_cast<ptrdiff_t>(args.n_gpus), devices.end());
  }

  if (devices.empty()) {
    std::cerr << "No CUDA devices visible -- nothing to benchmark.\n";
    return 1;
  }

  carbon::System sys(std::move(devices));
  const int n_gpus = static_cast<int>(sys.gpu_group().size());

  // Lattice mode: Components is fixed by the lattice; --components is ignored.
  RunArgs run_args = args;
  if (run_args.semantic == Semantic::Lattice) {
    if (run_args.lattice == LatticeKind::None) {
      std::cerr << "--semantic lattice requires --lattice <kind>\n";
      return 1;
    }
    run_args.components = {lattice_components(run_args.lattice)};
  }

  std::cout << "Carbon halo scalability benchmark\n"
            << "  cube_edge = " << run_args.cube_edge << "\n"
            << "  n_gpus    = " << n_gpus << "\n"
            << "  semantic  = " << to_string(run_args.semantic) << "\n"
            << "  lattice   = " << to_string(run_args.lattice) << "\n"
            << "  iters     = " << run_args.iters << " (warmup " << run_args.warmup << ")\n";

  std::vector<Result> rows;
  rows.reserve(run_args.components.size() * 2U);
  for (int c : run_args.components) {
    try {
      dispatch_components(run_args, sys, c, rows);
    } catch (const std::exception& e) {
      std::cerr << "Run with C=" << c << " failed: " << e.what() << "\n";
    }
  }

  print_table(rows);

  if (!args.csv_path.empty()) {
    const bool fresh = !std::ifstream(args.csv_path).good();
    std::ofstream f(args.csv_path, std::ios::app);
    if (!f.good()) {
      std::cerr << "Failed to open CSV: " << args.csv_path << "\n";
      return 1;
    }
    if (fresh) {
      write_csv_header(f);
    }
    for (const auto& r : rows) {
      write_csv_row(f, r);
    }
    std::cout << "Wrote " << rows.size() << " rows to " << args.csv_path << "\n";
  }

  return 0;
}
