// Current flow between two offset terminals on opposite faces of a box.
//
// Steady current flow in a conducting box solves  div(sigma grad(phi)) = 0.  Current J0 enters
// through a disc-shaped terminal on the top face and leaves through an equal terminal on the bottom
// face, shifted sideways. Every other part of the box surface is insulating, and a resistive sphere
// sits between the terminals.
//
// Boundary conditions, in terms of the outward-normal derivative dphi/dn (sigma = 1 at the faces):
//   top face, inside the inlet disc     dphi/dn = +J0      (Neumann, non-zero)
//   bottom face, inside the outlet disc dphi/dn = -J0      (Neumann, non-zero)
//   everywhere else                     dphi/dn = 0        (insulating)
//
// This is the case per-face constants (carbon::BoundaryConditions) cannot express: the flux differs
// from cell to cell on one face. It is declared once with carbon::BoundaryData, and the stencil
// kernel reads every neighbour through Partition::neighbour_bc with that data, with no edge
// branches. The conductivity field uses a second BoundaryData (zero flux on every face), so two
// fields on one grid carry different conditions.
//
// The program solves two cases and compares them:
//   terminals  the declared terminals above;
//   uniform    the same total current spread evenly over each whole face, which is all a single
//              constant per face can say.
//
// Checks (the program exits non-zero if either fails; both run on the host over small arrays):
//   - the Jacobi iteration has converged (relative update below 1e-6);
//   - all the injected current crosses the plane midway between the terminals, to within 1%.
//     The uniform case carries no sideways current at all.
//
// Output: a two-panel PPM of the potential on the mid-plane slice (left: terminals, right: uniform,
// one shared diverging colour scale), and with --raw DIR the raw float64 arrays (phi, jx, jz of both
// cases, indexed [z][y][x]) for plotting. Single process only.
//
// Usage:
//   ./carbon_example_offset_current_terminals [--size N] [--iters N] [--out image.ppm] [--raw DIR]
//
// Pure flux problem: the potential is defined up to a constant, so the images subtract the mean.

#include <carbon/boundary_condition_distributed.h>
#include <carbon/boundary_data_distributed.h>
#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense_distributed.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/skeleton.h>
#include <carbon/stencil.h>
#include <carbon/system.h>
#include <carbon/transfer.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using carbon::BoundaryData;
using carbon::DenseDistributed;
using carbon::Face;
using carbon::FaceBc;
using carbon::int64_3d;

constexpr double kInletFlux = 1.0;
constexpr double kInclusionSigma = 0.05;

// Geometry, scaled with the box side n so any --size keeps the same picture.
struct Geometry {
  int64_t n{0};
  int64_t cx{0};  // box axis / sphere centre
  int64_t xa{0};  // inlet terminal centre (top face), x
  int64_t xb{0};  // outlet terminal centre (bottom face), x
  double terminal_r{0.0};
  double sphere_r{0.0};

  explicit Geometry(int64_t side)
      : n(side),
        cx(side / 2),
        xa(side * 27 / 100),
        xb(side - 1 - side * 27 / 100),
        terminal_r(0.15 * static_cast<double>(side)),
        sphere_r(0.17 * static_cast<double>(side)) {}

  [[nodiscard]] auto
  in_terminal(int64_t x,
              int64_t y,
              int64_t centre_x) const -> bool {
    const double dx = static_cast<double>(x - centre_x);
    const double dy = static_cast<double>(y - cx);
    return dx * dx + dy * dy <= terminal_r * terminal_r;
  }

  [[nodiscard]] auto
  sigma(int64_t x,
        int64_t y,
        int64_t z) const -> double {
    const double dx = static_cast<double>(x - cx);
    const double dy = static_cast<double>(y - cx);
    const double dz = static_cast<double>(z - cx);
    return dx * dx + dy * dy + dz * dz <= sphere_r * sphere_r ? kInclusionSigma : 1.0;
  }

  // Trapezoid weight of a node on a node-centred face / plane: edge nodes carry half a cell.
  [[nodiscard]] auto
  weight(int64_t a) const -> double {
    return (a == 0 || a == n - 1) ? 0.5 : 1.0;
  }

  // Current entering through the inlet terminal, weighted like the discrete face integral.
  [[nodiscard]] auto
  injected_current() const -> double {
    double total = 0.0;
    for (int64_t y = 0; y < n; ++y) {
      for (int64_t x = 0; x < n; ++x) {
        if (in_terminal(x, y, xa)) {
          total += kInletFlux * weight(x) * weight(y);
        }
      }
    }
    return total;
  }
};

// One Jacobi sweep of div(sigma grad(phi)) = 0 with harmonic-mean face conductivity. Every
// neighbour of phi and sigma is read through neighbour_bc with that field's own boundary data.
template <typename Grid>
[[nodiscard]] auto
build_sweep(const Grid& grid,
            typename Grid::template Field<double,
                                          1>& phi_in,
            typename Grid::template Field<double,
                                          1>& phi_out,
            typename Grid::template Field<double,
                                          1>& sigma,
            const BoundaryData<1>& phi_bc,
            const BoundaryData<1>& sigma_bc,
            std::string name) -> carbon::Kernel {
  using Span = typename Grid::Span;
  using Idx = typename Grid::Idx;
  return carbon::Kernel::compute(
      grid,
      [&phi_in, &phi_out, &sigma, &phi_bc, &sigma_bc](carbon::KernelLoader& loader) {
        const auto pin = loader.read_stencil(phi_in);
        const auto sig = loader.read_stencil(sigma);
        const auto pout = loader.write(phi_out);
        const auto pv = phi_bc.view(loader.relative_gpu_id());
        const auto sv = sigma_bc.view(loader.relative_gpu_id());
        return [pin, sig, pout, pv, sv] CARBON_HOST_DEVICE(Span s, Idx idx) -> void {
          const int dirs[6][3] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                                  {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
          const double sc = sig.read(s, idx, 0);
          double num = 0.0;
          double den = 0.0;
          for (const auto& d : dirs) {
            const carbon::int8_3d o{static_cast<int8_t>(d[0]), static_cast<int8_t>(d[1]),
                                    static_cast<int8_t>(d[2])};
            const auto p = pin.neighbour_bc(s, pv, idx, o, 0);
            const auto m = sig.neighbour_bc(s, sv, idx, o, 0);
            const double sf = 2.0 * sc * m.value / (sc + m.value);
            num += sf * p.value;
            den += sf;
          }
          pout.write(s, idx, 0, num / den);
        };
      },
      std::move(name));
}

// Current density j = -sigma grad(phi): component 0 is j_x, component 1 is j_z.
template <typename Grid>
[[nodiscard]] auto
build_current(const Grid& grid,
              typename Grid::template Field<double,
                                            1>& phi,
              typename Grid::template Field<double,
                                            1>& sigma,
              typename Grid::template Field<double,
                                            2>& current,
              const BoundaryData<1>& phi_bc,
              std::string name) -> carbon::Kernel {
  using Span = typename Grid::Span;
  using Idx = typename Grid::Idx;
  return carbon::Kernel::compute(
      grid,
      [&phi, &sigma, &current, &phi_bc](carbon::KernelLoader& loader) {
        const auto pin = loader.read_stencil(phi);
        const auto sig = loader.read(sigma);
        const auto jout = loader.write(current);
        const auto pv = phi_bc.view(loader.relative_gpu_id());
        return [pin, sig, jout, pv] CARBON_HOST_DEVICE(Span s, Idx idx) -> void {
          const double sc = sig.read(s, idx, 0);
          const auto xp = pin.neighbour_bc(s, pv, idx, carbon::int8_3d{1, 0, 0}, 0);
          const auto xm = pin.neighbour_bc(s, pv, idx, carbon::int8_3d{-1, 0, 0}, 0);
          const auto zp = pin.neighbour_bc(s, pv, idx, carbon::int8_3d{0, 0, 1}, 0);
          const auto zm = pin.neighbour_bc(s, pv, idx, carbon::int8_3d{0, 0, -1}, 0);
          jout.write(s, idx, 0, -sc * 0.5 * (xp.value - xm.value));
          jout.write(s, idx, 1, -sc * 0.5 * (zp.value - zm.value));
        };
      },
      std::move(name));
}

struct CaseResult {
  std::vector<double> phi;  // [z][y][x], mean removed
  std::vector<double> jx;
  std::vector<double> jz;
  double plane_current{0.0};
  double relative_update{0.0};
  int sweeps{0};
};

// Copy the owned cells of a per-GPU field into one global [z][y][x] array.
template <int C>
void
gather(const DenseDistributed& grid,
       const DenseDistributed::Field<double,
                                     C>& field,
       int comp,
       int64_t n,
       std::vector<double>& out) {
  out.assign(static_cast<size_t>(n * n * n), 0.0);
  for (size_t gpu = 0; gpu < grid.gpu_count(); ++gpu) {
    const auto& span = grid.spans()[gpu];
    const int64_3d halo = span.halo();
    const int64_3d owned = span.logic_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t stride = span.linear_storage_elements();
    const double* h = field.cpu_buffers()[gpu].data();
    for (int64_t z = 0; z < owned.z; ++z) {
      for (int64_t y = 0; y < owned.y; ++y) {
        for (int64_t x = 0; x < owned.x; ++x) {
          const int64_3d g =
              span.global_location(span.make_idx(halo.x + x, halo.y + y, halo.z + z));
          const int64_t off = static_cast<int64_t>(comp) * stride + (halo.z + z) * pitch +
                              (halo.y + y) * span.haloed_extent().x + (halo.x + x);
          out[static_cast<size_t>((g.z * n + g.y) * n + g.x)] = h[off];
        }
      }
    }
  }
}

[[nodiscard]] auto
solve_case(carbon::System& sys,
           const Geometry& geo,
           bool terminals,
           int sweeps_requested,
           CaseResult& result) -> bool {
  const int64_t n = geo.n;
  const int64_3d global{n, n, n};
  const carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);
  DenseDistributed grid(sys, global, stencil);

  // The uniform case spreads the same total current over each whole face: the only thing one
  // constant per face can say.
  int64_t terminal_cells = 0;
  for (int64_t y = 0; y < n; ++y) {
    for (int64_t x = 0; x < n; ++x) {
      terminal_cells += geo.in_terminal(x, y, geo.xa) ? 1 : 0;
    }
  }
  const double uniform_flux =
      kInletFlux * static_cast<double>(terminal_cells) / static_cast<double>(n * n);

  BoundaryData<1> phi_bc(grid, terminals ? "phi_terminals" : "phi_uniform");
  phi_bc.fill([&](Face f, int64_3d g, int) -> FaceBc {
    if (f == Face::ZHigh) {
      if (!terminals) {
        return carbon::neumann(+uniform_flux);
      }
      return geo.in_terminal(g.x, g.y, geo.xa) ? carbon::neumann(+kInletFlux)
                                               : carbon::neumann(0.0);
    }
    if (f == Face::ZLow) {
      if (!terminals) {
        return carbon::neumann(-uniform_flux);
      }
      return geo.in_terminal(g.x, g.y, geo.xb) ? carbon::neumann(-kInletFlux)
                                               : carbon::neumann(0.0);
    }
    return carbon::neumann(0.0);
  });
  phi_bc.upload();

  // The conductivity carries different conditions from the potential on the same grid: zero flux
  // on every face, which makes its out-of-box value the in-box value.
  BoundaryData<1> sigma_bc(grid, "sigma_bc");
  sigma_bc.fill(carbon::BoundaryConditions::none()
                    .with(Face::XLow, carbon::neumann())
                    .with(Face::XHigh, carbon::neumann())
                    .with(Face::YLow, carbon::neumann())
                    .with(Face::YHigh, carbon::neumann())
                    .with(Face::ZLow, carbon::neumann())
                    .with(Face::ZHigh, carbon::neumann()));
  sigma_bc.upload();

  auto phi_a = grid.newField<double, 1>(carbon::MemorySide::HostDevice, "phi_a");
  auto phi_b = grid.newField<double, 1>(carbon::MemorySide::HostDevice, "phi_b");
  auto sigma = grid.newField<double, 1>(carbon::MemorySide::HostDevice, "sigma");
  auto current = grid.newField<double, 2>(carbon::MemorySide::HostDevice, "current");

  for (size_t gpu = 0; gpu < grid.gpu_count(); ++gpu) {
    const auto& span = grid.spans()[gpu];
    const int64_3d halo = span.halo();
    const int64_3d owned = span.logic_extent();
    const int64_3d haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    double* ha = phi_a.cpu_buffers()[gpu].data();
    double* hb = phi_b.cpu_buffers()[gpu].data();
    double* hs = sigma.cpu_buffers()[gpu].data();
    for (int64_t z = 0; z < haloed.z; ++z) {
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const bool own = x >= halo.x && x < halo.x + owned.x && y >= halo.y &&
                           y < halo.y + owned.y && z >= halo.z && z < halo.z + owned.z;
          const int64_3d g = span.global_location(span.make_idx(x, y, z));
          const int64_t off = z * pitch + y * haloed.x + x;
          ha[off] = hb[off] = 0.0;
          hs[off] = own ? geo.sigma(g.x, g.y, g.z) : 1.0;
        }
      }
    }
  }
  carbon::Stream stream = sys.stream();
  carbon::Transfer<double>(sys, phi_a.cpu_buffers(), phi_a.gpu_buffers())(stream);
  carbon::Transfer<double>(sys, phi_b.cpu_buffers(), phi_b.gpu_buffers())(stream);
  carbon::Transfer<double>(sys, sigma.cpu_buffers(), sigma.gpu_buffers())(stream);
  stream.wait();

  const int cycles = (sweeps_requested + 1) / 2;
  carbon::Skeleton pipe(sys);
  pipe.add(build_sweep(grid, phi_a, phi_b, sigma, phi_bc, sigma_bc, "sweep_ab"));
  pipe.add(build_sweep(grid, phi_b, phi_a, sigma, phi_bc, sigma_bc, "sweep_ba"));
  pipe.compile();
  for (int c = 0; c < cycles; ++c) {
    pipe.run();
  }
  pipe.wait();
  result.sweeps = cycles * 2;

  // Convergence: size of the last update (phi_a against phi_b, one sweep apart) relative to the size
  // of the solution. The checks run on the host: the arrays are small.
  carbon::Skeleton cur(sys);
  cur.add(build_current(grid, phi_a, sigma, current, phi_bc, "current_density"));
  cur.run();
  cur.wait();

  carbon::Transfer<double>(sys, phi_a.gpu_buffers(), phi_a.cpu_buffers())(stream);
  carbon::Transfer<double>(sys, phi_b.gpu_buffers(), phi_b.cpu_buffers())(stream);
  carbon::Transfer<double>(sys, current.gpu_buffers(), current.cpu_buffers())(stream);
  stream.wait();
  gather<1>(grid, phi_a, 0, n, result.phi);
  std::vector<double> phi_prev;
  gather<1>(grid, phi_b, 0, n, phi_prev);
  gather<2>(grid, current, 0, n, result.jx);
  gather<2>(grid, current, 1, n, result.jz);

  double diff2 = 0.0;
  double norm2 = 0.0;
  for (size_t i = 0; i < result.phi.size(); ++i) {
    const double d = result.phi[i] - phi_prev[i];
    diff2 += d * d;
    norm2 += result.phi[i] * result.phi[i];
  }
  result.relative_update = std::sqrt(diff2 / std::max(norm2, 1.0e-300));

  // Current crossing the plane x == cx: sum of j_x, trapezoid-weighted in y and z.
  for (int64_t z = 0; z < n; ++z) {
    for (int64_t y = 0; y < n; ++y) {
      result.plane_current +=
          geo.weight(y) * geo.weight(z) * result.jx[static_cast<size_t>((z * n + y) * n + geo.cx)];
    }
  }

  double mean = 0.0;
  for (double v : result.phi) {
    mean += v;
  }
  mean /= static_cast<double>(result.phi.size());
  for (double& v : result.phi) {
    v -= mean;
  }
  return true;
}

// Diverging blue-white-red map, t in [-1, 1].
void
diverging(double t,
          unsigned char rgb[3]) {
  const double c = std::clamp(t, -1.0, 1.0);
  const double neutral[3] = {240.0, 239.0, 236.0};
  const double blue[3] = {24.0, 79.0, 149.0};
  const double red[3] = {179.0, 45.0, 45.0};
  const double* end = c < 0.0 ? blue : red;
  const double a = std::fabs(c);
  for (int i = 0; i < 3; ++i) {
    rgb[i] = static_cast<unsigned char>(std::lround(neutral[i] + (end[i] - neutral[i]) * a));
  }
}

[[nodiscard]] auto
write_ppm(const std::string& path,
          const CaseResult& left,
          const CaseResult& right,
          int64_t n) -> bool {
  constexpr int64_t kScale = 8;
  constexpr int64_t kGap = 12;
  constexpr int64_t kBar = 16;
  const int64_t panel = n * kScale;
  const int64_t width = 2 * panel + kGap + kGap + kBar;
  const int64_t height = panel;
  double vmax = 0.0;
  for (const auto* r : {&left, &right}) {
    for (double v : r->phi) {
      vmax = std::max(vmax, std::fabs(v));
    }
  }
  std::vector<unsigned char> img(static_cast<size_t>(width * height * 3), 252);
  const int64_t y_mid = n / 2;
  const auto put = [&](int64_t px, int64_t py, const unsigned char rgb[3]) {
    const size_t i = static_cast<size_t>((py * width + px) * 3);
    img[i] = rgb[0];
    img[i + 1] = rgb[1];
    img[i + 2] = rgb[2];
  };
  const CaseResult* panels[2] = {&left, &right};
  for (int p = 0; p < 2; ++p) {
    const int64_t x0 = p * (panel + kGap);
    for (int64_t z = 0; z < n; ++z) {
      for (int64_t x = 0; x < n; ++x) {
        unsigned char rgb[3];
        diverging(panels[p]->phi[static_cast<size_t>((z * n + y_mid) * n + x)] / vmax, rgb);
        for (int64_t dy = 0; dy < kScale; ++dy) {
          for (int64_t dx = 0; dx < kScale; ++dx) {
            put(x0 + x * kScale + dx, (n - 1 - z) * kScale + dy, rgb);  // z up
          }
        }
      }
    }
  }
  const int64_t bar_x0 = 2 * panel + 2 * kGap;
  for (int64_t py = 0; py < height; ++py) {
    unsigned char rgb[3];
    diverging(1.0 - 2.0 * static_cast<double>(py) / static_cast<double>(height - 1), rgb);
    for (int64_t px = 0; px < kBar; ++px) {
      put(bar_x0 + px, py, rgb);
    }
  }
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    return false;
  }
  f << "P6\n" << width << " " << height << "\n255\n";
  f.write(reinterpret_cast<const char*>(img.data()), static_cast<std::streamsize>(img.size()));
  return static_cast<bool>(f);
}

void
write_raw(const std::string& dir,
          const std::string& tag,
          const CaseResult& r) {
  const auto dump = [&](const char* name, const std::vector<double>& v) {
    std::ofstream f(dir + "/" + tag + "_" + name + ".bin", std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(double)));
  };
  dump("phi", r.phi);
  dump("jx", r.jx);
  dump("jz", r.jz);
}

[[nodiscard]] auto
arg_value(int argc,
          char** argv,
          std::string_view flag) -> std::string {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string_view{argv[i]} == flag) {
      return std::string{argv[i + 1]};
    }
  }
  return {};
}

}  // namespace

auto
main(int argc,
     char** argv) -> int {
  carbon::init();
  if (carbon::device_count() <= 0) {
    CARBON_LOG_ERROR("offset_current_terminals: no devices visible");
    return EXIT_FAILURE;
  }
  int64_t size = 25;
  int iters = 4000;
  try {
    if (const std::string v = arg_value(argc, argv, "--size"); !v.empty()) {
      size = std::stoll(v);
    }
    if (const std::string v = arg_value(argc, argv, "--iters"); !v.empty()) {
      iters = std::stoi(v);
    }
  } catch (const std::exception&) {
    CARBON_LOG_ERROR("offset_current_terminals: --size and --iters take integers");
    return EXIT_FAILURE;
  }
  if (size < 15 || iters < 2) {
    CARBON_LOG_ERROR("offset_current_terminals: need --size >= 15 and --iters >= 2");
    return EXIT_FAILURE;
  }
  std::string out_path = arg_value(argc, argv, "--out");
  if (out_path.empty()) {
    out_path = "offset_current_terminals.ppm";
  }
  const std::string raw_dir = arg_value(argc, argv, "--raw");

  carbon::System sys(carbon::get_all_devices());
  if (sys.distributed() != nullptr) {
    CARBON_LOG_ERROR(
        "offset_current_terminals: single process only (the output gathers on rank 0)");
    return EXIT_FAILURE;
  }
  const Geometry geo(size);
  const double injected = geo.injected_current();

  CaseResult terminals;
  CaseResult uniform;
  try {
    if (!solve_case(sys, geo, /*terminals=*/true, iters, terminals) ||
        !solve_case(sys, geo, /*terminals=*/false, iters, uniform)) {
      return EXIT_FAILURE;
    }
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("offset_current_terminals: {}", e.what());
    return EXIT_FAILURE;
  }

  const double swing_t = *std::max_element(terminals.phi.begin(), terminals.phi.end());
  const double swing_u = *std::max_element(uniform.phi.begin(), uniform.phi.end());
  CARBON_LOG_INFO(
      "offset_current_terminals: {}^3 box, {} sweeps. Injected current {:.3f}. Current crossing "
      "the "
      "mid-plane: terminals {:.3f}, uniform {:.3f}. Potential swing +-{:.3f} (terminals) against "
      "+-{:.3f} (uniform). Relative update: {:.2e} / {:.2e}.",
      size, terminals.sweeps, injected, terminals.plane_current, uniform.plane_current, swing_t,
      swing_u, terminals.relative_update, uniform.relative_update);

  if (!write_ppm(out_path, terminals, uniform, size)) {
    CARBON_LOG_ERROR("offset_current_terminals: could not write {}", out_path);
    return EXIT_FAILURE;
  }
  if (!raw_dir.empty()) {
    write_raw(raw_dir, "terminals", terminals);
    write_raw(raw_dir, "uniform", uniform);
  }
  CARBON_LOG_INFO("offset_current_terminals: wrote {} (left: terminals, right: uniform)", out_path);

  bool ok = true;
  if (terminals.relative_update > 1.0e-6 || uniform.relative_update > 1.0e-6) {
    CARBON_LOG_ERROR("offset_current_terminals: not converged; raise --iters");
    ok = false;
  }
  if (std::fabs(terminals.plane_current - injected) > 0.01 * injected) {
    CARBON_LOG_ERROR(
        "offset_current_terminals: current crossing the mid-plane ({:.3f}) differs from the "
        "injected "
        "current ({:.3f}) by more than 1%",
        terminals.plane_current, injected);
    ok = false;
  }
  if (std::fabs(uniform.plane_current) > 0.01 * injected) {
    CARBON_LOG_ERROR(
        "offset_current_terminals: the uniform case should carry no sideways current ({:.3f})",
        uniform.plane_current);
    ok = false;
  }
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
