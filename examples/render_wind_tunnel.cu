// Ray-march visualization of a three-dimensional cylinder wake, driven by a real
// LBM solve.
//
// A D3Q27 lattice-Boltzmann simulation (BGK collision) of cross-flow past a
// circular cylinder is run to a shedding state, then two scalar fields are
// rendered top-down (looking along the span, +z):
//   pass 1 — |u|/ulb in Jet (freestream faded out);
//   pass 2 — Q-criterion (vortex cores, Q > 0) in Viridis;
// both overlaid on a Blinn-lit cylinder SDF from an analytic signed-distance field.
//
// The interactive viewer adds two transfer functions that read a second field
// per sample (@c RayMarchShading::ScalarFieldAttribute):
//   "Wake"                  — opacity from |u - u_inf|, hue from the signed
//                             streamwise deviation, so the free stream is
//                             transparent and slowed/accelerated fluid separate
//                             into the two lobes of a diverging colormap;
//   "Q-criterion (tinted)"  — opacity from Q, hue from local speed, telling the
//                             energetic near-body braids from decayed far cores
//                             that happen to share a Q value.
//
// The interactive viewer also offers density-splat variants of the passes
// (@ref carbon::make_density_kernel): same Carbon kernel-loader model as
// @c render_density / @c render_density_mgrid, no ray traversal — the path that
// ports directly onto adaptive @c MultiResBlock towers without a BVH.
//
// Why a cylinder (not a sphere): a sphere wake only sheds above Re ~ 270; a
// cylinder sheds for Re >~ 47 and turns three-dimensional past Re ~ 190 (the
// mode-A/B spanwise instabilities), so a genuinely 3D wake lands comfortably
// inside the stable band. The span (z) is *thick* and periodic — physically exact
// for a spanwise-homogeneous cylinder and the canonical DNS setup — and the
// free-stream is seeded with a small spanwise velocity ripple so the 3D mode is
// actually triggered (a z-invariant state would stay 2D forever). A larger Re is
// reached at fixed stability by resolving the body more (bigger D ⇒ larger nu at
// fixed Re ⇒ omega stays < 2), the same trade the XLB reference makes with a
// multi-resolution grid.
//
// Setup (defaults, all overridable): grid 768x384x128 (CARBON_WT_NX/NY/NZ),
// cylinder R = 32 (D = 64, blockage 16.7%) centred at (Nx/4, Ny/2). Uniform
// free-stream inlet + far-field transverse walls (velocity Dirichlet),
// density-anchored extrapolation outlet, PeriodicZ span, half-way bounce-back at
// the cylinder. Runs on a single GPU by default (see pick_render_devices);
// pass --multi-gpu or set CARBON_WT_MULTI_GPU=1 to use every visible GPU. Hot LBM
// kernels come from benchmarks/lbm_lid_driven_cavity/src/ unmodified.
//
// Build & run
// -----------
//   cmake --build build --target carbon_example_render_wind_tunnel
//   ./build/examples/carbon_example_render_wind_tunnel
//   ./build/examples/carbon_example_render_wind_tunnel --web --port 5000
//   ./build/examples/carbon_example_render_wind_tunnel --multi-gpu
// The third positional argument is the VTK XML ImageData (.vti) volume dump of the
// raw velocity / speed / Q-criterion fields, written next to the two renderings.

#include "cell_type.h"
#include "d3q19.h"
#include "d3q27.h"
#include "lbm_kernels.h"

#include <carbon/carbon.h>
#include <carbon/dense.h>
#include <carbon/halo.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/qcriterion.h>
#include <carbon/render.h>
#include <carbon/skeleton.h>
#include <carbon/stencil.h>
#include <carbon/system.h>
#include <carbon/transfer.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#ifdef CARBON_HAVE_VIEWER
#include "viewer/carbon_viewer.h"
#include "viewer/render_rgb_utils.h"
#endif

namespace {

enum class Layers : unsigned char {
  Speed,
  Q,
  /** @c |u - u_inf| : the free stream subtracted away, so only the body's wake is drawn. */
  Wake,
  /** Q-criterion opacity, local-speed hue. */
  QTinted,
  Both,
};

using carbon::bench::lbm::BgkCollision;
using carbon::bench::lbm::Classification;
using carbon::bench::lbm::D3Q27;
using carbon::bench::lbm::encode_cell;
using carbon::bench::lbm::LbmParameters;
using carbon::bench::lbm::make_compute_wall_mask_kernel;
using carbon::bench::lbm::make_iteration_kernel;
using carbon::bench::lbm::make_macroscopic_kernel;

#ifdef CARBON_HAVE_MPI_NCCL
[[nodiscard]] auto
world_rank(const carbon::System& sys) noexcept -> int {
  if (const auto* d = sys.distributed(); d != nullptr) {
    return d->world_rank();
  }
  return 0;
}
#else
[[nodiscard]] auto
world_rank(const carbon::System& /*sys*/) noexcept -> int {
  return 0;
}
#endif

// Signed distance to an infinite circular cylinder whose axis is the z (span)
// direction. Span-invariant: distance depends only on the (x, y) offset from
// the axis, so pz is unused.
[[nodiscard]] CARBON_HD auto
cylinder_sdf(float px,
             float py,
             float /*pz*/,
             float cx,
             float cy,
             float radius) noexcept -> float {
  const float dx = px - cx;
  const float dy = py - cy;
  return std::sqrt(dx * dx + dy * dy) - radius;
}

// Is this cell inside the solid cylinder obstacle (integer grid coords)?
[[nodiscard]] auto
inside_cylinder(const carbon::int64_3d& gc,
                float cx,
                float cy,
                float radius) noexcept -> bool {
  const float dx = static_cast<float>(gc.x) - cx;
  const float dy = static_cast<float>(gc.y) - cy;
  return (dx * dx + dy * dy) <= radius * radius;
}

// Second-order Hermite equilibrium for direction q at density rho = 1 and
// velocity u: f_q^eq = w_q * (1 + 3 c.u + 4.5 (c.u)^2 - 1.5 |u|^2).
template <typename Lattice>
[[nodiscard]] CARBON_HD auto
feq3(int q,
     float ux,
     float uy,
     float uz) noexcept -> float {
  const float w = Lattice::w(q);
  const auto cq = Lattice::c(q);
  const float cu =
      static_cast<float>(cq.x) * ux + static_cast<float>(cq.y) * uy + static_cast<float>(cq.z) * uz;
  const float usqr = ux * ux + uy * uy + uz * uz;
  return w * (1.0F + 3.0F * cu + 4.5F * cu * cu - 1.5F * usqr);
}

// Convenience: streamwise-only equilibrium (u = (ux, 0, 0)).
template <typename Lattice>
[[nodiscard]] CARBON_HD auto
feq_x(int q,
      float ux) noexcept -> float {
  return feq3<Lattice>(q, ux, 0.0F, 0.0F);
}

// Solid obstacle: the cylinder only. The spanwise (z) faces wrap periodically
// and the transverse (y) faces are far-field velocity boundaries re-imposed in
// the inlet/outlet kernel — neither is a bounce-back wall. This is an
// unconfined-style cross-flow past a cylinder rather than a bounded duct.
[[nodiscard]] auto
wt_is_wall(const carbon::int64_3d& gc,
           float cx,
           float cy,
           float radius) -> bool {
  return inside_cylinder(gc, cx, cy, radius);
}

[[nodiscard]] auto
wt_cell_value(const carbon::int64_3d& gc,
              float cx,
              float cy,
              float radius) -> int32_t {
  return wt_is_wall(gc, cx, cy, radius) ? encode_cell(Classification::BounceBack)
                                        : encode_cell(Classification::Bulk);
}

// Initial populations: the cylinder is seeded to 0; every fluid cell starts at
// the free-stream equilibrium. A thick spanwise (z) box turns this from a
// quasi-2D into a genuinely 3D cylinder wake — but only if the spanwise symmetry
// is actually broken: a purely (x, y) initial + boundary state stays z-invariant
// (i.e. 2D) forever, since nothing couples the span. So the free-stream is seeded
// with a small spanwise-varying transverse/vertical velocity ripple (two z
// harmonics, ≈ pert_amp·u_max) that the cylinder shear layers amplify into the
// real 3D mode-A/B instabilities of the wake. nz <= 1 (the legacy thin slab)
// disables the ripple.
template <typename Lattice>
[[nodiscard]] auto
wt_population_value(const carbon::int64_3d& gc,
                    int q,
                    float u_max,
                    float cx,
                    float cy,
                    float radius,
                    int64_t nz,
                    float pert_amp) -> float {
  if (wt_is_wall(gc, cx, cy, radius)) {
    return 0.0F;
  }
  float uy = 0.0F;
  float uz = 0.0F;
  if (nz > 1 && pert_amp > 0.0F) {
    constexpr float two_pi = 6.2831853F;
    const float zf = static_cast<float>(gc.z) / static_cast<float>(nz);
    uy = pert_amp * u_max * std::sin(two_pi * zf);
    uz = pert_amp * u_max * std::sin(2.0F * two_pi * zf + 1.0F);
  }
  return feq3<Lattice>(q, u_max, uy, uz);
}

// Inlet / far-field / outlet kernel re-imposed on a population field after each
// collide-stream step. Flow is along +x; the span (z) is periodic and handled
// by the halo, so only the four x/y faces get a BC here:
//
//   x == 0, y == 0, y == Ny-1 : free-stream velocity Dirichlet — clamp to the
//                               uniform equilibrium feq(1, (u_max, 0, 0)). The
//                               transverse faces are far-field rather than
//                               no-slip so the confined duct does not distort
//                               the wake; the cylinder is the only solid.
//   x == Nx - 1 (outlet)      : zeroth-order velocity extrapolation from the
//                               interior neighbour, then equilibrium re-imposed
//                               at rho = 1. Anchoring the outlet density keeps
//                               the open domain from drifting while still
//                               letting shed vortices convect out. (A DoNothing
//                               copy or convective outflow reads cleaner in
//                               theory but, paired with a non-anchoring inlet,
//                               lets the global density run away here.)
//
// x/y-neighbours are always local (only z is partitioned across GPUs) so no
// halo is needed. Built with dense.make_kernel; the benchmark's hot
// collide-stream / wall-mask / macroscopic kernels are reused unmodified.
template <typename Lattice>
[[nodiscard]] auto
build_inout_bc_kernel(const carbon::Dense& dense,
                      carbon::Dense::Field<float,
                                           Lattice::Q>& field,
                      carbon::int64_3d global,
                      float u_max) -> carbon::Kernel {
  return dense.make_kernel([&field, global, u_max](carbon::KernelLoader& loader) {
    auto io = loader.read_write(field);
    return [io, global, u_max] CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> void {
      const carbon::int64_3d gc = span.global_location(idx);
      if (gc.x == global.x - 1) {
        const carbon::int8_3d back{-1, 0, 0};
        float rho = 0.0F;
        float mx = 0.0F;
        float my = 0.0F;
        float mz = 0.0F;
#pragma unroll
        for (int q = 0; q < Lattice::Q; ++q) {
          const auto ngh = io.neighbour(span, idx, back, q);
          const float fq = ngh.valid ? ngh.value : feq_x<Lattice>(q, u_max);
          const auto cq = Lattice::c(q);
          rho += fq;
          mx += static_cast<float>(cq.x) * fq;
          my += static_cast<float>(cq.y) * fq;
          mz += static_cast<float>(cq.z) * fq;
        }
        const float inv_rho = (rho > 1.0e-6F) ? (1.0F / rho) : 0.0F;
        const float ux = mx * inv_rho;
        const float uy = my * inv_rho;
        const float uz = mz * inv_rho;
#pragma unroll
        for (int q = 0; q < Lattice::Q; ++q) {
          io.write(span, idx, q, feq3<Lattice>(q, ux, uy, uz));
        }
      } else if (gc.x == 0 || gc.y == 0 || gc.y == global.y - 1) {
#pragma unroll
        for (int q = 0; q < Lattice::Q; ++q) {
          io.write(span, idx, q, feq_x<Lattice>(q, u_max));
        }
      }
    };
  });
}

[[nodiscard]] auto
build_sdf_init_kernel(const carbon::Dense& dense,
                      carbon::Dense::Field<float,
                                           1>& sdf,
                      float cx,
                      float cy,
                      float radius) -> carbon::Kernel {
  return dense.make_kernel([&sdf, cx, cy, radius](carbon::KernelLoader& loader) {
    auto out = loader.write(sdf);
    return [out, cx, cy, radius]
           CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> void {
             const carbon::int64_3d gc = span.global_location(idx);
             const float sp = span.spacing();
             const carbon::float32_3d origin = span.origin();
             const float px = origin.x + static_cast<float>(gc.x) * sp;
             const float py = origin.y + static_cast<float>(gc.y) * sp;
             out.write(span, idx, 0, cylinder_sdf(px, py, 0.0F, cx, cy, radius));
           };
  });
}

[[nodiscard]] auto
composite_layers(const carbon::RayMarchImage& bottom,
                 const carbon::RayMarchImage& top,
                 carbon::float32_3d background) -> std::tuple<std::vector<float>,
                                                              std::vector<float>,
                                                              std::vector<float>> {
  const size_t n = bottom.pixel_count();
  std::vector<float> out_r(n);
  std::vector<float> out_g(n);
  std::vector<float> out_b(n);
  for (size_t i = 0; i < n; ++i) {
    const float bot_r = bottom.host_color_r()[i];
    const float bot_g = bottom.host_color_g()[i];
    const float bot_b = bottom.host_color_b()[i];
    const float bot_t = bottom.host_transmittance()[i];
    const float top_r = top.host_color_r()[i];
    const float top_g = top.host_color_g()[i];
    const float top_b = top.host_color_b()[i];
    const float top_t = top.host_transmittance()[i];
    const float mid_r = bot_r + bot_t * background.x;
    const float mid_g = bot_g + bot_t * background.y;
    const float mid_b = bot_b + bot_t * background.z;
    out_r[i] = top_r + top_t * mid_r;
    out_g[i] = top_g + top_t * mid_g;
    out_b[i] = top_b + top_t * mid_b;
  }
  return {std::move(out_r), std::move(out_g), std::move(out_b)};
}

[[nodiscard]] auto
write_rgb_ppm(std::string_view path,
              int width,
              int height,
              const std::vector<float>& color_r,
              const std::vector<float>& color_g,
              const std::vector<float>& color_b) -> bool {
  std::ofstream out(std::string(path), std::ios::binary);
  if (!out) {
    return false;
  }
  out << "P6\n" << width << ' ' << height << "\n255\n";
  const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
  for (size_t i = 0; i < n; ++i) {
    auto encode = [](float v) -> unsigned char {
      if (!std::isfinite(v)) {
        return 0;
      }
      const float c = (v < 0.0F) ? 0.0F : ((v > 1.0F) ? 1.0F : v);
      return static_cast<unsigned char>(std::lround(c * 255.0F));
    };
    out.put(static_cast<char>(encode(color_r[i])));
    out.put(static_cast<char>(encode(color_g[i])));
    out.put(static_cast<char>(encode(color_b[i])));
  }
  return static_cast<bool>(out);
}

// Gather a Dense float field's owned cells (across every local GPU) into one
// contiguous host array in VTI point order — x fastest, then y, then z, with the
// C components interleaved per point. cpu_buffers() must already hold device data
// (download with carbon::Transfer first). Only z is haloed on a carbon Dense grid,
// so x/y span the full extent; the owned z range [z0, z1) is returned via out-params
// (0..Nz for a single process, the rank-local slab range under MPI).
template <int C>
[[nodiscard]] auto
gather_owned(const carbon::Dense& dense,
             const carbon::Dense::Field<float,
                                        C>& field,
             int64_t& z0,
             int64_t& z1) -> std::vector<float> {
  const auto& spans = dense.spans();
  const size_t n_gpus = spans.size();
  z0 = std::numeric_limits<int64_t>::max();
  z1 = std::numeric_limits<int64_t>::min();
  for (size_t g = 0; g < n_gpus; ++g) {
    const int64_t owned_z0 = spans[g].z_offset() + spans[g].halo_z();
    z0 = std::min(z0, owned_z0);
    z1 = std::max(z1, owned_z0 + spans[g].logic_extent().z);
  }
  const int64_t nx = spans[0].haloed_extent().x;
  const int64_t ny = spans[0].haloed_extent().y;
  const int64_t nz = z1 - z0;
  std::vector<float> out(static_cast<size_t>(nx * ny * nz) * static_cast<size_t>(C), 0.0F);
  for (size_t g = 0; g < n_gpus; ++g) {
    const auto& span = spans[g];
    const int64_t halo_z = span.halo_z();
    const int64_t lz = span.logic_extent().z;
    const int64_t pitch = span.xy_slab_elements();
    const int64_t per_comp = span.linear_storage_elements();
    const int64_t owned_z0 = span.z_offset() + halo_z;
    const float* base = field.cpu_buffers()[g].data();
    for (int64_t lzi = 0; lzi < lz; ++lzi) {
      const int64_t zbuf = halo_z + lzi;
      const int64_t zloc = owned_z0 + lzi - z0;
      for (int64_t y = 0; y < ny; ++y) {
        for (int64_t x = 0; x < nx; ++x) {
          const size_t point = static_cast<size_t>(x + nx * (y + ny * zloc));
          for (int c = 0; c < C; ++c) {
            out[point * static_cast<size_t>(C) + static_cast<size_t>(c)] =
                base[static_cast<size_t>(c) * per_comp + zbuf * pitch + y * nx + x];
          }
        }
      }
    }
  }
  return out;
}

// One appended, raw, little-endian binary DataArray in a .vti file.
struct VtiArray {
  std::string name;
  int components;
  const std::vector<float>* data;
};

// Minimal VTK XML ImageData (.vti) writer: a uniform grid with one appended raw
// binary block per DataArray, each length-prefixed by a UInt64 (header_type). This
// is the modern XML format ParaView opens natively; carbon's io surface only emits
// legacy STRUCTURED_POINTS .vtk today, so the .vti request stays self-contained here.
[[nodiscard]] auto
write_vti(std::string_view path,
          int64_t nx,
          int64_t ny,
          int64_t z0,
          int64_t z1,
          const std::vector<VtiArray>& arrays) -> bool {
  std::ofstream out(std::string(path), std::ios::binary);
  if (!out) {
    return false;
  }
  const int64_t x1 = nx - 1;
  const int64_t y1 = ny - 1;
  const int64_t zhi = z1 - 1;
  out << "<?xml version=\"1.0\"?>\n"
      << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" "
         "header_type=\"UInt64\">\n"
      << "  <ImageData WholeExtent=\"0 " << x1 << " 0 " << y1 << ' ' << z0 << ' ' << zhi
      << "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n"
      << "    <Piece Extent=\"0 " << x1 << " 0 " << y1 << ' ' << z0 << ' ' << zhi << "\">\n"
      << "      <PointData>\n";
  uint64_t offset = 0;
  for (const auto& a : arrays) {
    out << "        <DataArray type=\"Float32\" Name=\"" << a.name << "\" NumberOfComponents=\""
        << a.components << "\" format=\"appended\" offset=\"" << offset << "\"/>\n";
    offset += static_cast<uint64_t>(sizeof(uint64_t)) + a.data->size() * sizeof(float);
  }
  out << "      </PointData>\n"
      << "    </Piece>\n"
      << "  </ImageData>\n"
      << "  <AppendedData encoding=\"raw\">\n_";
  for (const auto& a : arrays) {
    const uint64_t nbytes = a.data->size() * sizeof(float);
    out.write(reinterpret_cast<const char*>(&nbytes), sizeof(nbytes));
    out.write(reinterpret_cast<const char*>(a.data->data()), static_cast<std::streamsize>(nbytes));
  }
  out << "\n  </AppendedData>\n</VTKFile>\n";
  return static_cast<bool>(out);
}

// Devices for the (non-distributed) solve. This example runs the LBM + render on a
// SINGLE GPU by default purely for simplicity: a single 80 GB GPU holds a large 3D
// domain, so a multi-GPU z-split adds cross-device halo traffic without speeding up a
// render job. Multi-GPU is correct — the earlier "3D wake diverges on 2 GPUs" was a
// cross-skeleton ordering race (sk_ab / sk_ba on independent streams with no barrier
// between iterations), since fixed by anchoring every Skeleton on
// System::default_stream() with a start/end fork-join; the direction-filtered Lattice-semantic
// multi-GPU PeriodicZ halo itself is exercised by test_halo_periodic_z_lattice and
// passes. Set CARBON_WT_MULTI_GPU=1 or pass --multi-gpu to use every visible GPU
// (e.g. to fit a bigger domain across device memory). The distributed (MPI,
// one-rank-per-GPU) path uses DistTransfer and is unaffected.
[[nodiscard]] auto
pick_render_devices(bool want_multi) -> std::vector<carbon::detail::Gpu> {
  auto devices = carbon::get_all_devices();
  if (!want_multi && devices.size() > 1U) {
    return std::vector<carbon::detail::Gpu>{devices.front()};
  }
  return devices;
}

[[noreturn]] auto
print_usage(const char* prog) -> void {
  std::fprintf(stderr,
               "Usage: %s [OPTIONS] [velocity.ppm [q.ppm [fields.vti]]]\n"
               "\n"
               "Options:\n"
               "  --multi-gpu, --all-gpus   Partition the grid across every visible GPU\n"
               "                            (default: first GPU only). Same as\n"
               "                            CARBON_WT_MULTI_GPU=1.\n"
               "  --interactive             Open the desktop viewer (needs a display).\n"
#ifdef CARBON_HAVE_VIEWER_WEB
               "  --web                     Serve the browser viewer on --port.\n"
               "  --port PORT               Web viewer port (default 5000).\n"
#endif
               "  -h, --help                Print this help and exit.\n"
               "\n"
               "Environment (grid / physics overrides):\n"
               "  CARBON_WT_NX, CARBON_WT_NY, CARBON_WT_NZ   Domain extent\n"
               "  CARBON_WT_PERT                             Spanwise seed amplitude\n"
               "  CARBON_WT_MULTI_GPU=1                      Same as --multi-gpu\n",
               prog);
  std::exit(EXIT_SUCCESS);
}

}  // namespace

auto
main(int argc,
     char** argv) -> int {
  carbon::init();

  std::string out_path = "render_wind_tunnel.ppm";
  std::string q_out_path = "render_qcriterion.ppm";
  std::string vti_out_path = "render_wind_tunnel.vti";
  const std::string wake_out_path = "render_wake.ppm";
  const std::string q_tint_out_path = "render_qcriterion_tinted.ppm";
  const std::string cp_out_path = "render_cp.ppm";
  bool interactive = false;
  bool want_multi_gpu = false;
#ifdef CARBON_HAVE_VIEWER_WEB
  bool web = false;
  int web_port = 5000;
#endif
  int positional = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
    } else if (arg == "--multi-gpu" || arg == "--all-gpus") {
      want_multi_gpu = true;
    } else if (arg == "--interactive") {
      interactive = true;
#ifdef CARBON_HAVE_VIEWER_WEB
    } else if (arg == "--web") {
      web = true;
    } else if (arg == "--port" && i + 1 < argc) {
      web_port = std::atoi(argv[++i]);
#endif
    } else if (arg.empty() || arg[0] != '-') {
      if (positional == 0) {
        out_path = arg;
      } else if (positional == 1) {
        q_out_path = arg;
      } else if (positional == 2) {
        vti_out_path = arg;
      }
      ++positional;
    }
  }

  if (!want_multi_gpu) {
    const char* multi = std::getenv("CARBON_WT_MULTI_GPU");
    want_multi_gpu = (multi != nullptr && std::atoi(multi) != 0);
  }

#ifdef CARBON_HAVE_MPI_NCCL
  auto distributed = std::make_shared<const carbon::Distributed>();
  const int ranks = distributed->world_size();
  carbon::System sys = (ranks > 1) ? carbon::System(distributed)
                                   : carbon::System(pick_render_devices(want_multi_gpu));
#else
  carbon::System sys(pick_render_devices(want_multi_gpu));
#endif

  if (sys.gpu_group().size() == 0U) {
    CARBON_LOG_ERROR("render_wind_tunnel: no devices visible");
    return EXIT_FAILURE;
  }

  CARBON_LOG_INFO("render_wind_tunnel: {} GPU(s) selected ({} visible)", sys.gpu_group().size(),
                  carbon::get_all_devices().size());

  // Positional args and flags parsed above; remainder of main follows.
  // Cross-flow past a circular cylinder (axis = z / span). The x–y plane resolves
  // the von Kármán street; the span (z) wraps periodically (BoundaryMode::PeriodicZ)
  // — physically exact for a spanwise-homogeneous cylinder, and the canonical setup
  // for DNS of the 3D cylinder wake. A *thick* span (default 128 = 2 D) lets the
  // wake develop genuine three-dimensional structure (the mode-A/B spanwise
  // instabilities that set in past Re ≈ 190), rather than the earlier thin
  // quasi-2D slab. Diameter D = 2R = 64 over a 512-cell transverse box gives
  // blockage D/H = 12.5 %; the cylinder sits Nx/4 from the inlet, leaving ~12 D of
  // wake downstream. Extent is overridable at run time (CARBON_WT_NX / NY / NZ) so
  // the resolution can be traded against Re without a rebuild; a bigger D is how a
  // higher Re is reached while keeping omega safely below 2.
  int64_t k_nx = 768;
  int64_t k_ny = 384;
  int64_t k_nz = 128;
  if (const char* e = std::getenv("CARBON_WT_NX"); e != nullptr) {
    k_nx = std::atoll(e);
  }
  if (const char* e = std::getenv("CARBON_WT_NY"); e != nullptr) {
    k_ny = std::atoll(e);
  }
  if (const char* e = std::getenv("CARBON_WT_NZ"); e != nullptr) {
    k_nz = std::atoll(e);
  }
  const carbon::int64_3d extent{k_nx, k_ny, k_nz};
  // Amplitude of the spanwise velocity ripple that seeds the 3D instability
  // (fraction of u_max); CARBON_WT_PERT overrides. 0 reproduces the 2D seed.
  float k_pert_amp = 0.05F;
  if (const char* e = std::getenv("CARBON_WT_PERT"); e != nullptr) {
    k_pert_amp = static_cast<float>(std::atof(e));
  }
  const float radius = 32.0F;
  const float cx = static_cast<float>(extent.x / 4);
  // Nudge the centre a fraction of a cell off the symmetry axis to seed the
  // shedding instability; a perfectly symmetric setup can sit on the unstable
  // fixed point far longer than physical before numerical noise tips it.
  constexpr float k_shed_perturb = 2.5F;
  const float cy = 0.5F * static_cast<float>(extent.y - 1) + k_shed_perturb;
  constexpr float k_sdf_edge_width = 2.0F;
  const carbon::float32_3d background{1.0F, 1.0F, 1.0F};

  // LBM parameters, defined in the XLB idiom (see
  // XLB/examples/cfd/multires_windtunnel_3d.py): pick a lattice inlet speed
  // ulb and a diameter Reynolds number, then DERIVE nu and omega
  // (omega_finest = 1/(3 nu + 0.5)); size the run by "flow passes" — how many
  // times the flow crosses the domain length — rather than a raw iteration
  // count (num_steps = passes * Nx / ulb).
  //
  // A circular cylinder sheds a periodic von Kármán wake for Re ≳ 47, goes
  // three-dimensional past Re ≈ 190 (the mode-A/B spanwise instabilities), and
  // becomes an increasingly turbulent transitional wake beyond. With the
  // less-dissipative BGK collision (KBC's entropic limiter was suppressing the
  // shedding mode, leaving a near-steady wake) the grid stays finite well past
  // omega ≈ 1.8. Re = 450 with D = 64 (nu = 0.0142, omega ≈ 1.843) is chosen: on
  // the thick-span domain it develops a genuinely 3D wake — out-of-plane velocity
  // reaches ≈ 65 % of ulb and the transverse velocity varies along the span —
  // while the shedding stays coherent and omega stays below the BGK stability
  // limit of 2 for a well-resolved solve. A larger D is what buys the higher Re at
  // fixed omega: nu = ulb·D/Re, so doubling D halves nu's shrinkage. ulb = 0.10
  // gives Mach ≈ 0.17. (Override at run time via CARBON_WT_RE / CARBON_WT_ULB /
  // CARBON_WT_ITERS, and the domain via CARBON_WT_NX / NY / NZ.)
  double k_ulb = 0.10;
  double k_re = 450.0;
  if (const char* re_env = std::getenv("CARBON_WT_RE"); re_env != nullptr) {
    k_re = std::atof(re_env);
  }
  if (const char* ulb_env = std::getenv("CARBON_WT_ULB"); ulb_env != nullptr) {
    k_ulb = std::atof(ulb_env);
  }
  const double k_diameter = 2.0 * static_cast<double>(radius);
  const double nu_lb = k_ulb * k_diameter / k_re;
  const double k_omega = 1.0 / (3.0 * nu_lb + 0.5);
  constexpr double k_flow_passes = 4.0;
  int64_t k_iters = static_cast<int64_t>(k_flow_passes * static_cast<double>(extent.x) / k_ulb);
  if (const char* iters_env = std::getenv("CARBON_WT_ITERS"); iters_env != nullptr) {
    k_iters = std::atoll(iters_env);
  }
  const LbmParameters params{static_cast<float>(k_ulb), static_cast<float>(k_omega)};

  // Radius-1, 26-point D3Q27 halo stencil (drops the center population).
  std::vector<carbon::int32_3d> stencil_points;
  stencil_points.reserve(D3Q27::Q - 1);
  for (int q = 0; q < D3Q27::Q; ++q) {
    if (q == D3Q27::kCenter) {
      continue;
    }
    const auto cq = D3Q27::c(q);
    stencil_points.push_back({cq.x, cq.y, cq.z});
  }
  const carbon::Stencil stencil(stencil_points, /*remove_center=*/true);

  carbon::Dense dense(sys, extent, stencil, carbon::BoundaryMode::PeriodicZ);
  auto pop_a = dense.newField<float, D3Q27::Q>(carbon::MemorySide::HostDevice, "pop_a");
  auto pop_b = dense.newField<float, D3Q27::Q>(carbon::MemorySide::HostDevice, "pop_b");
  auto cells = dense.newField<int32_t, 1>(carbon::MemorySide::HostDevice, "cells");
  auto cells_scratch = dense.newField<int32_t, 1>(carbon::MemorySide::HostDevice, "cells_scratch");
  auto velocity = dense.newField<float, 3>(carbon::MemorySide::HostDevice).set_name("u");
  auto density = dense.newField<float, 1>(carbon::MemorySide::HostDevice).set_name("rho");
  auto qfield = dense.newField<float, 1>(carbon::MemorySide::HostDevice).set_name("q");
  // Positive part of Q, in its own field. It used to be written over rho, which was free
  // when nothing else read rho — the surface-pressure pass does, so the two now coexist.
  auto qpos = dense.newField<float, 1>(carbon::MemorySide::Device).set_name("qpos");
  auto sdf = dense.newField<float, 1>(carbon::MemorySide::Device).set_name("sdf");

  const carbon::Stream stream = sys.stream();

  // Host-side init: cell classification + uniform free-stream populations.
  auto pop_filler = [cx, cy, radius, nz = extent.z, pert = k_pert_amp,
                     u_max = static_cast<float>(k_ulb)](carbon::int64_3d gc, int q) -> float {
    return wt_population_value<D3Q27>(gc, q, u_max, cx, cy, radius, nz, pert);
  };
  pop_a.fill_owned_host(pop_filler);
  pop_b.fill_owned_host(pop_filler);
  auto cell_filler = [cx, cy, radius](carbon::int64_3d gc, int /*c*/) -> int32_t {
    return wt_cell_value(gc, cx, cy, radius);
  };
  cells.fill_owned_host(cell_filler);
  cells_scratch.fill_owned_host(cell_filler);

  const auto sdf_init = build_sdf_init_kernel(dense, sdf, cx, cy, radius);
  sdf_init.launch(stream);

  // Upload the CPU-seeded populations + cell fields to the GPU(s).
  carbon::Transfer<float>(sys, pop_a.cpu_buffers(), pop_a.gpu_buffers())(stream);
  carbon::Transfer<float>(sys, pop_b.cpu_buffers(), pop_b.gpu_buffers())(stream);
  carbon::Transfer<int32_t>(sys, cells.cpu_buffers(), cells.gpu_buffers())(stream);
  carbon::Transfer<int32_t>(sys, cells_scratch.cpu_buffers(), cells_scratch.gpu_buffers())(stream);
  stream.wait();

  // Precompute wall_neighbour_bits once (reads cells, writes cells_scratch via
  // the auto-halo + stencil pattern), then publish scratch back into cells.
  {
    carbon::Skeleton precompute(sys);
    precompute.add(
        make_compute_wall_mask_kernel<carbon::Dense, D3Q27>(dense, cells, cells_scratch));
    precompute.compile();
    precompute.run();
    precompute.wait();
    carbon::Transfer<int32_t>(sys, cells_scratch.gpu_buffers(), cells.gpu_buffers())(stream);
    stream.wait();
  }

  // Two ping-pong iteration skeletons (D3Q27 + KBC), run alternately. Each step
  // is collide-stream followed by the inlet/far-field/outlet BC on the freshly
  // written field, so the boundary condition is re-imposed every iteration.
  const float ulb_f = static_cast<float>(k_ulb);
  carbon::Skeleton sk_ab(sys);
  sk_ab.add(make_iteration_kernel<carbon::Dense, D3Q27, BgkCollision>(dense, pop_a, pop_b, cells,
                                                                      params, "wt_step_ab"));
  sk_ab.add(build_inout_bc_kernel<D3Q27>(dense, pop_b, extent, ulb_f));
  carbon::Skeleton sk_ba(sys);
  sk_ba.add(make_iteration_kernel<carbon::Dense, D3Q27, BgkCollision>(dense, pop_b, pop_a, cells,
                                                                      params, "wt_step_ba"));
  sk_ba.add(build_inout_bc_kernel<D3Q27>(dense, pop_a, extent, ulb_f));
  sk_ab.compile();
  sk_ba.compile();

  CARBON_LOG_INFO(
      "render_wind_tunnel: LBM D3Q27+BGK N={}x{}x{} Re={:.1f} u_max={:.4g} "
      "nu={:.5g} omega={:.4g} passes={:.1f} iters={}",
      extent.x, extent.y, extent.z, k_re, k_ulb, nu_lb, k_omega, k_flow_passes, k_iters);
  // No per-iteration drain: sk_ab and sk_ba both anchor on sys.default_stream(), and
  // each run() is bracketed by a start/end fork-join on that shared anchor, so the
  // ping-pong is ordered on the device. (Before the anchor landed, the two skeletons
  // sat on independent streams with nothing between them, and iteration N+1's
  // cross-GPU halo peer-copy could read a population iteration N was still writing —
  // which blew up a thick-span 3D wake on > 1 GPU.)
  for (int64_t it = 0; it < k_iters; ++it) {
    carbon::Skeleton& sk = (it % 2 == 0) ? sk_ab : sk_ba;
    sk.run();
  }
  sk_ab.wait();
  sk_ba.wait();

  // After k_iters (even), the freshest populations live in pop_a. Extract the
  // macroscopic velocity magnitude field for rendering.
  const auto macro =
      make_macroscopic_kernel<carbon::Dense, D3Q27>(dense, pop_a, cells, density, velocity, params);
  macro.launch(stream);
  stream.wait();

  {
    carbon::ReductionResult<float> u_max(sys);
    const auto u_max_k = dense.make_reduce_kernel<float>(
        [&velocity](carbon::KernelLoader& loader) {
          const auto field = loader.read(velocity);
          return [field] CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
            const float vx = field.read(span, idx, 0);
            const float vy = field.read(span, idx, 1);
            const float vz = field.read(span, idx, 2);
            const float s = std::sqrt(vx * vx + vy * vy + vz * vz);
            // NaN/inf would be silently dropped by Max (all comparisons false),
            // letting the finite inlet-clamp plane mask a diverged interior. Map
            // non-finite cells to a huge sentinel so divergence is visible.
            return std::isfinite(s) ? s : 1.0e30F;
          };
        },
        u_max, carbon::reduce::Max<float>{});
    u_max_k.launch(stream);
    u_max.sync(stream);
    CARBON_LOG_INFO("render_wind_tunnel: |u| max={:.6g} (u_max={:.4g})", u_max.host_value(), k_ulb);
  }

  const carbon::Halo<float, 3> vel_halo(velocity);
  const carbon::Halo<float, 1> sdf_halo(sdf);
  vel_halo(stream);
  sdf_halo(stream);
  stream.wait();

  const float inv_ulb = 1.0F / static_cast<float>(k_ulb);
  carbon::make_qcriterion_kernel(dense, velocity, qfield, inv_ulb).launch(stream);
  stream.wait();
  const carbon::Halo<float, 1> q_halo(qfield);
  q_halo(stream);
  stream.wait();

  const auto q_to_qpos = dense.make_kernel([&qfield, &qpos](carbon::KernelLoader& loader) {
    const auto q = loader.read(qfield);
    auto out = loader.write(qpos);
    return [q, out] CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> void {
      const float v = q.read(span, idx, 0);
      out.write(span, idx, 0, (v > 0.0F) ? v : 0.0F);
    };
  });
  q_to_qpos.launch(stream);
  stream.wait();
  const carbon::Halo<float, 1> rho_halo(density);
  rho_halo(stream);
  stream.wait();

  // Reference static density for the surface-pressure coefficient. Anchoring it to the
  // outlet's rho = 1 would be wrong at the body: the channel carries a viscous pressure drop
  // over the 3/4 of its length that lies downstream, so the ambient density at the cylinder
  // station sits measurably above 1 and every Cp computed against the outlet inherits that
  // offset. Take it instead from a slab a few diameters upstream, which is the numerical
  // equivalent of a wind tunnel's upstream static tap.
  float rho_inf = 1.0F;
  {
    carbon::ReductionResult<float> rho_sum(sys);
    // Far enough upstream that the body's own pressure field has decayed, but clear of the
    // x = 0 inlet plane, whose Dirichlet velocity condition perturbs the density there.
    const float ref_x0 = cx - 5.0F * radius;
    const float ref_x1 = cx - 3.0F * radius;
    const auto rho_ref_k = dense.make_reduce_kernel<float>(
        [&density, ref_x0, ref_x1](carbon::KernelLoader& loader) {
          const auto field = loader.read(density);
          return [field, ref_x0, ref_x1]
                 CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
                   const carbon::int64_3d gc = span.global_location(idx);
                   const float x = static_cast<float>(gc.x);
                   return (x < ref_x0 || x > ref_x1) ? 0.0F : field.read(span, idx, 0);
                 };
        },
        rho_sum, carbon::reduce::Sum<float>{});
    rho_ref_k.launch(stream);
    rho_sum.sync(stream);
    const auto x_lo = static_cast<int64_t>(std::ceil(ref_x0));
    const auto x_hi = static_cast<int64_t>(std::floor(ref_x1));
    const double cells = static_cast<double>(x_hi - x_lo + 1) * static_cast<double>(extent.y) *
                         static_cast<double>(extent.z);
    rho_inf = static_cast<float>(static_cast<double>(rho_sum.host_value()) / cells);
    CARBON_LOG_INFO("render_wind_tunnel: reference rho_inf={:.8g} (upstream slab x in [{}, {}])",
                    rho_inf, x_lo, x_hi);
  }

  float q_wake_max = 0.0F;
  {
    carbon::ReductionResult<float> q_pos_max(sys);
    const float wake_x0 = cx - 1.5F * radius;
    const float wake_x1 = cx + 20.0F * radius;
    const float wake_y0 = cy - 6.0F * radius;
    const float wake_y1 = cy + 6.0F * radius;
    const float wake_z0 = -1.0F;
    const float wake_z1 = static_cast<float>(extent.z);
    const auto q_wake_k = dense.make_reduce_kernel<float>(
        [&qpos, wake_x0, wake_x1, wake_y0, wake_y1, wake_z0,
         wake_z1](carbon::KernelLoader& loader) {
          const auto field = loader.read(qpos);
          return [field, wake_x0, wake_x1, wake_y0, wake_y1, wake_z0, wake_z1]
                 CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
                   const carbon::int64_3d gc = span.global_location(idx);
                   const float x = static_cast<float>(gc.x);
                   const float y = static_cast<float>(gc.y);
                   const float z = static_cast<float>(gc.z);
                   if (x < wake_x0 || x > wake_x1 || y < wake_y0 || y > wake_y1 || z < wake_z0 ||
                       z > wake_z1) {
                     return 0.0F;
                   }
                   return field.read(span, idx, 0);
                 };
        },
        q_pos_max, carbon::reduce::Max<float>{});
    q_wake_k.launch(stream);
    q_pos_max.sync(stream);
    q_wake_max = q_pos_max.host_value();
    CARBON_LOG_INFO("render_wind_tunnel: Q+ wake max={:.6g}", q_wake_max);
  }

  constexpr int k_frame_w = 1920;
  constexpr int k_frame_h = 1080;
  constexpr int k_samples_per_axis = 2;
  constexpr float k_splat_sigma_px = 1.75F;
  // Top-down view (look along -z, the span): flow runs +x left-to-right, +y is
  // up in frame, and the thin periodic span is integrated through — the classic
  // 2D presentation of a von Kármán street. Aspect ~ 23R × 12R ≈ 16:9.
  const float zmax = static_cast<float>(extent.z);
  const carbon::float32_3d bounds_min{cx - 3.0F * radius, cy - 6.0F * radius, 0.0F};
  const carbon::float32_3d bounds_max{cx + 20.0F * radius, cy + 6.0F * radius, zmax};

  // Clip to the wake box so the inlet/outlet/far-field planes never contribute
  // to the volume integral; the full span is kept.
  const carbon::float32_3d clip_min{cx - 2.5F * radius, cy - 6.0F * radius, -1.0F};
  const carbon::float32_3d clip_max{cx + 22.0F * radius, cy + 6.0F * radius, zmax + 1.0F};

  carbon::RayMarchImage vel_image(sys, k_frame_w, k_frame_h);
  carbon::RayMarchImage q_image(sys, k_frame_w, k_frame_h);
  carbon::RayMarchImage sphere_image(sys, k_frame_w, k_frame_h);
  carbon::DensityImage vel_density(sys, k_frame_w, k_frame_h);
  carbon::DensityImage q_density(sys, k_frame_w, k_frame_h);

  carbon::RayMarchOptions vel_opts{};
  vel_opts.step_size = 0.0F;
  vel_opts.max_steps = 512;
  vel_opts.shading = carbon::RayMarchShading::ScalarField;
  vel_opts.colormap = carbon::Colormap::Jet;
  // Full-field speed map: 0 (wake/recirc) to 1.5·ulb (accelerated shoulders).
  // The thin span integrates to a near-opaque 2D colour field.
  vel_opts.value_threshold = 0.0F;
  vel_opts.value_min = 0.0F;
  vel_opts.value_max = 1.5F;
  vel_opts.opacity_scale = 2.5F;
  vel_opts.use_clip_bounds = true;
  vel_opts.clip_bounds_min = clip_min;
  vel_opts.clip_bounds_max = clip_max;

  carbon::RayMarchOptions q_opts{};
  q_opts.max_steps = 1024;
  // Rendered on a white background, so contrast — not raw brightness — governs
  // legibility: the dark low-t (purple/navy) end of Viridis reads far more
  // clearly against white than its bright teal→yellow end, which blends into the
  // page. Two knobs, decoupled in the ray-march kernel: value_max sets only the
  // colour (t = s/value_max), while alpha grows with the raw Q times
  // opacity_scale. So anchor value_max to ~0.5·(wake peak) — keeping the whole
  // street in the dark, high-contrast band with the strong near-cylinder shear
  // layers saturating — and lift opacity_scale high enough that the weak
  // far-downstream cores still accumulate solid alpha through the integrated span.
  const float q_gain = 1.0F;
  const float q_ceil = (q_wake_max > 0.0F) ? (q_wake_max * q_gain * 0.5F) : 1.0F;
  q_opts.shading = carbon::RayMarchShading::ScalarField;
  q_opts.colormap = carbon::Colormap::Viridis;
  q_opts.value_threshold = 0.0F;
  q_opts.value_min = 0.0F;
  q_opts.value_max = q_ceil;
  q_opts.opacity_scale = 200.0F;
  q_opts.step_size = 0.15F;
  q_opts.use_clip_bounds = true;
  // Keep the transverse clip a hair inside the far-field y-walls (±5 R, not the
  // full ±6 R half-height): the uniform-velocity wall clamp leaves a thin shear
  // sheet of nonzero Q right at y = 0 / y = Ny-1 that would otherwise integrate
  // into a distracting frame-wide line across the render.
  q_opts.clip_bounds_min = {cx - 0.5F * radius, cy - 5.0F * radius, -1.0F};
  q_opts.clip_bounds_max = {cx + 22.0F * radius, cy + 5.0F * radius, zmax + 1.0F};

  // Same Q cloud, but hue reports the speed inside each core rather than the core strength.
  // Q alone says where the vortices are; this says which ones sit in fast-moving fluid,
  // which is what separates the energetic near-body braids from the slow, decayed cores far
  // downstream — they can share a Q value and be physically very different.
  carbon::RayMarchOptions q_tint_opts = q_opts;
  q_tint_opts.shading = carbon::RayMarchShading::ScalarFieldAttribute;
  q_tint_opts.colormap = carbon::Colormap::Jet;
  q_tint_opts.value_min = 0.0F;
  q_tint_opts.value_max = 1.5F;

  // Wake mode: opacity from the disturbance magnitude |u - u_inf|, hue from the *signed*
  // streamwise deviation (u_x - u_inf) through the diverging map. Undisturbed fluid sits at
  // zero on both, so the free stream is transparent and the picture contains only what the
  // cylinder did: blue where the flow was slowed (the recirculation and the vortex cores),
  // red over the shoulders where it was accelerated around the body. Rendering |u| instead
  // spends most of the colour range on the free stream and leaves the two signs
  // indistinguishable.
  const carbon::float32_3d free_stream{static_cast<float>(k_ulb), 0.0F, 0.0F};
  carbon::RayMarchOptions wake_opts{};
  wake_opts.step_size = 0.0F;
  wake_opts.max_steps = 512;
  wake_opts.shading = carbon::RayMarchShading::ScalarFieldAttribute;
  wake_opts.colormap = carbon::Colormap::CoolWarm;
  // 5 % of the free stream: below that the fluid is undisturbed to the eye, and dropping it
  // keeps the long upstream approach from hazing over the wake behind it.
  wake_opts.value_threshold = 0.05F;
  wake_opts.value_min = -0.8F;
  wake_opts.value_max = 0.8F;
  wake_opts.opacity_scale = 3.0F;
  wake_opts.use_clip_bounds = true;
  wake_opts.clip_bounds_min = clip_min;
  wake_opts.clip_bounds_max = clip_max;

  carbon::RayMarchOptions sphere_opts{};
  sphere_opts.step_size = 0.0F;
  sphere_opts.absorption_scale = 0.55F;
  sphere_opts.max_steps = 512;
  sphere_opts.shading = carbon::RayMarchShading::SurfaceBlinn;
  sphere_opts.light_dir = {0.35F, -0.55F, 0.75F};
  sphere_opts.ambient = 0.22F;
  sphere_opts.diffuse = 0.62F;
  sphere_opts.specular = 0.18F;
  sphere_opts.shininess = 48.0F;
  sphere_opts.isosurface_width = k_sdf_edge_width;
  sphere_opts.material_color = {0.78F, 0.80F, 0.84F};
  sphere_opts.use_clip_bounds = true;
  sphere_opts.clip_bounds_min = clip_min;
  sphere_opts.clip_bounds_max = clip_max;

  // Surface pressure coefficient on the cylinder skin. In LBM the equation of state is
  // p = rho * cs^2 with cs^2 = 1/3, so Cp = (p - p_inf) / (0.5 rho_inf U^2) collapses to
  // (rho - rho_inf) / (1.5 rho_inf U^2); the outlet anchors rho_inf = 1. Stagnation gives
  // Cp = +1 at the nose and the suction peak on the shoulders runs to about -2 or -3, which
  // sets the symmetric-ish range used for the colormap.
  const float k_cp_scale = 1.0F / (1.5F * rho_inf * static_cast<float>(k_ulb * k_ulb));
  // Radius of the probe ring, in cells beyond the wall. The wall is half-way bounce-back, so
  // the cells the surfel band covers straddle it and those inside the body never hold a fluid
  // state at all; the probe has to clear that layer. Probing at a fixed *radius* rather than
  // by a fixed step from each surfel matters twice over: a fixed step rounded to the lattice
  // under-reaches on the diagonals (a 2-cell step along (0.71, 0.71) rounds to (1, 1), only
  // 1.41 cells of reach, which lands back inside the wall), and every reading comes off the
  // same station, which is what makes the values comparable around the body.
  constexpr float k_cp_probe_radius = 2.5F;
  // Stagnation pins the top of the scale at +1 exactly; the bottom is the suction side,
  // deeper than -1 and Reynolds-dependent, so the map is deliberately not symmetric. -2.5
  // covers the measured suction peak (about -2 at this Re, logged per run as Cp(theta))
  // without spending a third of the colour range on values the flow never reaches.
  constexpr float k_cp_min = -2.5F;
  constexpr float k_cp_max = 1.0F;
  const auto render_cp_for_camera = [&](const carbon::Camera& camera, carbon::SurfelImage& cp_out) {
    cp_out.clear(stream);
    const auto cp_splat = carbon::make_surfel_kernel(
        dense, cp_out, camera,
        [&density, cx, cy, radius, k_cp_scale, rho_inf](carbon::KernelLoader& loader) {
          const auto rho = loader.read(density);
          return [rho, cx, cy, radius, k_cp_scale,
                  rho_inf] CARBON_HD(carbon::Dense::Span span,
                                     carbon::Dense::Idx idx) -> carbon::SurfelSample {
            const carbon::int64_3d gc = span.global_location(idx);
            const float px = static_cast<float>(gc.x) + 0.5F - cx;
            const float py = static_cast<float>(gc.y) + 0.5F - cy;
            const float r = std::sqrt(px * px + py * py);
            carbon::SurfelSample sample{};
            sample.sdf = r - radius;
            // Outward radial normal. The cylinder axis is z, so the normal has no z
            // component and the probe below never crosses a slab boundary — the one place
            // this body's geometry buys a simplification the general case would not get.
            sample.normal = (r > 1.0e-6F) ? carbon::float32_3d{px / r, py / r, 0.0F}
                                          : carbon::float32_3d{1.0F, 0.0F, 0.0F};
            const float probe_r = radius + k_cp_probe_radius;
            const carbon::int8_3d probe{
                static_cast<int8_t>(std::lround(cx + sample.normal.x * probe_r - 0.5F) - gc.x),
                static_cast<int8_t>(std::lround(cy + sample.normal.y * probe_r - 0.5F) - gc.y), 0};
            const carbon::NeighbourValue<float> probed = rho.neighbour(span, idx, probe, 0);
            sample.attribute = probed.valid ? ((probed.value - rho_inf) * k_cp_scale) : 0.0F;
            return sample;
          };
        },
        // A generous pixel cap: the Cp camera sits close to the body, so one cell of skin
        // covers several pixels and the default cap leaves the near end of the cylinder
        // speckled with background. The cap only bounds how many pixels a surfel may be
        // tested against — the footprint's actual size still comes from the voxel.
        /*band_width=*/1.5F, /*splat_radius_px=*/8.0F);
    cp_splat.launch(stream);
    cp_out.combine(stream);
    stream.wait();
  };

  const auto render_sphere_for_camera = [&](const carbon::Camera& camera,
                                            carbon::RayMarchImage& sphere_out) {
    sphere_out.clear(stream);
    const auto sphere_march = carbon::make_ray_march_kernel(
        dense, sphere_out, camera, sphere_opts, [&sdf](carbon::KernelLoader& loader) {
          const auto field = loader.read(sdf);
          return [field] CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
            return field.read(span, idx, 0);
          };
        });
    sphere_march.launch(stream);
    sphere_out.combine(stream);
    sphere_out.dist_combine();
    stream.wait();
  };

  const auto render_density_for_camera = [&](const carbon::Camera& camera,
                                             carbon::DensityImage& vel_out,
                                             carbon::DensityImage& q_out, Layers layers) {
    const bool want_speed = (layers == Layers::Speed || layers == Layers::Both);
    const bool want_wake = (layers == Layers::Wake);
    const bool want_q = (layers != Layers::Speed && layers != Layers::Wake);

    if (want_speed || want_wake) {
      vel_out.clear(stream);
    }
    if (want_q) {
      q_out.clear(stream);
    }

    if (want_speed || want_wake) {
      // Additive splatting has no per-pixel colour channel, so the wake variant here can
      // only show the disturbance magnitude; the signed streamwise split is march-only.
      const auto vel_splat = carbon::make_density_kernel(
          dense, vel_out, camera,
          [&velocity, inv_ulb, clip_min, clip_max, want_wake,
           free_stream](carbon::KernelLoader& loader) {
            const auto field = loader.read(velocity);
            return [field, inv_ulb, clip_min, clip_max, want_wake, free_stream]
                   CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
                     const carbon::int64_3d gc = span.global_location(idx);
                     const float x = static_cast<float>(gc.x);
                     const float y = static_cast<float>(gc.y);
                     const float z = static_cast<float>(gc.z);
                     if (x < clip_min.x || x > clip_max.x || y < clip_min.y || y > clip_max.y ||
                         z < clip_min.z || z > clip_max.z) {
                       return 0.0F;
                     }
                     const carbon::float32_3d u{field.read(span, idx, 0), field.read(span, idx, 1),
                                                field.read(span, idx, 2)};
                     if (want_wake) {
                       const float s = carbon::render::speed_deviation(u, free_stream, inv_ulb);
                       return (s > 0.05F) ? s : 0.0F;
                     }
                     return std::sqrt(u.x * u.x + u.y * u.y + u.z * u.z) * inv_ulb;
                   };
          },
          {}, 1.0F, k_samples_per_axis, k_splat_sigma_px);
      vel_splat.launch(stream);
      vel_out.combine(stream);
    }

    if (want_q) {
      const carbon::float32_3d q_clip_min = q_opts.clip_bounds_min;
      const carbon::float32_3d q_clip_max = q_opts.clip_bounds_max;
      const auto q_splat = carbon::make_density_kernel(
          dense, q_out, camera,
          [&qpos, cx, cy, radius, q_gain, q_clip_min, q_clip_max](carbon::KernelLoader& loader) {
            const auto field = loader.read(qpos);
            return [field, cx, cy, radius, q_gain, q_clip_min, q_clip_max]
                   CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
                     const carbon::int64_3d gc = span.global_location(idx);
                     const float x = static_cast<float>(gc.x);
                     const float y = static_cast<float>(gc.y);
                     const float z = static_cast<float>(gc.z);
                     if (x < q_clip_min.x || x > q_clip_max.x || y < q_clip_min.y ||
                         y > q_clip_max.y || z < q_clip_min.z || z > q_clip_max.z) {
                       return 0.0F;
                     }
                     const float dx = x - cx;
                     const float dy = y - cy;
                     if (dx * dx + dy * dy < radius * radius) {
                       return 0.0F;
                     }
                     const float q = field.read(span, idx, 0);
                     if (q <= 0.0F) {
                       return 0.0F;
                     }
                     return q * q_gain;
                   };
          },
          {}, 1.0F, k_samples_per_axis, k_splat_sigma_px);
      q_splat.launch(stream);
      q_out.combine(stream);
    }
    stream.wait();
  };

  const auto render_for_camera = [&](const carbon::Camera& camera, carbon::RayMarchImage& vel_out,
                                     carbon::RayMarchImage& q_out,
                                     carbon::RayMarchImage& sphere_out,
                                     Layers layers = Layers::Both) {
    // Wake shares vel_out with Speed and QTinted shares q_out with Q: each pair draws the
    // same quantity through a different transfer function, so they never coexist in a frame.
    const bool want_speed = (layers == Layers::Speed || layers == Layers::Both);
    const bool want_wake = (layers == Layers::Wake);
    const bool want_q = (layers == Layers::Q || layers == Layers::Both);
    const bool want_q_tint = (layers == Layers::QTinted);

    if (want_speed || want_wake) {
      vel_out.clear(stream);
    }
    if (want_q || want_q_tint) {
      q_out.clear(stream);
    }
    sphere_out.clear(stream);

    if (want_speed) {
      const auto vel_march = carbon::make_ray_march_kernel(
          dense, vel_out, camera, vel_opts, [&velocity, inv_ulb](carbon::KernelLoader& loader) {
            const auto field = loader.read(velocity);
            return [field, inv_ulb]
                   CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
                     const float vx = field.read(span, idx, 0);
                     const float vy = field.read(span, idx, 1);
                     const float vz = field.read(span, idx, 2);
                     return std::sqrt(vx * vx + vy * vy + vz * vz) * inv_ulb;
                   };
          });
      vel_march.launch(stream);
      vel_out.combine(stream);
      vel_out.dist_combine();
    }

    if (want_wake) {
      const auto wake_march = carbon::make_ray_march_kernel(
          dense, vel_out, camera, wake_opts,
          [&velocity, inv_ulb, free_stream](carbon::KernelLoader& loader) {
            const auto field = loader.read(velocity);
            return [field, inv_ulb, free_stream]
                   CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
                     const carbon::float32_3d u{field.read(span, idx, 0), field.read(span, idx, 1),
                                                field.read(span, idx, 2)};
                     return carbon::render::speed_deviation(u, free_stream, inv_ulb);
                   };
          },
          [&velocity, inv_ulb, free_stream](carbon::KernelLoader& loader) {
            const auto field = loader.read(velocity);
            return [field, inv_ulb, free_stream]
                   CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
                     return (field.read(span, idx, 0) - free_stream.x) * inv_ulb;
                   };
          });
      wake_march.launch(stream);
      vel_out.combine(stream);
      vel_out.dist_combine();
    }

    if (want_q_tint) {
      const auto q_tint_march = carbon::make_ray_march_kernel(
          dense, q_out, camera, q_tint_opts,
          [&qpos, cx, cy, radius, q_gain](carbon::KernelLoader& loader) {
            const auto field = loader.read(qpos);
            return [field, cx, cy, radius, q_gain]
                   CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
                     const carbon::int64_3d gc = span.global_location(idx);
                     const float dx = static_cast<float>(gc.x) - cx;
                     const float dy = static_cast<float>(gc.y) - cy;
                     if (dx * dx + dy * dy < radius * radius) {
                       return 0.0F;
                     }
                     const float q = field.read(span, idx, 0);
                     return (q > 0.0F) ? (q * q_gain) : 0.0F;
                   };
          },
          [&velocity, inv_ulb](carbon::KernelLoader& loader) {
            const auto field = loader.read(velocity);
            return [field, inv_ulb]
                   CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
                     const float vx = field.read(span, idx, 0);
                     const float vy = field.read(span, idx, 1);
                     const float vz = field.read(span, idx, 2);
                     return std::sqrt(vx * vx + vy * vy + vz * vz) * inv_ulb;
                   };
          });
      q_tint_march.launch(stream);
      q_out.combine(stream);
      q_out.dist_combine();
    }

    if (want_q) {
      const auto q_march = carbon::make_ray_march_kernel(
          dense, q_out, camera, q_opts,
          [&qpos, cx, cy, radius, q_gain](carbon::KernelLoader& loader) {
            const auto field = loader.read(qpos);
            return [field, cx, cy, radius, q_gain]
                   CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
                     const carbon::int64_3d gc = span.global_location(idx);
                     const float dx = static_cast<float>(gc.x) - cx;
                     const float dy = static_cast<float>(gc.y) - cy;
                     if (dx * dx + dy * dy < radius * radius) {
                       return 0.0F;
                     }
                     const float q = field.read(span, idx, 0);
                     if (q <= 0.0F) {
                       return 0.0F;
                     }
                     return q * q_gain;
                   };
          });
      q_march.launch(stream);
      q_out.combine(stream);
      q_out.dist_combine();
    }

    render_sphere_for_camera(camera, sphere_out);
  };

#ifdef CARBON_HAVE_VIEWER
  const std::vector<std::string> render_styles = {
      "Speed",         "Q-criterion",         "Wake (u - u_inf)", "Q-criterion (speed-tinted)",
      "Speed (splat)", "Q-criterion (splat)", "Wake (splat)",     "Surface Cp"};
  constexpr int k_first_splat_mode = 4;
  constexpr int k_mode_cp = 7;
  // Which scalar each viewer mode draws. The splat modes repeat the first three; the tinted
  // Q mode has no splat twin because additive splatting carries no per-pixel colour.
  const auto style_layers = [](int mode) -> Layers {
    switch (mode) {
      case 1:
      case 5:
        return Layers::Q;
      case 2:
      case 6:
        return Layers::Wake;
      case 3:
        return Layers::QTinted;
      default:
        return Layers::Speed;
    }
  };
  std::vector<float> splat_rgb_scratch;
  carbon::SurfelImage cp_surfel(sys, k_frame_w, k_frame_h, /*with_attribute=*/true);
  const auto render_cb = [&](const carbon::Camera& camera, int /*w*/, int /*h*/, int mode,
                             std::vector<float>& rgb_out) {
    if (mode == k_mode_cp) {
      render_cp_for_camera(camera, cp_surfel);
      carbon::viewer::rgb_from_surfel_attribute(cp_surfel, camera, carbon::Colormap::CoolWarm,
                                                k_cp_min, k_cp_max, rgb_out);
      return;
    }
    const bool splat = (mode >= k_first_splat_mode);
    const Layers layers = style_layers(mode);
    const bool on_q_image = (layers == Layers::Q || layers == Layers::QTinted);
    if (splat) {
      render_density_for_camera(camera, vel_density, q_density, layers);
      render_sphere_for_camera(camera, sphere_image);
      carbon::viewer::rgb_from_density_colormap(
          on_q_image ? q_density : vel_density,
          on_q_image ? carbon::Colormap::Viridis : carbon::Colormap::Jet, splat_rgb_scratch);
      carbon::viewer::rgb_composite_ray_march_over_rgb(splat_rgb_scratch, sphere_image, rgb_out);
    } else {
      render_for_camera(camera, vel_image, q_image, sphere_image, layers);
      carbon::viewer::rgb_from_composited_layers(on_q_image ? q_image : vel_image, sphere_image,
                                                 background, rgb_out);
    }
  };
  const auto orbit = carbon::viewer::OrbitCamera::from_frame_bounds(
      bounds_min, bounds_max, {0.0F, 0.0F, 1.0F}, {0.0F, 1.0F, 0.0F}, 0.785398F);

#ifdef CARBON_HAVE_VIEWER_WEB
  if (web) {
    // The browser stream is downsampled to a power-of-two texture, so a lighter render
    // resolution keeps the wake interactive without visible quality loss.
    constexpr int k_web_w = 1280;
    constexpr int k_web_h = 720;
    carbon::RayMarchImage web_vel_image(sys, k_web_w, k_web_h);
    carbon::RayMarchImage web_q_image(sys, k_web_w, k_web_h);
    carbon::RayMarchImage web_sphere_image(sys, k_web_w, k_web_h);
    carbon::DensityImage web_vel_density(sys, k_web_w, k_web_h);
    carbon::DensityImage web_q_density(sys, k_web_w, k_web_h);
    carbon::SurfelImage web_cp_surfel(sys, k_web_w, k_web_h, /*with_attribute=*/true);
    const auto web_render_cb = [&](const carbon::Camera& camera, int /*w*/, int /*h*/, int mode,
                                   std::vector<float>& rgb_out) {
      if (mode == k_mode_cp) {
        render_cp_for_camera(camera, web_cp_surfel);
        carbon::viewer::rgb_from_surfel_attribute(web_cp_surfel, camera, carbon::Colormap::CoolWarm,
                                                  k_cp_min, k_cp_max, rgb_out);
        return;
      }
      const bool splat = (mode >= k_first_splat_mode);
      const Layers layers = style_layers(mode);
      const bool on_q_image = (layers == Layers::Q || layers == Layers::QTinted);
      if (splat) {
        render_density_for_camera(camera, web_vel_density, web_q_density, layers);
        render_sphere_for_camera(camera, web_sphere_image);
        carbon::viewer::rgb_from_density_colormap(
            on_q_image ? web_q_density : web_vel_density,
            on_q_image ? carbon::Colormap::Viridis : carbon::Colormap::Jet, splat_rgb_scratch);
        carbon::viewer::rgb_composite_ray_march_over_rgb(splat_rgb_scratch, web_sphere_image,
                                                         rgb_out);
      } else {
        render_for_camera(camera, web_vel_image, web_q_image, web_sphere_image, layers);
        carbon::viewer::rgb_from_composited_layers(on_q_image ? web_q_image : web_vel_image,
                                                   web_sphere_image, background, rgb_out);
      }
    };
    carbon::viewer::Viewer viewer(k_web_w, k_web_h, "render_wind_tunnel");
    carbon::viewer::Viewer::WebOptions wopts;
    wopts.port = web_port;
    wopts.render_modes = render_styles;
    viewer.run_web(sys, orbit, web_render_cb, wopts);
    return EXIT_SUCCESS;
  }
#endif

  bool use_interactive = interactive;
  if (use_interactive && world_rank(sys) == 0 && !carbon::viewer::Viewer::display_available()) {
    CARBON_LOG_WARN("render_wind_tunnel: no display; falling back to file output");
    use_interactive = false;
  }
  if (use_interactive) {
    carbon::viewer::Viewer viewer(k_frame_w, k_frame_h, "render_wind_tunnel");
    viewer.run(sys, orbit, render_cb, render_styles);
    return EXIT_SUCCESS;
  }
#endif

  const carbon::Camera camera =
      carbon::Camera::frame_bounds(bounds_min, bounds_max, {0.0F, 0.0F, 1.0F}, {0.0F, 1.0F, 0.0F},
                                   0.785398F, k_frame_w, k_frame_h);
  render_for_camera(camera, vel_image, q_image, sphere_image);

  // Export the raw fields as a VTI volume alongside the rendering: download the
  // velocity + Q-criterion fields to their host mirrors, then gather the owned
  // cells into one uniform grid. Every rank writes its own shard under MPI (the
  // rank-local z slab range, `.rankN` suffix); a single process writes the whole
  // volume to one file.
  carbon::Transfer<float>(sys, velocity.gpu_buffers(), velocity.cpu_buffers())(stream);
  carbon::Transfer<float>(sys, qfield.gpu_buffers(), qfield.cpu_buffers())(stream);
  carbon::Transfer<float>(sys, density.gpu_buffers(), density.cpu_buffers())(stream);
  stream.wait();

  {
    int64_t z0 = 0;
    int64_t z1 = 0;
    const std::vector<float> vel_host = gather_owned<3>(dense, velocity, z0, z1);
    const std::vector<float> q_host = gather_owned<1>(dense, qfield, z0, z1);
    const std::vector<float> rho_host = gather_owned<1>(dense, density, z0, z1);
    std::vector<float> speed_host(q_host.size());
    for (size_t i = 0; i < speed_host.size(); ++i) {
      const float vx = vel_host[i * 3 + 0];
      const float vy = vel_host[i * 3 + 1];
      const float vz = vel_host[i * 3 + 2];
      speed_host[i] = std::sqrt(vx * vx + vy * vy + vz * vz);
    }
    const int64_t nx = dense.spans()[0].haloed_extent().x;
    const int64_t ny = dense.spans()[0].haloed_extent().y;

    // Geometry marker so the domain is legible in ParaView without a separate
    // legend: 0 = open fluid, 1 = cylinder obstacle, 2 = far-field / inlet /
    // outlet velocity boundary (the x = 0 / x = Nx-1 / y = 0 / y = Ny-1 faces).
    // The span faces (z) are periodic and stay fluid.
    std::vector<float> geo_host(q_host.size(), 0.0F);
    for (int64_t zl = 0; zl < z1 - z0; ++zl) {
      const int64_t gz = z0 + zl;
      for (int64_t y = 0; y < ny; ++y) {
        for (int64_t x = 0; x < nx; ++x) {
          const carbon::int64_3d gc{x, y, gz};
          float marker = 0.0F;
          if (inside_cylinder(gc, cx, cy, radius)) {
            marker = 1.0F;
          } else if (gc.x == 0 || gc.x == extent.x - 1 || gc.y == 0 || gc.y == extent.y - 1) {
            marker = 2.0F;
          }
          geo_host[static_cast<size_t>(x + nx * (y + ny * zl))] = marker;
        }
      }
    }
    std::string vti_shard = vti_out_path;
#ifdef CARBON_HAVE_MPI_NCCL
    if (const auto* d = sys.distributed(); d != nullptr && d->world_size() > 1) {
      vti_shard += ".rank" + std::to_string(d->world_rank());
    }
#endif
    const std::vector<VtiArray> arrays{{"velocity", 3, &vel_host},
                                       {"speed", 1, &speed_host},
                                       {"q", 1, &q_host},
                                       {"rho", 1, &rho_host},
                                       {"geometry", 1, &geo_host}};
    if (!write_vti(vti_shard, nx, ny, z0, z1, arrays)) {
      CARBON_LOG_ERROR("render_wind_tunnel: failed to write {}", vti_shard);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("render_wind_tunnel: wrote {} ({}x{}x{} pts, velocity+speed+q+rho+geometry)",
                    vti_shard, nx, ny, z1 - z0);

    // Surface-pressure profile around the probe ring at mid-span, as a text table. The Cp
    // picture is a qualitative view of exactly these numbers, and a cylinder has a textbook
    // profile (+1 at the nose, a suction minimum near 70-80 degrees, a flat base), so this
    // is the cheapest way to tell a plausible-looking render from a correct one.
    if (world_rank(sys) == 0 && z1 - z0 > 0) {
      const int64_t zl = (z1 - z0) / 2;
      const float probe_r = radius + k_cp_probe_radius;
      std::ostringstream profile;
      profile.setf(std::ios::showpos);
      profile.precision(2);
      profile.setf(std::ios::fixed);
      for (int a = 0; a < 24; ++a) {
        const float theta = static_cast<float>(a) * (2.0F * 3.14159265F / 24.0F);
        const auto ix = static_cast<int64_t>(std::lround(cx + probe_r * std::cos(theta) - 0.5F));
        const auto iy = static_cast<int64_t>(std::lround(cy + probe_r * std::sin(theta) - 0.5F));
        const float rho_p = rho_host[static_cast<size_t>(ix + nx * (iy + ny * zl))];
        profile << ' ' << std::noshowpos
                << static_cast<int>(std::lround(theta * (180.0F / 3.14159265F))) << ':'
                << std::showpos << ((rho_p - rho_inf) * k_cp_scale);
      }
      CARBON_LOG_INFO("render_wind_tunnel: Cp(theta) at r={:.1f}, mid-span:{}", probe_r,
                      profile.str());
    }
  }

  if (world_rank(sys) == 0) {
    const auto [vel_r, vel_g, vel_b] = composite_layers(vel_image, sphere_image, background);
    if (!write_rgb_ppm(out_path, k_frame_w, k_frame_h, vel_r, vel_g, vel_b)) {
      CARBON_LOG_ERROR("render_wind_tunnel: failed to write {}", out_path);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("render_wind_tunnel: wrote {}", out_path);

    const auto [q_r, q_g, q_b] = composite_layers(q_image, sphere_image, background);
    if (!write_rgb_ppm(q_out_path, k_frame_w, k_frame_h, q_r, q_g, q_b)) {
      CARBON_LOG_ERROR("render_wind_tunnel: failed to write {}", q_out_path);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("render_wind_tunnel: wrote {}", q_out_path);
  }

  const auto write_attribute_still = [&](Layers layers, const std::string& path) -> bool {
    render_for_camera(camera, vel_image, q_image, sphere_image, layers);
    if (world_rank(sys) != 0) {
      return true;
    }
    const carbon::RayMarchImage& src = (layers == Layers::Wake) ? vel_image : q_image;
    const auto [r, g, b] = composite_layers(src, sphere_image, background);
    if (!write_rgb_ppm(path, k_frame_w, k_frame_h, r, g, b)) {
      CARBON_LOG_ERROR("render_wind_tunnel: failed to write {}", path);
      return false;
    }
    CARBON_LOG_INFO("render_wind_tunnel: wrote {}", path);
    return true;
  };
  if (!write_attribute_still(Layers::Wake, wake_out_path) ||
      !write_attribute_still(Layers::QTinted, q_tint_out_path)) {
    return EXIT_FAILURE;
  }

  {
    // Surface pressure on the body itself rather than a field in the fluid: a surfel splat,
    // so it needs no ray traversal and ports to the sparse grid families unchanged.
    //
    // Its own camera, not the top-down one the volume passes use. The cylinder's normals are
    // all radial in xy, i.e. perpendicular to a view down the z axis, so from there not one
    // of them faces the eye and back-face culling correctly leaves nothing but the silhouette
    // rim. Seeing pressure painted on a surface requires looking at its flank.
    const carbon::Camera cp_camera = carbon::Camera::frame_bounds(
        {cx - 1.6F * radius, cy - 1.6F * radius, 0.0F},
        {cx + 1.6F * radius, cy + 1.6F * radius, zmax}, {-0.62F, 0.58F, 0.49F}, {0.0F, 0.0F, 1.0F},
        0.785398F, k_frame_w, k_frame_h);
    carbon::SurfelImage cp_image(sys, k_frame_w, k_frame_h, /*with_attribute=*/true);
    render_cp_for_camera(cp_camera, cp_image);
    if (!cp_image.write_attribute_ppm(cp_out_path, carbon::Colormap::CoolWarm, k_cp_min,
                                      k_cp_max)) {
      CARBON_LOG_ERROR("render_wind_tunnel: failed to write {}", cp_out_path);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("render_wind_tunnel: wrote {}", cp_out_path);
  }

  return EXIT_SUCCESS;
}
