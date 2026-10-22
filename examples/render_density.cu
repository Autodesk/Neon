// Parallel density rendering tutorial: analytic sphere SDF on a 512³ Dense grid,
// Gaussian screen splats + smooth SDF transfer, 4K (3840×2160) PGM export.
//
// Build & run
// -----------
//   cmake --build build --target carbon_example_render_density
//   ./build/examples/carbon_example_render_density
//   ./build/examples/carbon_example_render_density render.pgm
//
// With MPI (combine folds cross-rank contributions; only rank 0 writes the image):
//   cmake --build build-mpi --target carbon_example_render_density
//   mpiexec -n 1 ./build-mpi/examples/carbon_example_render_density

#include <carbon/carbon.h>
#include <carbon/dense.h>
#include <carbon/halo.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/render.h>
#include <carbon/stencil.h>
#include <carbon/system.h>

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef CARBON_HAVE_VIEWER
#include "viewer/carbon_viewer.h"
#include "viewer/render_rgb_utils.h"
#endif

namespace {

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

[[nodiscard]] CARBON_HD auto
sphere_sdf(float px,
           float py,
           float pz,
           float cx,
           float cy,
           float cz,
           float radius) noexcept -> float {
  const float dx = px - cx;
  const float dy = py - cy;
  const float dz = pz - cz;
  return std::sqrt(dx * dx + dy * dy + dz * dz) - radius;
}

// Capped cylinder aligned with the x-axis, centred at (cx, cy, cz). Breaks the
// sphere's radial symmetry so orbiting the camera visibly changes the view.
[[nodiscard]] CARBON_HD auto
cylinder_sdf(float px,
             float py,
             float pz,
             float cx,
             float cy,
             float cz,
             float radius,
             float half_len) noexcept -> float {
  const float ay = py - cy;
  const float az = pz - cz;
  const float radial = std::sqrt(ay * ay + az * az) - radius;
  const float axial = std::fabs(px - cx) - half_len;
  const float outside_r = std::fmax(radial, 0.0F);
  const float outside_a = std::fmax(axial, 0.0F);
  const float outside = std::sqrt(outside_r * outside_r + outside_a * outside_a);
  const float inside = std::fmin(std::fmax(radial, axial), 0.0F);
  return outside + inside;
}

[[nodiscard]] auto
build_sdf_init_kernel(const carbon::Dense& dense,
                      carbon::Dense::Field<float,
                                           1>& sdf,
                      float cx,
                      float cy,
                      float cz,
                      float radius) -> carbon::Kernel {
  return dense.make_kernel([&sdf, cx, cy, cz, radius](carbon::KernelLoader& loader) {
    auto out = loader.write(sdf);
    return [out, cx, cy, cz, radius]
           CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> void {
             const carbon::int64_3d gc = span.global_location(idx);
             const float sp = span.spacing();
             const carbon::float32_3d origin = span.origin();
             const float px = origin.x + static_cast<float>(gc.x) * sp;
             const float py = origin.y + static_cast<float>(gc.y) * sp;
             const float pz = origin.z + static_cast<float>(gc.z) * sp;
             const float sphere = sphere_sdf(px, py, pz, cx, cy, cz, radius);
             // half-length kept < domain half-width so the caps stay inside the grid; otherwise the
             // rod would be an (uncapped) tunnel at the domain boundary and ray-march sees a hole.
             const float cyl = cylinder_sdf(px, py, pz, cx, cy, cz, radius * 0.35F, radius * 1.2F);
             out.write(span, idx, 0, std::fmin(sphere, cyl));
           };
  });
}

}  // namespace

auto
main(int argc,
     char** argv) -> int {
  carbon::init();

#ifdef CARBON_HAVE_MPI_NCCL
  auto distributed = std::make_shared<const carbon::Distributed>();
  const int ranks = distributed->world_size();
  carbon::System sys =
      (ranks > 1) ? carbon::System(distributed) : carbon::System(carbon::get_all_devices());
#else
  carbon::System sys(carbon::get_all_devices());
#endif

  if (sys.gpu_group().size() == 0U) {
    CARBON_LOG_ERROR("render_density: no devices visible");
    return EXIT_FAILURE;
  }

  std::string out_path = "render_density.pgm";
  bool interactive = false;
#ifdef CARBON_HAVE_VIEWER_WEB
  bool web = false;
  int web_port = 5000;
#endif
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--interactive") {
      interactive = true;
#ifdef CARBON_HAVE_VIEWER_WEB
    } else if (arg == "--web") {
      web = true;
    } else if (arg == "--port" && i + 1 < argc) {
      web_port = std::atoi(argv[++i]);
#endif
    } else if (out_path == "render_density.pgm" || arg[0] != '-') {
      out_path = arg;
    }
  }

  const carbon::int64_3d extent{512, 512, 512};
  const float cx = 0.5F * static_cast<float>(extent.x - 1);
  const float cy = 0.5F * static_cast<float>(extent.y - 1);
  const float cz = 0.5F * static_cast<float>(extent.z - 1);
  const float radius = 0.35F * static_cast<float>(extent.x);
  constexpr int k_samples_per_axis = 2;
  constexpr float k_sdf_edge_width = 3.0F;
  constexpr float k_splat_sigma_px = 1.75F;

  // Radius 2 (rather than the 6 face neighbours alone) so the grid carries a two-cell z halo: the
  // ray-march normal is a central difference of trilinear samples, which reaches one cell further
  // than the sample itself. With a one-cell halo that gradient is invalid exactly at a per-GPU slab
  // seam, and both neighbours skip the hit -- a one-pixel transparent cut across the surface.
  carbon::Stencil stencil(std::vector<carbon::int32_3d>{{1, 0, 0},
                                                        {-1, 0, 0},
                                                        {0, 1, 0},
                                                        {0, -1, 0},
                                                        {0, 0, 1},
                                                        {0, 0, -1},
                                                        {0, 0, 2},
                                                        {0, 0, -2}},
                          /*remove_center=*/true);

  const carbon::Dense dense(sys, extent, stencil, carbon::BoundaryMode::Open, {}, 1.0F);
  auto sdf = dense.newField<float, 1>(carbon::MemorySide::Device).set_name("sdf");

  const carbon::Stream stream = sys.stream();

  // The init kernel writes owned cells only, so halo cells (per-GPU slab seams and the Open domain
  // edges) would keep their uninitialized contents. The ray-march samples across cell boundaries
  // and reads those as sdf ~ 0, i.e. a spurious isosurface sheet at every seam. Seed the whole
  // allocation "far outside" first; the real halo values arrive with the exchange below.
  // 0x4C4C4C4C reads back as ~5.4e7 -- positive and far outside any isosurface band.
  constexpr int k_sdf_far_byte = 0x4C;
  for (size_t g = 0; g < sys.gpu_group().size(); ++g) {
    const size_t bytes =
        static_cast<size_t>(dense.spans()[g].linear_storage_elements()) * sizeof(float);
    sys.async_memset(sdf.gpu_buffers()[g], k_sdf_far_byte, bytes, stream);
  }
  stream.wait();

  const auto init = build_sdf_init_kernel(dense, sdf, cx, cy, cz, radius);
  init.launch(stream);
  stream.wait();

  // The surface (ray-march) visualization samples the SDF gradient across cells, so it needs the
  // one-cell halo exchanged (harmless for the single-GPU density splat path).
  const carbon::Halo<float, 1> halo(sdf);
  halo(stream);
  stream.wait();

  constexpr int k_frame_w = 3840;
  constexpr int k_frame_h = 2160;
  // Bounding box == the simulation domain (grid extent); the sphere + rod live inside it.
  const carbon::float32_3d bounds_min{0.0F, 0.0F, 0.0F};
  const carbon::float32_3d bounds_max{
      static_cast<float>(extent.x - 1),
      static_cast<float>(extent.y - 1),
      static_cast<float>(extent.z - 1),
  };

  carbon::DensityImage image(sys, k_frame_w, k_frame_h);

  const auto render_for_camera = [&](const carbon::Camera& camera,
                                     carbon::DensityImage& out_image) {
    out_image.clear(stream);
    const auto splat = carbon::make_density_kernel(
        dense, out_image, camera,
        [&sdf, k_sdf_edge_width](carbon::KernelLoader& loader) {
          const auto field = loader.read(sdf);
          return [field, k_sdf_edge_width]
                 CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
                   const float sdf = field.read(span, idx, 0);
                   return carbon::render::sdf_smooth_density(sdf, k_sdf_edge_width);
                 };
        },
        {}, 1.0F, k_samples_per_axis, k_splat_sigma_px);
    splat.launch(stream);
    out_image.combine(stream);
    stream.wait();
  };

  // Non-density visualization: ray-march the same SDF into an opaque Blinn-shaded surface.
  carbon::RayMarchOptions rm_opts{};
  rm_opts.step_size = 0.0F;
  rm_opts.absorption_scale = 0.45F;
  // 0.5-cell steps across the full 512^3 domain diagonal need ~1800 samples; give headroom so the
  // surface is never truncated mid-ray (skipped samples outside the band are near-free).
  rm_opts.max_steps = 4096;
  rm_opts.shading = carbon::RayMarchShading::SurfaceBlinn;
  rm_opts.light_dir = {0.35F, -0.8F, 0.5F};
  rm_opts.light_dir2 = {-0.45F, 0.55F, -0.35F};
  rm_opts.ambient = 0.18F;
  rm_opts.diffuse = 0.68F;
  rm_opts.fill_diffuse = 0.50F;
  rm_opts.specular = 0.20F;
  rm_opts.shininess = 48.0F;
  rm_opts.isosurface_width = k_sdf_edge_width;
  rm_opts.material_color = {0.831F, 0.953F, 0.188F};
  rm_opts.add_background = true;
  rm_opts.background = {0.949F, 0.957F, 0.953F};
  // No clip bounds: the SDF is valid across the whole domain (caps are inside the grid) and empty
  // space is already skipped by the isosurface band test, so a tight clip box only introduces a
  // spurious translucent quad where its entry face grazes the isosurface band.
  rm_opts.use_clip_bounds = false;

  const auto surface_for_camera = [&](const carbon::Camera& camera,
                                      carbon::RayMarchImage& out_image) {
    out_image.clear(stream);
    const auto march = carbon::make_ray_march_kernel(
        dense, out_image, camera, rm_opts, [&sdf](carbon::KernelLoader& loader) {
          const auto field = loader.read(sdf);
          return [field] CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
            return field.read(span, idx, 0);
          };
        });
    march.launch(stream);
    out_image.combine(stream);
    out_image.dist_combine();
    stream.wait();
  };

  constexpr float k_surfel_band = k_sdf_edge_width;
  // Upper bound only: the kernel sizes each footprint from the projected voxel, so this just has
  // to leave enough headroom for the closest zoom the viewer allows.
  constexpr float k_surfel_radius_px = 8.0F;

  const auto surfel_for_camera = [&](const carbon::Camera& camera, carbon::SurfelImage& out_image) {
    out_image.clear(stream);
    const auto splat = carbon::make_surfel_kernel(
        dense, out_image, camera,
        [&sdf](carbon::KernelLoader& loader) {
          const auto field = loader.read(sdf);
          return [field] CARBON_HD(carbon::Dense::Span span,
                                   carbon::Dense::Idx idx) -> carbon::SurfelSample {
            carbon::SurfelSample sample{};
            sample.sdf = field.read(span, idx, 0);
            sample.normal = {0.0F, 0.0F, 1.0F};
            // Only cells inside the splat kernel's band contribute a surfel, and the kernel
            // rejects the rest as soon as this returns. Testing the centre value here first
            // keeps the six gradient reads off the domain interior, which on a 512^3 grid is
            // the overwhelming majority of the ~134M cells.
            if (sample.sdf > k_surfel_band || sample.sdf < -k_surfel_band) {
              return sample;
            }

            const carbon::int64_3d cell = idx.cell();
            const carbon::int64_3d haloed = span.haloed_extent();
            const float sp = span.spacing();
            const auto read_at = [&](int64_t bx, int64_t by, int64_t bz) -> float {
              return field.read(span, span.make_idx(bx, by, bz), 0);
            };
            // Dense stores a halo on z only, so haloed_extent().x / .y are the global domain
            // bounds: an unclamped x/y neighbour read at a domain face walks into the
            // adjacent row, and at cell 0 addresses before the row base. Clamp the sample
            // pair and divide by the distance actually spanned, which degrades cleanly to a
            // one-sided difference on the face. z needs no such care — it is the axis that
            // carries the halo, and the exchange above has filled it.
            const auto lower = [](int64_t v) -> int64_t {
              return (v > 0) ? v - 1 : 0;
            };
            const auto upper = [](int64_t v, int64_t limit) -> int64_t {
              return (v < limit - 1) ? v + 1 : limit - 1;
            };
            const auto slope = [sp](float hi, float lo, int64_t cells) -> float {
              return (cells > 0) ? (hi - lo) / (static_cast<float>(cells) * sp) : 0.0F;
            };

            const int64_t x_lo = lower(cell.x);
            const int64_t x_hi = upper(cell.x, haloed.x);
            const int64_t y_lo = lower(cell.y);
            const int64_t y_hi = upper(cell.y, haloed.y);
            const int64_t z_lo = lower(cell.z);
            const int64_t z_hi = upper(cell.z, haloed.z);

            const float nx =
                slope(read_at(x_hi, cell.y, cell.z), read_at(x_lo, cell.y, cell.z), x_hi - x_lo);
            const float ny =
                slope(read_at(cell.x, y_hi, cell.z), read_at(cell.x, y_lo, cell.z), y_hi - y_lo);
            const float nz =
                slope(read_at(cell.x, cell.y, z_hi), read_at(cell.x, cell.y, z_lo), z_hi - z_lo);

            const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (len > 1.0e-6F) {
              sample.normal = {nx / len, ny / len, nz / len};
            }
            return sample;
          };
        },
        k_surfel_band, k_surfel_radius_px);
    splat.launch(stream);
    out_image.combine(stream);
    stream.wait();
  };

#ifdef CARBON_HAVE_VIEWER
  const std::vector<std::string> render_styles = {
      "Shaded", "Solid green", "Grayscale", "Heatmap", "Surface (ray-march)", "Surface (surfel)"};
  const int ray_march_mode = static_cast<int>(render_styles.size()) - 2;
  const int surfel_mode = static_cast<int>(render_styles.size()) - 1;
  carbon::RayMarchImage rm_image(sys, k_frame_w, k_frame_h);
  carbon::SurfelImage surfel_image(sys, k_frame_w, k_frame_h);
  const auto render_cb = [&](const carbon::Camera& camera, int /*w*/, int /*h*/, int mode,
                             std::vector<float>& rgb_out) {
    if (mode == surfel_mode) {
      surfel_for_camera(camera, surfel_image);
      carbon::viewer::rgb_from_surfel_studio(surfel_image, camera, rgb_out);
    } else if (mode == ray_march_mode) {
      surface_for_camera(camera, rm_image);
      carbon::viewer::rgb_from_ray_march_studio(rm_image, rgb_out);
    } else {
      render_for_camera(camera, image);
      carbon::viewer::rgb_from_density(image, rgb_out, mode);
    }
  };
  const auto orbit = carbon::viewer::OrbitCamera::from_frame_bounds(
      bounds_min, bounds_max, {0.0F, -1.0F, 0.0F}, {0.0F, 0.0F, 1.0F}, 0.785398F);

#ifdef CARBON_HAVE_VIEWER_WEB
  if (web) {
    // The browser stream is downsampled to a power-of-two texture, so a lighter render resolution
    // keeps the shaded colormap post-process interactive without visible quality loss.
    constexpr int k_web_w = 1600;
    constexpr int k_web_h = 900;
    carbon::DensityImage web_image(sys, k_web_w, k_web_h);
    carbon::RayMarchImage web_rm_image(sys, k_web_w, k_web_h);
    carbon::SurfelImage web_surfel_image(sys, k_web_w, k_web_h);
    const auto web_render_cb = [&](const carbon::Camera& camera, int /*w*/, int /*h*/, int mode,
                                   std::vector<float>& rgb_out) {
      if (mode == surfel_mode) {
        surfel_for_camera(camera, web_surfel_image);
        carbon::viewer::rgb_from_surfel_studio(web_surfel_image, camera, rgb_out);
      } else if (mode == ray_march_mode) {
        surface_for_camera(camera, web_rm_image);
        carbon::viewer::rgb_from_ray_march_studio(web_rm_image, rgb_out);
      } else {
        render_for_camera(camera, web_image);
        carbon::viewer::rgb_from_density(web_image, rgb_out, mode);
      }
    };
    carbon::viewer::Viewer viewer(k_web_w, k_web_h, "render_density");
    carbon::viewer::Viewer::WebOptions opts;
    opts.port = web_port;
    opts.render_modes = render_styles;
    viewer.run_web(sys, orbit, web_render_cb, opts);
    return EXIT_SUCCESS;
  }
#endif

  bool use_interactive = interactive;
  if (use_interactive && world_rank(sys) == 0 && !carbon::viewer::Viewer::display_available()) {
    CARBON_LOG_WARN("render_density: no display; falling back to file output");
    use_interactive = false;
  }
  if (use_interactive) {
    carbon::viewer::Viewer viewer(k_frame_w, k_frame_h, "render_density");
    viewer.run(sys, orbit, render_cb, render_styles);
    return EXIT_SUCCESS;
  }
#endif

  render_for_camera(
      carbon::Camera::frame_bounds(bounds_min, bounds_max, {0.0F, -1.0F, 0.0F}, {0.0F, 0.0F, 1.0F},
                                   0.785398F, k_frame_w, k_frame_h),
      image);

  if (world_rank(sys) == 0) {
    if (!image.write_pgm(out_path)) {
      CARBON_LOG_ERROR("render_density: failed to write {}", out_path);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("render_density: wrote {}", out_path);
  }

  return EXIT_SUCCESS;
}
