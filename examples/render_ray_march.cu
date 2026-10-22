// Parallel ray-march rendering: analytic sphere SDF on a 512³ Dense grid,
// Blinn–Phong surface shading, multi-GPU / MPI segment compositing, 4K P6 export.
//
// Build & run
// -----------
//   cmake --build build --target carbon_example_render_ray_march
//   ./build/examples/carbon_example_render_ray_march
//   ./build/examples/carbon_example_render_ray_march render_ray_march.ppm
//
// With MPI (combine folds cross-rank segments; only rank 0 writes the image):
//   cmake --build build-mpi --target carbon_example_render_ray_march
//   mpiexec -n 2 ./build-mpi/examples/carbon_example_render_ray_march

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
             out.write(span, idx, 0, sphere_sdf(px, py, pz, cx, cy, cz, radius));
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
    CARBON_LOG_ERROR("render_ray_march: no devices visible");
    return EXIT_FAILURE;
  }

  std::string out_path = "render_ray_march.ppm";
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
    } else if (out_path == "render_ray_march.ppm" || arg[0] != '-') {
      out_path = arg;
    }
  }

  const carbon::int64_3d extent{512, 512, 512};
  const float cx = 0.5F * static_cast<float>(extent.x - 1);
  const float cy = 0.5F * static_cast<float>(extent.y - 1);
  const float cz = 0.5F * static_cast<float>(extent.z - 1);
  const float radius = 0.35F * static_cast<float>(extent.x);
  constexpr float k_sdf_edge_width = 3.0F;

  carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);

  const carbon::Dense dense(sys, extent, stencil, carbon::BoundaryMode::Open, {}, 1.0F);
  auto sdf = dense.newField<float, 1>(carbon::MemorySide::Device).set_name("sdf");

  const carbon::Stream stream = sys.stream();
  const auto init = build_sdf_init_kernel(dense, sdf, cx, cy, cz, radius);
  init.launch(stream);
  stream.wait();

  const carbon::Halo<float, 1> halo(sdf);
  halo(stream);
  stream.wait();

  constexpr int k_frame_w = 3840;
  constexpr int k_frame_h = 2160;
  const carbon::float32_3d bounds_min{cx - radius, cy - radius, cz - radius};
  const carbon::float32_3d bounds_max{cx + radius, cy + radius, cz + radius};

  carbon::RayMarchOptions opts{};
  opts.step_size = 0.0F;
  opts.absorption_scale = 0.45F;
  opts.max_steps = 1024;
  opts.shading = carbon::RayMarchShading::SurfaceBlinn;
  opts.light_dir = {0.35F, -0.8F, 0.5F};
  opts.light_dir2 = {-0.45F, 0.55F, -0.35F};
  opts.ambient = 0.18F;
  opts.diffuse = 0.68F;
  opts.fill_diffuse = 0.50F;
  opts.specular = 0.20F;
  opts.shininess = 48.0F;
  opts.isosurface_width = k_sdf_edge_width;
  opts.material_color = {0.25F, 0.88F, 0.82F};
  opts.add_background = true;
  opts.background = {1.0F, 1.0F, 1.0F};
  opts.use_clip_bounds = true;
  opts.clip_bounds_min = {bounds_min.x - k_sdf_edge_width, bounds_min.y - k_sdf_edge_width,
                          bounds_min.z - k_sdf_edge_width};
  opts.clip_bounds_max = {bounds_max.x + k_sdf_edge_width, bounds_max.y + k_sdf_edge_width,
                          bounds_max.z + k_sdf_edge_width};

  carbon::RayMarchImage image(sys, k_frame_w, k_frame_h);

  const auto render_for_camera = [&](const carbon::Camera& camera,
                                     carbon::RayMarchImage& out_image) {
    out_image.clear(stream);
    const auto march = carbon::make_ray_march_kernel(
        dense, out_image, camera, opts, [&sdf](carbon::KernelLoader& loader) {
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

#ifdef CARBON_HAVE_VIEWER
  const auto render_cb = [&](const carbon::Camera& camera, int /*w*/, int /*h*/, int /*mode*/,
                             std::vector<float>& rgb_out) {
    render_for_camera(camera, image);
    carbon::viewer::rgb_from_ray_march(image, opts.add_background, opts.background, rgb_out);
  };
  const auto orbit = carbon::viewer::OrbitCamera::from_frame_bounds(
      bounds_min, bounds_max, {0.0F, -1.0F, 0.0F}, {0.0F, 0.0F, 1.0F}, 0.785398F);

#ifdef CARBON_HAVE_VIEWER_WEB
  if (web) {
    carbon::viewer::Viewer viewer(k_frame_w, k_frame_h, "render_ray_march");
    carbon::viewer::Viewer::WebOptions wopts;
    wopts.port = web_port;
    viewer.run_web(sys, orbit, render_cb, wopts);
    return EXIT_SUCCESS;
  }
#endif

  bool use_interactive = interactive;
  if (use_interactive && world_rank(sys) == 0 && !carbon::viewer::Viewer::display_available()) {
    CARBON_LOG_WARN("render_ray_march: no display; falling back to file output");
    use_interactive = false;
  }
  if (use_interactive) {
    carbon::viewer::Viewer viewer(k_frame_w, k_frame_h, "render_ray_march");
    viewer.run(sys, orbit, render_cb);
    return EXIT_SUCCESS;
  }
#endif

  render_for_camera(
      carbon::Camera::frame_bounds(bounds_min, bounds_max, {0.0F, -1.0F, 0.0F}, {0.0F, 0.0F, 1.0F},
                                   0.785398F, k_frame_w, k_frame_h),
      image);

  if (world_rank(sys) == 0) {
    if (!image.write_pgm(out_path, carbon::DensityNormalize::MinMax)) {
      CARBON_LOG_ERROR("render_ray_march: failed to write {}", out_path);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("render_ray_march: wrote {}", out_path);
  }

  return EXIT_SUCCESS;
}
