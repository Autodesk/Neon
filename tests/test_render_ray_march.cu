// Unit tests for ray-march rendering: sphere visibility, multi-GPU composite parity,
// absorption (centre transmittance vs edge), and frustum miss.

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
#include <numeric>

namespace {

[[nodiscard]] auto
frame_sum(const carbon::RayMarchImage& image) -> double {
  double acc = 0.0;
  for (float v : image.host_color()) {
    acc += static_cast<double>(v);
  }
  return acc;
}

[[nodiscard]] auto
build_unit_sphere_sdf_kernel(const carbon::Dense& dense,
                             carbon::Dense::Field<float,
                                                  1>& field,
                             float cx,
                             float cy,
                             float cz,
                             float radius) -> carbon::Kernel {
  return dense.make_kernel([&field, cx, cy, cz, radius](carbon::KernelLoader& loader) {
    auto out = loader.write(field);
    return [out, cx, cy, cz, radius]
           CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> void {
             const carbon::int64_3d gc = span.global_location(idx);
             const float sp = span.spacing();
             const carbon::float32_3d origin = span.origin();
             const float px = origin.x + static_cast<float>(gc.x) * sp;
             const float py = origin.y + static_cast<float>(gc.y) * sp;
             const float pz = origin.z + static_cast<float>(gc.z) * sp;
             const float dx = px - cx;
             const float dy = py - cy;
             const float dz = pz - cz;
             const float sdf = std::sqrt(dx * dx + dy * dy + dz * dz) - radius;
             out.write(span, idx, 0, sdf);
           };
  });
}

[[nodiscard]] auto
build_unit_density_kernel(const carbon::Dense& dense,
                          carbon::Dense::Field<float,
                                               1>& field) -> carbon::Kernel {
  return dense.make_kernel([&field](carbon::KernelLoader& loader) {
    auto out = loader.write(field);
    return [out] CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> void {
      out.write(span, idx, 0, 1.0F);
    };
  });
}

[[nodiscard]] auto
domain_bounds(const carbon::Dense& dense) -> std::pair<carbon::float32_3d,
                                                       carbon::float32_3d> {
  const auto ge = dense.global_extent();
  const float sp = dense.spacing();
  const carbon::float32_3d origin = dense.origin();
  return {
      origin,
      {origin.x + static_cast<float>(ge.x) * sp, origin.y + static_cast<float>(ge.y) * sp,
       origin.z + static_cast<float>(ge.z) * sp},
  };
}

[[nodiscard]] auto
frame_camera(const carbon::Dense& dense,
             int width,
             int height) -> carbon::Camera {
  const auto [bounds_min, bounds_max] = domain_bounds(dense);
  return carbon::Camera::frame_bounds(bounds_min, bounds_max, {0.0F, -1.0F, 0.0F},
                                      {0.0F, 0.0F, 1.0F}, 1.2F, width, height);
}

[[nodiscard]] auto
ray_march_opts(float absorption = 2.0F) -> carbon::RayMarchOptions {
  carbon::RayMarchOptions opts{};
  opts.absorption_scale = absorption;
  opts.max_steps = 512;
  opts.shading = carbon::RayMarchShading::Emission;
  return opts;
}

[[nodiscard]] auto
blinn_sphere_opts(float absorption = 2.0F) -> carbon::RayMarchOptions {
  carbon::RayMarchOptions opts = ray_march_opts(absorption);
  opts.shading = carbon::RayMarchShading::SurfaceBlinn;
  opts.light_dir = {0.3F, -0.85F, 0.45F};
  opts.ambient = 0.12F;
  opts.diffuse = 0.62F;
  opts.specular = 0.45F;
  opts.shininess = 64.0F;
  opts.isosurface_width = 3.0F;
  opts.material_color = {0.92F, 0.94F, 1.0F};
  return opts;
}

[[nodiscard]] auto
volume_gradient_opts(float absorption = 2.0F) -> carbon::RayMarchOptions {
  carbon::RayMarchOptions opts = ray_march_opts(absorption);
  opts.shading = carbon::RayMarchShading::VolumeGradient;
  opts.light_dir = {0.0F, -1.0F, 0.3F};
  opts.ambient = 0.15F;
  opts.diffuse = 0.75F;
  opts.material_color = {0.85F, 0.9F, 1.0F};
  return opts;
}

[[nodiscard]] auto
pixel_luminance(const carbon::RayMarchImage& image,
                int x,
                int y,
                int width) -> float {
  const size_t off = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
  return image.host_color()[off];
}

void
sync_field_halos(carbon::Dense::Field<float,
                                      1>& field,
                 const carbon::Stream& stream) {
  const carbon::Halo<float, 1> halo(field);
  halo(stream);
  stream.wait();
}

void
test_camera_ray_fan(const carbon::Camera& camera) {
  const int w = camera.width();
  const int h = camera.height();
  float dir_y_min = 1.0F;
  float dir_y_max = -1.0F;
  int ok = 0;
  for (int py = 0; py < h; py += (h > 256 ? h / 32 : 1)) {
    carbon::float32_3d origin{};
    carbon::float32_3d dir{};
    if (!camera.ray_through_pixel(w / 2, py, origin, dir)) {
      continue;
    }
    ++ok;
    dir_y_min = (dir.y < dir_y_min) ? dir.y : dir_y_min;
    dir_y_max = (dir.y > dir_y_max) ? dir.y : dir_y_max;
  }
  if (ok < 2 || dir_y_max - dir_y_min < 0.05F) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_ray_march camera_ray_fan: expected dir.y spread across rows (min {} max {} "
        "ok {})",
        dir_y_min, dir_y_max, ok);
  }
}

void
test_sphere_visible_large(carbon::System& sys,
                          const carbon::Stream& stream) {
  const carbon::int64_3d extent{512, 512, 512};
  const float cx = 0.5F * static_cast<float>(extent.x - 1);
  const float cy = 0.5F * static_cast<float>(extent.y - 1);
  const float cz = 0.5F * static_cast<float>(extent.z - 1);
  const float radius = 0.35F * static_cast<float>(extent.x);

  carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);

  const carbon::Dense dense(sys, extent, stencil, carbon::BoundaryMode::Open, {}, 1.0F);
  auto sdf = dense.newField<float, 1>(carbon::MemorySide::Device);

  const auto init = build_unit_sphere_sdf_kernel(dense, sdf, cx, cy, cz, radius);
  init.launch(stream);
  stream.wait();
  sync_field_halos(sdf, stream);

  constexpr int k_w = 256;
  constexpr int k_h = 256;
  const carbon::float32_3d bounds_min{cx - radius, cy - radius, cz - radius};
  const carbon::float32_3d bounds_max{cx + radius, cy + radius, cz + radius};
  const carbon::Camera camera = carbon::Camera::frame_bounds(
      bounds_min, bounds_max, {0.0F, -1.0F, 0.0F}, {0.0F, 0.0F, 1.0F}, 0.785398F, k_w, k_h);
  carbon::RayMarchImage image(sys, k_w, k_h);
  image.clear(stream);

  const auto march = carbon::make_ray_march_kernel(
      dense, image, camera, ray_march_opts(), [&sdf](carbon::KernelLoader& loader) {
        const auto field = loader.read(sdf);
        return [field] CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
          return carbon::render::sdf_smooth_density(field.read(span, idx, 0), 3.0F);
        };
      });

  march.launch(stream);
  image.combine(stream);
  image.dist_combine();

  if (frame_sum(image) <= 0.0) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_ray_march sphere_visible_large: expected visible frame, got zero sum");
  }
}

void
test_sphere_visible(carbon::System& sys,
                    const carbon::Stream& stream) {
  const carbon::int64_3d extent{64, 64, 64};
  const float cx = 0.5F * static_cast<float>(extent.x - 1);
  const float cy = 0.5F * static_cast<float>(extent.y - 1);
  const float cz = 0.5F * static_cast<float>(extent.z - 1);
  const float radius = 0.3F * static_cast<float>(extent.x);

  carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);

  const carbon::Dense dense(sys, extent, stencil, carbon::BoundaryMode::Open, {}, 1.0F);
  auto sdf = dense.newField<float, 1>(carbon::MemorySide::Device);

  const auto init = build_unit_sphere_sdf_kernel(dense, sdf, cx, cy, cz, radius);
  init.launch(stream);
  stream.wait();

  constexpr int k_w = 128;
  constexpr int k_h = 128;
  const carbon::Camera camera = frame_camera(dense, k_w, k_h);
  carbon::RayMarchImage image(sys, k_w, k_h);
  image.clear(stream);

  const auto march = carbon::make_ray_march_kernel(
      dense, image, camera, ray_march_opts(), [&sdf](carbon::KernelLoader& loader) {
        const auto field = loader.read(sdf);
        return [field] CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
          return carbon::render::sdf_smooth_density(field.read(span, idx, 0), 1.0F);
        };
      });

  march.launch(stream);
  image.combine(stream);
  image.dist_combine();

  if (frame_sum(image) <= 0.0) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_ray_march sphere_visible: expected visible frame, got zero sum");
  }
}

void
test_slab_composite(carbon::System& sys,
                    const carbon::Stream& stream) {
  if (sys.gpu_group().size() < 2U) {
    return;
  }

  const carbon::int64_3d extent{32, 32, 32};
  carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);

  const carbon::Dense dense_multi(sys, extent, stencil, carbon::BoundaryMode::Open, {}, 1.0F);
  auto field_multi = dense_multi.newField<float, 1>(carbon::MemorySide::Device);
  const auto fill_multi = build_unit_density_kernel(dense_multi, field_multi);
  fill_multi.launch(stream);
  stream.wait();
  sync_field_halos(field_multi, stream);

  carbon::System sys_single(
      carbon::detail::GpuGroup(std::vector<carbon::detail::Gpu>{sys.gpu_group().devices()[0]}));
  const carbon::Stream stream_single = sys_single.stream();
  const carbon::Dense dense_single(sys_single, extent, stencil, carbon::BoundaryMode::Open, {},
                                   1.0F);
  auto field_single = dense_single.newField<float, 1>(carbon::MemorySide::Device);
  const auto fill_single = build_unit_density_kernel(dense_single, field_single);
  fill_single.launch(stream_single);
  stream_single.wait();

  constexpr int k_w = 64;
  constexpr int k_h = 64;
  const carbon::Camera camera = frame_camera(dense_multi, k_w, k_h);
  const auto opts = ray_march_opts(0.5F);

  carbon::RayMarchImage image_multi(sys, k_w, k_h);
  image_multi.clear(stream);
  const auto march_multi = carbon::make_ray_march_kernel(
      dense_multi, image_multi, camera, opts, [&field_multi](carbon::KernelLoader& loader) {
        const auto field = loader.read(field_multi);
        return [field] CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
          return field.read(span, idx, 0);
        };
      });
  march_multi.launch(stream);
  image_multi.combine(stream);
  image_multi.dist_combine();

  carbon::RayMarchImage image_single(sys_single, k_w, k_h);
  image_single.clear(stream_single);
  const auto march_single = carbon::make_ray_march_kernel(
      dense_single, image_single, camera, opts, [&field_single](carbon::KernelLoader& loader) {
        const auto field = loader.read(field_single);
        return [field] CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
          return field.read(span, idx, 0);
        };
      });
  march_single.launch(stream_single);
  image_single.combine(stream_single);
  image_single.dist_combine();

  const double got = frame_sum(image_multi);
  const double expected = frame_sum(image_single);
  const double denom = std::max(expected, 1.0e-6);
  const double rel = std::abs(got - expected) / denom;
  if (rel > 0.15) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_ray_march slab_composite: expected sum {} got {} (rel err {})", expected, got,
        rel);
  }
}

void
test_multi_gpu_center_row(carbon::System& sys,
                          const carbon::Stream& stream) {
  if (sys.gpu_group().size() < 2U) {
    return;
  }

  const carbon::int64_3d extent{512, 512, 512};
  const float cx = 0.5F * static_cast<float>(extent.x - 1);
  const float cy = 0.5F * static_cast<float>(extent.y - 1);
  const float cz = 0.5F * static_cast<float>(extent.z - 1);
  const float radius = 0.35F * static_cast<float>(extent.x);

  carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);

  const carbon::Dense dense_multi(sys, extent, stencil, carbon::BoundaryMode::Open, {}, 1.0F);
  auto sdf_multi = dense_multi.newField<float, 1>(carbon::MemorySide::Device);

  const auto init_multi = build_unit_sphere_sdf_kernel(dense_multi, sdf_multi, cx, cy, cz, radius);
  init_multi.launch(stream);
  stream.wait();
  sync_field_halos(sdf_multi, stream);

  carbon::System sys_single(
      carbon::detail::GpuGroup(std::vector<carbon::detail::Gpu>{sys.gpu_group().devices()[0]}));
  const carbon::Stream stream_single = sys_single.stream();
  const carbon::Dense dense_single(sys_single, extent, stencil, carbon::BoundaryMode::Open, {},
                                   1.0F);
  auto sdf_single = dense_single.newField<float, 1>(carbon::MemorySide::Device);

  const auto init_single =
      build_unit_sphere_sdf_kernel(dense_single, sdf_single, cx, cy, cz, radius);
  init_single.launch(stream_single);
  stream_single.wait();

  constexpr int k_w = 3840;
  constexpr int k_h = 2160;
  const carbon::float32_3d bounds_min{cx - radius, cy - radius, cz - radius};
  const carbon::float32_3d bounds_max{cx + radius, cy + radius, cz + radius};
  const carbon::Camera camera = carbon::Camera::frame_bounds(
      bounds_min, bounds_max, {0.0F, -1.0F, 0.0F}, {0.0F, 0.0F, 1.0F}, 0.785398F, k_w, k_h);
  const auto opts = blinn_sphere_opts();

  carbon::RayMarchImage image_multi(sys, k_w, k_h);
  image_multi.clear(stream);
  const auto march_multi = carbon::make_ray_march_kernel(
      dense_multi, image_multi, camera, opts, [&sdf_multi](carbon::KernelLoader& loader) {
        const auto field = loader.read(sdf_multi);
        return [field] CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
          return field.read(span, idx, 0);
        };
      });
  march_multi.launch(stream);
  image_multi.combine(stream);
  image_multi.dist_combine();

  carbon::RayMarchImage image_single(sys_single, k_w, k_h);
  image_single.clear(stream_single);
  const auto march_single = carbon::make_ray_march_kernel(
      dense_single, image_single, camera, opts, [&sdf_single](carbon::KernelLoader& loader) {
        const auto field = loader.read(sdf_single);
        return [field] CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
          return field.read(span, idx, 0);
        };
      });
  march_single.launch(stream_single);
  image_single.combine(stream_single);
  image_single.dist_combine();

  const int cy_px = k_h / 2;
  const int cx_px = k_w / 2;
  const float center = pixel_luminance(image_multi, cx_px, cy_px, k_w);
  if (center < 0.45F) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_ray_march multi_gpu_center_row: expected bright center pixel, got {}", center);
  }

  float max_rgb_err = 0.0F;
  constexpr int k_seam_half = 16;
  for (int x = cx_px - k_seam_half; x <= cx_px + k_seam_half; ++x) {
    if (x < 0 || x >= k_w) {
      continue;
    }
    const float lum = pixel_luminance(image_multi, x, cy_px, k_w);
    if (lum < 0.25F) {
      continue;
    }
    const size_t off =
        static_cast<size_t>(cy_px) * static_cast<size_t>(k_w) + static_cast<size_t>(x);
    const float dr = std::abs(image_multi.host_color_r()[off] - image_single.host_color_r()[off]);
    const float dg = std::abs(image_multi.host_color_g()[off] - image_single.host_color_g()[off]);
    const float db = std::abs(image_multi.host_color_b()[off] - image_single.host_color_b()[off]);
    const float err = std::max({dr, dg, db});
    max_rgb_err = (err > max_rgb_err) ? err : max_rgb_err;
  }
  if (max_rgb_err > 0.2F) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_ray_march multi_gpu_center_row: multi vs single RGB err {} exceeds threshold",
        max_rgb_err);
  }
}

void
test_surface_not_filled(carbon::System& sys,
                        const carbon::Stream& stream) {
  const carbon::int64_3d extent{64, 64, 64};
  const float cx = 0.5F * static_cast<float>(extent.x - 1);
  const float cy = 0.5F * static_cast<float>(extent.y - 1);
  const float cz = 0.5F * static_cast<float>(extent.z - 1);
  const float radius = 0.28F * static_cast<float>(extent.x);

  carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);

  const carbon::Dense dense(sys, extent, stencil, carbon::BoundaryMode::Open, {}, 1.0F);
  auto sdf = dense.newField<float, 1>(carbon::MemorySide::Device);

  const auto init = build_unit_sphere_sdf_kernel(dense, sdf, cx, cy, cz, radius);
  init.launch(stream);
  stream.wait();

  constexpr int k_w = 128;
  constexpr int k_h = 128;
  const carbon::Camera camera = frame_camera(dense, k_w, k_h);
  carbon::RayMarchImage image(sys, k_w, k_h);
  image.clear(stream);

  carbon::RayMarchOptions opts = ray_march_opts(4.0F);
  const auto march = carbon::make_ray_march_kernel(
      dense, image, camera, opts, [&sdf](carbon::KernelLoader& loader) {
        const auto field = loader.read(sdf);
        return [field] CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
          return carbon::render::sdf_smooth_density(field.read(span, idx, 0), 1.0F);
        };
      });

  march.launch(stream);
  image.combine(stream);
  image.dist_combine();

  float t_min = 1.0F;
  float t_max = 0.0F;
  for (float t : image.host_transmittance()) {
    t_min = (t < t_min) ? t : t_min;
    t_max = (t > t_max) ? t : t_max;
  }
  if (!(t_min < t_max - 0.05F)) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_ray_march surface_not_filled: expected transmittance variation (min {} max "
        "{})",
        t_min, t_max);
  }
}

void
test_surface_blinn_lit_side(carbon::System& sys,
                            const carbon::Stream& stream) {
  const carbon::int64_3d extent{512, 512, 512};
  const float cx = 0.5F * static_cast<float>(extent.x - 1);
  const float cy = 0.5F * static_cast<float>(extent.y - 1);
  const float cz = 0.5F * static_cast<float>(extent.z - 1);
  const float radius = 0.35F * static_cast<float>(extent.x);

  carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);

  const carbon::Dense dense(sys, extent, stencil, carbon::BoundaryMode::Open, {}, 1.0F);
  auto sdf = dense.newField<float, 1>(carbon::MemorySide::Device);

  const auto init = build_unit_sphere_sdf_kernel(dense, sdf, cx, cy, cz, radius);
  init.launch(stream);
  stream.wait();
  sync_field_halos(sdf, stream);

  constexpr int k_w = 512;
  constexpr int k_h = 512;
  const carbon::float32_3d bounds_min{cx - radius, cy - radius, cz - radius};
  const carbon::float32_3d bounds_max{cx + radius, cy + radius, cz + radius};
  const carbon::Camera camera = carbon::Camera::frame_bounds(
      bounds_min, bounds_max, {0.0F, -1.0F, 0.0F}, {0.0F, 0.0F, 1.0F}, 0.785398F, k_w, k_h);
  carbon::RayMarchImage image(sys, k_w, k_h);
  image.clear(stream);

  auto opts = blinn_sphere_opts();
  opts.light_dir = {0.0F, -1.0F, 0.2F};

  const auto march = carbon::make_ray_march_kernel(
      dense, image, camera, opts, [&sdf](carbon::KernelLoader& loader) {
        const auto field = loader.read(sdf);
        return [field] CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
          return field.read(span, idx, 0);
        };
      });

  march.launch(stream);
  image.combine(stream);
  image.dist_combine();

  const int top_y = k_h / 4;
  const int bottom_y = (3 * k_h) / 4;
  const int mid_x = k_w / 2;
  const float lit = pixel_luminance(image, mid_x, top_y, k_w);
  const float dark = pixel_luminance(image, mid_x, bottom_y, k_w);
  if (!(lit > dark + 0.08F)) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_ray_march surface_blinn_lit_side: expected lit top {} > dark bottom {}", lit,
        dark);
  }
}

void
test_volume_gradient_mode(carbon::System& sys,
                          const carbon::Stream& stream) {
  const carbon::int64_3d extent{128, 128, 128};
  const float cx = 0.5F * static_cast<float>(extent.x - 1);
  const float cy = 0.5F * static_cast<float>(extent.y - 1);
  const float cz = 0.5F * static_cast<float>(extent.z - 1);
  const float radius = 0.35F * static_cast<float>(extent.x);

  carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);

  const carbon::Dense dense(sys, extent, stencil, carbon::BoundaryMode::Open, {}, 1.0F);
  auto sdf = dense.newField<float, 1>(carbon::MemorySide::Device);

  const auto init = build_unit_sphere_sdf_kernel(dense, sdf, cx, cy, cz, radius);
  init.launch(stream);
  stream.wait();
  sync_field_halos(sdf, stream);

  constexpr int k_w = 256;
  constexpr int k_h = 256;
  const carbon::Camera camera = frame_camera(dense, k_w, k_h);
  carbon::RayMarchImage image(sys, k_w, k_h);
  image.clear(stream);

  const auto march = carbon::make_ray_march_kernel(
      dense, image, camera, volume_gradient_opts(), [&sdf](carbon::KernelLoader& loader) {
        const auto field = loader.read(sdf);
        return [field] CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
          return carbon::render::sdf_smooth_density(field.read(span, idx, 0), 2.0F);
        };
      });

  march.launch(stream);
  image.combine(stream);
  image.dist_combine();

  double mean_r = 0.0;
  double mean_g = 0.0;
  double mean_b = 0.0;
  const size_t n = image.host_color_r().size();
  for (size_t i = 0; i < n; ++i) {
    mean_r += static_cast<double>(image.host_color_r()[i]);
    mean_g += static_cast<double>(image.host_color_g()[i]);
    mean_b += static_cast<double>(image.host_color_b()[i]);
  }
  mean_r /= static_cast<double>(n);
  mean_g /= static_cast<double>(n);
  mean_b /= static_cast<double>(n);

  double var_r = 0.0;
  double var_g = 0.0;
  double var_b = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double dr = static_cast<double>(image.host_color_r()[i]) - mean_r;
    const double dg = static_cast<double>(image.host_color_g()[i]) - mean_g;
    const double db = static_cast<double>(image.host_color_b()[i]) - mean_b;
    var_r += dr * dr;
    var_g += dg * dg;
    var_b += db * db;
  }
  var_r /= static_cast<double>(n);
  var_g /= static_cast<double>(n);
  var_b /= static_cast<double>(n);

  const double total_var = var_r + var_g + var_b;
  if (total_var < 1.0e-4) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_ray_march volume_gradient_mode: expected non-uniform RGB, var {}", total_var);
  }
}

void
test_scalar_field_threshold(carbon::System& sys,
                            const carbon::Stream& stream) {
  const carbon::int64_3d extent{32, 32, 32};
  carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);

  const carbon::Dense dense(sys, extent, stencil, carbon::BoundaryMode::Open, {}, 1.0F);
  auto field = dense.newField<float, 1>(carbon::MemorySide::Device);
  const auto fill = build_unit_density_kernel(dense, field);
  fill.launch(stream);
  stream.wait();

  constexpr int k_w = 64;
  constexpr int k_h = 64;
  const carbon::Camera camera = frame_camera(dense, k_w, k_h);

  auto run_threshold = [&](float threshold) -> float {
    carbon::RayMarchImage image(sys, k_w, k_h);
    image.clear(stream);
    carbon::RayMarchOptions opts{};
    opts.absorption_scale = 1.0F;
    opts.max_steps = 256;
    opts.shading = carbon::RayMarchShading::ScalarField;
    opts.colormap = carbon::Colormap::Viridis;
    opts.value_threshold = threshold;
    opts.value_min = 0.0F;
    opts.value_max = 1.0F;
    opts.opacity_scale = 4.0F;
    const auto march = carbon::make_ray_march_kernel(
        dense, image, camera, opts, [&field](carbon::KernelLoader& loader) {
          const auto f = loader.read(field);
          return [f] CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
            (void)f;
            (void)span;
            (void)idx;
            return 0.5F;
          };
        });
    march.launch(stream);
    image.combine(stream);
    image.dist_combine();
    const size_t off =
        static_cast<size_t>(k_h / 2) * static_cast<size_t>(k_w) + static_cast<size_t>(k_w / 2);
    return image.host_transmittance()[off];
  };

  const float trans_above = run_threshold(0.8F);
  const float trans_below = run_threshold(0.3F);
  if (trans_above < 0.95F) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_ray_march scalar_field_threshold: expected transparent above threshold, "
        "trans {}",
        trans_above);
  }
  if (trans_below > 0.5F) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_ray_march scalar_field_threshold: expected opaque below threshold, trans {}",
        trans_below);
  }
}

void
test_colormap_monotonic() {
  const carbon::float32_3d c0 = carbon::render::colormap_eval(carbon::Colormap::Viridis, 0.0F);
  const carbon::float32_3d c1 = carbon::render::colormap_eval(carbon::Colormap::Viridis, 1.0F);
  if (c0.x >= c1.x || c0.y >= c1.y) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_ray_march colormap_monotonic: expected Viridis endpoints to ramp up in r/g");
  }

  const carbon::float32_3d j0 = carbon::render::colormap_eval(carbon::Colormap::Jet, 0.0F);
  const carbon::float32_3d j1 = carbon::render::colormap_eval(carbon::Colormap::Jet, 1.0F);
  if (j0.x >= j1.x || j0.z <= j1.z) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_ray_march colormap_monotonic: expected Jet endpoints blue->red");
  }

  const carbon::float32_3d g0 = carbon::render::colormap_eval(carbon::Colormap::Grayscale, 0.0F);
  const carbon::float32_3d g1 = carbon::render::colormap_eval(carbon::Colormap::Grayscale, 1.0F);
  if (g0.x != 0.0F || g1.x != 1.0F) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_ray_march colormap_monotonic: expected Grayscale endpoints 0 and 1");
  }

  // CoolWarm is diverging, so the property to pin is not monotonicity but the shape: cool
  // blue end, warm red end, and a near-neutral pivot at the midpoint. A sequential map
  // substituted here by mistake would still ramp end to end but would not be grey in the
  // middle.
  const carbon::float32_3d w0 = carbon::render::colormap_eval(carbon::Colormap::CoolWarm, 0.0F);
  const carbon::float32_3d wm = carbon::render::colormap_eval(carbon::Colormap::CoolWarm, 0.5F);
  const carbon::float32_3d w1 = carbon::render::colormap_eval(carbon::Colormap::CoolWarm, 1.0F);
  if (!(w0.z > w0.x) || !(w1.x > w1.z)) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_ray_march colormap_monotonic: expected CoolWarm endpoints blue->red (got "
        "({}, {}, {}) and ({}, {}, {}))",
        w0.x, w0.y, w0.z, w1.x, w1.y, w1.z);
  }
  const float pivot_spread = std::max({wm.x, wm.y, wm.z}) - std::min({wm.x, wm.y, wm.z});
  if (wm.x < 0.5F || pivot_spread > 0.05F) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_ray_march colormap_monotonic: expected CoolWarm midpoint near-neutral and "
        "light (got ({}, {}, {}), spread {})",
        wm.x, wm.y, wm.z, pivot_spread);
  }
}

void
test_speed_deviation() {
  const carbon::float32_3d free_stream{0.1F, 0.0F, 0.0F};
  const float inv = 1.0F / 0.1F;
  if (carbon::render::speed_deviation(free_stream, free_stream, inv) != 0.0F) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_ray_march speed_deviation: undisturbed fluid must read exactly zero");
  }
  // Same magnitude, rotated 90 degrees: a speed-only difference would call this undisturbed.
  const carbon::float32_3d deflected{0.0F, 0.1F, 0.0F};
  const float d = carbon::render::speed_deviation(deflected, free_stream, inv);
  if (std::abs(d - std::sqrt(2.0F)) > 1.0e-5F) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_ray_march speed_deviation: expected sqrt(2) for a 90-degree deflection at "
        "free-stream magnitude, got {}",
        d);
  }
  const carbon::float32_3d halved{0.05F, 0.0F, 0.0F};
  if (std::abs(carbon::render::speed_deviation(halved, free_stream, inv) - 0.5F) > 1.0e-5F) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_ray_march speed_deviation: expected 0.5 for a half-speed wake");
  }
}

/**
 * ScalarFieldAttribute must take opacity from the primary field and colour from the second
 * one. A ball of uniform density carrying an x-ramp attribute separates the two: the colour
 * has to vary across the ball even though its density does not, and pixels outside the ball
 * have to stay fully transparent even though the attribute there is large — which is what
 * fails if the threshold or the alpha accidentally read the attribute.
 */
void
test_scalar_field_attribute(carbon::System& sys,
                            const carbon::Stream& stream) {
  const carbon::int64_3d extent{96, 96, 96};
  const float c = 0.5F * static_cast<float>(extent.x - 1);
  const float radius = 0.25F * static_cast<float>(extent.x);

  const carbon::Dense dense(sys, extent, carbon::Stencil({}), carbon::BoundaryMode::Open, {}, 1.0F);
  auto rho = dense.newField<float, 1>(carbon::MemorySide::Device);
  auto attr = dense.newField<float, 1>(carbon::MemorySide::Device);

  const carbon::Kernel fill =
      dense.make_kernel([&rho, &attr, c, radius](carbon::KernelLoader& loader) {
        auto out_rho = loader.write(rho);
        auto out_attr = loader.write(attr);
        return [out_rho, out_attr, c, radius]
               CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> void {
                 const carbon::int64_3d gc = span.global_location(idx);
                 const float dx = static_cast<float>(gc.x) - c;
                 const float dy = static_cast<float>(gc.y) - c;
                 const float dz = static_cast<float>(gc.z) - c;
                 const bool inside = (dx * dx + dy * dy + dz * dz) <= (radius * radius);
                 out_rho.write(span, idx, 0, inside ? 1.0F : 0.0F);
                 out_attr.write(span, idx, 0, dx);
               };
      });
  fill.launch(stream);
  stream.wait();

  constexpr int k_w = 160;
  constexpr int k_h = 160;
  const carbon::Camera camera = carbon::Camera::frame_bounds(
      {0.0F, 0.0F, 0.0F},
      {static_cast<float>(extent.x), static_cast<float>(extent.y), static_cast<float>(extent.z)},
      {0.0F, 0.0F, 1.0F}, {0.0F, 1.0F, 0.0F}, 0.785398F, k_w, k_h);

  carbon::RayMarchOptions opts{};
  opts.shading = carbon::RayMarchShading::ScalarFieldAttribute;
  opts.colormap = carbon::Colormap::CoolWarm;
  opts.value_threshold = 0.5F;
  opts.value_min = -radius;
  opts.value_max = radius;
  opts.opacity_scale = 4.0F;
  opts.max_steps = 512;

  carbon::RayMarchImage image(sys, k_w, k_h);
  image.clear(stream);
  const carbon::Kernel march = carbon::make_ray_march_kernel(
      dense, image, camera, opts,
      [&rho](carbon::KernelLoader& loader) {
        const auto f = loader.read(rho);
        return [f] CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
          return f.read(span, idx, 0);
        };
      },
      [&attr](carbon::KernelLoader& loader) {
        const auto f = loader.read(attr);
        return [f] CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
          return f.read(span, idx, 0);
        };
      });
  march.launch(stream);
  image.combine(stream);
  image.dist_combine();

  const auto& red = image.host_color_r();
  const auto& blue = image.host_color_b();
  const auto& trans = image.host_transmittance();
  double left_rb = 0.0;
  double right_rb = 0.0;
  int left_n = 0;
  int right_n = 0;
  for (size_t o = 0; o < trans.size(); ++o) {
    if (trans[o] > 0.5F) {
      continue;
    }
    const double rb = static_cast<double>(red[o]) - static_cast<double>(blue[o]);
    if (static_cast<int>(o % static_cast<size_t>(k_w)) < k_w / 2) {
      left_rb += rb;
      ++left_n;
    } else {
      right_rb += rb;
      ++right_n;
    }
  }
  if (left_n < 100 || right_n < 100) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_ray_march scalar_field_attribute: too few covered pixels (left {}, right {})",
        left_n, right_n);
  }
  if (!(left_rb / left_n < 0.0) || !(right_rb / right_n > 0.0)) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_ray_march scalar_field_attribute: expected the attribute ramp to split the "
        "ball blue/red (mean r-b: left {}, right {})",
        left_rb / left_n, right_rb / right_n);
  }

  // The four frame corners look through the box but miss the ball. The attribute is at its
  // most positive on the right-hand pair, so opacity leaking from the attribute would show
  // up here first.
  for (const size_t corner : {size_t{0}, static_cast<size_t>(k_w - 1),
                              static_cast<size_t>(k_w) * static_cast<size_t>(k_h - 1),
                              static_cast<size_t>(k_w) * static_cast<size_t>(k_h) - 1}) {
    if (trans[corner] != 1.0F) {
      carbon::log::throw_with_log<std::runtime_error>(
          "test_render_ray_march scalar_field_attribute: corner pixel {} misses the ball but is "
          "not transparent (trans {}); opacity must come from the primary field",
          corner, trans[corner]);
    }
  }
}

/**
 * The depth plane must carry the distance to the first sample that contributed opacity, not
 * the distance at which the ray entered the domain box. A sphere makes the two easy to tell
 * apart: the box entry is flat across the frame, the sphere's front surface bulges towards
 * the camera at the centre, and outside the silhouette the ray crosses the whole box while
 * hitting nothing at all.
 */
void
test_hit_depth(carbon::System& sys,
               const carbon::Stream& stream) {
  const carbon::int64_3d extent{96, 96, 96};
  const float c = 0.5F * static_cast<float>(extent.x - 1);
  const float radius = 0.3F * static_cast<float>(extent.x);

  carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);

  const carbon::Dense dense(sys, extent, stencil, carbon::BoundaryMode::Open, {}, 1.0F);
  auto sdf = dense.newField<float, 1>(carbon::MemorySide::Device);
  build_unit_sphere_sdf_kernel(dense, sdf, c, c, c, radius).launch(stream);
  stream.wait();
  sync_field_halos(sdf, stream);

  constexpr int k_w = 128;
  constexpr int k_h = 128;
  // Frame the whole domain, not just the sphere, so the corners are box-but-no-sphere rays.
  const carbon::Camera camera = carbon::Camera::frame_bounds(
      {0.0F, 0.0F, 0.0F},
      {static_cast<float>(extent.x), static_cast<float>(extent.y), static_cast<float>(extent.z)},
      {0.0F, 0.0F, 1.0F}, {0.0F, 1.0F, 0.0F}, 0.785398F, k_w, k_h);

  carbon::RayMarchOptions opts = blinn_sphere_opts();
  carbon::RayMarchImage image(sys, k_w, k_h);
  image.clear(stream);
  const carbon::Kernel march = carbon::make_ray_march_kernel(
      dense, image, camera, opts, [&sdf](carbon::KernelLoader& loader) {
        const auto f = loader.read(sdf);
        return [f] CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
          return f.read(span, idx, 0);
        };
      });
  march.launch(stream);
  image.combine(stream);
  image.dist_combine();

  const auto& depth = image.host_depth();
  const size_t centre =
      static_cast<size_t>(k_h / 2) * static_cast<size_t>(k_w) + static_cast<size_t>(k_w / 2);
  if (!std::isfinite(depth[centre])) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_ray_march hit_depth: expected a finite depth through the sphere centre");
  }
  for (const size_t corner : {size_t{0}, static_cast<size_t>(k_w - 1),
                              static_cast<size_t>(k_w) * static_cast<size_t>(k_h - 1)}) {
    if (std::isfinite(depth[corner])) {
      carbon::log::throw_with_log<std::runtime_error>(
          "test_render_ray_march hit_depth: corner pixel {} crosses the box but misses the "
          "sphere, expected inf depth, got {}",
          corner, depth[corner]);
    }
  }

  // The centre ray passes through the sphere centre, so its first hit sits one radius in
  // front of it. The isosurface has a band width, so allow a couple of band widths of slack.
  carbon::float32_3d eye{};
  carbon::float32_3d dir{};
  if (!camera.ray_through_pixel(k_w / 2, k_h / 2, eye, dir)) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_ray_march hit_depth: centre ray is degenerate");
  }
  const float ex = eye.x - c;
  const float ey = eye.y - c;
  const float ez = eye.z - c;
  const float expected = std::sqrt(ex * ex + ey * ey + ez * ez) - radius;
  const float tol = 3.0F * opts.isosurface_width;
  if (std::abs(depth[centre] - expected) > tol) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_ray_march hit_depth: expected the centre hit one radius in front of the "
        "sphere centre ({} +/- {}), got {}",
        expected, tol, depth[centre]);
  }
}

/**
 * SurfaceScalar takes its geometry from the primary field and its colour from a second one.
 * The sphere SDF supplies the surface; the attribute is a linear ramp in x mapped through
 * CoolWarm, so the rendered surface must go blue on the low-x side and red on the high-x
 * side. Comparing the two sides rather than absolute values keeps the check independent of
 * the lighting term, which is what makes it a test of the attribute plumbing specifically:
 * a SurfaceScalar that silently fell back to the flat material colour would light both
 * sides the same hue.
 */
void
test_surface_scalar_attribute(carbon::System& sys,
                              const carbon::Stream& stream) {
  const carbon::int64_3d extent{96, 96, 96};
  const float c = 0.5F * static_cast<float>(extent.x - 1);
  const float radius = 0.3F * static_cast<float>(extent.x);

  carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);

  const carbon::Dense dense(sys, extent, stencil, carbon::BoundaryMode::Open, {}, 1.0F);
  auto sdf = dense.newField<float, 1>(carbon::MemorySide::Device);
  auto attr = dense.newField<float, 1>(carbon::MemorySide::Device);

  build_unit_sphere_sdf_kernel(dense, sdf, c, c, c, radius).launch(stream);
  const carbon::Kernel fill_attr = dense.make_kernel([&attr, c](carbon::KernelLoader& loader) {
    auto out = loader.write(attr);
    return [out, c] CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> void {
      out.write(span, idx, 0, static_cast<float>(span.global_location(idx).x) - c);
    };
  });
  fill_attr.launch(stream);
  stream.wait();
  sync_field_halos(sdf, stream);
  sync_field_halos(attr, stream);

  constexpr int k_w = 192;
  constexpr int k_h = 192;
  // Viewed down -z with +x to the right, so the ramp runs across the image.
  const carbon::Camera camera = carbon::Camera::frame_bounds(
      {c - radius, c - radius, c - radius}, {c + radius, c + radius, c + radius},
      {0.0F, 0.0F, 1.0F}, {0.0F, 1.0F, 0.0F}, 0.785398F, k_w, k_h);

  carbon::RayMarchOptions opts = blinn_sphere_opts();
  opts.shading = carbon::RayMarchShading::SurfaceScalar;
  opts.colormap = carbon::Colormap::CoolWarm;
  opts.value_min = -radius;
  opts.value_max = radius;
  opts.material_color = {1.0F, 1.0F, 1.0F};

  carbon::RayMarchImage image(sys, k_w, k_h);
  image.clear(stream);
  const carbon::Kernel march = carbon::make_ray_march_kernel(
      dense, image, camera, opts,
      [&sdf](carbon::KernelLoader& loader) {
        const auto f = loader.read(sdf);
        return [f] CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
          return f.read(span, idx, 0);
        };
      },
      [&attr](carbon::KernelLoader& loader) {
        const auto f = loader.read(attr);
        return [f] CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
          return f.read(span, idx, 0);
        };
      });
  march.launch(stream);
  image.combine(stream);
  image.dist_combine();

  // Average the two halves of the middle band, over pixels the surface actually covered.
  const auto& red = image.host_color_r();
  const auto& blue = image.host_color_b();
  const auto& trans = image.host_transmittance();
  double left_rb = 0.0;
  double right_rb = 0.0;
  int left_n = 0;
  int right_n = 0;
  for (int y = k_h / 4; y < (3 * k_h) / 4; ++y) {
    for (int x = 0; x < k_w; ++x) {
      const size_t o = static_cast<size_t>(y) * static_cast<size_t>(k_w) + static_cast<size_t>(x);
      if (trans[o] > 0.5F) {
        continue;
      }
      const double rb = static_cast<double>(red[o]) - static_cast<double>(blue[o]);
      if (x < k_w / 2) {
        left_rb += rb;
        ++left_n;
      } else {
        right_rb += rb;
        ++right_n;
      }
    }
  }
  if (left_n < 100 || right_n < 100) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_ray_march surface_scalar_attribute: too few covered pixels (left {}, right "
        "{}); the surface did not render",
        left_n, right_n);
  }

  const double left_mean = left_rb / static_cast<double>(left_n);
  const double right_mean = right_rb / static_cast<double>(right_n);
  if (!(left_mean < 0.0) || !(right_mean > 0.0)) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_ray_march surface_scalar_attribute: expected the attribute ramp to tint the "
        "low-x side blue and the high-x side red (mean r-b: left {}, right {})",
        left_mean, right_mean);
  }
}

void
test_frustum_miss(carbon::System& sys,
                  const carbon::Stream& stream) {
  const carbon::int64_3d extent{32, 32, 32};
  carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);

  const carbon::Dense dense(sys, extent, stencil, carbon::BoundaryMode::Open, {}, 1.0F);
  auto field = dense.newField<float, 1>(carbon::MemorySide::Device);
  const auto fill = build_unit_density_kernel(dense, field);
  fill.launch(stream);
  stream.wait();

  constexpr int k_w = 64;
  constexpr int k_h = 64;
  const carbon::Camera camera =
      carbon::Camera::perspective({32.0F, 32.0F, -200.0F}, {32.0F, 32.0F, -400.0F},
                                  {0.0F, 1.0F, 0.0F}, 1.0F, 0.1F, 500.0F, k_w, k_h);

  carbon::RayMarchImage image(sys, k_w, k_h);
  image.clear(stream);

  const auto march = carbon::make_ray_march_kernel(
      dense, image, camera, ray_march_opts(), [&field](carbon::KernelLoader& loader) {
        const auto f = loader.read(field);
        return [f] CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
          return f.read(span, idx, 0);
        };
      });

  march.launch(stream);
  image.combine(stream);
  image.dist_combine();

  if (frame_sum(image) != 0.0) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_ray_march frustum_miss: expected zero frame sum, got {}", frame_sum(image));
  }
}

}  // namespace

auto
main() -> int {
  carbon::init();
  carbon::System sys(carbon::get_all_devices());
  if (sys.gpu_group().size() == 0U) {
    CARBON_LOG_ERROR("test_render_ray_march: no devices visible");
    return EXIT_FAILURE;
  }

  const carbon::Stream stream = sys.stream();

  {
    const carbon::int64_3d extent{512, 512, 512};
    const float cx = 0.5F * static_cast<float>(extent.x - 1);
    const float radius = 0.35F * static_cast<float>(extent.x);
    const carbon::Camera camera = carbon::Camera::frame_bounds(
        {cx - radius, cx - radius, cx - radius}, {cx + radius, cx + radius, cx + radius},
        {0.0F, -1.0F, 0.0F}, {0.0F, 0.0F, 1.0F}, 0.785398F, 3840, 2160);
    test_camera_ray_fan(camera);
  }

  test_sphere_visible(sys, stream);
  test_sphere_visible_large(sys, stream);
  test_multi_gpu_center_row(sys, stream);
  test_surface_blinn_lit_side(sys, stream);
  test_volume_gradient_mode(sys, stream);
  test_hit_depth(sys, stream);
  test_speed_deviation();
  test_surface_scalar_attribute(sys, stream);
  test_scalar_field_attribute(sys, stream);
  test_scalar_field_threshold(sys, stream);
  test_colormap_monotonic();
  test_slab_composite(sys, stream);
  test_surface_not_filled(sys, stream);
  test_frustum_miss(sys, stream);

  return EXIT_SUCCESS;
}
