// Unit tests for surfel splat surface rendering on Dense grids: silhouette, hole-free
// coverage, depth/normal fidelity, multi-GPU bit-equality, and determinism.

#include <carbon/carbon.h>
#include <carbon/dense.h>
#include <carbon/halo.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/render.h>
#include <carbon/stencil.h>
#include <carbon/system.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace {

constexpr float k_band = 1.5F;
constexpr float k_splat_radius_px = 3.0F;

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

void
sync_field_halos(carbon::Dense::Field<float,
                                      1>& field,
                 const carbon::Stream& stream) {
  const carbon::Halo<float, 1> halo(field);
  halo(stream);
  stream.wait();
}

[[nodiscard]] auto
render_sphere_surfel(carbon::System& sys,
                     const carbon::Stream& stream,
                     const carbon::Dense& dense,
                     carbon::Dense::Field<float,
                                          1>& sdf,
                     float cx,
                     float cy,
                     float cz,
                     float /*radius*/,
                     const carbon::Camera& camera) -> carbon::SurfelImage {
  carbon::SurfelImage image(sys, camera.width(), camera.height());
  image.clear(stream);

  const auto splat = carbon::make_surfel_kernel(
      dense, image, camera,
      [&sdf, cx, cy, cz](carbon::KernelLoader& loader) {
        const auto f = loader.read(sdf);
        return [f, cx, cy, cz]
               CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> carbon::SurfelSample {
                 const carbon::int64_3d gc = span.global_location(idx);
                 const float sp = span.spacing();
                 const carbon::float32_3d origin = span.origin();
                 const float px = origin.x + (static_cast<float>(gc.x) + 0.5F) * sp - cx;
                 const float py = origin.y + (static_cast<float>(gc.y) + 0.5F) * sp - cy;
                 const float pz = origin.z + (static_cast<float>(gc.z) + 0.5F) * sp - cz;
                 const float len = std::sqrt(px * px + py * py + pz * pz);
                 carbon::SurfelSample sample{};
                 sample.sdf = f.read(span, idx, 0);
                 if (len > 1.0e-6F) {
                   sample.normal = {px / len, py / len, pz / len};
                 } else {
                   sample.normal = {0.0F, 0.0F, 1.0F};
                 }
                 return sample;
               };
      },
      k_band, k_splat_radius_px);
  splat.launch(stream);
  image.combine(stream);
  stream.wait();
  return image;
}

[[nodiscard]] auto
normal_angle_deg(const carbon::float32_3d& a,
                 const carbon::float32_3d& b) -> float {
  const float dot = std::clamp(a.x * b.x + a.y * b.y + a.z * b.z, -1.0F, 1.0F);
  return std::acos(dot) * (180.0F / static_cast<float>(M_PI));
}

// A surfel only ever represents the patch of surface inside its own footprint. If its depth is
// extrapolated beyond that patch (the tangent plane grazes the view ray) it can win pixels it does
// not cover and punch through nearer geometry, so compare the resolved depth against the analytic
// ray / sphere intersection.
void
test_depth_matches_analytic_sphere(carbon::System& sys,
                                   const carbon::Stream& stream) {
  const carbon::int64_3d extent{96, 96, 96};
  const float cx = 0.5F * static_cast<float>(extent.x - 1);
  const float cy = 0.5F * static_cast<float>(extent.y - 1);
  const float cz = 0.5F * static_cast<float>(extent.z - 1);
  const float radius = 0.30F * static_cast<float>(extent.x);

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

  constexpr int k_w = 160;
  constexpr int k_h = 160;
  const carbon::float32_3d bounds_min{cx - radius, cy - radius, cz - radius};
  const carbon::float32_3d bounds_max{cx + radius, cy + radius, cz + radius};
  // Oblique view: normals sweep every octant, so grazing footprints are exercised.
  const carbon::Camera camera = carbon::Camera::frame_bounds(
      bounds_min, bounds_max, {-0.58F, -0.38F, -0.72F}, {0.0F, 0.0F, 1.0F}, 0.611F, k_w, k_h);

  const carbon::SurfelImage image =
      render_sphere_surfel(sys, stream, dense, sdf, cx, cy, cz, radius, camera);

  const carbon::float32_3d centre{cx, cy, cz};
  float worst_dev = 0.0F;
  int compared = 0;

  for (int py = 0; py < k_h; ++py) {
    for (int px = 0; px < k_w; ++px) {
      const size_t off =
          static_cast<size_t>(py) * static_cast<size_t>(k_w) + static_cast<size_t>(px);
      if (!image.has_fragment(off)) {
        continue;
      }
      carbon::float32_3d origin{};
      carbon::float32_3d dir{};
      if (!camera.ray_through_pixel(px, py, origin, dir)) {
        continue;
      }
      const float ox = origin.x - centre.x;
      const float oy = origin.y - centre.y;
      const float oz = origin.z - centre.z;
      const float b = ox * dir.x + oy * dir.y + oz * dir.z;
      const float c = ox * ox + oy * oy + oz * oz - radius * radius;
      const float disc = b * b - c;
      // Skip the silhouette annulus: there the analytic hit is a grazing tangent and one cell of
      // discretization moves it by many cells along the ray.
      if (disc <= 0.09F * radius * radius) {
        continue;
      }
      const float t_front = -b - std::sqrt(disc);
      if (t_front <= 0.0F) {
        continue;
      }
      worst_dev = std::max(worst_dev, std::abs(image.host_depth()[off] - t_front));
      ++compared;
    }
  }

  if (compared < 500) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_surfel analytic depth: too few comparable pixels ({})", compared);
  }
  // The surface is a discretized SDF snapped along a central-difference normal, so a couple of
  // cells of slack is expected; bleed-through shows up as tens of cells.
  if (worst_dev > 4.0F) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_surfel analytic depth: worst deviation {} exceeds 4 cells over {} pixels",
        worst_dev, compared);
  }
}

void
test_silhouette_and_holes(carbon::System& sys,
                          const carbon::Stream& stream) {
  const carbon::int64_3d extent{64, 64, 64};
  const float cx = 0.5F * static_cast<float>(extent.x - 1);
  const float cy = 0.5F * static_cast<float>(extent.y - 1);
  const float cz = 0.5F * static_cast<float>(extent.z - 1);
  const float radius = 0.30F * static_cast<float>(extent.x);

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
  const carbon::float32_3d bounds_min{cx - radius, cy - radius, cz - radius};
  const carbon::float32_3d bounds_max{cx + radius, cy + radius, cz + radius};
  const carbon::Camera camera = carbon::Camera::frame_bounds(
      bounds_min, bounds_max, {0.0F, -1.0F, 0.0F}, {0.0F, 0.0F, 1.0F}, 0.785398F, k_w, k_h);

  const carbon::SurfelImage image =
      render_sphere_surfel(sys, stream, dense, sdf, cx, cy, cz, radius, camera);

  const int cx_px = k_w / 2;
  const int cy_px = k_h / 2;
  const int outer_r = static_cast<int>(0.42F * static_cast<float>(std::min(k_w, k_h)));
  const int inner_r = static_cast<int>(0.24F * static_cast<float>(std::min(k_w, k_h)));

  int inside_hits = 0;
  int inside_total = 0;
  int outside_hits = 0;
  int outside_total = 0;

  for (int py = 0; py < k_h; ++py) {
    for (int px = 0; px < k_w; ++px) {
      const int dx = px - cx_px;
      const int dy = py - cy_px;
      const int r2 = dx * dx + dy * dy;
      const size_t off =
          static_cast<size_t>(py) * static_cast<size_t>(k_w) + static_cast<size_t>(px);
      const bool hit = image.has_fragment(off);
      if (r2 <= inner_r * inner_r) {
        ++inside_total;
        if (hit) {
          ++inside_hits;
        }
      }
      if (r2 >= outer_r * outer_r) {
        ++outside_total;
        if (hit) {
          ++outside_hits;
        }
      }
    }
  }

  if (inside_total == 0 || inside_hits < inside_total) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_surfel silhouette/holes: expected full coverage inside inner disk ({}/{} "
        "pixels hit)",
        inside_hits, inside_total);
  }
  if (outside_hits > outside_total / 20) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_surfel silhouette: too many hits outside outer disk ({}/{})", outside_hits,
        outside_total);
  }
}

void
test_depth_and_normals(carbon::System& sys,
                       const carbon::Stream& stream) {
  const carbon::int64_3d extent{64, 64, 64};
  const float cx = 0.5F * static_cast<float>(extent.x - 1);
  const float cy = 0.5F * static_cast<float>(extent.y - 1);
  const float cz = 0.5F * static_cast<float>(extent.z - 1);
  const float radius = 0.30F * static_cast<float>(extent.x);

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
  const carbon::float32_3d bounds_min{cx - radius, cy - radius, cz - radius};
  const carbon::float32_3d bounds_max{cx + radius, cy + radius, cz + radius};
  const carbon::Camera camera = carbon::Camera::frame_bounds(
      bounds_min, bounds_max, {0.0F, -1.0F, 0.0F}, {0.0F, 0.0F, 1.0F}, 0.785398F, k_w, k_h);

  const carbon::SurfelImage image =
      render_sphere_surfel(sys, stream, dense, sdf, cx, cy, cz, radius, camera);

  const int cx_px = k_w / 2;
  const int cy_px = k_h / 2;
  const int sample_r = 8;
  double depth_acc = 0.0;
  double normal_acc = 0.0;
  int count = 0;

  for (int dy = -sample_r; dy <= sample_r; ++dy) {
    for (int dx = -sample_r; dx <= sample_r; ++dx) {
      const int px = cx_px + dx;
      const int py = cy_px + dy;
      if (px < 0 || py < 0 || px >= k_w || py >= k_h) {
        continue;
      }
      const size_t off =
          static_cast<size_t>(py) * static_cast<size_t>(k_w) + static_cast<size_t>(px);
      if (!image.has_fragment(off)) {
        continue;
      }
      const float depth = image.host_depth()[off];
      const carbon::float32_3d n = image.host_normal()[off];
      const carbon::float32_3d expected_n{0.0F, -1.0F, 0.0F};
      depth_acc += static_cast<double>(depth);
      normal_acc += static_cast<double>(normal_angle_deg(n, expected_n));
      ++count;
    }
  }

  if (count < 10) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_surfel depth/normals: too few centre samples ({})", count);
  }

  const float mean_depth = static_cast<float>(depth_acc / static_cast<double>(count));
  if (mean_depth <= 0.0F || !std::isfinite(mean_depth)) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_surfel depth: expected positive finite ray depth, got mean {}", mean_depth);
  }

  double outer_depth_acc = 0.0;
  int outer_count = 0;
  const int outer_sample_r = sample_r + 6;
  for (int dy = -outer_sample_r; dy <= outer_sample_r; ++dy) {
    for (int dx = -outer_sample_r; dx <= outer_sample_r; ++dx) {
      const int r2 = dx * dx + dy * dy;
      if (r2 < (sample_r + 2) * (sample_r + 2) || r2 > outer_sample_r * outer_sample_r) {
        continue;
      }
      const int px = cx_px + dx;
      const int py = cy_px + dy;
      if (px < 0 || py < 0 || px >= k_w || py >= k_h) {
        continue;
      }
      const size_t off =
          static_cast<size_t>(py) * static_cast<size_t>(k_w) + static_cast<size_t>(px);
      if (!image.has_fragment(off)) {
        continue;
      }
      outer_depth_acc += static_cast<double>(image.host_depth()[off]);
      ++outer_count;
    }
  }
  if (outer_count > 0) {
    const float outer_mean = static_cast<float>(outer_depth_acc / static_cast<double>(outer_count));
    if (mean_depth >= outer_mean) {
      carbon::log::throw_with_log<std::runtime_error>(
          "test_render_surfel depth: expected centre closer than limb (centre {} limb {})",
          mean_depth, outer_mean);
    }
  }

  const float mean_normal_err = static_cast<float>(normal_acc / static_cast<double>(count));
  if (mean_normal_err > 25.0F) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_surfel normals: mean angular error {} deg exceeds 25 deg", mean_normal_err);
  }
}

void
test_determinism(carbon::System& sys,
                 const carbon::Stream& stream) {
  const carbon::int64_3d extent{48, 48, 48};
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

  constexpr int k_w = 96;
  constexpr int k_h = 96;
  const carbon::float32_3d bounds_min{cx - radius, cy - radius, cz - radius};
  const carbon::float32_3d bounds_max{cx + radius, cy + radius, cz + radius};
  const carbon::Camera camera = carbon::Camera::frame_bounds(
      bounds_min, bounds_max, {0.0F, -1.0F, 0.0F}, {0.0F, 0.0F, 1.0F}, 0.785398F, k_w, k_h);

  const carbon::SurfelImage a =
      render_sphere_surfel(sys, stream, dense, sdf, cx, cy, cz, radius, camera);
  const carbon::SurfelImage b =
      render_sphere_surfel(sys, stream, dense, sdf, cx, cy, cz, radius, camera);

  if (a.host_visibility() != b.host_visibility()) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_surfel determinism: visibility buffers differ between identical renders");
  }
}

void
test_multi_gpu_bit_equality(carbon::System& sys,
                            const carbon::Stream& stream) {
  if (sys.gpu_group().size() < 2U) {
    return;
  }

  const carbon::int64_3d extent{32, 32, 32};
  const float cx = 0.5F * static_cast<float>(extent.x - 1);
  const float cy = 0.5F * static_cast<float>(extent.y - 1);
  const float cz = 0.5F * static_cast<float>(extent.z - 1);
  const float radius = 0.30F * static_cast<float>(extent.x);

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

  constexpr int k_w = 96;
  constexpr int k_h = 96;
  const carbon::float32_3d bounds_min{cx - radius, cy - radius, cz - radius};
  const carbon::float32_3d bounds_max{cx + radius, cy + radius, cz + radius};
  const carbon::Camera camera = carbon::Camera::frame_bounds(
      bounds_min, bounds_max, {0.0F, -1.0F, 0.0F}, {0.0F, 0.0F, 1.0F}, 0.785398F, k_w, k_h);

  const carbon::SurfelImage multi =
      render_sphere_surfel(sys, stream, dense_multi, sdf_multi, cx, cy, cz, radius, camera);
  const carbon::SurfelImage single = render_sphere_surfel(sys_single, stream_single, dense_single,
                                                          sdf_single, cx, cy, cz, radius, camera);

  if (multi.host_visibility() != single.host_visibility()) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_surfel multi_gpu: visibility buffers differ from single-GPU reference");
  }
}

void
test_frustum_miss(carbon::System& sys,
                  const carbon::Stream& stream) {
  const carbon::int64_3d extent{32, 32, 32};
  const float cx = 0.5F * static_cast<float>(extent.x - 1);
  const float cy = 0.5F * static_cast<float>(extent.y - 1);
  const float cz = 0.5F * static_cast<float>(extent.z - 1);
  const float radius = 0.30F * static_cast<float>(extent.x);

  carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);

  const carbon::Dense dense(sys, extent, stencil, carbon::BoundaryMode::Open, {}, 1.0F);
  auto sdf = dense.newField<float, 1>(carbon::MemorySide::Device);
  const auto init = build_unit_sphere_sdf_kernel(dense, sdf, cx, cy, cz, radius);
  init.launch(stream);
  stream.wait();

  constexpr int k_w = 64;
  constexpr int k_h = 64;
  const carbon::Camera camera =
      carbon::Camera::perspective({32.0F, 32.0F, -200.0F}, {32.0F, 32.0F, -400.0F},
                                  {0.0F, 1.0F, 0.0F}, 1.0F, 0.1F, 500.0F, k_w, k_h);

  const carbon::SurfelImage image =
      render_sphere_surfel(sys, stream, dense, sdf, cx, cy, cz, radius, camera);

  for (unsigned long long key : image.host_visibility()) {
    if (key != carbon::SurfelImage::k_empty_key) {
      carbon::log::throw_with_log<std::runtime_error>(
          "test_render_surfel frustum_miss: expected empty frame");
    }
  }
}

// The packed key carries the normal as 16:16 octahedral coordinates. A decode that drops the
// lower-hemisphere sign silently collapses every nz < 0 normal onto the equator, which shades as
// a radial cone rather than a curved surface, so pin the round-trip over the whole sphere.
void
test_octahedral_normal_round_trip() {
  constexpr int k_theta_steps = 60;
  constexpr int k_phi_steps = 120;
  float worst_err_deg = 0.0F;

  for (int i = 0; i <= k_theta_steps; ++i) {
    for (int j = 0; j <= k_phi_steps; ++j) {
      const float theta =
          static_cast<float>(M_PI) * static_cast<float>(i) / static_cast<float>(k_theta_steps);
      const float phi =
          2.0F * static_cast<float>(M_PI) * static_cast<float>(j) / static_cast<float>(k_phi_steps);
      const carbon::float32_3d n{
          std::sin(theta) * std::cos(phi),
          std::sin(theta) * std::sin(phi),
          std::cos(theta),
      };
      const uint32_t enc = carbon::detail::encode_octahedral_normal(n.x, n.y, n.z);
      const carbon::float32_3d decoded = carbon::detail::decode_octahedral_normal(enc);
      worst_err_deg = std::max(worst_err_deg, normal_angle_deg(n, decoded));
    }
  }

  if (worst_err_deg > 1.0F) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_surfel octahedral round-trip: worst angular error {} deg exceeds 1 deg",
        worst_err_deg);
  }
}

// Both halves of the sortable-float key, including the negative range. Depth is positive by
// construction, so the negative branch of the decoder went unexercised until the attribute
// plane started carrying signed data through the same packing — and it was wrong.
void
test_sortable_float_round_trip() {
  const std::vector<float> values{0.0F,     -0.0F,     1.0F,     -1.0F,    3.75F, -3.75F,
                                  1.0e-7F,  -1.0e-7F,  1.0e7F,   -1.0e7F,  0.5F,  -2.5F,
                                  123.456F, -123.456F, 1.0e-30F, -1.0e-30F};
  for (const float v : values) {
    const uint32_t enc = carbon::detail::float_to_sortable_uint(v);
    const float back = carbon::detail::sortable_uint_to_float(enc);
    if (back != v) {
      carbon::log::throw_with_log<std::runtime_error>(
          "test_render_surfel sortable round-trip: {} encoded to {} decoded to {}", v, enc, back);
    }
  }

  // The point of the encoding is that unsigned order matches float order across the sign
  // change, which is what lets a single atomicMin resolve the nearest fragment.
  std::vector<float> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  for (size_t i = 1; i < sorted.size(); ++i) {
    if (sorted[i - 1] < sorted[i] && !(carbon::detail::float_to_sortable_uint(sorted[i - 1]) <
                                       carbon::detail::float_to_sortable_uint(sorted[i]))) {
      carbon::log::throw_with_log<std::runtime_error>(
          "test_render_surfel sortable round-trip: order not preserved between {} and {}",
          sorted[i - 1], sorted[i]);
    }
  }
}

}  // namespace

/**
 * The attribute plane has to report the value belonging to the fragment the pixel is
 * showing, which is only interesting when the surface is self-occluding: a sphere carrying
 * an attribute that varies along the view axis puts a near value and a far value on the same
 * pixel, and only the near one may survive. Reading back the front hemisphere's value
 * everywhere is what proves the two planes agree on a winner; if the attribute plane
 * tie-broke on its own it would take the smaller value, which here is the hidden back face.
 */
void
test_attribute_plane(carbon::System& sys,
                     const carbon::Stream& stream) {
  const carbon::int64_3d extent{96, 96, 96};
  const float cx = 0.5F * static_cast<float>(extent.x - 1);
  const float cy = 0.5F * static_cast<float>(extent.y - 1);
  const float cz = 0.5F * static_cast<float>(extent.z - 1);
  const float radius = 0.30F * static_cast<float>(extent.x);

  carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);

  const carbon::Dense dense(sys, extent, stencil, carbon::BoundaryMode::Open, {}, 1.0F);

  constexpr int k_w = 160;
  constexpr int k_h = 160;
  // Eye on the +z side, so the attribute (z - cz) splits cleanly into a visible near half
  // and a hidden far half. That orientation is what makes the check discriminating: an
  // attribute plane that tie-broke on its own would take the *smallest* value per pixel,
  // which is exactly the occluded back face.
  const carbon::Camera camera = carbon::Camera::frame_bounds(
      {cx - radius, cy - radius, cz - radius}, {cx + radius, cy + radius, cz + radius},
      {0.0F, 0.0F, 1.0F}, {0.0F, 1.0F, 0.0F}, 0.611F, k_w, k_h);

  carbon::SurfelImage image(sys, k_w, k_h, /*with_attribute=*/true);
  image.clear(stream);
  // Analytic SDF and normal at the cell centre rather than the sampled field, so the snapped
  // surfel lands exactly on the sphere. The pointwise identity below is then limited only by
  // the normal's 16:16 octahedral quantization, which is what makes it a tight check; a
  // sampled SDF would leave half a cell of slack that has nothing to do with the plumbing
  // under test.
  const carbon::Kernel splat = carbon::make_surfel_kernel(
      dense, image, camera,
      [cx, cy, cz, radius](carbon::KernelLoader& /*loader*/) {
        return [cx, cy, cz, radius]
               CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> carbon::SurfelSample {
                 const carbon::int64_3d gc = span.global_location(idx);
                 const float sp = span.spacing();
                 const carbon::float32_3d origin = span.origin();
                 const float px = origin.x + (static_cast<float>(gc.x) + 0.5F) * sp - cx;
                 const float py = origin.y + (static_cast<float>(gc.y) + 0.5F) * sp - cy;
                 const float pz = origin.z + (static_cast<float>(gc.z) + 0.5F) * sp - cz;
                 const float len = std::sqrt(px * px + py * py + pz * pz);
                 carbon::SurfelSample sample{};
                 sample.sdf = len - radius;
                 sample.normal = (len > 1.0e-6F) ? carbon::float32_3d{px / len, py / len, pz / len}
                                                 : carbon::float32_3d{0.0F, 0.0F, 1.0F};
                 // The z offset of the point the surfel is snapped to, not of the cell
                 // centre: splat_at moves the surfel onto the isosurface along its normal.
                 sample.attribute = pz - sample.sdf * sample.normal.z;
                 return sample;
               };
      },
      k_band, k_splat_radius_px);
  splat.launch(stream);
  image.combine(stream);
  stream.wait();

  const auto& attribute = image.host_attribute();
  int covered = 0;
  int back_face = 0;
  int tied = 0;
  double worst = 0.0;
  for (size_t i = 0; i < image.pixel_count(); ++i) {
    if (!image.has_fragment(i)) {
      continue;
    }
    ++covered;
    if (attribute[i] < 0.0F) {
      ++back_face;
    }
    // The attribute is the surfel's own z offset, and the normal's z component is that
    // offset over the radius: the two must agree, which pins the pairing per pixel rather
    // than just in aggregate.
    const double expected = static_cast<double>(image.host_normal()[i].z) * radius;
    const double err = std::abs(static_cast<double>(attribute[i]) - expected);
    if (err > 0.05) {
      ++tied;
    }
    worst = std::max(worst, err);
  }

  if (covered < 1000) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_surfel attribute_plane: only {} covered pixels; the sphere did not render",
        covered);
  }
  if (back_face != 0) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_surfel attribute_plane: {} of {} pixels report a back-face attribute; the "
        "attribute plane is not selecting the same fragment as the visibility plane",
        back_face, covered);
  }
  // Both sides describe the same snapped point, so away from depth ties the only slack is
  // the 16:16 octahedral quantization of the normal scaled by the radius — well under 0.05
  // here. The exception is the documented one: where two fragments land at bit-identical
  // depth the planes tie-break independently and can pair a normal with another fragment's
  // attribute. That happens on the polar cap facing the camera, where the surface is most
  // nearly perpendicular to the view and neighbouring surfels' tangent planes agree to the
  // last bit; measured at 17 of 5189 covered pixels. Bound both the incidence and the size,
  // so a genuine plumbing regression (which breaks every pixel, not a third of a percent)
  // still fails loudly.
  if (tied > covered / 100) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_surfel attribute_plane: {} of {} pixels disagree beyond quantization; "
        "expected only rare exact depth ties",
        tied, covered);
  }
  if (worst > 1.0) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_surfel attribute_plane: attribute and normal disagree by up to {} (expected "
        "attr == normal.z * radius, radius {})",
        worst, radius);
  }

  // Without the plane the accessor must fail loudly rather than hand back zeros.
  carbon::SurfelImage plain(sys, 8, 8);
  bool threw = false;
  try {
    (void)plain.host_attribute();
  } catch (const std::logic_error&) {
    threw = true;
  }
  if (!threw) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_surfel attribute_plane: host_attribute must throw without with_attribute");
  }
}

auto
main() -> int {
  carbon::init();
  carbon::System sys(carbon::get_all_devices());
  if (sys.gpu_group().size() == 0U) {
    CARBON_LOG_ERROR("test_render_surfel: no devices visible");
    return EXIT_FAILURE;
  }

  const carbon::Stream stream = sys.stream();

  test_octahedral_normal_round_trip();
  test_sortable_float_round_trip();
  test_depth_matches_analytic_sphere(sys, stream);
  test_silhouette_and_holes(sys, stream);
  test_depth_and_normals(sys, stream);
  test_determinism(sys, stream);
  test_multi_gpu_bit_equality(sys, stream);
  test_frustum_miss(sys, stream);
  test_attribute_plane(sys, stream);

  return EXIT_SUCCESS;
}
