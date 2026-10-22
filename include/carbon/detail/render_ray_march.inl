// detail/render_ray_march.inl — ray-march API for <carbon/render.h>.

#include <carbon/detail/dense/partition.h>
#include <carbon/detail/ray_march_kernel_impl.h>
#include <carbon/detail/render_sample.h>

#include <type_traits>
#include <utility>

namespace carbon {
namespace detail {

[[nodiscard]] CARBON_HD constexpr auto
ray_inf() noexcept -> float {
#ifdef __CUDA_ARCH__
  return __int_as_float(0x7f800000);
#elif defined(__HIP_DEVICE_COMPILE__)
  return __int_as_float(0x7f800000);
#else
  return std::numeric_limits<float>::infinity();
#endif
}

struct RayMarchSegment {
  float color_r{0.0F};
  float color_g{0.0F};
  float color_b{0.0F};
  float trans{1.0F};
  /** Ray parameter of the first sample that contributed opacity; @c inf if none did. */
  float depth{std::numeric_limits<float>::infinity()};
};

[[nodiscard]] CARBON_HD inline auto
intersect_ray_aabb(float ox,
                   float oy,
                   float oz,
                   float dx,
                   float dy,
                   float dz,
                   float bmin_x,
                   float bmin_y,
                   float bmin_z,
                   float bmax_x,
                   float bmax_y,
                   float bmax_z,
                   float& t_enter,
                   float& t_exit) noexcept -> bool {
  t_enter = 0.0F;
  t_exit = detail::ray_inf();

  auto clip_axis = [&](float o, float d, float bmin, float bmax) -> bool {
    if (std::abs(d) > 1.0e-8F) {
      float t0 = (bmin - o) / d;
      float t1 = (bmax - o) / d;
      if (t0 > t1) {
        const float tmp = t0;
        t0 = t1;
        t1 = tmp;
      }
      t_enter = (t0 > t_enter) ? t0 : t_enter;
      t_exit = (t1 < t_exit) ? t1 : t_exit;
      return t_enter < t_exit;
    }
    return o >= bmin && o < bmax;
  };

  if (!clip_axis(ox, dx, bmin_x, bmax_x)) {
    return false;
  }
  if (!clip_axis(oy, dy, bmin_y, bmax_y)) {
    return false;
  }
  if (!clip_axis(oz, dz, bmin_z, bmax_z)) {
    return false;
  }
  return t_enter < t_exit;
}

[[nodiscard]] CARBON_HD inline auto
intersect_ray_z_slab(float oz,
                     float dz,
                     float slab_z0,
                     float slab_z1,
                     float& t_enter,
                     float& t_exit) noexcept -> bool {
  if (std::abs(dz) > 1.0e-8F) {
    const float t0 = (slab_z0 - oz) / dz;
    const float t1 = (slab_z1 - oz) / dz;
    t_enter = (t0 < t1) ? t0 : t1;
    t_exit = (t0 > t1) ? t0 : t1;
    return t_enter < t_exit;
  }
  return oz >= slab_z0 && oz < slab_z1;
}

inline auto
composite_segments_front_to_back(std::vector<RayMarchSegment>& segments,
                                 float& out_r,
                                 float& out_g,
                                 float& out_b,
                                 float& out_trans) -> void {
  std::sort(segments.begin(), segments.end(),
            [](const RayMarchSegment& a, const RayMarchSegment& b) {
              return a.depth < b.depth;
            });
  float r_acc = 0.0F;
  float g_acc = 0.0F;
  float b_acc = 0.0F;
  float t_acc = 1.0F;
  for (const RayMarchSegment& seg : segments) {
    if (!std::isfinite(seg.depth)) {
      continue;
    }
    r_acc += t_acc * seg.color_r;
    g_acc += t_acc * seg.color_g;
    b_acc += t_acc * seg.color_b;
    t_acc *= seg.trans;
  }
  out_r = r_acc;
  out_g = g_acc;
  out_b = b_acc;
  out_trans = t_acc;
}

[[nodiscard]] CARBON_HD inline auto
luminance(float r,
          float g,
          float b) noexcept -> float {
  return 0.299F * r + 0.587F * g + 0.114F * b;
}

CARBON_HD inline auto
shade_blinn(float nx,
            float ny,
            float nz,
            float lx,
            float ly,
            float lz,
            float lx2,
            float ly2,
            float lz2,
            float fill_diffuse,
            float vx,
            float vy,
            float vz,
            float ambient,
            float diffuse,
            float specular,
            float shininess,
            float mat_r,
            float mat_g,
            float mat_b,
            float& out_r,
            float& out_g,
            float& out_b) noexcept -> void {
  const float ndotl = nx * lx + ny * ly + nz * lz;
  const float diff = (ndotl > 0.0F) ? ndotl : 0.0F;
  const float ndotl2 = nx * lx2 + ny * ly2 + nz * lz2;
  const float diff2 = (fill_diffuse > 0.0F && ndotl2 > 0.0F) ? ndotl2 : 0.0F;
  const float hx = lx + vx;
  const float hy = ly + vy;
  const float hz = lz + vz;
  const float hlen = std::sqrt(hx * hx + hy * hy + hz * hz);
  float spec = 0.0F;
  if (hlen > 1.0e-8F) {
    const float inv_h = 1.0F / hlen;
    const float ndoth = nx * hx * inv_h + ny * hy * inv_h + nz * hz * inv_h;
    if (ndoth > 0.0F) {
      spec = std::pow(ndoth, shininess);
    }
  }
  const float shade = ambient + diffuse * diff + fill_diffuse * diff2 + specular * spec;
  out_r = shade * mat_r;
  out_g = shade * mat_g;
  out_b = shade * mat_b;
}

CARBON_HD inline auto
shade_volume_gradient(float nx,
                      float ny,
                      float nz,
                      float lx,
                      float ly,
                      float lz,
                      float lx2,
                      float ly2,
                      float lz2,
                      float fill_diffuse,
                      float ambient,
                      float diffuse,
                      float mat_r,
                      float mat_g,
                      float mat_b,
                      float& out_r,
                      float& out_g,
                      float& out_b) noexcept -> void {
  const float ndotl = nx * lx + ny * ly + nz * lz;
  const float diff = std::abs(ndotl);
  const float ndotl2 = nx * lx2 + ny * ly2 + nz * lz2;
  const float diff2 = (fill_diffuse > 0.0F) ? std::abs(ndotl2) : 0.0F;
  const float shade = ambient + diffuse * diff + fill_diffuse * diff2;
  out_r = shade * mat_r;
  out_g = shade * mat_g;
  out_b = shade * mat_b;
}

}  // namespace detail

namespace render {

// FieldSample / FieldGradientSample / trilinear_detail now live in
// <carbon/detail/render_sample.h> so all four grid families share one implementation.

template <typename Span,
          typename UserFn>
[[nodiscard]] CARBON_HD inline auto
sample_density_field(const Span& span,
                     UserFn user_fn,
                     float32_3d world) -> FieldSample {
  const float sp = span.spacing();
  const float32_3d origin = span.origin();
  const int64_3d ge = span.global_extent();

  const float fx = (world.x - origin.x) / sp;
  const float fy = (world.y - origin.y) / sp;
  const float fz = (world.z - origin.z) / sp;

  const int ix0 = static_cast<int>(std::floor(fx));
  const int iy0 = static_cast<int>(std::floor(fy));
  const int iz0 = static_cast<int>(std::floor(fz));

  const int64_t z_storage_lo = span.z_offset();
  const int64_t z_storage_hi = span.z_offset() + span.haloed_extent().z;

  return trilinear_detail::sample_clamped(
      ix0, iy0, iz0, fx - static_cast<float>(ix0), fy - static_cast<float>(iy0),
      fz - static_cast<float>(iz0), ge.x, ge.y, z_storage_lo, z_storage_hi,
      [&](int gx, int gy, int gz) -> float {
        const auto idx = span.make_idx(static_cast<int64_t>(gx), static_cast<int64_t>(gy),
                                       static_cast<int64_t>(gz) - span.z_offset());
        return user_fn(span, idx);
      });
}

template <typename T,
          int Components,
          typename Span>
CARBON_HD inline auto
sample_field_trilinear(const detail::dense::Partition<T,
                                                      Components>& partition,
                       const Span& span,
                       float32_3d world) -> float {
  const float sp = span.spacing();
  const float32_3d origin = span.origin();
  const int64_3d ge = span.global_extent();

  const float fx = (world.x - origin.x) / sp;
  const float fy = (world.y - origin.y) / sp;
  const float fz = (world.z - origin.z) / sp;

  const int ix0 = static_cast<int>(std::floor(fx));
  const int iy0 = static_cast<int>(std::floor(fy));
  const int iz0 = static_cast<int>(std::floor(fz));

  const int64_t z_storage_lo = span.z_offset();
  const int64_t z_storage_hi = span.z_offset() + span.haloed_extent().z;

  const FieldSample sample = trilinear_detail::sample_clamped(
      ix0, iy0, iz0, fx - static_cast<float>(ix0), fy - static_cast<float>(iy0),
      fz - static_cast<float>(iz0), ge.x, ge.y, z_storage_lo, z_storage_hi,
      [&](int gx, int gy, int gz) -> float {
        const auto idx = span.make_idx(static_cast<int64_t>(gx), static_cast<int64_t>(gy),
                                       static_cast<int64_t>(gz) - span.z_offset());
        return static_cast<float>(partition.read(span, idx, 0));
      });
  return sample.valid ? sample.value : 0.0F;
}

template <typename Span,
          typename UserFn>
CARBON_HD inline auto
sample_density_trilinear(const Span& span,
                         UserFn user_fn,
                         float32_3d world) -> float {
  const FieldSample sample = sample_density_field(span, user_fn, world);
  return sample.valid ? sample.value : 0.0F;
}

template <typename Span,
          typename UserFn>
[[nodiscard]] CARBON_HD inline auto
sample_density_gradient_field(const Span& span,
                              UserFn user_fn,
                              float32_3d world) -> FieldGradientSample {
  const float h = 0.5F * span.spacing();
  const float inv_2h = 1.0F / (2.0F * h);
  const float32_3d xp{world.x + h, world.y, world.z};
  const float32_3d xm{world.x - h, world.y, world.z};
  const float32_3d yp{world.x, world.y + h, world.z};
  const float32_3d ym{world.x, world.y - h, world.z};
  const float32_3d zp{world.x, world.y, world.z + h};
  const float32_3d zm{world.x, world.y, world.z - h};

  const FieldSample sx_p = sample_density_field(span, user_fn, xp);
  const FieldSample sx_m = sample_density_field(span, user_fn, xm);
  const FieldSample sy_p = sample_density_field(span, user_fn, yp);
  const FieldSample sy_m = sample_density_field(span, user_fn, ym);
  const FieldSample sz_p = sample_density_field(span, user_fn, zp);
  const FieldSample sz_m = sample_density_field(span, user_fn, zm);

  if (!sx_p.valid || !sx_m.valid || !sy_p.valid || !sy_m.valid || !sz_p.valid || !sz_m.valid) {
    return {};
  }

  return {
      {(sx_p.value - sx_m.value) * inv_2h, (sy_p.value - sy_m.value) * inv_2h,
       (sz_p.value - sz_m.value) * inv_2h},
      true,
  };
}

template <typename Span,
          typename UserFn>
CARBON_HD inline auto
sample_density_gradient_trilinear(const Span& span,
                                  UserFn user_fn,
                                  float32_3d world) -> float32_3d {
  const FieldGradientSample sample = sample_density_gradient_field(span, user_fn, world);
  return sample.valid ? sample.value : float32_3d{};
}

}  // namespace render

namespace detail {

/**
 * Per-pixel ray march over one GPU's share of a grid.
 *
 * Grid-family-agnostic: everything it needs about the data — world bounds to clip against,
 * the owned world-z band this GPU may integrate, the cell spacing, and the field value and
 * gradient at an arbitrary world point — comes from a @c Sampler (see
 * @c detail/render_sample.h ), so the same march serves Dense, DenseDisg, Block and
 * MultiResBlock. Samples outside the sampler's addressable region report invalid and are
 * skipped, which is also what makes a sparse grid's holes render as holes rather than as
 * zero-valued matter.
 */
/**
 * Stand-in attribute sampler for the single-loader @c make_ray_march_kernel overload: there
 * is no second field, so every attribute query misses and @c SurfaceScalar degrades to
 * @c SurfaceBlinn on the flat material colour.
 */
struct NoAttributeSampler {
  [[nodiscard]] CARBON_HD auto
  sample(float32_3d) const noexcept -> ::carbon::render::FieldSample {
    return {};
  }
};

template <typename Sampler, typename AttributeSampler = NoAttributeSampler>
struct RayMarchPixelFn {
  Sampler sampler;
  AttributeSampler attribute_sampler;
  float* seg_color_r;
  float* seg_color_g;
  float* seg_color_b;
  float* seg_trans;
  float* seg_depth;
  Camera camera;
  RayMarchOptions opts;
  int frame_w;
  int frame_h;

  CARBON_HD auto
  write_miss(size_t offset) const -> void {
    seg_color_r[offset] = 0.0F;
    seg_color_g[offset] = 0.0F;
    seg_color_b[offset] = 0.0F;
    seg_trans[offset] = 1.0F;
    seg_depth[offset] = detail::ray_inf();
  }

  CARBON_HD auto
  operator()(int px,
             int py) const -> void {
    const size_t offset =
        static_cast<size_t>(py) * static_cast<size_t>(frame_w) + static_cast<size_t>(px);

    float32_3d origin{};
    float32_3d dir{};
    if (!camera.ray_through_pixel(px, py, origin, dir)) {
      write_miss(offset);
      return;
    }

    const float sp = sampler.spacing();
    const float32_3d bmin = sampler.world_bounds_min();
    const float32_3d bmax = sampler.world_bounds_max();
    const float slab_z0 = sampler.owned_z_lo();
    const float slab_z1 = sampler.owned_z_hi();
    const float bmin_x = bmin.x;
    const float bmin_y = bmin.y;
    const float bmin_z = bmin.z;
    const float bmax_x = bmax.x;
    const float bmax_y = bmax.y;
    const float bmax_z = bmax.z;

    float clip_bmin_x = bmin_x;
    float clip_bmin_y = bmin_y;
    float clip_bmin_z = bmin_z;
    float clip_bmax_x = bmax_x;
    float clip_bmax_y = bmax_y;
    float clip_bmax_z = bmax_z;
    if (opts.use_clip_bounds) {
      clip_bmin_x = (opts.clip_bounds_min.x > bmin_x) ? opts.clip_bounds_min.x : bmin_x;
      clip_bmin_y = (opts.clip_bounds_min.y > bmin_y) ? opts.clip_bounds_min.y : bmin_y;
      clip_bmin_z = (opts.clip_bounds_min.z > bmin_z) ? opts.clip_bounds_min.z : bmin_z;
      clip_bmax_x = (opts.clip_bounds_max.x < bmax_x) ? opts.clip_bounds_max.x : bmax_x;
      clip_bmax_y = (opts.clip_bounds_max.y < bmax_y) ? opts.clip_bounds_max.y : bmax_y;
      clip_bmax_z = (opts.clip_bounds_max.z < bmax_z) ? opts.clip_bounds_max.z : bmax_z;
    }

    float t_aabb_enter = 0.0F;
    float t_aabb_exit = detail::ray_inf();
    if (!intersect_ray_aabb(origin.x, origin.y, origin.z, dir.x, dir.y, dir.z, clip_bmin_x,
                            clip_bmin_y, clip_bmin_z, clip_bmax_x, clip_bmax_y, clip_bmax_z,
                            t_aabb_enter, t_aabb_exit)) {
      write_miss(offset);
      return;
    }

    float t_slab_enter = 0.0F;
    float t_slab_exit = detail::ray_inf();
    if (!intersect_ray_z_slab(origin.z, dir.z, slab_z0, slab_z1, t_slab_enter, t_slab_exit)) {
      write_miss(offset);
      return;
    }

    float t_enter = (t_aabb_enter > t_slab_enter) ? t_aabb_enter : t_slab_enter;
    const float t_exit = (t_aabb_exit < t_slab_exit) ? t_aabb_exit : t_slab_exit;
    t_enter = (t_enter > 0.0F) ? t_enter : 0.0F;
    if (t_enter >= t_exit) {
      write_miss(offset);
      return;
    }

    const float step = (opts.step_size > 0.0F) ? opts.step_size : (0.5F * sp);
    // Snap the first sample onto the global t = k * step lattice. Each GPU only marches its own
    // z-slab, so starting at the slab's own entry restarts the step phase at every seam: the last
    // sample below the seam and the first above it end up closer together than @c step , the band
    // is integrated twice over that sliver, and the surface shows a darker line where it crosses a
    // slab boundary. Anchoring every slab to the same lattice makes the union of samples across
    // slabs exactly the set a single continuous march would visit.
    const float t_start = std::ceil(t_enter / step) * step;
    const float sigma = opts.absorption_scale;
    const float lx = opts.light_dir.x;
    const float ly = opts.light_dir.y;
    const float lz = opts.light_dir.z;
    const float lx2 = opts.light_dir2.x;
    const float ly2 = opts.light_dir2.y;
    const float lz2 = opts.light_dir2.z;
    const float fill_diffuse = opts.fill_diffuse;
    const float vx = -dir.x;
    const float vy = -dir.y;
    const float vz = -dir.z;
    const float mat_r = opts.material_color.x;
    const float mat_g = opts.material_color.y;
    const float mat_b = opts.material_color.z;

    float t_acc = 1.0F;
    float r_acc = 0.0F;
    float g_acc = 0.0F;
    float b_acc = 0.0F;
    float t_hit = detail::ray_inf();
    int steps = 0;
    for (float t = t_start; t < t_exit && steps < opts.max_steps; t += step, ++steps) {
      const float32_3d world{
          origin.x + dir.x * t,
          origin.y + dir.y * t,
          origin.z + dir.z * t,
      };

      float sample_r = 0.0F;
      float sample_g = 0.0F;
      float sample_b = 0.0F;
      float rho = 0.0F;
      float alpha = 0.0F;

      if (opts.shading == RayMarchShading::Emission) {
        const ::carbon::render::FieldSample s = sampler.sample(world);
        if (!s.valid) {
          continue;
        }
        rho = s.value;
        alpha = 1.0F - std::exp(-sigma * rho * step);
        sample_r = rho * mat_r;
        sample_g = rho * mat_g;
        sample_b = rho * mat_b;
      } else if (opts.shading == RayMarchShading::VolumeGradient) {
        const ::carbon::render::FieldSample s = sampler.sample(world);
        if (!s.valid || s.value <= 0.0F) {
          continue;
        }
        rho = s.value;
        const ::carbon::render::FieldGradientSample grad_sample = sampler.gradient(world);
        if (!grad_sample.valid) {
          continue;
        }
        const float32_3d grad = grad_sample.value;
        float gn = std::sqrt(grad.x * grad.x + grad.y * grad.y + grad.z * grad.z);
        if (gn > 1.0e-8F) {
          const float inv_gn = 1.0F / gn;
          shade_volume_gradient(grad.x * inv_gn, grad.y * inv_gn, grad.z * inv_gn, lx, ly, lz, lx2,
                                ly2, lz2, fill_diffuse, opts.ambient, opts.diffuse, mat_r, mat_g,
                                mat_b, sample_r, sample_g, sample_b);
        } else {
          sample_r = opts.ambient * mat_r;
          sample_g = opts.ambient * mat_g;
          sample_b = opts.ambient * mat_b;
        }
        alpha = 1.0F - std::exp(-sigma * rho * step);
      } else if (opts.shading == RayMarchShading::ScalarField ||
                 opts.shading == RayMarchShading::ScalarFieldAttribute) {
        const ::carbon::render::FieldSample sample = sampler.sample(world);
        if (!sample.valid || sample.value < opts.value_threshold) {
          continue;
        }
        const float s = sample.value;
        // Opacity always tracks the primary field; only what the colormap reads changes.
        float colour_value = s;
        if (opts.shading == RayMarchShading::ScalarFieldAttribute) {
          const ::carbon::render::FieldSample attr = attribute_sampler.sample(world);
          if (!attr.valid) {
            continue;
          }
          colour_value = attr.value;
        }
        const float range = opts.value_max - opts.value_min;
        const float t = (range > 0.0F) ? ((colour_value - opts.value_min) / range) : 0.0F;
        const float32_3d c = ::carbon::render::colormap_eval(opts.colormap, t);
        sample_r = c.x;
        sample_g = c.y;
        sample_b = c.z;
        alpha = 1.0F - std::exp(-opts.opacity_scale * (s - opts.value_threshold) * step);
        rho = s;
      } else {
        const ::carbon::render::FieldSample field_sample = sampler.sample(world);
        if (!field_sample.valid) {
          continue;
        }
        const float field = field_sample.value;
        const float width = (opts.isosurface_width > 0.0F) ? opts.isosurface_width : sp;
        if (std::abs(field) > 4.0F * width) {
          continue;
        }
        const float band_arg = field / width;
        const float band = std::exp(-band_arg * band_arg);
        if (band < 0.001F) {
          continue;
        }
        alpha = 1.0F - std::exp(-sigma * band * step);
        if (alpha < 1.0e-5F) {
          continue;
        }
        const ::carbon::render::FieldGradientSample grad_sample = sampler.gradient(world);
        if (!grad_sample.valid) {
          continue;
        }

        // Shape from the primary field, colour from the attribute field. A miss on the
        // attribute (no second loader, or a hole in a sparse grid) keeps the flat material
        // colour, so the surface stays lit rather than turning black.
        float surf_r = mat_r;
        float surf_g = mat_g;
        float surf_b = mat_b;
        if (opts.shading == RayMarchShading::SurfaceScalar) {
          const ::carbon::render::FieldSample attr = attribute_sampler.sample(world);
          if (attr.valid) {
            const float range = opts.value_max - opts.value_min;
            const float t = (range > 0.0F) ? ((attr.value - opts.value_min) / range) : 0.0F;
            const float32_3d c = ::carbon::render::colormap_eval(opts.colormap, t);
            surf_r = c.x;
            surf_g = c.y;
            surf_b = c.z;
          }
        }

        const float32_3d grad = grad_sample.value;
        float gn = std::sqrt(grad.x * grad.x + grad.y * grad.y + grad.z * grad.z);
        if (gn > 1.0e-8F) {
          const float inv_gn = 1.0F / gn;
          shade_blinn(grad.x * inv_gn, grad.y * inv_gn, grad.z * inv_gn, lx, ly, lz, lx2, ly2, lz2,
                      fill_diffuse, vx, vy, vz, opts.ambient, opts.diffuse, opts.specular,
                      opts.shininess, surf_r, surf_g, surf_b, sample_r, sample_g, sample_b);
        } else {
          sample_r = opts.ambient * surf_r;
          sample_g = opts.ambient * surf_g;
          sample_b = opts.ambient * surf_b;
        }
        (void)band;
      }

      if (opts.shading == RayMarchShading::Emission) {
        r_acc += t_acc * sample_r * step;
        g_acc += t_acc * sample_g * step;
        b_acc += t_acc * sample_b * step;
      } else {
        r_acc += t_acc * sample_r * alpha;
        g_acc += t_acc * sample_g * alpha;
        b_acc += t_acc * sample_b * alpha;
      }
      // Depth is where the ray first picks up opacity, not where it entered the box. An
      // in-bounds sample that happens to be empty (zero density, below threshold) must not
      // claim the pixel, or every ray that merely crosses the domain would report a hit and
      // depth compositing against another layer would occlude on empty space.
      const bool contributes =
          (opts.shading == RayMarchShading::Emission) ? (rho > 0.0F) : (alpha > 0.0F);
      if (contributes && !(t_hit < detail::ray_inf())) {
        t_hit = t;
      }
      t_acc *= (1.0F - alpha);
      if (t_acc < 1.0e-4F) {
        break;
      }
      (void)rho;
    }

    seg_color_r[offset] = r_acc;
    seg_color_g[offset] = g_acc;
    seg_color_b[offset] = b_acc;
    seg_trans[offset] = t_acc;
    seg_depth[offset] = t_hit;
  }
};

}  // namespace detail

inline RayMarchImage::RayMarchImage(const System& sys,
                                    int width,
                                    int height)
    : sys_(sys),
      width_(width),
      height_(height) {
  if (width <= 0 || height <= 0) {
    log::throw_with_log<std::invalid_argument>(
        "RayMarchImage: width and height must be positive (got {} x {})", width, height);
  }

  const size_t gpu_count = sys.gpu_group().size();
  if (gpu_count == 0U) {
    log::throw_with_log<std::runtime_error>(
        "RayMarchImage: System has no GPUs (gpu_group().size() == 0)");
  }

  pixel_count_ = static_cast<size_t>(width_) * static_cast<size_t>(height_);
  std::vector<size_t> counts(gpu_count, pixel_count_);
  gpu_segment_color_r_ = sys.buffer<float>(counts);
  gpu_segment_color_g_ = sys.buffer<float>(counts);
  gpu_segment_color_b_ = sys.buffer<float>(counts);
  gpu_segment_trans_ = sys.buffer<float>(counts);
  gpu_segment_depth_ = sys.buffer<float>(counts);

  host_gpu_color_r_.reserve(gpu_count);
  host_gpu_color_g_.reserve(gpu_count);
  host_gpu_color_b_.reserve(gpu_count);
  host_gpu_trans_.reserve(gpu_count);
  host_gpu_depth_.reserve(gpu_count);
  for (size_t g = 0; g < gpu_count; ++g) {
    host_gpu_color_r_.push_back(Cpu::buffer<float>(pixel_count_));
    host_gpu_color_g_.push_back(Cpu::buffer<float>(pixel_count_));
    host_gpu_color_b_.push_back(Cpu::buffer<float>(pixel_count_));
    host_gpu_trans_.push_back(Cpu::buffer<float>(pixel_count_));
    host_gpu_depth_.push_back(Cpu::buffer<float>(pixel_count_));
  }

  host_rank_color_r_.assign(pixel_count_, 0.0F);
  host_rank_color_g_.assign(pixel_count_, 0.0F);
  host_rank_color_b_.assign(pixel_count_, 0.0F);
  host_rank_trans_.assign(pixel_count_, 1.0F);
  host_rank_depth_.assign(pixel_count_, std::numeric_limits<float>::infinity());
  host_combined_color_r_.assign(pixel_count_, 0.0F);
  host_combined_color_g_.assign(pixel_count_, 0.0F);
  host_combined_color_b_.assign(pixel_count_, 0.0F);
  host_combined_luminance_.assign(pixel_count_, 0.0F);
  host_combined_trans_.assign(pixel_count_, 1.0F);
  host_combined_depth_.assign(pixel_count_, std::numeric_limits<float>::infinity());
}

inline auto
RayMarchImage::clear(const Stream& stream) const -> void {
  const size_t nbytes = pixel_count_ * sizeof(float);
  for (size_t g = 0; g < gpu_count(); ++g) {
    sys_.async_memset(gpu_segment_color_r_[g], 0, nbytes, stream);
    sys_.async_memset(gpu_segment_color_g_[g], 0, nbytes, stream);
    sys_.async_memset(gpu_segment_color_b_[g], 0, nbytes, stream);
  }
}

inline auto
RayMarchImage::gpu_segment_color_r(size_t relative_gpu_id) const -> float* {
  if (relative_gpu_id >= gpu_count()) {
    log::throw_with_log<std::out_of_range>(
        "RayMarchImage::gpu_segment_color_r: relative_gpu_id {} >= gpu_count {}", relative_gpu_id,
        gpu_count());
  }
  return gpu_segment_color_r_[relative_gpu_id].data();
}

inline auto
RayMarchImage::gpu_segment_color_g(size_t relative_gpu_id) const -> float* {
  if (relative_gpu_id >= gpu_count()) {
    log::throw_with_log<std::out_of_range>(
        "RayMarchImage::gpu_segment_color_g: relative_gpu_id {} >= gpu_count {}", relative_gpu_id,
        gpu_count());
  }
  return gpu_segment_color_g_[relative_gpu_id].data();
}

inline auto
RayMarchImage::gpu_segment_color_b(size_t relative_gpu_id) const -> float* {
  if (relative_gpu_id >= gpu_count()) {
    log::throw_with_log<std::out_of_range>(
        "RayMarchImage::gpu_segment_color_b: relative_gpu_id {} >= gpu_count {}", relative_gpu_id,
        gpu_count());
  }
  return gpu_segment_color_b_[relative_gpu_id].data();
}

inline auto
RayMarchImage::gpu_segment_transmittance(size_t relative_gpu_id) const -> float* {
  if (relative_gpu_id >= gpu_count()) {
    log::throw_with_log<std::out_of_range>(
        "RayMarchImage::gpu_segment_transmittance: relative_gpu_id {} >= gpu_count {}",
        relative_gpu_id, gpu_count());
  }
  return gpu_segment_trans_[relative_gpu_id].data();
}

inline auto
RayMarchImage::gpu_segment_depth(size_t relative_gpu_id) const -> float* {
  if (relative_gpu_id >= gpu_count()) {
    log::throw_with_log<std::out_of_range>(
        "RayMarchImage::gpu_segment_depth: relative_gpu_id {} >= gpu_count {}", relative_gpu_id,
        gpu_count());
  }
  return gpu_segment_depth_[relative_gpu_id].data();
}

inline auto
RayMarchImage::combine(const Stream& stream) -> void {
  const size_t nbytes = pixel_count_ * sizeof(float);
  for (size_t g = 0; g < gpu_count(); ++g) {
    sys_.async_copy(gpu_segment_color_r_[g], host_gpu_color_r_[g], nbytes, stream);
    sys_.async_copy(gpu_segment_color_g_[g], host_gpu_color_g_[g], nbytes, stream);
    sys_.async_copy(gpu_segment_color_b_[g], host_gpu_color_b_[g], nbytes, stream);
    sys_.async_copy(gpu_segment_trans_[g], host_gpu_trans_[g], nbytes, stream);
    sys_.async_copy(gpu_segment_depth_[g], host_gpu_depth_[g], nbytes, stream);
  }
  stream.wait();

  std::vector<detail::RayMarchSegment> segments;
  segments.reserve(gpu_count());
  for (size_t i = 0; i < pixel_count_; ++i) {
    segments.clear();
    for (size_t g = 0; g < gpu_count(); ++g) {
      const float t_in = host_gpu_depth_[g].data()[i];
      if (!std::isfinite(t_in)) {
        continue;
      }
      segments.push_back({host_gpu_color_r_[g].data()[i], host_gpu_color_g_[g].data()[i],
                          host_gpu_color_b_[g].data()[i], host_gpu_trans_[g].data()[i], t_in});
    }
    float r = 0.0F;
    float g = 0.0F;
    float b = 0.0F;
    float t = 1.0F;
    if (!segments.empty()) {
      detail::composite_segments_front_to_back(segments, r, g, b, t);
      float t_min = std::numeric_limits<float>::infinity();
      for (const detail::RayMarchSegment& seg : segments) {
        t_min = std::min(t_min, seg.depth);
      }
      host_rank_depth_[i] = t_min;
    } else {
      host_rank_depth_[i] = std::numeric_limits<float>::infinity();
    }
    host_rank_color_r_[i] = r;
    host_rank_color_g_[i] = g;
    host_rank_color_b_[i] = b;
    host_rank_trans_[i] = t;
  }

  combined_ready_ = false;
}

inline auto
RayMarchImage::dist_combine() -> void {
  if (sys_.distributed() != nullptr) {
#ifdef CARBON_HAVE_MPI_NCCL
    const size_t n = pixel_count_;
    const int world_size = sys_.distributed()->world_size();
    std::vector<float> all_color_r;
    std::vector<float> all_color_g;
    std::vector<float> all_color_b;
    std::vector<float> all_trans;
    std::vector<float> all_depth;
    all_color_r.assign(static_cast<size_t>(world_size) * n, 0.0F);
    all_color_g.assign(static_cast<size_t>(world_size) * n, 0.0F);
    all_color_b.assign(static_cast<size_t>(world_size) * n, 0.0F);
    all_trans.assign(static_cast<size_t>(world_size) * n, 1.0F);
    all_depth.assign(static_cast<size_t>(world_size) * n, std::numeric_limits<float>::infinity());
    render::dist_gather_segments(sys_, host_rank_color_r_.data(), host_rank_color_g_.data(),
                                 host_rank_color_b_.data(), host_rank_trans_.data(),
                                 host_rank_depth_.data(), all_color_r.data(), all_color_g.data(),
                                 all_color_b.data(), all_trans.data(), all_depth.data(), n);

    std::vector<detail::RayMarchSegment> segments;
    segments.reserve(static_cast<size_t>(world_size));
    for (size_t i = 0; i < n; ++i) {
      segments.clear();
      for (int r = 0; r < world_size; ++r) {
        const size_t off = static_cast<size_t>(r) * n + i;
        const float t_in = all_depth[off];
        if (!std::isfinite(t_in)) {
          continue;
        }
        segments.push_back(
            {all_color_r[off], all_color_g[off], all_color_b[off], all_trans[off], t_in});
      }
      float cr = 0.0F;
      float cg = 0.0F;
      float cb = 0.0F;
      float t = 1.0F;
      float depth = std::numeric_limits<float>::infinity();
      if (!segments.empty()) {
        detail::composite_segments_front_to_back(segments, cr, cg, cb, t);
        for (const detail::RayMarchSegment& seg : segments) {
          depth = std::min(depth, seg.depth);
        }
      }
      host_combined_color_r_[i] = cr;
      host_combined_color_g_[i] = cg;
      host_combined_color_b_[i] = cb;
      host_combined_trans_[i] = t;
      host_combined_depth_[i] = depth;
    }
#else
    log::throw_with_log<std::runtime_error>(
        "RayMarchImage::dist_combine: System has Distributed but carbon lacks MPI support");
#endif
  } else {
    host_combined_color_r_ = host_rank_color_r_;
    host_combined_color_g_ = host_rank_color_g_;
    host_combined_color_b_ = host_rank_color_b_;
    host_combined_trans_ = host_rank_trans_;
    host_combined_depth_ = host_rank_depth_;
  }

  host_combined_luminance_.resize(pixel_count_);
  for (size_t i = 0; i < pixel_count_; ++i) {
    host_combined_luminance_[i] = detail::luminance(
        host_combined_color_r_[i], host_combined_color_g_[i], host_combined_color_b_[i]);
  }
  combined_ready_ = true;
}

inline auto
RayMarchImage::set_export_options(bool add_background,
                                  float32_3d background,
                                  float exposure) noexcept -> void {
  export_add_background_ = add_background;
  export_background_ = background;
  export_exposure_ = exposure;
}

inline auto
RayMarchImage::sync(const Stream& stream) const -> void {
  stream.wait();
}

inline auto
RayMarchImage::write_pgm(std::string_view path,
                         DensityNormalize mode) const -> bool {
  int rank = 0;
#ifdef CARBON_HAVE_MPI_NCCL
  if (const Distributed* dist = sys_.distributed(); dist != nullptr) {
    rank = dist->world_rank();
  }
#endif
  if (rank != 0) {
    return true;
  }

  if (!combined_ready_) {
    return false;
  }

  float mn = 0.0F;
  float mx = 1.0F;
  const float bg_lum =
      export_add_background_
          ? detail::luminance(export_background_.x, export_background_.y, export_background_.z)
          : 0.0F;
  constexpr float k_bg_lum_eps = 0.03F;
  if (mode == DensityNormalize::MinMax) {
    mn = std::numeric_limits<float>::infinity();
    mx = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < pixel_count_; ++i) {
      float r = host_combined_color_r_[i];
      float g = host_combined_color_g_[i];
      float b = host_combined_color_b_[i];
      if (export_add_background_) {
        const float t = host_combined_trans_[i];
        r += t * export_background_.x;
        g += t * export_background_.y;
        b += t * export_background_.z;
      }
      const float lum = detail::luminance(r, g, b);
      if (std::isfinite(lum) && lum > 1.0e-6F) {
        if (export_add_background_ && lum >= bg_lum - k_bg_lum_eps) {
          continue;
        }
        mn = std::min(mn, lum);
        mx = std::max(mx, lum);
      }
    }
    if (!std::isfinite(mn) || !std::isfinite(mx) || mx <= mn) {
      mn = 0.0F;
      mx = 1.0F;
    }
  }

  const float inv = (mode == DensityNormalize::MinMax && mx > mn) ? (1.0F / (mx - mn)) : 1.0F;

  std::ofstream out(std::string(path), std::ios::binary);
  if (!out) {
    return false;
  }

  out << "P6\n" << width_ << ' ' << height_ << "\n255\n";
  for (size_t i = 0; i < pixel_count_; ++i) {
    float r = host_combined_color_r_[i];
    float g = host_combined_color_g_[i];
    float b = host_combined_color_b_[i];
    const float raw_lum = detail::luminance(r, g, b);
    const float trans = host_combined_trans_[i];
    if (export_add_background_) {
      if (trans > 0.97F && raw_lum < 0.05F) {
        r = export_background_.x;
        g = export_background_.y;
        b = export_background_.z;
      } else {
        r += trans * export_background_.x;
        g += trans * export_background_.y;
        b += trans * export_background_.z;
      }
    }
    auto encode = [&](float v) -> unsigned char {
      if (!std::isfinite(v)) {
        return 0;
      }
      float t = v;
      if (mode == DensityNormalize::Clamp01) {
        t = v * export_exposure_;
      } else if (mode == DensityNormalize::MinMax) {
        const float lum = detail::luminance(r, g, b);
        if (export_add_background_ && lum >= bg_lum - k_bg_lum_eps) {
          t = v;
        } else if (std::abs(lum) > 1.0e-8F) {
          t = v * ((lum - mn) * inv) / lum;
        } else {
          t = 0.0F;
        }
      }
      const float c = std::clamp(t, 0.0F, 1.0F);
      return static_cast<unsigned char>(std::lround(c * 255.0F));
    };
    out.put(static_cast<char>(encode(r)));
    out.put(static_cast<char>(encode(g)));
    out.put(static_cast<char>(encode(b)));
  }
  return static_cast<bool>(out);
}

namespace detail {

/**
 * Probe: a multi-resolution tower (@c MultiResBlock ) rather than a flat @c Block . Both
 * answer @c false to @c grid_has_dense_iteration_v , so the ray march needs this second
 * question to pick between the single-level and per-level locator.
 */
template <typename Grid>
inline constexpr bool grid_is_multi_res_v = requires(const Grid& g) {
  g.depth();
};

/**
 * Build the per-GPU sampler the ray march reads the field through. Dense and DenseDisg take
 * their world frame from the @c Span they already carry; @c Block and @c MultiResBlock are
 * pure index space and take it from @c RayMarchOptions::world_origin / @c world_spacing .
 */
template <typename Grid,
          typename UserFn>
[[nodiscard]] inline auto
make_ray_march_sampler(const Grid& grid,
                       size_t gpu_id,
                       const RayMarchOptions& opts,
                       UserFn user_fn) {
  if constexpr (grid_has_dense_iteration_v<Grid>) {
    return ::carbon::render::make_sampler(grid.spans()[gpu_id], std::move(user_fn));
  } else if constexpr (grid_is_multi_res_v<Grid>) {
    return ::carbon::render::make_sampler(
        grid.locator(gpu_id, opts.world_origin, opts.world_spacing), std::move(user_fn));
  } else {
    return ::carbon::render::make_sampler(
        grid.locator(gpu_id, opts.world_origin, opts.world_spacing), grid.span(gpu_id),
        std::move(user_fn));
  }
}

/**
 * Cell spacing and domain diagonal in world units, used to raise @c max_steps to at least
 * what a corner-to-corner ray needs. Returns @c false when the grid has no slabs to measure.
 */
template <typename Grid>
[[nodiscard]] inline auto
ray_march_world_frame(const Grid& grid,
                      const RayMarchOptions& opts,
                      float& spacing,
                      float& diagonal) -> bool {
  int64_3d extent{};
  if constexpr (grid_has_dense_iteration_v<Grid>) {
    if (grid.spans().empty()) {
      return false;
    }
    spacing = grid.spans().front().spacing();
    extent = grid.spans().front().global_extent();
  } else {
    spacing = opts.world_spacing;
    if constexpr (grid_is_multi_res_v<Grid>) {
      extent = grid.finest_extent();
    } else {
      extent = grid.global_extent();
    }
  }
  diagonal = spacing * std::sqrt(static_cast<float>(extent.x * extent.x + extent.y * extent.y +
                                                    extent.z * extent.z));
  return spacing > 0.0F;
}

/** Placeholder passed by the single-loader @c make_ray_march_kernel overload. */
struct NoAttributeLoader {};

}  // namespace detail

/**
 * Ray-march kernel with a second, @b attribute field.
 *
 * @p density_loader supplies the field the march integrates — the density, or the signed
 * distance whose zero level set the surface modes trace. @p attribute_loader supplies a
 * second scalar sampled at the same world point, which @ref RayMarchShading::SurfaceScalar
 * maps through @ref RayMarchOptions::colormap to colour the surface. Both loaders have the
 * same shape (a @c KernelLoader& in, a @c (Span, Idx) -> float callable out) and are sampled
 * through the same per-family sampler, so the attribute may live on any field over the same
 * grid, or be computed on the fly from several.
 *
 * Every other shading mode ignores the attribute.
 */
template <typename Grid,
          typename DensityLoader,
          typename AttributeLoader>
inline auto
make_ray_march_kernel(const Grid& grid,
                      RayMarchImage& image,
                      Camera camera,
                      RayMarchOptions opts,
                      DensityLoader&& density_loader,
                      AttributeLoader&& attribute_loader) -> Kernel {
  if (opts.max_steps < 1) {
    log::throw_with_log<std::invalid_argument>(
        "make_ray_march_kernel: max_steps must be >= 1 (got {})", opts.max_steps);
  }
  if (opts.absorption_scale < 0.0F) {
    log::throw_with_log<std::invalid_argument>(
        "make_ray_march_kernel: absorption_scale must be >= 0 (got {})", opts.absorption_scale);
  }

  RayMarchOptions effective_opts = opts;
  float ld_x = effective_opts.light_dir.x;
  float ld_y = effective_opts.light_dir.y;
  float ld_z = effective_opts.light_dir.z;
  if (detail::render_math::normalize3(ld_x, ld_y, ld_z)) {
    effective_opts.light_dir = {ld_x, ld_y, ld_z};
  }
  float ld2_x = effective_opts.light_dir2.x;
  float ld2_y = effective_opts.light_dir2.y;
  float ld2_z = effective_opts.light_dir2.z;
  if (detail::render_math::normalize3(ld2_x, ld2_y, ld2_z)) {
    effective_opts.light_dir2 = {ld2_x, ld2_y, ld2_z};
  } else {
    effective_opts.fill_diffuse = 0.0F;
  }

  image.set_export_options(effective_opts.add_background, effective_opts.background,
                           effective_opts.export_exposure);

  float ref_spacing = 0.0F;
  float diag = 0.0F;
  if (detail::ray_march_world_frame(grid, effective_opts, ref_spacing, diag)) {
    const float step =
        (effective_opts.step_size > 0.0F) ? effective_opts.step_size : (0.5F * ref_spacing);
    const int needed = static_cast<int>(std::ceil(diag / step)) + 1;
    if (effective_opts.max_steps < needed) {
      effective_opts.max_steps = needed;
    }
  }

  // Force the brick-index table to be built here, on one host thread. The loading lambda
  // below runs inside an OpenMP parallel region over GPUs, and while the once_flag makes a
  // concurrent first build safe, the builder itself drives copies for every GPU and would
  // then be doing so with the other worker threads blocked behind it.
  if constexpr (!detail::grid_has_dense_iteration_v<Grid>) {
    (void)grid.locator(0, effective_opts.world_origin, effective_opts.world_spacing);
  }

  auto loading_fn = [&image, camera, effective_opts, &grid,
                     density_loader = std::forward<DensityLoader>(density_loader),
                     attribute_loader =
                         std::forward<AttributeLoader>(attribute_loader)](KernelLoader& loader) {
    auto user_fn = density_loader(loader);
    auto sampler = detail::make_ray_march_sampler(grid, loader.relative_gpu_id(), effective_opts,
                                                  std::move(user_fn));
    auto attr_sampler = [&]() {
      if constexpr (std::is_same_v<std::decay_t<AttributeLoader>, detail::NoAttributeLoader>) {
        return detail::NoAttributeSampler{};
      } else {
        return detail::make_ray_march_sampler(grid, loader.relative_gpu_id(), effective_opts,
                                              attribute_loader(loader));
      }
    }();
    return detail::RayMarchPixelFn<decltype(sampler), decltype(attr_sampler)>{
        std::move(sampler),
        std::move(attr_sampler),
        image.gpu_segment_color_r(loader.relative_gpu_id()),
        image.gpu_segment_color_g(loader.relative_gpu_id()),
        image.gpu_segment_color_b(loader.relative_gpu_id()),
        image.gpu_segment_transmittance(loader.relative_gpu_id()),
        image.gpu_segment_depth(loader.relative_gpu_id()),
        camera,
        effective_opts,
        camera.width(),
        camera.height(),
    };
  };

  using LoadingFn = decltype(loading_fn);
  return Kernel::from_impl(std::make_shared<detail::RayMarchKernelImpl<Grid, LoadingFn>>(
      grid, image, camera, effective_opts, std::move(loading_fn)));
}

/** Ray-march kernel over a single field; see the two-loader overload above. */
template <typename Grid,
          typename DensityLoader>
inline auto
make_ray_march_kernel(const Grid& grid,
                      RayMarchImage& image,
                      Camera camera,
                      RayMarchOptions opts,
                      DensityLoader&& density_loader) -> Kernel {
  return make_ray_march_kernel(grid, image, camera, opts,
                               std::forward<DensityLoader>(density_loader),
                               detail::NoAttributeLoader{});
}

}  // namespace carbon
