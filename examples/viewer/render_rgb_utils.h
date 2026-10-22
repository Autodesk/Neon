#pragma once

#include <carbon/render.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace carbon::viewer {

namespace detail {

[[nodiscard]] inline auto
luminance(float r,
          float g,
          float b) noexcept -> float {
  return 0.299F * r + 0.587F * g + 0.114F * b;
}

/** Shared near-white studio backdrop with a radial vignette (matches the density modes). */
inline auto
studio_backdrop(int col,
                int row,
                int w,
                int h,
                float& r,
                float& g,
                float& b) noexcept -> void {
  const float inv_w = 1.0F / static_cast<float>(std::max(1, w - 1));
  const float inv_h = 1.0F / static_cast<float>(std::max(1, h - 1));
  const float aspect = static_cast<float>(w) / static_cast<float>(std::max(1, h));
  const float nx = ((static_cast<float>(col) * inv_w) * 2.0F - 1.0F) * aspect;
  const float ny = (static_cast<float>(row) * inv_h) * 2.0F - 1.0F;
  const float vig = std::clamp(0.055F * (nx * nx + ny * ny), 0.0F, 0.10F);
  r = (1.0F - vig) * 0.99F;
  g = (1.0F - vig);
  b = (1.0F - vig) * 0.98F;
}

}  // namespace detail

/**
 * @c RayMarchImage surface -> interleaved RGB composited over the shared studio backdrop, so the
 * surface style's background matches the density styles' vignette (no MinMax renormalization).
 */
inline auto
rgb_from_ray_march_studio(const RayMarchImage& image,
                          std::vector<float>& rgb_out) -> void {
  const int w = std::max(1, image.width());
  const int h = std::max(1, image.height());
  const size_t n = image.pixel_count();
  rgb_out.resize(n * 3U);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int row = 0; row < h; ++row) {
    for (int col = 0; col < w; ++col) {
      const size_t i = static_cast<size_t>(row) * w + col;
      float bg_r = 0.0F;
      float bg_g = 0.0F;
      float bg_b = 0.0F;
      detail::studio_backdrop(col, row, w, h, bg_r, bg_g, bg_b);
      const float t = image.host_transmittance()[i];
      rgb_out[i * 3U + 0U] = std::clamp(image.host_color_r()[i] + t * bg_r, 0.0F, 1.0F);
      rgb_out[i * 3U + 1U] = std::clamp(image.host_color_g()[i] + t * bg_g, 0.0F, 1.0F);
      rgb_out[i * 3U + 2U] = std::clamp(image.host_color_b()[i] + t * bg_b, 0.0F, 1.0F);
    }
  }
}

/** Planar @c RayMarchImage host colours -> interleaved RGB in @c [0, 1] (MinMax luminance). */
inline auto
rgb_from_ray_march(const RayMarchImage& image,
                   bool add_background,
                   float32_3d background,
                   std::vector<float>& rgb_out) -> void {
  const size_t n = image.pixel_count();
  rgb_out.resize(n * 3U);

  float mn = std::numeric_limits<float>::infinity();
  float mx = -std::numeric_limits<float>::infinity();
  const float bg_lum =
      add_background ? detail::luminance(background.x, background.y, background.z) : 0.0F;
  constexpr float k_bg_lum_eps = 0.03F;

  for (size_t i = 0; i < n; ++i) {
    float r = image.host_color_r()[i];
    float g = image.host_color_g()[i];
    float b = image.host_color_b()[i];
    if (add_background) {
      const float t = image.host_transmittance()[i];
      r += t * background.x;
      g += t * background.y;
      b += t * background.z;
    }
    const float lum = detail::luminance(r, g, b);
    if (std::isfinite(lum) && lum > 1.0e-6F) {
      if (add_background && lum >= bg_lum - k_bg_lum_eps) {
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
  const float inv = (mx > mn) ? (1.0F / (mx - mn)) : 1.0F;

  for (size_t i = 0; i < n; ++i) {
    float r = image.host_color_r()[i];
    float g = image.host_color_g()[i];
    float b = image.host_color_b()[i];
    if (add_background) {
      const float t = image.host_transmittance()[i];
      r += t * background.x;
      g += t * background.y;
      b += t * background.z;
    }
    const float lum = detail::luminance(r, g, b);
    float t = (lum - mn) * inv;
    t = std::clamp(t, 0.0F, 1.0F);
    const float scale = (lum > 1.0e-8F) ? (t / lum) : 0.0F;
    rgb_out[i * 3U + 0U] = r * scale;
    rgb_out[i * 3U + 1U] = g * scale;
    rgb_out[i * 3U + 2U] = b * scale;
  }
}

/** Composite two ray-march layers (bottom then top over background) into interleaved RGB. */
inline auto
rgb_from_composited_layers(const RayMarchImage& bottom,
                           const RayMarchImage& top,
                           float32_3d background,
                           std::vector<float>& rgb_out) -> void {
  const size_t n = bottom.pixel_count();
  rgb_out.resize(n * 3U);
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
    rgb_out[i * 3U + 0U] = top_r + top_t * mid_r;
    rgb_out[i * 3U + 1U] = top_g + top_t * mid_g;
    rgb_out[i * 3U + 2U] = top_b + top_t * mid_b;
  }
}

/**
 * @c SurfelImage host normals -> interleaved RGB over the shared studio backdrop (Blinn, matches
 * ray-march surface shading).
 */
inline auto
rgb_from_surfel_studio(const SurfelImage& image,
                       const Camera& camera,
                       std::vector<float>& rgb_out,
                       float32_3d light_dir = {0.35F,
                                               -0.8F,
                                               0.5F},
                       float32_3d light_dir2 = {-0.45F,
                                                0.55F,
                                                -0.35F},
                       float32_3d material = {0.831F,
                                              0.953F,
                                              0.188F},
                       float ambient = 0.18F,
                       float diffuse = 0.68F,
                       float fill_diffuse = 0.50F,
                       float specular = 0.20F,
                       float shininess = 48.0F) -> void {
  const int w = std::max(1, image.width());
  const int h = std::max(1, image.height());
  const size_t n = image.pixel_count();
  rgb_out.resize(n * 3U);

  float lx = light_dir.x;
  float ly = light_dir.y;
  float lz = light_dir.z;
  const float llen = std::sqrt(lx * lx + ly * ly + lz * lz);
  if (llen > 1.0e-6F) {
    lx /= llen;
    ly /= llen;
    lz /= llen;
  }

  float lx2 = light_dir2.x;
  float ly2 = light_dir2.y;
  float lz2 = light_dir2.z;
  const float llen2 = std::sqrt(lx2 * lx2 + ly2 * ly2 + lz2 * lz2);
  if (llen2 > 1.0e-6F) {
    lx2 /= llen2;
    ly2 /= llen2;
    lz2 /= llen2;
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int row = 0; row < h; ++row) {
    for (int col = 0; col < w; ++col) {
      const size_t i = static_cast<size_t>(row) * static_cast<size_t>(w) + static_cast<size_t>(col);
      float bg_r = 0.0F;
      float bg_g = 0.0F;
      float bg_b = 0.0F;
      detail::studio_backdrop(col, row, w, h, bg_r, bg_g, bg_b);
      if (!image.has_fragment(i)) {
        rgb_out[i * 3U + 0U] = bg_r;
        rgb_out[i * 3U + 1U] = bg_g;
        rgb_out[i * 3U + 2U] = bg_b;
        continue;
      }
      float32_3d ray_origin{};
      float32_3d ray_dir{};
      if (!camera.ray_through_pixel(col, row, ray_origin, ray_dir)) {
        rgb_out[i * 3U + 0U] = bg_r;
        rgb_out[i * 3U + 1U] = bg_g;
        rgb_out[i * 3U + 2U] = bg_b;
        continue;
      }
      const float32_3d n = image.host_normal()[i];
      const float vx = -ray_dir.x;
      const float vy = -ray_dir.y;
      const float vz = -ray_dir.z;
      const float ndotl = n.x * lx + n.y * ly + n.z * lz;
      const float diff = (ndotl > 0.0F) ? ndotl : 0.0F;
      const float ndotl2 = n.x * lx2 + n.y * ly2 + n.z * lz2;
      const float diff2 = (fill_diffuse > 0.0F && ndotl2 > 0.0F) ? ndotl2 : 0.0F;
      const float hx = lx + vx;
      const float hy = ly + vy;
      const float hz = lz + vz;
      const float hlen = std::sqrt(hx * hx + hy * hy + hz * hz);
      float spec = 0.0F;
      if (hlen > 1.0e-8F) {
        const float inv_h = 1.0F / hlen;
        const float ndoth = n.x * hx * inv_h + n.y * hy * inv_h + n.z * hz * inv_h;
        if (ndoth > 0.0F) {
          spec = std::pow(ndoth, shininess);
        }
      }
      const float shade = ambient + diffuse * diff + fill_diffuse * diff2 + specular * spec;
      rgb_out[i * 3U + 0U] = std::clamp(material.x * shade, 0.0F, 1.0F);
      rgb_out[i * 3U + 1U] = std::clamp(material.y * shade, 0.0F, 1.0F);
      rgb_out[i * 3U + 2U] = std::clamp(material.z * shade, 0.0F, 1.0F);
    }
  }
}

/**
 * As @ref rgb_from_surfel_studio , but the albedo is a per-pixel @ref Colormap lookup of the
 * image's attribute plane over @c [value_min, value_max] instead of one flat material. Shades
 * with a white material first and multiplies the colour in afterwards, so the lighting is
 * identical to the plain surfel mode and only the albedo changes. No-op without an attribute
 * plane.
 */
inline auto
rgb_from_surfel_attribute(const SurfelImage& image,
                          const Camera& camera,
                          Colormap colormap,
                          float value_min,
                          float value_max,
                          std::vector<float>& rgb_out) -> void {
  rgb_from_surfel_studio(image, camera, rgb_out, {0.35F, -0.8F, 0.5F}, {-0.45F, 0.55F, -0.35F},
                         {1.0F, 1.0F, 1.0F});
  if (!image.has_attribute()) {
    return;
  }
  const std::vector<float>& attr = image.host_attribute();
  const float range = value_max - value_min;
  const size_t n = image.pixel_count();
  for (size_t i = 0; i < n; ++i) {
    if (!image.has_fragment(i)) {
      continue;
    }
    const float t = (range > 0.0F) ? ((attr[i] - value_min) / range) : 0.0F;
    const float32_3d c = render::colormap_eval(colormap, t);
    rgb_out[i * 3U + 0U] *= c.x;
    rgb_out[i * 3U + 1U] *= c.y;
    rgb_out[i * 3U + 2U] *= c.z;
  }
}

namespace detail {

/** Perceptual light-mint -> deep-green ramp for the density albedo. */
inline auto
green_ramp(float t,
           float& r,
           float& g,
           float& b) noexcept -> void {
  // Three-stop ramp: light lime -> vivid lime (212,243,48) -> deep lime (low end kept saturated
  // enough to read against the near-white studio backdrop).
  constexpr float s0[3] = {0.910F, 0.972F, 0.553F};
  constexpr float s1[3] = {0.831F, 0.953F, 0.188F};
  constexpr float s2[3] = {0.400F, 0.478F, 0.075F};
  if (t < 0.5F) {
    const float u = t * 2.0F;
    r = s0[0] + (s1[0] - s0[0]) * u;
    g = s0[1] + (s1[1] - s0[1]) * u;
    b = s0[2] + (s1[2] - s0[2]) * u;
  } else {
    const float u = (t - 0.5F) * 2.0F;
    r = s1[0] + (s2[0] - s1[0]) * u;
    g = s1[1] + (s2[1] - s1[1]) * u;
    b = s1[2] + (s2[2] - s1[2]) * u;
  }
}

/** Perceptual blue -> cyan -> green -> yellow -> red heatmap ramp. */
inline auto
heatmap_ramp(float t,
             float& r,
             float& g,
             float& b) noexcept -> void {
  constexpr float stops[5][3] = {
      {0.23F, 0.30F, 0.75F}, {0.00F, 0.70F, 0.80F}, {0.10F, 0.70F, 0.22F},
      {0.95F, 0.85F, 0.15F}, {0.85F, 0.15F, 0.10F},
  };
  const float s = std::clamp(t, 0.0F, 1.0F) * 4.0F;
  const int i0 = std::min(3, static_cast<int>(s));
  const float u = s - static_cast<float>(i0);
  r = stops[i0][0] + (stops[i0 + 1][0] - stops[i0][0]) * u;
  g = stops[i0][1] + (stops[i0 + 1][1] - stops[i0][1]) * u;
  b = stops[i0][2] + (stops[i0 + 1][2] - stops[i0][2]) * u;
}

}  // namespace detail

/**
 * @c DensityImage host frame -> interleaved RGB with a scalar colormap over the studio backdrop.
 *
 * Unlike @ref rgb_from_density this path is meant for simulation fields (speed, Q, …) where the
 * splat kernel already encoded the physics; min-max normalization drives @p colormap_eval .
 */
inline auto
rgb_from_density_colormap(const DensityImage& image,
                          carbon::Colormap cmap,
                          std::vector<float>& rgb_out) -> void {
  const int w = std::max(1, image.width());
  const int h = std::max(1, image.height());
  const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);
  rgb_out.resize(n * 3U);

  const std::vector<float>& frame = image.host_frame();

  float mn = std::numeric_limits<float>::infinity();
  float mx = -std::numeric_limits<float>::infinity();
  for (float v : frame) {
    if (std::isfinite(v)) {
      mn = std::min(mn, v);
      mx = std::max(mx, v);
    }
  }
  if (!std::isfinite(mn) || !std::isfinite(mx) || mx <= mn) {
    mn = 0.0F;
    mx = 1.0F;
  }
  const float inv = (mx > mn) ? (1.0F / (mx - mn)) : 1.0F;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int row = 0; row < h; ++row) {
    for (int col = 0; col < w; ++col) {
      const size_t i = static_cast<size_t>(row) * static_cast<size_t>(w) + static_cast<size_t>(col);
      detail::studio_backdrop(col, row, w, h, rgb_out[i * 3U + 0U], rgb_out[i * 3U + 1U],
                              rgb_out[i * 3U + 2U]);

      const float v = frame[i];
      if (!std::isfinite(v) || v <= 0.0F) {
        continue;
      }
      const float t = std::clamp((v - mn) * inv, 0.0F, 1.0F);
      const float a = std::clamp(std::sqrt(t) * 1.15F, 0.0F, 1.0F);
      const carbon::float32_3d c = carbon::render::colormap_eval(cmap, t);
      const float bg_r = rgb_out[i * 3U + 0U];
      const float bg_g = rgb_out[i * 3U + 1U];
      const float bg_b = rgb_out[i * 3U + 2U];
      rgb_out[i * 3U + 0U] = bg_r + a * (c.x - bg_r);
      rgb_out[i * 3U + 1U] = bg_g + a * (c.y - bg_g);
      rgb_out[i * 3U + 2U] = bg_b + a * (c.z - bg_b);
    }
  }
}

/** Composite a ray-march layer (top) over an existing interleaved RGB buffer (bottom). */
inline auto
rgb_composite_ray_march_over_rgb(const std::vector<float>& base_rgb,
                                 const RayMarchImage& top,
                                 std::vector<float>& rgb_out) -> void {
  const size_t n = top.pixel_count();
  rgb_out.resize(n * 3U);
  for (size_t i = 0; i < n; ++i) {
    const float mid_r = base_rgb[i * 3U + 0U];
    const float mid_g = base_rgb[i * 3U + 1U];
    const float mid_b = base_rgb[i * 3U + 2U];
    const float top_r = top.host_color_r()[i];
    const float top_g = top.host_color_g()[i];
    const float top_b = top.host_color_b()[i];
    const float top_t = top.host_transmittance()[i];
    rgb_out[i * 3U + 0U] = top_r + top_t * mid_r;
    rgb_out[i * 3U + 1U] = top_g + top_t * mid_g;
    rgb_out[i * 3U + 2U] = top_b + top_t * mid_b;
  }
}

/** Render styles selectable at runtime; index matches the viewer's "Render style" combo. */
enum class RenderStyle : int {
  Shaded = 0,      ///< Lambert + rim shaded green over a studio backdrop with a contact shadow.
  SolidGreen = 1,  ///< Flat green, opacity by density (no shading).
  Grayscale = 2,   ///< Dark density on white (classic line-integral look).
  Heatmap = 3,     ///< Perceptual blue->red heatmap of density.
};

/**
 * @c DensityImage host frame -> interleaved RGB in @c [0, 1], colored per @p mode.
 *
 * For @c RenderStyle::Shaded the scalar density is treated as a screen-space height field: a
 * Sobel-style gradient yields a pseudo-normal that drives Lambert + rim shading over a perceptual
 * green ramp, so the volume reads as a lit 3D form, with a soft contact shadow and a studio
 * vignette. The other styles are cheaper flat / grayscale / heatmap colorizations of the same
 * normalized density.
 */
inline auto
rgb_from_density(const DensityImage& image,
                 std::vector<float>& rgb_out,
                 int mode = 0) -> void {
  const int w = std::max(1, image.width());
  const int h = std::max(1, image.height());
  const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);
  rgb_out.resize(n * 3U);

  const std::vector<float>& frame = image.host_frame();

  float mn = std::numeric_limits<float>::infinity();
  float mx = -std::numeric_limits<float>::infinity();
  for (float v : frame) {
    if (std::isfinite(v)) {
      mn = std::min(mn, v);
      mx = std::max(mx, v);
    }
  }
  if (!std::isfinite(mn) || !std::isfinite(mx) || mx <= mn) {
    mn = 0.0F;
    mx = 1.0F;
  }
  const float inv = (mx > mn) ? (1.0F / (mx - mn)) : 1.0F;

  // Pass 1: normalized density (height field) + per-pixel opacity.
  std::vector<float> t_buf(n, 0.0F);
  std::vector<float> a_buf(n, 0.0F);
  for (size_t i = 0; i < n; ++i) {
    const float v = frame[i];
    float t = 0.0F;
    if (std::isfinite(v)) {
      t = std::clamp((v - mn) * inv, 0.0F, 1.0F);
    }
    t_buf[i] = t;
    // Ease-in opacity so faint outer density still registers instead of washing into the backdrop.
    a_buf[i] = std::clamp(std::sqrt(t) * 1.15F, 0.0F, 1.0F);
  }

  // Cheaper flat / grayscale / heatmap styles: a single vignette + composite pass, no shading.
  if (mode != 0) {
    const float inv_w2 = 1.0F / static_cast<float>(std::max(1, w - 1));
    const float inv_h2 = 1.0F / static_cast<float>(std::max(1, h - 1));
    const float aspect2 = static_cast<float>(w) / static_cast<float>(h);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int row = 0; row < h; ++row) {
      for (int col = 0; col < w; ++col) {
        const size_t i = static_cast<size_t>(row) * w + col;
        const float nx = ((static_cast<float>(col) * inv_w2) * 2.0F - 1.0F) * aspect2;
        const float ny = (static_cast<float>(row) * inv_h2) * 2.0F - 1.0F;
        const float vig = std::clamp(0.055F * (nx * nx + ny * ny), 0.0F, 0.10F);
        const float bg_r = (1.0F - vig) * 0.99F;
        const float bg_g = (1.0F - vig);
        const float bg_b = (1.0F - vig) * 0.98F;

        const float t = t_buf[i];
        const float a = a_buf[i];
        float obj_r = 0.0F;
        float obj_g = 0.0F;
        float obj_b = 0.0F;
        if (mode == static_cast<int>(RenderStyle::SolidGreen)) {
          obj_r = 0.831F;
          obj_g = 0.953F;
          obj_b = 0.188F;
        } else if (mode == static_cast<int>(RenderStyle::Grayscale)) {
          const float gray = 0.12F + 0.10F * (1.0F - t);
          obj_r = gray;
          obj_g = gray;
          obj_b = gray;
        } else {  // Heatmap
          detail::heatmap_ramp(t, obj_r, obj_g, obj_b);
        }

        rgb_out[i * 3U + 0U] = bg_r + a * (obj_r - bg_r);
        rgb_out[i * 3U + 1U] = bg_g + a * (obj_g - bg_g);
        rgb_out[i * 3U + 2U] = bg_b + a * (obj_b - bg_b);
      }
    }
    return;
  }

  // Coarse blurred coverage for the contact shadow (cheap: small grid + separable box blur).
  constexpr int k_shadow_ds = 6;
  const int sw = (w + k_shadow_ds - 1) / k_shadow_ds;
  const int sh = (h + k_shadow_ds - 1) / k_shadow_ds;
  std::vector<float> cov(static_cast<size_t>(sw) * sh, 0.0F);
  for (int sy = 0; sy < sh; ++sy) {
    for (int sx = 0; sx < sw; ++sx) {
      float acc = 0.0F;
      int cnt = 0;
      for (int dy = 0; dy < k_shadow_ds; ++dy) {
        const int py = sy * k_shadow_ds + dy;
        if (py >= h) {
          break;
        }
        for (int dx = 0; dx < k_shadow_ds; ++dx) {
          const int px = sx * k_shadow_ds + dx;
          if (px >= w) {
            break;
          }
          acc += a_buf[static_cast<size_t>(py) * w + px];
          ++cnt;
        }
      }
      cov[static_cast<size_t>(sy) * sw + sx] = (cnt > 0) ? (acc / static_cast<float>(cnt)) : 0.0F;
    }
  }
  const auto box_blur = [&](std::vector<float>& src) {
    std::vector<float> tmp(src.size(), 0.0F);
    constexpr int rad = 2;
    for (int y = 0; y < sh; ++y) {
      for (int x = 0; x < sw; ++x) {
        float acc = 0.0F;
        int cnt = 0;
        for (int k = -rad; k <= rad; ++k) {
          const int xx = x + k;
          if (xx >= 0 && xx < sw) {
            acc += src[static_cast<size_t>(y) * sw + xx];
            ++cnt;
          }
        }
        tmp[static_cast<size_t>(y) * sw + x] = acc / static_cast<float>(cnt);
      }
    }
    for (int y = 0; y < sh; ++y) {
      for (int x = 0; x < sw; ++x) {
        float acc = 0.0F;
        int cnt = 0;
        for (int k = -rad; k <= rad; ++k) {
          const int yy = y + k;
          if (yy >= 0 && yy < sh) {
            acc += tmp[static_cast<size_t>(yy) * sw + x];
            ++cnt;
          }
        }
        src[static_cast<size_t>(y) * sw + x] = acc / static_cast<float>(cnt);
      }
    }
  };
  box_blur(cov);
  box_blur(cov);

  const float inv_w = 1.0F / static_cast<float>(std::max(1, w - 1));
  const float inv_h = 1.0F / static_cast<float>(std::max(1, h - 1));
  const float aspect = static_cast<float>(w) / static_cast<float>(h);

  // Light from upper-left-front (screen: -x left, -y up, +z toward viewer).
  constexpr float lx = -0.42F;
  constexpr float ly = -0.55F;
  constexpr float lz = 0.72F;
  // Shadow is cast opposite the light's screen projection (down-right).
  const int shadow_off_x = static_cast<int>(0.028F * static_cast<float>(w));
  const int shadow_off_y = static_cast<int>(0.045F * static_cast<float>(h));

  // Pass 2: shade.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int row = 0; row < h; ++row) {
    for (int col = 0; col < w; ++col) {
      const size_t i = static_cast<size_t>(row) * w + col;

      const float nx = ((static_cast<float>(col) * inv_w) * 2.0F - 1.0F) * aspect;
      const float ny = (static_cast<float>(row) * inv_h) * 2.0F - 1.0F;
      const float vig = std::clamp(0.055F * (nx * nx + ny * ny), 0.0F, 0.10F);
      float bg_r = (1.0F - vig) * 0.99F;
      float bg_g = (1.0F - vig);
      float bg_b = (1.0F - vig) * 0.98F;

      // Contact shadow: sample blurred coverage offset toward the shadow direction.
      const int scx = std::clamp((col - shadow_off_x) / k_shadow_ds, 0, sw - 1);
      const int scy = std::clamp((row - shadow_off_y) / k_shadow_ds, 0, sh - 1);
      const float shadow = cov[static_cast<size_t>(scy) * sw + scx];
      const float shade_bg = 1.0F - 0.32F * std::clamp(shadow, 0.0F, 1.0F);
      bg_r *= shade_bg;
      bg_g *= shade_bg;
      bg_b *= shade_bg;

      const float a = a_buf[i];
      if (a <= 0.001F) {
        rgb_out[i * 3U + 0U] = bg_r;
        rgb_out[i * 3U + 1U] = bg_g;
        rgb_out[i * 3U + 2U] = bg_b;
        continue;
      }

      // Pseudo-normal from the density height field.
      const int xm = (col > 0) ? col - 1 : col;
      const int xp = (col < w - 1) ? col + 1 : col;
      const int ym = (row > 0) ? row - 1 : row;
      const int yp = (row < h - 1) ? row + 1 : row;
      const float gx =
          t_buf[static_cast<size_t>(row) * w + xp] - t_buf[static_cast<size_t>(row) * w + xm];
      const float gy =
          t_buf[static_cast<size_t>(yp) * w + col] - t_buf[static_cast<size_t>(ym) * w + col];
      constexpr float bump = 7.0F;
      float pnx = -gx * bump;
      float pny = -gy * bump;
      float pnz = 1.0F;
      const float pinv = 1.0F / std::sqrt(pnx * pnx + pny * pny + pnz * pnz);
      pnx *= pinv;
      pny *= pinv;
      pnz *= pinv;

      const float lambert = std::clamp(pnx * lx + pny * ly + pnz * lz, 0.0F, 1.0F);
      const float rim = std::pow(1.0F - std::clamp(pnz, 0.0F, 1.0F), 3.0F);

      float ar = 0.0F;
      float ag = 0.0F;
      float ab = 0.0F;
      detail::green_ramp(t_buf[i], ar, ag, ab);
      const float lit = 0.42F + 0.72F * lambert;
      float obj_r = ar * lit + 0.22F * rim;
      float obj_g = ag * lit + 0.30F * rim;
      float obj_b = ab * lit + 0.22F * rim;
      obj_r = std::clamp(obj_r, 0.0F, 1.0F);
      obj_g = std::clamp(obj_g, 0.0F, 1.0F);
      obj_b = std::clamp(obj_b, 0.0F, 1.0F);

      rgb_out[i * 3U + 0U] = bg_r + a * (obj_r - bg_r);
      rgb_out[i * 3U + 1U] = bg_g + a * (obj_g - bg_g);
      rgb_out[i * 3U + 2U] = bg_b + a * (obj_b - bg_b);
    }
  }
}

}  // namespace carbon::viewer
