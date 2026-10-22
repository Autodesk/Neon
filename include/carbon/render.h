#pragma once

/**
 * Parallel density rendering for Carbon grid families.
 *
 * Each voxel thread evaluates a scalar density, then projects a regular
 * @c samples_per_axis³ sub-grid of points within the cell (not just the centre)
 * through a perspective @ref Camera and atomically accumulates into a per-GPU
 * framebuffer (each sub-sample carries @c density / samples_per_axis³ so total
 * voxel mass is preserved). After
 * @ref DensityImage::combine , per-pixel sums are folded across local GPUs and
 * (when @c System::distributed() is set) across MPI ranks on the host. Rank 0
 * may write the combined frame as a Netpbm @c P5 grayscale image via
 * @ref DensityImage::write_pgm , or as a two-colour @c P6 image via
 * @ref DensityImage::write_ppm .
 */

#include <carbon/cpu.h>
#include <carbon/cuda_annotation.h>
#include <carbon/detail/atomic.h>
#include <carbon/detail/dense/partition.h>
#include <carbon/detail/launch_traits.h>
#include <carbon/floating3d.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/reduce_dist.h>
#include <carbon/render_dist.h>
#include <carbon/system.h>

#ifdef CARBON_HAVE_MPI_NCCL
#include <carbon/distributed.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace carbon {

/**
 * Perspective camera for voxel splatting. Stores a row-major view-projection
 * matrix and framebuffer dimensions. All members are usable from device code.
 */
struct Camera {
  float m[16]{};
  float inv_m[16]{};
  float32_3d eye_{};
  int width_{0};
  int height_{0};

  /** Build a perspective camera looking from @p eye toward @p target. */
  [[nodiscard]] static auto
  perspective(float32_3d eye,
              float32_3d target,
              float32_3d up,
              float fov_y_rad,
              float znear,
              float zfar,
              int width,
              int height) -> Camera;

  /**
   * Build a perspective camera that frames an axis-aligned bounding box.
   *
   * Places the eye on the ray @c center + normalize(eye_offset_dir) * distance so
   * the box fits in the vertical and horizontal fields of view, then chooses
   * @c znear / @c zfar to enclose the box along the view axis. @p eye_offset_dir
   * must be non-zero (typical default @c {0, -1, 0} : camera below, looking up).
   *
   * @p padding scales distance and the far plane (> @c 1 adds margin around the box).
   */
  [[nodiscard]] static auto
  frame_bounds(float32_3d bounds_min,
               float32_3d bounds_max,
               float32_3d eye_offset_dir,
               float32_3d up,
               float fov_y_rad,
               int width,
               int height,
               float padding = 1.15F) -> Camera;

  [[nodiscard]] CARBON_HD constexpr auto
  width() const noexcept -> int {
    return width_;
  }

  [[nodiscard]] CARBON_HD constexpr auto
  height() const noexcept -> int {
    return height_;
  }

  /**
   * Project a world-space point to pixel coordinates.
   *
   * @param world  World-space position (cell-centre convention upstream).
   * @param px     Output pixel column in @c [0, width) .
   * @param py     Output pixel row in @c [0, height) .
   * @param depth  Output clip-space depth @c clip.z / clip.w .
   * @return @c false when the point is behind the camera or outside the NDC frustum.
   */
  [[nodiscard]] CARBON_HD auto
  project(float32_3d world,
          int& px,
          int& py,
          float& depth) const -> bool;

  /**
   * Like @ref project but returns continuous pixel coordinates @c (fx, fy) at
   * sub-pixel precision (pixel centres sit at @c i + 0.5 ).
   */
  [[nodiscard]] CARBON_HD auto
  project_pixel(float32_3d world,
                float& fx,
                float& fy,
                float& depth) const -> bool;

  /**
   * Like @ref project_pixel, but without the lateral frustum test: @p fx / @p fy are filled in
   * even when the projection lands outside the frame. Only a point behind the eye or outside
   * the near/far depth range is rejected.
   *
   * Splatting needs this. A splat covers a footprint around its projected centre, so a surfel
   * whose centre falls just outside the frame can still cover pixels inside it — culling on
   * the centre erases those footprints and thins the silhouette along all four frame edges,
   * most visibly when the camera is close enough that footprints span several pixels.
   *
   * Callers take on the clipping obligation: every pixel actually written must be
   * bounds-checked (see @c SurfelSplatCellFn::try_atomic_min ).
   */
  [[nodiscard]] CARBON_HD auto
  project_pixel_unclipped(float32_3d world,
                          float& fx,
                          float& fy,
                          float& depth) const -> bool;

  /**
   * Reconstruct a world-space ray through pixel @p (px, py) using the stored inverse
   * view-projection matrix and eye position. @p origin receives the eye; @p dir is
   * normalized. Returns @c false when the ray direction degenerates.
   */
  [[nodiscard]] CARBON_HD auto
  ray_through_pixel(int px,
                    int py,
                    float32_3d& origin,
                    float32_3d& dir) const -> bool;
};

namespace render {

/**
 * Smooth density from a signed-distance sample. @p edge_width is the SDF scale
 * over which the transfer rolls off (same units as the field). @p edge_width <= 0
 * recovers the hard step @c (sdf < 0 ? 1 : 0) .
 */
[[nodiscard]] CARBON_HD inline auto
sdf_smooth_density(float sdf,
                   float edge_width) -> float {
  if (edge_width <= 0.0F) {
    return sdf < 0.0F ? 1.0F : 0.0F;
  }
  return 0.5F * (1.0F - std::tanh(sdf / edge_width));
}

/**
 * Disturbance magnitude @c |u - u_inf| , scaled by @p inv_reference_speed (pass
 * @c 1/u_inf to read the result in free-stream units).
 *
 * Rendering raw @c |u| in an external flow wastes the whole dynamic range on the free
 * stream, which fills most of the domain at a value the eye then has to subtract by
 * itself. Subtracting the free stream first makes undisturbed fluid read exactly zero, so a
 * small @c value_threshold turns it fully transparent and only the wake survives —
 * the standard way to isolate a body's disturbance field, and the reason wake
 * visualizations look empty upstream and dense downstream.
 *
 * @p free_stream is a vector, not a speed: near a body the flow turns as well as slows, and
 * a deflected-but-undiminished region is part of the wake even though @c |u| is unchanged.
 */
[[nodiscard]] CARBON_HD inline auto
speed_deviation(float32_3d u,
                float32_3d free_stream,
                float inv_reference_speed) noexcept -> float {
  const float dx = u.x - free_stream.x;
  const float dy = u.y - free_stream.y;
  const float dz = u.z - free_stream.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz) * inv_reference_speed;
}

}  // namespace render

/** Normalization applied by @ref DensityImage::write_pgm and @ref RayMarchImage::write_pgm . */
enum class DensityNormalize {
  MinMax,
  Clamp01,
};

/** Named colormap for @ref RayMarchShading::ScalarField magnitude mapping. */
enum class Colormap {
  Grayscale,
  Viridis,
  Jet,
  /**
   * Diverging blue-white-red. Unlike the three above it is meant to be read around its
   * @b midpoint, so pair it with a symmetric @c [value_min, value_max] : the white band then
   * marks zero and the two lobes are signed. This is the map for quantities that are
   * meaningfully positive or negative — a free-stream-subtracted wake velocity, a pressure
   * coefficient, a vorticity component — where a sequential map like Viridis hides the sign
   * change that is the whole point.
   */
  CoolWarm,
};

namespace render {

[[nodiscard]] CARBON_HD inline auto
clamp01(float v) noexcept -> float {
  return (v < 0.0F) ? 0.0F : ((v > 1.0F) ? 1.0F : v);
}

/**
 * Map @p t in @c [0, 1] (clamped) through @p cmap to linear RGB. Shared by every consumer of
 * @ref Colormap — the ray-march shading modes, the surfel attribute export, the viewer's
 * host-side density colouring — so a colormap looks the same whichever path drew it.
 */
[[nodiscard]] CARBON_HD inline auto
colormap_eval(Colormap cmap,
              float t) -> float32_3d {
  t = clamp01(t);
  if (cmap == Colormap::Grayscale) {
    return {t, t, t};
  }
  if (cmap == Colormap::Jet) {
    const float r = clamp01(1.5F - std::abs(4.0F * t - 3.0F));
    const float g = clamp01(1.5F - std::abs(4.0F * t - 2.0F));
    const float b = clamp01(1.5F - std::abs(4.0F * t - 1.0F));
    return {r, g, b};
  }
  if (cmap == Colormap::CoolWarm) {
    // Piecewise-linear ramp between Moreland's cool-warm endpoints through a near-white
    // pivot. Linear in RGB rather than in his Msh space: the endpoint and midpoint colours
    // are his, the interpolation between them is not, which is visually close enough for a
    // shading tint and costs two lerps instead of a colour-space round trip per sample.
    const float32_3d cool{0.230F, 0.299F, 0.754F};
    const float32_3d pivot{0.865F, 0.865F, 0.865F};
    const float32_3d warm{0.706F, 0.016F, 0.150F};
    if (t < 0.5F) {
      const float u = t * 2.0F;
      return {
          cool.x + (pivot.x - cool.x) * u,
          cool.y + (pivot.y - cool.y) * u,
          cool.z + (pivot.z - cool.z) * u,
      };
    }
    const float u = (t - 0.5F) * 2.0F;
    return {
        pivot.x + (warm.x - pivot.x) * u,
        pivot.y + (warm.y - pivot.y) * u,
        pivot.z + (warm.z - pivot.z) * u,
    };
  }
  const float r =
      0.277727327223877F +
      t * (0.117099384668018F +
           t * (0.254071029900719F +
                t * (-0.034120559300536F + t * (-0.051204190415638F + t * 0.008067089065F))));
  const float g =
      0.005001332034898F +
      t * (0.329654930121324F +
           t * (0.864162171632568F +
                t * (-0.553760031088362F + t * (0.108239631329586F + t * (-0.006048466085F)))));
  const float b =
      0.334040553645506F +
      t * (1.22606402536436F +
           t * (-0.374045550182304F +
                t * (-0.208202026173045F + t * (0.102158279808045F + t * (-0.031983167602847F)))));
  return {clamp01(r), clamp01(g), clamp01(b)};
}

}  // namespace render

/** Ray-march shading model applied during front-to-back integration. */
enum class RayMarchShading {
  /** Scalar emission: integrate @c ρ along the ray (replicated to RGB on export). */
  Emission,
  /** Volumetric gradient shading: pseudo-normal from @c ∇ρ , Lambertian term @c |N·L| . */
  VolumeGradient,
  /** SDF isosurface Blinn–Phong: narrow band around @c sdf ≈ 0 with @c ∇sdf normal. */
  SurfaceBlinn,
  /** Thresholded scalar field: @c value < value_threshold is transparent; colormap maps
   * @c [value_min, value_max] to RGB. */
  ScalarField,
  /**
   * @ref ScalarField opacity driven by a @b second field's colour: the threshold and the
   * accumulated alpha still come from the primary field, but @c [value_min, value_max] and
   * the colormap read an @b attribute scalar sampled at the same point.
   *
   * The volumetric counterpart of @ref SurfaceScalar , and the mode for "where is this
   * quantity, and what is another quantity doing there" — a Q-criterion cloud whose opacity
   * marks the vortex cores and whose hue reports the local speed inside them. Requires the
   * two-loader @ref make_ray_march_kernel overload; with the single-loader one every sample
   * is dropped and the frame comes back empty, deliberately, since the mode has no meaning
   * without its second field.
   */
  ScalarFieldAttribute,
  /**
   * @ref SurfaceBlinn geometry driven by a @b second field: the isosurface, its band and its
   * normal all come from the primary field exactly as in @c SurfaceBlinn , but the material
   * colour at each hit is the @ref RayMarchOptions::colormap lookup of an @b attribute scalar
   * sampled at the same point, over @c [value_min, value_max] .
   *
   * This is the "shape from one quantity, colour from another" mode: a Q-criterion isosurface
   * tinted by local speed, or a wall isosurface tinted by pressure. Requires the two-loader
   * @ref make_ray_march_kernel overload; with the single-loader one there is no attribute to
   * read and every hit falls back to @ref RayMarchOptions::material_color , which renders as
   * plain @c SurfaceBlinn .
   */
  SurfaceScalar,
};

/** Ray-march integration parameters (world-space units). */
struct RayMarchOptions {
  /** Fixed step in world units; @c 0 selects @c 0.5 * grid spacing at launch. */
  float step_size{0.0F};
  /** Beer–Lambert absorption scale @c σ in @c alpha = 1 - exp(-σ ρ dt) . */
  float absorption_scale{1.0F};
  /** Maximum samples along each clipped slab segment. */
  int max_steps{512};
  /** Shading model; default @c Emission preserves pre-RGB behaviour. */
  RayMarchShading shading{RayMarchShading::Emission};
  /** World-space key-light direction (normalized by @ref make_ray_march_kernel when non-zero). */
  float32_3d light_dir{0.0F, -1.0F, 0.0F};
  /** Optional fill-light direction; zero vector disables the second term. */
  float32_3d light_dir2{};
  /** Blinn–Phong / Lambert coefficients (scalar; multiplied by @ref material_color ). */
  float ambient{0.15F};
  float diffuse{0.65F};
  /** Diffuse weight for @ref light_dir2 (specular stays on the key light only). */
  float fill_diffuse{0.0F};
  float specular{0.35F};
  float shininess{48.0F};
  /** @c SurfaceBlinn isosurface band width in world units (same scale as the field). */
  float isosurface_width{3.0F};
  /** RGB tint applied to shaded samples. */
  float32_3d material_color{1.0F, 1.0F, 1.0F};
  /** Optional background colour added as @c T * background when @ref add_background . */
  float32_3d background{};
  /** When @c true , @ref RayMarchImage::write_pgm uses @c color + T * background . */
  bool add_background{false};
  /** When @c use_clip_bounds , intersect the grid AABB with this world box before marching. */
  bool use_clip_bounds{false};
  float32_3d clip_bounds_min{};
  float32_3d clip_bounds_max{};
  /** Linear scale applied by @ref write_pgm when @c DensityNormalize::Clamp01 . */
  float export_exposure{1.0F};
  /** @c ScalarField : samples below this value contribute zero alpha (transparent). */
  float value_threshold{0.0F};
  /** @c ScalarField : lower bound mapped to colormap @c t = 0 . */
  float value_min{0.0F};
  /** @c ScalarField : upper bound mapped to colormap @c t = 1 . */
  float value_max{1.0F};
  /** @c ScalarField : magnitude-to-RGB lookup table. */
  Colormap colormap{Colormap::Viridis};
  /** @c ScalarField : Beer–Lambert scale in @c alpha = 1 - exp(-opacity_scale * (s - threshold) * dt) . */
  float opacity_scale{1.0F};
  /**
   * World frame for the sparse families (@c Block , @c MultiResBlock ), whose @c Span is
   * pure index space and carries no origin or spacing of its own. Voxel @c v then sits at
   * @c world_origin + (v + 0.5) * world_spacing . Ignored by @c Dense / @c DenseDisg , which
   * take their frame from the @c Span they were built with.
   */
  float32_3d world_origin{};
  float world_spacing{1.0F};
};

/**
 * Multi-GPU (and MPI-aware) density framebuffer. Owns one @c width × @c height
 * float buffer per local GPU plus host mirrors for combine.
 */
class DensityImage {
public:
  DensityImage(const System& sys,
               int width,
               int height);

  /** Zero every per-GPU framebuffer slot asynchronously on @p stream . */
  auto
  clear(const Stream& stream) const -> void;

  /** Device pointer to this GPU's framebuffer ( @c width * height floats ). */
  [[nodiscard]] auto
  gpu_framebuffer(size_t relative_gpu_id) const -> float*;

  /**
   * Async D→H each GPU framebuffer, sum into @ref host_frame , and (when
   * distributed) @c MPI_Allreduce the host buffer in place.
   */
  auto
  combine(const Stream& stream) -> void;

  /** Block until @p stream is idle (pair after @ref combine ). */
  auto
  sync(const Stream& stream) const -> void;

  [[nodiscard]] auto
  host_frame() const noexcept -> const std::vector<float>& {
    return host_combined_;
  }

  [[nodiscard]] auto
  width() const noexcept -> int {
    return width_;
  }

  [[nodiscard]] auto
  height() const noexcept -> int {
    return height_;
  }

  [[nodiscard]] auto
  pixel_count() const noexcept -> size_t {
    return pixel_count_;
  }

  [[nodiscard]] auto
  gpu_count() const noexcept -> size_t {
    return gpu_framebuffers_.size();
  }

  /**
   * Write the combined host frame as Netpbm @c P5 on rank 0 only. Returns @c false
   * on I/O failure or when @ref combine has not run.
   */
  [[nodiscard]] auto
  write_pgm(std::string_view path,
            DensityNormalize mode = DensityNormalize::MinMax) const -> bool;

  /**
   * Write the combined host frame as Netpbm @c P6 (RGB) on rank 0 only. Density is
   * tone-mapped to @c t in [0, 1] exactly as @ref write_pgm does, then shaded as
   * @c lerp(background, ink, t) — so empty pixels take @p background and the densest
   * take @p ink . The default is dense-black on white; pass @p ink to tint the
   * field (e.g. green ink on a white page). Returns @c false on I/O failure or when
   * @ref combine has not run.
   */
  [[nodiscard]] auto
  write_ppm(std::string_view path,
            float32_3d ink = {0.0F,
                              0.0F,
                              0.0F},
            float32_3d background = {1.0F,
                                     1.0F,
                                     1.0F},
            DensityNormalize mode = DensityNormalize::MinMax) const -> bool;

private:
  const System& sys_;
  int width_{0};
  int height_{0};
  size_t pixel_count_{0U};
  Buffer<float> gpu_framebuffers_{};
  std::vector<detail::Allocation<float>> host_gpu_mirrors_{};
  std::vector<float> host_combined_{};
  bool combined_ready_{false};
};

/** Per-voxel payload for @ref make_surfel_kernel . */
struct SurfelSample {
  float sdf{};
  float32_3d normal{};
  /**
   * Optional scalar carried to the winning pixel — a surface pressure coefficient, a wall
   * shear, a temperature. Only recorded when the @ref SurfelImage was built with
   * @c with_attribute ; read it back through @ref SurfelImage::host_attribute .
   */
  float attribute{};
};

/**
 * Multi-GPU (and MPI-aware) surfel visibility buffer. Each local GPU atomically writes a
 * packed 64-bit key per pixel (sortable depth in the high 32 bits, octahedral normal in
 * the low 32). @ref combine folds with @c min across GPUs and MPI ranks, then decodes
 * @ref host_depth and @ref host_normal .
 *
 * With @c with_attribute a second plane is kept, packed as the same sortable depth over
 * @ref SurfelSample::attribute and folded by the same @c min . Because depth occupies the
 * high 32 bits of both planes, the nearest fragment wins both, so the attribute a pixel
 * reports is the one belonging to the surface it is showing.
 *
 * Two distinct fragments at @b bit-identical depth are the one exception: the planes then
 * tie-break on their low bits independently and can pair one fragment's normal with
 * another's attribute. This shows up where the surface is most nearly perpendicular to the
 * view and neighbouring surfels' tangent planes agree to the last bit — measured at 0.3 % of
 * covered pixels on a sphere in @c test_render_surfel . Both candidates are at the same
 * depth, so neither answer is wrong, but an attribute with a steep gradient across the
 * surface will show it as a sparse speckle there.
 */
class SurfelImage {
public:
  /** Empty-pixel sentinel written by @ref clear and returned when no fragment wins. */
  static constexpr unsigned long long k_empty_key = ~0ULL;

  /**
   * @p with_attribute allocates the second plane (one more @c uint64 per pixel per GPU plus
   * its D→H copy). Off by default so the common shaded-surface use pays nothing for it; with
   * it off, @ref SurfelSample::attribute is ignored and @ref host_attribute throws rather
   * than handing back zeros that look like data.
   */
  SurfelImage(const System& sys,
              int width,
              int height,
              bool with_attribute = false);

  /** Fill every per-GPU visibility slot with @ref k_empty_key on @p stream . */
  auto
  clear(const Stream& stream) const -> void;

  /** Device pointer to this GPU's visibility buffer ( @c width * height keys ). */
  [[nodiscard]] auto
  gpu_visibility(size_t relative_gpu_id) const -> unsigned long long*;

  /** Device pointer to this GPU's attribute plane, or @c nullptr without @c with_attribute . */
  [[nodiscard]] auto
  gpu_attribute(size_t relative_gpu_id) const -> unsigned long long*;

  [[nodiscard]] auto
  has_attribute() const noexcept -> bool {
    return with_attribute_;
  }

  /**
   * Async D→H each GPU buffer, @c min -fold into @ref host_visibility , optionally
   * @c MPI_Allreduce , then decode @ref host_depth / @ref host_normal .
   */
  auto
  combine(const Stream& stream) -> void;

  auto
  sync(const Stream& stream) const -> void;

  [[nodiscard]] auto
  host_visibility() const noexcept -> const std::vector<unsigned long long>& {
    return host_combined_vis_;
  }

  [[nodiscard]] auto
  host_depth() const noexcept -> const std::vector<float>& {
    return host_depth_;
  }

  [[nodiscard]] auto
  host_normal() const noexcept -> const std::vector<float32_3d>& {
    return host_normal_;
  }

  /**
   * Winning fragment's @ref SurfelSample::attribute per pixel; undefined where
   * @ref has_fragment is @c false . Throws when the image was built without
   * @c with_attribute .
   */
  [[nodiscard]] auto
  host_attribute() const -> const std::vector<float>&;

  [[nodiscard]] auto
  has_fragment(size_t pixel_index) const noexcept -> bool {
    return pixel_index < host_combined_vis_.size() &&
           host_combined_vis_[pixel_index] != k_empty_key;
  }

  [[nodiscard]] auto
  width() const noexcept -> int {
    return width_;
  }

  [[nodiscard]] auto
  height() const noexcept -> int {
    return height_;
  }

  [[nodiscard]] auto
  pixel_count() const noexcept -> size_t {
    return pixel_count_;
  }

  [[nodiscard]] auto
  gpu_count() const noexcept -> size_t {
    return gpu_vis_.size();
  }

  /**
   * Write a Lambert-shaded @c P6 frame on rank 0 only. Empty pixels take @p background ;
   * surface pixels use @ref host_normal against @p light_dir . Returns @c false when
   * @ref combine has not run or on I/O failure.
   */
  [[nodiscard]] auto
  write_ppm(std::string_view path,
            float32_3d light_dir = {0.3F,
                                    -0.85F,
                                    0.45F},
            float32_3d material = {0.831F,
                                   0.953F,
                                   0.188F},
            float32_3d background = {0.949F,
                                     0.957F,
                                     0.953F}) const -> bool;

  /**
   * Write a @c P6 frame colouring each surface pixel by @ref host_attribute through
   * @p colormap over @c [value_min, value_max] , modulated by the same Lambert term as the
   * shaded overload so the geometry stays readable under the data. Empty pixels take
   * @p background . Returns @c false without @c with_attribute , before @ref combine , or on
   * I/O failure.
   */
  [[nodiscard]] auto
  write_attribute_ppm(std::string_view path,
                      Colormap colormap,
                      float value_min,
                      float value_max,
                      float32_3d light_dir = {0.3F,
                                              -0.85F,
                                              0.45F},
                      float32_3d background = {0.949F,
                                               0.957F,
                                               0.953F}) const -> bool;

private:
  const System& sys_;
  int width_{0};
  int height_{0};
  size_t pixel_count_{0U};
  bool with_attribute_{false};
  Buffer<unsigned long long> gpu_vis_{};
  Buffer<unsigned long long> gpu_attr_{};
  std::vector<detail::Allocation<unsigned long long>> host_gpu_mirrors_{};
  std::vector<detail::Allocation<unsigned long long>> host_gpu_attr_mirrors_{};
  std::vector<unsigned long long> host_combined_vis_{};
  std::vector<unsigned long long> host_combined_attr_{};
  std::vector<float> host_depth_{};
  std::vector<float32_3d> host_normal_{};
  std::vector<float> host_attribute_{};
  bool combined_ready_{false};
};

/**
 * segments (@c color , @c transmittance , @c depth ); @ref combine merges local GPUs,
 * @ref dist_combine folds cross-rank segments in depth order.
 */
class RayMarchImage {
public:
  RayMarchImage(const System& sys,
                int width,
                int height);

  auto
  clear(const Stream& stream) const -> void;

  [[nodiscard]] auto
  gpu_segment_color_r(size_t relative_gpu_id) const -> float*;

  [[nodiscard]] auto
  gpu_segment_color_g(size_t relative_gpu_id) const -> float*;

  [[nodiscard]] auto
  gpu_segment_color_b(size_t relative_gpu_id) const -> float*;

  [[nodiscard]] auto
  gpu_segment_transmittance(size_t relative_gpu_id) const -> float*;

  [[nodiscard]] auto
  gpu_segment_depth(size_t relative_gpu_id) const -> float*;

  /** D→H local GPU segments and merge same-rank slabs per pixel by @c depth . */
  auto
  combine(const Stream& stream) -> void;

  /** MPI_Allgather rank-local segments and composite in depth order (no-op without MPI). */
  auto
  dist_combine() -> void;

  auto
  set_export_options(bool add_background,
                     float32_3d background,
                     float exposure = 1.0F) noexcept -> void;

  auto
  sync(const Stream& stream) const -> void;

  /** Per-pixel luminance @c 0.299R + 0.587G + 0.114B after @ref dist_combine . */
  [[nodiscard]] auto
  host_color() const noexcept -> const std::vector<float>& {
    return host_combined_luminance_;
  }

  [[nodiscard]] auto
  host_color_r() const noexcept -> const std::vector<float>& {
    return host_combined_color_r_;
  }

  [[nodiscard]] auto
  host_color_g() const noexcept -> const std::vector<float>& {
    return host_combined_color_g_;
  }

  [[nodiscard]] auto
  host_color_b() const noexcept -> const std::vector<float>& {
    return host_combined_color_b_;
  }

  [[nodiscard]] auto
  host_transmittance() const noexcept -> const std::vector<float>& {
    return host_combined_trans_;
  }

  /**
   * Per-pixel distance from the camera origin to the first sample that contributed opacity,
   * @c inf where the ray never picked any up. Same meaning and units as
   * @ref SurfelImage::host_depth , so a ray-marched layer and a surfel layer can be
   * depth-tested against each other pixel by pixel instead of composited in a fixed order.
   *
   * Valid after @ref dist_combine .
   */
  [[nodiscard]] auto
  host_depth() const noexcept -> const std::vector<float>& {
    return host_combined_depth_;
  }

  [[nodiscard]] auto
  width() const noexcept -> int {
    return width_;
  }

  [[nodiscard]] auto
  height() const noexcept -> int {
    return height_;
  }

  [[nodiscard]] auto
  pixel_count() const noexcept -> size_t {
    return pixel_count_;
  }

  [[nodiscard]] auto
  gpu_count() const noexcept -> size_t {
    return gpu_segment_color_r_.size();
  }

  /**
   * Write the combined host frame as Netpbm @c P6 (RGB) on rank 0 only. Min–max tone mapping
   * uses a shared luminance scale so hue is preserved. Returns @c false on I/O failure or when
   * @ref dist_combine has not run.
   */
  [[nodiscard]] auto
  write_pgm(std::string_view path,
            DensityNormalize mode = DensityNormalize::MinMax) const -> bool;

private:
  const System& sys_;
  int width_{0};
  int height_{0};
  size_t pixel_count_{0U};
  Buffer<float> gpu_segment_color_r_{};
  Buffer<float> gpu_segment_color_g_{};
  Buffer<float> gpu_segment_color_b_{};
  Buffer<float> gpu_segment_trans_{};
  Buffer<float> gpu_segment_depth_{};
  std::vector<detail::Allocation<float>> host_gpu_color_r_{};
  std::vector<detail::Allocation<float>> host_gpu_color_g_{};
  std::vector<detail::Allocation<float>> host_gpu_color_b_{};
  std::vector<detail::Allocation<float>> host_gpu_trans_{};
  std::vector<detail::Allocation<float>> host_gpu_depth_{};
  std::vector<float> host_rank_color_r_{};
  std::vector<float> host_rank_color_g_{};
  std::vector<float> host_rank_color_b_{};
  std::vector<float> host_rank_trans_{};
  std::vector<float> host_rank_depth_{};
  std::vector<float> host_combined_color_r_{};
  std::vector<float> host_combined_color_g_{};
  std::vector<float> host_combined_color_b_{};
  std::vector<float> host_combined_luminance_{};
  std::vector<float> host_combined_trans_{};
  std::vector<float> host_combined_depth_{};
  bool combined_ready_{false};
  bool export_add_background_{false};
  float32_3d export_background_{};
  float export_exposure_{1.0F};
};

namespace render {

/**
 * Trilinear sample of a scalar field stored on a dense z-slab partition at @p world .
 * Returns @c 0 when @p world lies outside the global @c xy bounds or this span's owned @c z .
 */
template <typename T,
          int Components,
          typename Span>
[[nodiscard]] CARBON_HD auto
sample_field_trilinear(const detail::dense::Partition<T,
                                                      Components>& partition,
                       const Span& span,
                       float32_3d world) -> float;

/**
 * Trilinear sample of a user density callable @p user_fn at @p world . @p user_fn must be
 * invocable as @c (Span, Idx) -> float .
 */
template <typename Span,
          typename UserFn>
[[nodiscard]] CARBON_HD auto
sample_density_trilinear(const Span& span,
                         UserFn user_fn,
                         float32_3d world) -> float;

/**
 * Central-difference gradient of @p user_fn at @p world (spacing-aware step @c 0.5 * spacing ).
 */
template <typename Span,
          typename UserFn>
[[nodiscard]] CARBON_HD auto
sample_density_gradient_trilinear(const Span& span,
                                  UserFn user_fn,
                                  float32_3d world) -> float32_3d;

}  // namespace render

namespace detail {

template <typename Grid, typename UserFn>
struct DensitySplatCellFn {
  UserFn user_fn;
  float* fb;
  Camera camera;
  int frame_w;
  int frame_h;
  float32_3d block_origin;
  float block_spacing;
  int samples_per_axis;
  float splat_sigma_px;

  CARBON_HD auto
  splat_at(float32_3d world,
           float splat_density) const -> void {
    float fx = 0.0F;
    float fy = 0.0F;
    float depth = 0.0F;
    if (!camera.project_pixel(world, fx, fy, depth)) {
      return;
    }

    if (splat_sigma_px <= 0.0F) {
      const int px = static_cast<int>(fx);
      const int py = static_cast<int>(fy);
      if (px < 0 || py < 0 || px >= frame_w || py >= frame_h) {
        return;
      }
      const size_t offset =
          static_cast<size_t>(py) * static_cast<size_t>(frame_w) + static_cast<size_t>(px);
      (void)atomic_add(fb + offset, splat_density);
      return;
    }

    const float sigma = splat_sigma_px;
    const float inv_2sigma2 = 1.0F / (2.0F * sigma * sigma);
    const float reach = 3.0F * sigma;
    const int px_min = static_cast<int>(std::ceil(fx - reach - 0.5F));
    const int px_max = static_cast<int>(std::floor(fx + reach - 0.5F));
    const int py_min = static_cast<int>(std::ceil(fy - reach - 0.5F));
    const int py_max = static_cast<int>(std::floor(fy + reach - 0.5F));

    for (int py = py_min; py <= py_max; ++py) {
      if (py < 0 || py >= frame_h) {
        continue;
      }
      const float off_y = fy - (static_cast<float>(py) + 0.5F);
      for (int px = px_min; px <= px_max; ++px) {
        if (px < 0 || px >= frame_w) {
          continue;
        }
        const float off_x = fx - (static_cast<float>(px) + 0.5F);
        const float w = std::exp(-(off_x * off_x + off_y * off_y) * inv_2sigma2);
        if (w < 1.0e-4F) {
          continue;
        }
        const size_t offset =
            static_cast<size_t>(py) * static_cast<size_t>(frame_w) + static_cast<size_t>(px);
        (void)atomic_add(fb + offset, splat_density * w);
      }
    }
  }

  CARBON_HD auto
  operator()(typename Grid::Span span,
             typename Grid::Idx idx) const -> void {
    const float density = user_fn(span, idx);
    if (density == 0.0F) {
      return;
    }

    const int n = samples_per_axis;
    const float n_f = static_cast<float>(n);
    const float splat_density = density / (n_f * n_f * n_f);

    if constexpr (grid_has_dense_iteration_v<Grid>) {
      const int64_3d gc = span.global_location(idx);
      const float sp = span.spacing();
      const float32_3d origin = span.origin();
      for (int iz = 0; iz < n; ++iz) {
        for (int iy = 0; iy < n; ++iy) {
          for (int ix = 0; ix < n; ++ix) {
            const float32_3d world{
                origin.x + (static_cast<float>(gc.x) + (static_cast<float>(ix) + 0.5F) / n_f) * sp,
                origin.y + (static_cast<float>(gc.y) + (static_cast<float>(iy) + 0.5F) / n_f) * sp,
                origin.z + (static_cast<float>(gc.z) + (static_cast<float>(iz) + 0.5F) / n_f) * sp,
            };
            splat_at(world, splat_density);
          }
        }
      }
    } else {
      const int64_3d gc = span.global_location(idx);
      for (int iz = 0; iz < n; ++iz) {
        for (int iy = 0; iy < n; ++iy) {
          for (int ix = 0; ix < n; ++ix) {
            const float32_3d world{
                block_origin.x +
                    (static_cast<float>(gc.x) + (static_cast<float>(ix) + 0.5F) / n_f) *
                        block_spacing,
                block_origin.y +
                    (static_cast<float>(gc.y) + (static_cast<float>(iy) + 0.5F) / n_f) *
                        block_spacing,
                block_origin.z +
                    (static_cast<float>(gc.z) + (static_cast<float>(iz) + 0.5F) / n_f) *
                        block_spacing,
            };
            splat_at(world, splat_density);
          }
        }
      }
    }
  }
};

}  // namespace detail

/**
 * Build a compute @ref Kernel that splats per-voxel densities into @p image .
 *
 * @p density_loader follows the Neon two-level idiom: a host loading lambda that
 * returns a @c CARBON_HD per-cell callable @c (Span, Idx) -> float .
 *
 * @p block_origin and @p block_spacing apply only to @c Block grids (which carry
 * integer global coordinates on @c Span ); @c Dense / @c DenseDisg use
 * @c Span::origin() / @c Span::spacing() .
 *
 * @p samples_per_axis controls how many evenly spaced points per cell axis are
 * projected ( @c 1 = cell centre only; @c 2 = 8 sub-samples ). Each sub-sample
 * receives @c density / samples_per_axis³ so the integrated mass per voxel is
 * unchanged.
 *
 * @p splat_sigma_px sets the screen-space Gaussian splat radius in pixels
 * ( @c 0 = single-pixel nearest-neighbour; @c 1.5 – @c 2 typical for anti-aliasing ).
 */
template <typename Grid,
          typename DensityLoader>
[[nodiscard]] auto
make_density_kernel(const Grid& grid,
                    DensityImage& image,
                    Camera camera,
                    DensityLoader&& density_loader,
                    float32_3d block_origin = {},
                    float block_spacing = 1.0F,
                    int samples_per_axis = 1,
                    float splat_sigma_px = 0.0F) -> Kernel;

/**
 * Build a compute @ref Kernel that splats near-surface voxels into @p image .
 *
 * @p surfel_loader returns a @c CARBON_HD per-cell @ref SurfelSample . Voxels with
 * @c abs(sdf) > band_width are skipped. Each surviving voxel is snapped to the
 * isosurface along its normal, projected, and written with a packed @c atomicMin depth
 * test over its footprint.
 *
 * The footprint is a world-space tangent disk one voxel across (with a ~1 pixel screen-space
 * floor so it never falls between samples when the grid is zoomed out), so it shrinks and
 * foreshortens on its own as the camera moves. @p splat_radius_px is only an upper bound on how
 * many pixels one voxel may be tested against — raise it to allow closer zoom, lower it to cap
 * the per-voxel cost. It does not set the footprint's shape.
 */
template <typename Grid,
          typename SurfelLoader>
[[nodiscard]] auto
make_surfel_kernel(const Grid& grid,
                   SurfelImage& image,
                   Camera camera,
                   SurfelLoader&& surfel_loader,
                   float band_width,
                   float splat_radius_px = 1.5F,
                   float32_3d block_origin = {},
                   float block_spacing = 1.0F) -> Kernel;

/**
 * Build a pixel-parallel ray-march @ref Kernel for @c Dense / @c DenseDisg grids.
 *
 * Each local GPU integrates only its global z-slab along each ray and writes slab segments
 * into @p image . Call @ref RayMarchImage::combine and @ref RayMarchImage::dist_combine
 * before @ref RayMarchImage::write_pgm .
 */
template <typename Grid,
          typename DensityLoader>
[[nodiscard]] auto
make_ray_march_kernel(const Grid& grid,
                      RayMarchImage& image,
                      Camera camera,
                      RayMarchOptions opts,
                      DensityLoader&& density_loader) -> Kernel;

}  // namespace carbon

#include <carbon/detail/render.inl>
#include <carbon/detail/render_ray_march.inl>
