// detail/render.inl — implementation for <carbon/render.h>. Included exactly once
// from the public header tail; not a standalone include.

namespace carbon {
namespace detail {
namespace render_math {

[[nodiscard]] inline auto
dot3(float ax,
     float ay,
     float az,
     float bx,
     float by,
     float bz) noexcept -> float {
  return ax * bx + ay * by + az * bz;
}

inline auto
cross3(float ax,
       float ay,
       float az,
       float bx,
       float by,
       float bz,
       float& ox,
       float& oy,
       float& oz) noexcept -> void {
  ox = ay * bz - az * by;
  oy = az * bx - ax * bz;
  oz = ax * by - ay * bx;
}

[[nodiscard]] CARBON_HD inline auto
normalize3(float& x,
           float& y,
           float& z) noexcept -> bool {
  const float len = std::sqrt(x * x + y * y + z * z);
  if (len <= 0.0F) {
    return false;
  }
  const float inv = 1.0F / len;
  x *= inv;
  y *= inv;
  z *= inv;
  return true;
}

inline auto
mat4_multiply(float out[16],
              const float a[16],
              const float b[16]) noexcept -> void {
  float tmp[16]{};
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      float sum = 0.0F;
      for (int k = 0; k < 4; ++k) {
        sum += a[i * 4 + k] * b[k * 4 + j];
      }
      tmp[i * 4 + j] = sum;
    }
  }
  for (int i = 0; i < 16; ++i) {
    out[i] = tmp[i];
  }
}

[[nodiscard]] inline auto
mat4_invert(float out[16],
            const float m[16]) noexcept -> bool {
  float aug[4][8]{};
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      aug[i][j] = m[i * 4 + j];
      aug[i][j + 4] = (i == j) ? 1.0F : 0.0F;
    }
  }
  for (int col = 0; col < 4; ++col) {
    int pivot = col;
    for (int row = col + 1; row < 4; ++row) {
      if (std::abs(aug[row][col]) > std::abs(aug[pivot][col])) {
        pivot = row;
      }
    }
    if (std::abs(aug[pivot][col]) < 1.0e-12F) {
      return false;
    }
    if (pivot != col) {
      for (int j = 0; j < 8; ++j) {
        const float tmp = aug[col][j];
        aug[col][j] = aug[pivot][j];
        aug[pivot][j] = tmp;
      }
    }
    const float inv_p = 1.0F / aug[col][col];
    for (int j = 0; j < 8; ++j) {
      aug[col][j] *= inv_p;
    }
    for (int row = 0; row < 4; ++row) {
      if (row == col) {
        continue;
      }
      const float factor = aug[row][col];
      for (int j = 0; j < 8; ++j) {
        aug[row][j] -= factor * aug[col][j];
      }
    }
  }
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      out[i * 4 + j] = aug[i][j + 4];
    }
  }
  return true;
}

inline auto
look_at(float out[16],
        float32_3d eye,
        float32_3d target,
        float32_3d up) noexcept -> void {
  float fx = target.x - eye.x;
  float fy = target.y - eye.y;
  float fz = target.z - eye.z;
  if (!normalize3(fx, fy, fz)) {
    fx = 0.0F;
    fy = 0.0F;
    fz = -1.0F;
  }

  float sx = fy * up.z - fz * up.y;
  float sy = fz * up.x - fx * up.z;
  float sz = fx * up.y - fy * up.x;
  if (!normalize3(sx, sy, sz)) {
    // Forward is parallel to up, so the caller's up carries no side direction. Substitute the
    // world axis least aligned with forward: it cannot be parallel to forward, so the basis
    // stays orthonormal. A fixed fallback cannot promise that — for a view along the same axis
    // as the fallback the side axis collapses onto forward, the third axis comes out zero and
    // Camera::perspective throws on the singular view-projection.
    const float ax = std::fabs(fx);
    const float ay = std::fabs(fy);
    const float az = std::fabs(fz);
    float hx = 0.0F;
    float hy = 0.0F;
    float hz = 0.0F;
    if (ax <= ay && ax <= az) {
      hx = 1.0F;
    } else if (ay <= az) {
      hy = 1.0F;
    } else {
      hz = 1.0F;
    }
    sx = fy * hz - fz * hy;
    sy = fz * hx - fx * hz;
    sz = fx * hy - fy * hx;
    // |f . h| <= 1/sqrt(3) by construction, so this cross product has length >= 0.81 and
    // normalizing it cannot fail.
    static_cast<void>(normalize3(sx, sy, sz));
  }

  float ux = sy * fz - sz * fy;
  float uy = sz * fx - sx * fz;
  float uz = sx * fy - sy * fx;

  out[0] = sx;
  out[1] = sy;
  out[2] = sz;
  out[3] = -dot3(sx, sy, sz, eye.x, eye.y, eye.z);
  out[4] = ux;
  out[5] = uy;
  out[6] = uz;
  out[7] = -dot3(ux, uy, uz, eye.x, eye.y, eye.z);
  out[8] = -fx;
  out[9] = -fy;
  out[10] = -fz;
  out[11] = dot3(fx, fy, fz, eye.x, eye.y, eye.z);
  out[12] = 0.0F;
  out[13] = 0.0F;
  out[14] = 0.0F;
  out[15] = 1.0F;
}

inline auto
perspective(float out[16],
            float fov_y_rad,
            float aspect,
            float znear,
            float zfar) noexcept -> void {
  const float f = 1.0F / std::tan(0.5F * fov_y_rad);
  const float a = f / aspect;
  const float c = (zfar + znear) / (znear - zfar);
  const float d = (2.0F * znear * zfar) / (znear - zfar);

  out[0] = a;
  out[1] = 0.0F;
  out[2] = 0.0F;
  out[3] = 0.0F;
  out[4] = 0.0F;
  out[5] = f;
  out[6] = 0.0F;
  out[7] = 0.0F;
  out[8] = 0.0F;
  out[9] = 0.0F;
  out[10] = c;
  out[11] = d;
  out[12] = 0.0F;
  out[13] = 0.0F;
  out[14] = -1.0F;
  out[15] = 0.0F;
}

}  // namespace render_math
}  // namespace detail

inline auto
Camera::perspective(float32_3d eye,
                    float32_3d target,
                    float32_3d up,
                    float fov_y_rad,
                    float znear,
                    float zfar,
                    int width,
                    int height) -> Camera {
  if (width <= 0 || height <= 0) {
    log::throw_with_log<std::invalid_argument>(
        "Camera::perspective: width and height must be positive (got {} x {})", width, height);
  }
  if (znear <= 0.0F || zfar <= znear) {
    log::throw_with_log<std::invalid_argument>(
        "Camera::perspective: require 0 < znear < zfar (got znear={}, zfar={})", znear, zfar);
  }

  float view[16]{};
  float proj[16]{};
  detail::render_math::look_at(view, eye, target, up);
  detail::render_math::perspective(
      proj, fov_y_rad, static_cast<float>(width) / static_cast<float>(height), znear, zfar);
  Camera cam{};
  detail::render_math::mat4_multiply(cam.m, proj, view);
  cam.width_ = width;
  cam.height_ = height;
  cam.eye_ = eye;
  if (!detail::render_math::mat4_invert(cam.inv_m, cam.m)) {
    log::throw_with_log<std::runtime_error>(
        "Camera::perspective: view-projection matrix is singular");
  }
  return cam;
}

inline auto
Camera::frame_bounds(float32_3d bounds_min,
                     float32_3d bounds_max,
                     float32_3d eye_offset_dir,
                     float32_3d up,
                     float fov_y_rad,
                     int width,
                     int height,
                     float padding) -> Camera {
  if (width <= 0 || height <= 0) {
    log::throw_with_log<std::invalid_argument>(
        "Camera::frame_bounds: width and height must be positive (got {} x {})", width, height);
  }
  if (padding <= 0.0F) {
    log::throw_with_log<std::invalid_argument>(
        "Camera::frame_bounds: padding must be positive (got {})", padding);
  }
  if (bounds_min.x > bounds_max.x || bounds_min.y > bounds_max.y || bounds_min.z > bounds_max.z) {
    log::throw_with_log<std::invalid_argument>(
        "Camera::frame_bounds: bounds_min must be component-wise <= bounds_max");
  }

  float dir_x = eye_offset_dir.x;
  float dir_y = eye_offset_dir.y;
  float dir_z = eye_offset_dir.z;
  if (!detail::render_math::normalize3(dir_x, dir_y, dir_z)) {
    log::throw_with_log<std::invalid_argument>(
        "Camera::frame_bounds: eye_offset_dir must be non-zero");
  }

  const float32_3d center{
      0.5F * (bounds_min.x + bounds_max.x),
      0.5F * (bounds_min.y + bounds_max.y),
      0.5F * (bounds_min.z + bounds_max.z),
  };
  const float size_x = bounds_max.x - bounds_min.x;
  const float size_y = bounds_max.y - bounds_min.y;
  const float size_z = bounds_max.z - bounds_min.z;
  const float radius = 0.5F * std::sqrt(size_x * size_x + size_y * size_y + size_z * size_z);

  const float aspect = static_cast<float>(width) / static_cast<float>(height);
  const float tan_half_y = std::tan(0.5F * fov_y_rad);
  const float tan_half_x = tan_half_y * aspect;
  const float dist_y = radius / tan_half_y;
  const float dist_x = radius / tan_half_x;
  const float dist = std::max(dist_y, dist_x) * padding;

  const float32_3d target = center;
  const float32_3d eye{
      center.x + dir_x * dist,
      center.y + dir_y * dist,
      center.z + dir_z * dist,
  };

  const float znear = std::max(0.01F, (dist - radius) * 0.1F);
  const float zfar = dist + radius * 2.0F + std::max({size_x, size_y, size_z});

  return perspective(eye, target, up, fov_y_rad, znear, zfar, width, height);
}

CARBON_HD inline auto
Camera::project_pixel_unclipped(float32_3d world,
                                float& fx,
                                float& fy,
                                float& depth) const -> bool {
  const float x = world.x;
  const float y = world.y;
  const float z = world.z;

  const float clip_x = m[0] * x + m[1] * y + m[2] * z + m[3];
  const float clip_y = m[4] * x + m[5] * y + m[6] * z + m[7];
  const float clip_z = m[8] * x + m[9] * y + m[10] * z + m[11];
  const float clip_w = m[12] * x + m[13] * y + m[14] * z + m[15];

  if (clip_w <= 0.0F) {
    return false;
  }

  const float inv_w = 1.0F / clip_w;
  const float ndc_x = clip_x * inv_w;
  const float ndc_y = clip_y * inv_w;
  depth = clip_z * inv_w;

  if (depth < -1.0F || depth > 1.0F) {
    return false;
  }

  fx = (ndc_x * 0.5F + 0.5F) * static_cast<float>(width_);
  fy = (1.0F - (ndc_y * 0.5F + 0.5F)) * static_cast<float>(height_);
  return true;
}

CARBON_HD inline auto
Camera::project_pixel(float32_3d world,
                      float& fx,
                      float& fy,
                      float& depth) const -> bool {
  if (!project_pixel_unclipped(world, fx, fy, depth)) {
    return false;
  }
  // Lateral frustum test, in pixel space. Equivalent to the ndc_x / ndc_y in [-1, 1] test this
  // used to spell inline: the mapping above is affine and monotone, sending -1 to 0 and +1 to
  // width_ (height_ on y, with the y flip), so the inclusive bounds correspond exactly.
  return fx >= 0.0F && fx <= static_cast<float>(width_) && fy >= 0.0F &&
         fy <= static_cast<float>(height_);
}

CARBON_HD inline auto
Camera::project(float32_3d world,
                int& px,
                int& py,
                float& depth) const -> bool {
  float fx = 0.0F;
  float fy = 0.0F;
  if (!project_pixel(world, fx, fy, depth)) {
    return false;
  }

  px = static_cast<int>(fx);
  py = static_cast<int>(fy);
  if (px < 0 || py < 0 || px >= width_ || py >= height_) {
    return false;
  }
  return true;
}

CARBON_HD inline auto
Camera::ray_through_pixel(int px,
                          int py,
                          float32_3d& origin,
                          float32_3d& dir) const -> bool {
  origin = eye_;

  const float ndc_x = ((static_cast<float>(px) + 0.5F) / static_cast<float>(width_)) * 2.0F - 1.0F;
  const float ndc_y = 1.0F - ((static_cast<float>(py) + 0.5F) / static_cast<float>(height_)) * 2.0F;

  float32_3d near_world{};
  float32_3d far_world{};
  for (int z_pass = 0; z_pass < 2; ++z_pass) {
    const float ndc_z = (z_pass == 0) ? -1.0F : 1.0F;
    const float clip[4] = {ndc_x, ndc_y, ndc_z, 1.0F};
    float world_h[4]{};
    for (int i = 0; i < 4; ++i) {
      float sum = 0.0F;
      for (int j = 0; j < 4; ++j) {
        sum += inv_m[i * 4 + j] * clip[j];
      }
      world_h[i] = sum;
    }
    if (std::abs(world_h[3]) < 1.0e-8F) {
      return false;
    }
    const float inv_w = 1.0F / world_h[3];
    float32_3d& out = (z_pass == 0) ? near_world : far_world;
    out.x = world_h[0] * inv_w;
    out.y = world_h[1] * inv_w;
    out.z = world_h[2] * inv_w;
  }

  dir.x = far_world.x - near_world.x;
  dir.y = far_world.y - near_world.y;
  dir.z = far_world.z - near_world.z;
  return detail::render_math::normalize3(dir.x, dir.y, dir.z);
}

inline DensityImage::DensityImage(const System& sys,
                                  int width,
                                  int height)
    : sys_(sys),
      width_(width),
      height_(height) {
  if (width <= 0 || height <= 0) {
    log::throw_with_log<std::invalid_argument>(
        "DensityImage: width and height must be positive (got {} x {})", width, height);
  }

  const size_t gpu_count = sys.gpu_group().size();
  if (gpu_count == 0U) {
    log::throw_with_log<std::runtime_error>(
        "DensityImage: System has no GPUs (gpu_group().size() == 0)");
  }

  pixel_count_ = static_cast<size_t>(width_) * static_cast<size_t>(height_);
  std::vector<size_t> counts(gpu_count, pixel_count_);
  gpu_framebuffers_ = sys.buffer<float>(counts);

  host_gpu_mirrors_.reserve(gpu_count);
  for (size_t g = 0; g < gpu_count; ++g) {
    host_gpu_mirrors_.push_back(Cpu::buffer<float>(pixel_count_));
  }
  host_combined_.assign(pixel_count_, 0.0F);
}

inline auto
DensityImage::clear(const Stream& stream) const -> void {
  const size_t nbytes = pixel_count_ * sizeof(float);
  for (size_t g = 0; g < gpu_count(); ++g) {
    sys_.async_memset(gpu_framebuffers_[g], 0, nbytes, stream);
  }
}

inline auto
DensityImage::gpu_framebuffer(size_t relative_gpu_id) const -> float* {
  if (relative_gpu_id >= gpu_count()) {
    log::throw_with_log<std::out_of_range>(
        "DensityImage::gpu_framebuffer: relative_gpu_id {} >= gpu_count {}", relative_gpu_id,
        gpu_count());
  }
  return gpu_framebuffers_[relative_gpu_id].data();
}

inline auto
DensityImage::combine(const Stream& stream) -> void {
  const size_t nbytes = pixel_count_ * sizeof(float);
  for (size_t g = 0; g < gpu_count(); ++g) {
    sys_.async_copy(gpu_framebuffers_[g], host_gpu_mirrors_[g], nbytes, stream);
  }
  stream.wait();

  std::fill(host_combined_.begin(), host_combined_.end(), 0.0F);
  for (size_t g = 0; g < gpu_count(); ++g) {
    const float* src = host_gpu_mirrors_[g].data();
    for (size_t i = 0; i < pixel_count_; ++i) {
      host_combined_[i] += src[i];
    }
  }

  if (sys_.distributed() != nullptr) {
    reduce::dist_host_allreduce_in_place(sys_, host_combined_.data(), pixel_count_,
                                         reduce::ReduceOpKind::Sum,
                                         reduce::data_kind<float>::value);
  }

  combined_ready_ = true;
}

inline auto
DensityImage::sync(const Stream& stream) const -> void {
  stream.wait();
}

namespace render_export {

// Netpbm writers only run on rank 0; other ranks report success without writing.
[[nodiscard]] inline auto
is_writer_rank(const System& sys) -> bool {
#ifdef CARBON_HAVE_MPI_NCCL
  if (const Distributed* dist = sys.distributed(); dist != nullptr) {
    return dist->world_rank() == 0;
  }
#endif
  (void)sys;
  return true;
}

// Tone-map scale shared by every DensityImage writer, so a grayscale and a colour
// export of the same frame map density to the same normalized t.
struct ToneMap {
  float offset{0.0F};
  float inv_span{1.0F};

  [[nodiscard]] auto
  apply(float v) const -> float {
    return std::clamp((v - offset) * inv_span, 0.0F, 1.0F);
  }
};

[[nodiscard]] inline auto
tone_map(const std::vector<float>& frame,
         DensityNormalize mode) -> ToneMap {
  if (mode != DensityNormalize::MinMax) {
    return {};
  }
  float mn = std::numeric_limits<float>::infinity();
  float mx = -std::numeric_limits<float>::infinity();
  for (float v : frame) {
    if (std::isfinite(v)) {
      mn = std::min(mn, v);
      mx = std::max(mx, v);
    }
  }
  if (!std::isfinite(mn) || !std::isfinite(mx) || mx <= mn) {
    return {};
  }
  return {mn, 1.0F / (mx - mn)};
}

[[nodiscard]] inline auto
to_byte(float unit) -> unsigned char {
  return static_cast<unsigned char>(std::lround(std::clamp(unit, 0.0F, 1.0F) * 255.0F));
}

}  // namespace render_export

inline auto
DensityImage::write_pgm(std::string_view path,
                        DensityNormalize mode) const -> bool {
  if (!render_export::is_writer_rank(sys_)) {
    return true;
  }
  if (!combined_ready_) {
    return false;
  }

  const auto map = render_export::tone_map(host_combined_, mode);

  std::ofstream out(std::string(path), std::ios::binary);
  if (!out) {
    return false;
  }

  out << "P5\n" << width_ << ' ' << height_ << "\n255\n";
  for (float v : host_combined_) {
    const unsigned char b = std::isfinite(v) ? render_export::to_byte(map.apply(v)) : 0;
    out.put(static_cast<char>(b));
  }
  return static_cast<bool>(out);
}

inline auto
DensityImage::write_ppm(std::string_view path,
                        float32_3d ink,
                        float32_3d background,
                        DensityNormalize mode) const -> bool {
  if (!render_export::is_writer_rank(sys_)) {
    return true;
  }
  if (!combined_ready_) {
    return false;
  }

  const auto map = render_export::tone_map(host_combined_, mode);

  std::ofstream out(std::string(path), std::ios::binary);
  if (!out) {
    return false;
  }

  out << "P6\n" << width_ << ' ' << height_ << "\n255\n";
  for (float v : host_combined_) {
    const float t = std::isfinite(v) ? map.apply(v) : 0.0F;
    const std::array<char, 3> rgb{
        static_cast<char>(render_export::to_byte(background.x + t * (ink.x - background.x))),
        static_cast<char>(render_export::to_byte(background.y + t * (ink.y - background.y))),
        static_cast<char>(render_export::to_byte(background.z + t * (ink.z - background.z)))};
    out.write(rgb.data(), static_cast<std::streamsize>(rgb.size()));
  }
  return static_cast<bool>(out);
}

namespace detail {

/** Map a @c float depth to a @c uint32 key whose unsigned order matches numeric @c float order. */
[[nodiscard]] CARBON_HD inline auto
float_to_sortable_uint(float v) noexcept -> uint32_t {
  const uint32_t bits = __builtin_bit_cast(uint32_t, v);
  return bits ^ ((-(bits >> 31U)) | 0x80000000U);
}

/**
 * Inverse of @ref float_to_sortable_uint . The encoder flips the sign bit of a positive
 * float and every bit of a negative one, so the decoder has to branch on the @b encoded top
 * bit — set means the value was positive — and undo whichever transform was applied.
 */
[[nodiscard]] inline auto
sortable_uint_to_float(uint32_t u) noexcept -> float {
  const uint32_t mask = ((u >> 31U) - 1U) | 0x80000000U;
  return __builtin_bit_cast(float, u ^ mask);
}

/** Pack a unit normal into 16:16 octahedral coordinates in the low 32 bits of a surfel key. */
[[nodiscard]] CARBON_HD inline auto
encode_octahedral_normal(float nx,
                         float ny,
                         float nz) noexcept -> uint32_t {
  const float ax = std::abs(nx);
  const float ay = std::abs(ny);
  const float az = std::abs(nz);
  const float denom = ax + ay + az;
  if (denom <= 0.0F) {
    return 0U;
  }
  float ox = nx / denom;
  float oy = ny / denom;
  if (nz < 0.0F) {
    const float sx = ox;
    const float sy = oy;
    ox = (1.0F - std::abs(sy)) * (sx >= 0.0F ? 1.0F : -1.0F);
    oy = (1.0F - std::abs(sx)) * (sy >= 0.0F ? 1.0F : -1.0F);
  }
  const auto clamp01 = [](float v) -> float {
    return v < 0.0F ? 0.0F : (v > 1.0F ? 1.0F : v);
  };
  const uint32_t u = static_cast<uint32_t>(std::lround(clamp01(ox * 0.5F + 0.5F) * 65535.0F));
  const uint32_t v = static_cast<uint32_t>(std::lround(clamp01(oy * 0.5F + 0.5F) * 65535.0F));
  return (u << 16U) | v;
}

/** Decode @ref encode_octahedral_normal . */
[[nodiscard]] inline auto
decode_octahedral_normal(uint32_t enc) noexcept -> float32_3d {
  const float u = static_cast<float>(enc >> 16U) / 65535.0F;
  const float v = static_cast<float>(enc & 0xFFFFU) / 65535.0F;
  float ox = u * 2.0F - 1.0F;
  float oy = v * 2.0F - 1.0F;
  float oz = 1.0F - std::abs(ox) - std::abs(oy);
  // oz keeps its negative value: it carries the lower-hemisphere sign that the fold below
  // moves out of (ox, oy). Zeroing it collapses every nz < 0 normal onto the equator.
  if (oz < 0.0F) {
    const float sx = ox;
    const float sy = oy;
    ox = (1.0F - std::abs(sy)) * (sx >= 0.0F ? 1.0F : -1.0F);
    oy = (1.0F - std::abs(sx)) * (sy >= 0.0F ? 1.0F : -1.0F);
  }
  if (!render_math::normalize3(ox, oy, oz)) {
    return {0.0F, 0.0F, 1.0F};
  }
  return {ox, oy, oz};
}

template <typename Grid, typename UserFn>
struct SurfelSplatCellFn {
  // Footprint radius of one surfel, in cells. A voxel's circumscribed sphere is 0.87 cells, so
  // anything above that overlaps its neighbours and keeps the surface hole-free; the excess also
  // caps how far a grazing tangent plane may extrapolate its depth.
  static constexpr float k_surfel_radius_cells = 1.5F;

  UserFn user_fn;
  unsigned long long* vis;
  /** Attribute plane, or @c nullptr when the image was built without one. */
  unsigned long long* attr;
  Camera camera;
  int frame_w;
  int frame_h;
  float32_3d block_origin;
  float block_spacing;
  float band_width;
  float splat_radius_px;

  CARBON_HD auto
  try_atomic_min(int px,
                 int py,
                 unsigned long long key,
                 float attribute) const -> void {
    if (px < 0 || py < 0 || px >= frame_w || py >= frame_h) {
      return;
    }
    const size_t offset =
        static_cast<size_t>(py) * static_cast<size_t>(frame_w) + static_cast<size_t>(px);
    (void)atomic_min(vis + offset, key);
    if (attr != nullptr) {
      // Same depth in the high 32 bits, attribute instead of the normal in the low 32, so the
      // two planes select the same fragment without needing a second pass or a lock.
      const unsigned long long attr_key =
          (key & 0xFFFFFFFF00000000ULL) |
          static_cast<unsigned long long>(float_to_sortable_uint(attribute));
      (void)atomic_min(attr + offset, attr_key);
    }
  }

  CARBON_HD auto
  splat_key_at(int px,
               int py,
               float32_3d world,
               float nx,
               float ny,
               float nz,
               float surfel_radius) const -> unsigned long long {
    float32_3d origin{};
    float32_3d dir{};
    if (!camera.ray_through_pixel(px, py, origin, dir)) {
      return SurfelImage::k_empty_key;
    }
    const float denom = dir.x * nx + dir.y * ny + dir.z * nz;
    if (std::abs(denom) < 1.0e-6F) {
      return SurfelImage::k_empty_key;
    }
    const float depth =
        ((world.x - origin.x) * nx + (world.y - origin.y) * ny + (world.z - origin.z) * nz) / denom;
    if (depth <= 0.0F || !std::isfinite(depth)) {
      return SurfelImage::k_empty_key;
    }

    // A surfel stands for one voxel of surface. Keep only the pixels whose tangent-plane hit falls
    // inside that disk: it bounds the depth to the patch the surfel actually covers (a grazing
    // plane otherwise extrapolates tens of cells and outvotes nearer geometry) and gives the
    // footprint its correct foreshortened, elliptical shape.
    const float hx = origin.x + dir.x * depth - world.x;
    const float hy = origin.y + dir.y * depth - world.y;
    const float hz = origin.z + dir.z * depth - world.z;
    if (hx * hx + hy * hy + hz * hz > surfel_radius * surfel_radius) {
      return SurfelImage::k_empty_key;
    }

    const uint32_t depth_bits = float_to_sortable_uint(depth);
    const uint32_t normal_bits = encode_octahedral_normal(nx, ny, nz);
    return (static_cast<unsigned long long>(depth_bits) << 32U) |
           static_cast<unsigned long long>(normal_bits);
  }

  CARBON_HD auto
  splat_at(float32_3d world,
           float32_3d normal,
           float surfel_radius,
           float attribute) const -> void {
    float fx = 0.0F;
    float fy = 0.0F;
    float clip_depth = 0.0F;
    // Unclipped on purpose: this surfel's footprint can straddle the frame edge with its centre
    // outside it, and every pixel written below goes through try_atomic_min, which clips. Using
    // the clipping projection here trimmed the outer rim of the silhouette.
    if (!camera.project_pixel_unclipped(world, fx, fy, clip_depth)) {
      return;
    }

    float nx = normal.x;
    float ny = normal.y;
    float nz = normal.z;
    if (!render_math::normalize3(nx, ny, nz)) {
      return;
    }

    const float to_eye_x = camera.eye_.x - world.x;
    const float to_eye_y = camera.eye_.y - world.y;
    const float to_eye_z = camera.eye_.z - world.z;
    if (nx * to_eye_x + ny * to_eye_y + nz * to_eye_z <= 0.0F) {
      return;
    }

    const int cx = static_cast<int>(fx);
    const int cy = static_cast<int>(fy);

    if (splat_radius_px <= 0.0F) {
      const unsigned long long key = splat_key_at(cx, cy, world, nx, ny, nz, surfel_radius);
      if (key != SurfelImage::k_empty_key) {
        try_atomic_min(cx, cy, key, attribute);
      }
      return;
    }

    // World span of one pixel at this surfel's depth, from the divergence of two adjacent pixel
    // rays. It converts the voxel-sized footprint into the pixel radius that actually needs
    // testing, and floors the footprint at ~1 pixel so a sub-pixel voxel still lands on a sample.
    float radius_world = surfel_radius;
    int radius = static_cast<int>(std::ceil(splat_radius_px));
    float32_3d o0{};
    float32_3d d0{};
    float32_3d o1{};
    float32_3d d1{};
    if (camera.ray_through_pixel(cx, cy, o0, d0) && camera.ray_through_pixel(cx + 1, cy, o1, d1)) {
      const float ex = world.x - camera.eye_.x;
      const float ey = world.y - camera.eye_.y;
      const float ez = world.z - camera.eye_.z;
      const float eye_dist = std::sqrt(ex * ex + ey * ey + ez * ez);
      const float sx = d1.x - d0.x;
      const float sy = d1.y - d0.y;
      const float sz = d1.z - d0.z;
      const float world_per_px = std::sqrt(sx * sx + sy * sy + sz * sz) * eye_dist;
      if (world_per_px > 1.0e-8F) {
        radius_world = std::fmax(surfel_radius, 0.75F * world_per_px);
        const int needed = static_cast<int>(std::ceil(radius_world / world_per_px));
        radius = needed < radius ? needed : radius;
      }
    }

    // Now that the centre is projected unclipped, reject the surfels whose whole footprint
    // misses the frame. Without this a surfel far off to one side would still walk its entire
    // (2r+1)^2 neighbourhood only for try_atomic_min to discard every pixel.
    if (cx + radius < 0 || cy + radius < 0 || cx - radius >= frame_w || cy - radius >= frame_h) {
      return;
    }

    const int r2 = radius * radius;
    for (int dy = -radius; dy <= radius; ++dy) {
      for (int dx = -radius; dx <= radius; ++dx) {
        if (dx * dx + dy * dy > r2) {
          continue;
        }
        const int px = cx + dx;
        const int py = cy + dy;
        const unsigned long long key = splat_key_at(px, py, world, nx, ny, nz, radius_world);
        if (key != SurfelImage::k_empty_key) {
          try_atomic_min(px, py, key, attribute);
        }
      }
    }
  }

  CARBON_HD auto
  operator()(typename Grid::Span span,
             typename Grid::Idx idx) const -> void {
    const SurfelSample sample = user_fn(span, idx);
    if (std::abs(sample.sdf) > band_width) {
      return;
    }

    float nx = sample.normal.x;
    float ny = sample.normal.y;
    float nz = sample.normal.z;
    if (!render_math::normalize3(nx, ny, nz)) {
      return;
    }

    if constexpr (grid_has_dense_iteration_v<Grid>) {
      const int64_3d gc = span.global_location(idx);
      const float sp = span.spacing();
      const float32_3d origin = span.origin();
      float32_3d world{
          origin.x + (static_cast<float>(gc.x) + 0.5F) * sp,
          origin.y + (static_cast<float>(gc.y) + 0.5F) * sp,
          origin.z + (static_cast<float>(gc.z) + 0.5F) * sp,
      };
      world.x -= sample.sdf * nx;
      world.y -= sample.sdf * ny;
      world.z -= sample.sdf * nz;
      splat_at(world, {nx, ny, nz}, k_surfel_radius_cells * sp, sample.attribute);
    } else {
      const int64_3d gc = span.global_location(idx);
      float32_3d world{
          block_origin.x + (static_cast<float>(gc.x) + 0.5F) * block_spacing,
          block_origin.y + (static_cast<float>(gc.y) + 0.5F) * block_spacing,
          block_origin.z + (static_cast<float>(gc.z) + 0.5F) * block_spacing,
      };
      world.x -= sample.sdf * nx;
      world.y -= sample.sdf * ny;
      world.z -= sample.sdf * nz;
      splat_at(world, {nx, ny, nz}, k_surfel_radius_cells * block_spacing, sample.attribute);
    }
  }
};

}  // namespace detail

inline SurfelImage::SurfelImage(const System& sys,
                                int width,
                                int height,
                                bool with_attribute)
    : sys_(sys),
      width_(width),
      height_(height),
      with_attribute_(with_attribute) {
  if (width <= 0 || height <= 0) {
    log::throw_with_log<std::invalid_argument>(
        "SurfelImage: width and height must be positive (got {} x {})", width, height);
  }

  const size_t gpu_count = sys.gpu_group().size();
  if (gpu_count == 0U) {
    log::throw_with_log<std::runtime_error>(
        "SurfelImage: System has no GPUs (gpu_group().size() == 0)");
  }

  pixel_count_ = static_cast<size_t>(width_) * static_cast<size_t>(height_);
  std::vector<size_t> counts(gpu_count, pixel_count_);
  gpu_vis_ = sys.buffer<unsigned long long>(counts);

  host_gpu_mirrors_.reserve(gpu_count);
  for (size_t g = 0; g < gpu_count; ++g) {
    host_gpu_mirrors_.push_back(Cpu::buffer<unsigned long long>(pixel_count_));
  }
  host_combined_vis_.assign(pixel_count_, k_empty_key);
  host_depth_.assign(pixel_count_, std::numeric_limits<float>::infinity());
  host_normal_.assign(pixel_count_, float32_3d{0.0F, 0.0F, 1.0F});

  if (with_attribute_) {
    gpu_attr_ = sys.buffer<unsigned long long>(counts);
    host_gpu_attr_mirrors_.reserve(gpu_count);
    for (size_t g = 0; g < gpu_count; ++g) {
      host_gpu_attr_mirrors_.push_back(Cpu::buffer<unsigned long long>(pixel_count_));
    }
    host_combined_attr_.assign(pixel_count_, k_empty_key);
    host_attribute_.assign(pixel_count_, 0.0F);
  }
}

inline auto
SurfelImage::clear(const Stream& stream) const -> void {
  const size_t nbytes = pixel_count_ * sizeof(unsigned long long);
  for (size_t g = 0; g < gpu_count(); ++g) {
    sys_.async_memset(gpu_vis_[g], 0xFF, nbytes, stream);
    if (with_attribute_) {
      sys_.async_memset(gpu_attr_[g], 0xFF, nbytes, stream);
    }
  }
}

inline auto
SurfelImage::gpu_visibility(size_t relative_gpu_id) const -> unsigned long long* {
  if (relative_gpu_id >= gpu_count()) {
    log::throw_with_log<std::out_of_range>(
        "SurfelImage::gpu_visibility: relative_gpu_id {} >= gpu_count {}", relative_gpu_id,
        gpu_count());
  }
  return gpu_vis_[relative_gpu_id].data();
}

inline auto
SurfelImage::gpu_attribute(size_t relative_gpu_id) const -> unsigned long long* {
  if (!with_attribute_) {
    return nullptr;
  }
  if (relative_gpu_id >= gpu_count()) {
    log::throw_with_log<std::out_of_range>(
        "SurfelImage::gpu_attribute: relative_gpu_id {} >= gpu_count {}", relative_gpu_id,
        gpu_count());
  }
  return gpu_attr_[relative_gpu_id].data();
}

inline auto
SurfelImage::host_attribute() const -> const std::vector<float>& {
  if (!with_attribute_) {
    log::throw_with_log<std::logic_error>(
        "SurfelImage::host_attribute: this image was built without with_attribute, so no "
        "attribute plane exists");
  }
  return host_attribute_;
}

inline auto
SurfelImage::combine(const Stream& stream) -> void {
  const size_t nbytes = pixel_count_ * sizeof(unsigned long long);
  for (size_t g = 0; g < gpu_count(); ++g) {
    sys_.async_copy(gpu_vis_[g], host_gpu_mirrors_[g], nbytes, stream);
    if (with_attribute_) {
      sys_.async_copy(gpu_attr_[g], host_gpu_attr_mirrors_[g], nbytes, stream);
    }
  }
  stream.wait();

  std::fill(host_combined_vis_.begin(), host_combined_vis_.end(), k_empty_key);
  for (size_t g = 0; g < gpu_count(); ++g) {
    const unsigned long long* src = host_gpu_mirrors_[g].data();
    for (size_t i = 0; i < pixel_count_; ++i) {
      host_combined_vis_[i] = std::min(host_combined_vis_[i], src[i]);
    }
  }

  if (sys_.distributed() != nullptr) {
    reduce::dist_host_allreduce_in_place(sys_, host_combined_vis_.data(), pixel_count_,
                                         reduce::ReduceOpKind::Min,
                                         reduce::data_kind<std::uint64_t>::value);
  }

  if (with_attribute_) {
    // Same min-fold, same depth in the high bits: the fragment that wins the visibility
    // plane wins this one too, so the attribute belongs to the surface the pixel shows.
    std::fill(host_combined_attr_.begin(), host_combined_attr_.end(), k_empty_key);
    for (size_t g = 0; g < gpu_count(); ++g) {
      const unsigned long long* src = host_gpu_attr_mirrors_[g].data();
      for (size_t i = 0; i < pixel_count_; ++i) {
        host_combined_attr_[i] = std::min(host_combined_attr_[i], src[i]);
      }
    }
    if (sys_.distributed() != nullptr) {
      reduce::dist_host_allreduce_in_place(sys_, host_combined_attr_.data(), pixel_count_,
                                           reduce::ReduceOpKind::Min,
                                           reduce::data_kind<std::uint64_t>::value);
    }
    host_attribute_.resize(pixel_count_);
    for (size_t i = 0; i < pixel_count_; ++i) {
      const unsigned long long key = host_combined_attr_[i];
      host_attribute_[i] =
          (key == k_empty_key)
              ? 0.0F
              : detail::sortable_uint_to_float(static_cast<uint32_t>(key & 0xFFFFFFFFULL));
    }
  }

  host_depth_.resize(pixel_count_);
  host_normal_.resize(pixel_count_);
  for (size_t i = 0; i < pixel_count_; ++i) {
    const unsigned long long key = host_combined_vis_[i];
    if (key == k_empty_key) {
      host_depth_[i] = std::numeric_limits<float>::infinity();
      host_normal_[i] = {0.0F, 0.0F, 1.0F};
      continue;
    }
    const uint32_t depth_bits = static_cast<uint32_t>(key >> 32U);
    const uint32_t normal_bits = static_cast<uint32_t>(key & 0xFFFFFFFFULL);
    host_depth_[i] = detail::sortable_uint_to_float(depth_bits);
    host_normal_[i] = detail::decode_octahedral_normal(normal_bits);
  }

  combined_ready_ = true;
}

inline auto
SurfelImage::sync(const Stream& stream) const -> void {
  stream.wait();
}

inline auto
SurfelImage::write_attribute_ppm(std::string_view path,
                                 Colormap colormap,
                                 float value_min,
                                 float value_max,
                                 float32_3d light_dir,
                                 float32_3d background) const -> bool {
  if (!render_export::is_writer_rank(sys_)) {
    return true;
  }
  if (!with_attribute_ || !combined_ready_) {
    return false;
  }

  float lx = light_dir.x;
  float ly = light_dir.y;
  float lz = light_dir.z;
  if (!detail::render_math::normalize3(lx, ly, lz)) {
    lx = 0.3F;
    ly = -0.85F;
    lz = 0.45F;
  }

  std::ofstream out(std::string(path), std::ios::binary);
  if (!out) {
    return false;
  }

  const float range = value_max - value_min;
  out << "P6\n" << width_ << ' ' << height_ << "\n255\n";
  for (size_t i = 0; i < pixel_count_; ++i) {
    float r = background.x;
    float g = background.y;
    float b = background.z;
    if (host_combined_vis_[i] != k_empty_key) {
      const float t = (range > 0.0F) ? ((host_attribute_[i] - value_min) / range) : 0.0F;
      const float32_3d c = render::colormap_eval(colormap, t);
      // A gentler Lambert range than the shaded overload: here the colour carries the data
      // and the shading only has to keep the body's curvature readable underneath it.
      const float32_3d n = host_normal_[i];
      const float ndotl = std::max(0.0F, n.x * lx + n.y * ly + n.z * lz);
      const float shade = 0.55F + 0.45F * ndotl;
      r = c.x * shade;
      g = c.y * shade;
      b = c.z * shade;
    }
    const std::array<char, 3> rgb{static_cast<char>(render_export::to_byte(r)),
                                  static_cast<char>(render_export::to_byte(g)),
                                  static_cast<char>(render_export::to_byte(b))};
    out.write(rgb.data(), static_cast<std::streamsize>(rgb.size()));
  }
  return static_cast<bool>(out);
}

inline auto
SurfelImage::write_ppm(std::string_view path,
                       float32_3d light_dir,
                       float32_3d material,
                       float32_3d background) const -> bool {
  if (!render_export::is_writer_rank(sys_)) {
    return true;
  }
  if (!combined_ready_) {
    return false;
  }

  float lx = light_dir.x;
  float ly = light_dir.y;
  float lz = light_dir.z;
  if (!detail::render_math::normalize3(lx, ly, lz)) {
    lx = 0.3F;
    ly = -0.85F;
    lz = 0.45F;
  }

  std::ofstream out(std::string(path), std::ios::binary);
  if (!out) {
    return false;
  }

  out << "P6\n" << width_ << ' ' << height_ << "\n255\n";
  for (size_t i = 0; i < pixel_count_; ++i) {
    float r = background.x;
    float g = background.y;
    float b = background.z;
    if (host_combined_vis_[i] != k_empty_key) {
      const float32_3d n = host_normal_[i];
      const float ndotl = std::max(0.0F, n.x * lx + n.y * ly + n.z * lz);
      const float shade = 0.18F + 0.72F * ndotl;
      r = material.x * shade;
      g = material.y * shade;
      b = material.z * shade;
    }
    const std::array<char, 3> rgb{static_cast<char>(render_export::to_byte(r)),
                                  static_cast<char>(render_export::to_byte(g)),
                                  static_cast<char>(render_export::to_byte(b))};
    out.write(rgb.data(), static_cast<std::streamsize>(rgb.size()));
  }
  return static_cast<bool>(out);
}

template <typename Grid,
          typename DensityLoader>
inline auto
make_density_kernel(const Grid& grid,
                    DensityImage& image,
                    Camera camera,
                    DensityLoader&& density_loader,
                    float32_3d block_origin,
                    float block_spacing,
                    int samples_per_axis,
                    float splat_sigma_px) -> Kernel {
  if (samples_per_axis < 1) {
    log::throw_with_log<std::invalid_argument>(
        "make_density_kernel: samples_per_axis must be >= 1 (got {})", samples_per_axis);
  }
  if (splat_sigma_px < 0.0F) {
    log::throw_with_log<std::invalid_argument>(
        "make_density_kernel: splat_sigma_px must be >= 0 (got {})", splat_sigma_px);
  }

  return grid.make_kernel(
      [&image, camera, density_loader = std::forward<DensityLoader>(density_loader), block_origin,
       block_spacing, samples_per_axis, splat_sigma_px](KernelLoader& loader) {
        auto user_fn = density_loader(loader);
        return detail::DensitySplatCellFn<Grid, decltype(user_fn)>{
            std::move(user_fn),
            image.gpu_framebuffer(loader.relative_gpu_id()),
            camera,
            camera.width(),
            camera.height(),
            block_origin,
            block_spacing,
            samples_per_axis,
            splat_sigma_px,
        };
      });
}

template <typename Grid,
          typename SurfelLoader>
inline auto
make_surfel_kernel(const Grid& grid,
                   SurfelImage& image,
                   Camera camera,
                   SurfelLoader&& surfel_loader,
                   float band_width,
                   float splat_radius_px,
                   float32_3d block_origin,
                   float block_spacing) -> Kernel {
  if (band_width <= 0.0F) {
    log::throw_with_log<std::invalid_argument>(
        "make_surfel_kernel: band_width must be > 0 (got {})", band_width);
  }
  if (splat_radius_px < 0.0F) {
    log::throw_with_log<std::invalid_argument>(
        "make_surfel_kernel: splat_radius_px must be >= 0 (got {})", splat_radius_px);
  }

  return grid.make_kernel([&image, camera,
                           surfel_loader = std::forward<SurfelLoader>(surfel_loader), block_origin,
                           block_spacing, band_width, splat_radius_px](KernelLoader& loader) {
    auto user_fn = surfel_loader(loader);
    return detail::SurfelSplatCellFn<Grid, decltype(user_fn)>{
        std::move(user_fn),
        image.gpu_visibility(loader.relative_gpu_id()),
        image.gpu_attribute(loader.relative_gpu_id()),
        camera,
        camera.width(),
        camera.height(),
        block_origin,
        block_spacing,
        band_width,
        splat_radius_px,
    };
  });
}

}  // namespace carbon
