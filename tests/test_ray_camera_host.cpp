// Quick host diagnostic for Camera::ray_through_pixel (not registered in CTest).
#include <carbon/render.h>

#include <cmath>
#include <cstdio>

int
main() {
  const float cx = 0.5F * 511.0F;
  const float radius = 0.35F * 512.0F;
  const carbon::float32_3d bounds_min{cx - radius, cx - radius, cx - radius};
  const carbon::float32_3d bounds_max{cx + radius, cx + radius, cx + radius};
  const carbon::Camera camera = carbon::Camera::frame_bounds(
      bounds_min, bounds_max, {0.0F, -1.0F, 0.0F}, {0.0F, 0.0F, 1.0F}, 0.785398F, 3840, 2160);

  for (int py : {0, 500, 1080, 1500, 2159}) {
    carbon::float32_3d origin{};
    carbon::float32_3d dir{};
    const bool ok = camera.ray_through_pixel(1920, py, origin, dir);
    std::printf("py=%d ok=%d dir=(%.4f, %.4f, %.4f)\n", py, ok ? 1 : 0, dir.x, dir.y, dir.z);
  }

  const carbon::float32_3d sphere_center{cx, cx, cx};
  float fx = 0.0F;
  float fy = 0.0F;
  float depth = 0.0F;
  const bool proj = camera.project_pixel(sphere_center, fx, fy, depth);
  std::printf("sphere center projects ok=%d fx=%.1f fy=%.1f\n", proj ? 1 : 0, fx, fy);

  if (proj) {
    carbon::float32_3d origin{};
    carbon::float32_3d dir{};
    const int px = static_cast<int>(fx);
    const int py = static_cast<int>(fy);
    const bool ray = camera.ray_through_pixel(px, py, origin, dir);
    std::printf("ray at projected px=%d py=%d ok=%d dir=(%.4f, %.4f, %.4f)\n", px, py, ray ? 1 : 0,
                dir.x, dir.y, dir.z);
  }
  return 0;
}
