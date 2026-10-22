// Unit tests for parallel density rendering: energy conservation, multi-GPU combine,
// and frustum culling.

#include <carbon/carbon.h>
#include <carbon/dense.h>
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
owned_cell_count(const carbon::Dense& dense) -> int64_t {
  const auto ge = dense.global_extent();
  return ge.x * ge.y * ge.z;
}

[[nodiscard]] auto
frame_sum(const carbon::DensityImage& image) -> double {
  double acc = 0.0;
  for (float v : image.host_frame()) {
    acc += static_cast<double>(v);
  }
  return acc;
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

void
test_frame_bounds(carbon::System& sys,
                  const carbon::Dense& dense,
                  carbon::Dense::Field<float,
                                       1>& field,
                  const carbon::Stream& stream) {
  const auto fill = build_unit_density_kernel(dense, field);
  fill.launch(stream);
  stream.wait();

  const carbon::Camera camera = frame_camera(dense, 256, 256);

  carbon::DensityImage image(sys, 256, 256);
  image.clear(stream);

  const auto splat =
      carbon::make_density_kernel(dense, image, camera, [&field](carbon::KernelLoader& loader) {
        const auto part = loader.read(field);
        return [part] CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
          return part.read(span, idx, 0);
        };
      });

  splat.launch(stream);
  image.combine(stream);

  if (frame_sum(image) <= 0.0) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_density frame_bounds: expected visible frame, got zero sum");
  }
}

void
test_energy_conservation(carbon::System& sys,
                         const carbon::Dense& dense,
                         carbon::Dense::Field<float,
                                              1>& field,
                         const carbon::Stream& stream) {
  const auto fill = build_unit_density_kernel(dense, field);
  fill.launch(stream);
  stream.wait();

  const carbon::Camera camera = frame_camera(dense, 256, 256);

  carbon::DensityImage image(sys, 256, 256);
  image.clear(stream);

  const auto splat =
      carbon::make_density_kernel(dense, image, camera, [&field](carbon::KernelLoader& loader) {
        const auto part = loader.read(field);
        return [part] CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
          return part.read(span, idx, 0);
        };
      });

  splat.launch(stream);
  image.combine(stream);

  const double expected = static_cast<double>(owned_cell_count(dense));
  const double got = frame_sum(image);
  const double rel = std::abs(got - expected) / expected;
  if (rel > 1.0e-4) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_density energy: expected sum {} got {} (rel err {})", expected, got, rel);
  }
}

void
test_frustum_culling(carbon::System& sys,
                     const carbon::Dense& dense,
                     carbon::Dense::Field<float,
                                          1>& field,
                     const carbon::Stream& stream) {
  const auto fill = build_unit_density_kernel(dense, field);
  fill.launch(stream);
  stream.wait();

  const carbon::Camera away =
      carbon::Camera::perspective({32.0F, 32.0F, -200.0F}, {32.0F, 32.0F, -400.0F},
                                  {0.0F, 1.0F, 0.0F}, 1.0F, 0.1F, 500.0F, 128, 128);

  carbon::DensityImage image(sys, 128, 128);
  image.clear(stream);

  const auto splat =
      carbon::make_density_kernel(dense, image, away, [&field](carbon::KernelLoader& loader) {
        const auto part = loader.read(field);
        return [part] CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
          return part.read(span, idx, 0);
        };
      });

  splat.launch(stream);
  image.combine(stream);

  const double got = frame_sum(image);
  if (got != 0.0) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_density frustum: expected zero frame sum, got {}", got);
  }
}

void
test_multi_gpu_combine(carbon::System& sys,
                       const carbon::Dense& dense,
                       carbon::Dense::Field<float,
                                            1>& field,
                       const carbon::Stream& stream) {
  if (sys.gpu_group().size() < 2U) {
    return;
  }

  const auto fill = build_unit_density_kernel(dense, field);
  fill.launch(stream);
  stream.wait();

  const carbon::Camera camera = frame_camera(dense, 256, 256);

  carbon::DensityImage image(sys, 256, 256);
  image.clear(stream);

  const auto splat =
      carbon::make_density_kernel(dense, image, camera, [&field](carbon::KernelLoader& loader) {
        const auto part = loader.read(field);
        return [part] CARBON_HD(carbon::Dense::Span span, carbon::Dense::Idx idx) -> float {
          return part.read(span, idx, 0);
        };
      });

  splat.launch(stream);
  image.combine(stream);

  const double expected = static_cast<double>(owned_cell_count(dense));
  const double got = frame_sum(image);
  const double rel = std::abs(got - expected) / expected;
  if (rel > 1.0e-4) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_density multi_gpu: expected sum {} got {} on {} GPUs", expected, got,
        sys.gpu_group().size());
  }
}

}  // namespace

auto
main() -> int {
  carbon::init();
  carbon::System sys(carbon::get_all_devices());
  if (sys.gpu_group().size() == 0U) {
    return EXIT_SUCCESS;
  }

  const carbon::int64_3d extent{64, 64, 64};
  carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);

  const carbon::Dense dense(sys, extent, stencil);
  auto field = dense.newField<float, 1>(carbon::MemorySide::Device);
  const carbon::Stream stream = sys.stream();

  test_frame_bounds(sys, dense, field, stream);
  test_energy_conservation(sys, dense, field, stream);
  test_frustum_culling(sys, dense, field, stream);
  test_multi_gpu_combine(sys, dense, field, stream);

  return EXIT_SUCCESS;
}
