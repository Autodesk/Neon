#pragma once

#include <carbon/data_view.h>
#include <carbon/detail/grid_concepts.h>
#include <carbon/detail/kernel_impl.h>
#include <carbon/detail/launch_traits.h>
#include <carbon/detail/render_launch.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>

#include <carbon/detail/backend/select.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace carbon::detail {

template <typename Grid, typename LoadingFn>
class RayMarchKernelImpl final : public KernelImpl {
public:
  RayMarchKernelImpl(Grid grid,
                     RayMarchImage& image,
                     Camera camera,
                     RayMarchOptions opts,
                     LoadingFn loading_fn)
      : KernelImpl("ray_march"),
        grid_(std::move(grid)),
        image_(&image),
        camera_(camera),
        opts_(opts),
        loading_fn_(std::move(loading_fn)) {}

  [[nodiscard]] auto
  kind() const noexcept -> KernelKind override {
    return KernelKind::Compute;
  }

  [[nodiscard]] auto
  runs_on_device() const noexcept -> bool override {
    return true;
  }

  auto
  launch(const ::carbon::Stream& stream,
         DataView view) -> void override {
    this->launch_on_device(stream, view);
  }

  auto
  launch_on_device(const ::carbon::Stream& stream,
                   DataView /*view*/) -> void override {
#if CARBON_BACKEND_IS_CUDALIKE && CARBON_CUDALIKE_DEVICE_TU
    const size_t n = grid_.gpu_count();
    if (n == 0U) {
      return;
    }
    if (stream.size() != n) {
      log::throw_with_log<std::logic_error>(
          "make_ray_march_kernel::launch: stream group size {} must equal gpu_count() {}",
          stream.size(), n);
    }

    const int frame_w = image_->width();
    const int frame_h = image_->height();
    std::vector<gpu_error_t> launch_errs(n, gpu_success);
#pragma omp parallel for num_threads(static_cast<int>(n)) if (n > 1)
    for (size_t i = 0; i < n; ++i) {
      KernelLoader ctx{i, KernelTarget::Device, KernelLoaderPhase::Launch};
      auto pixel_fn = loading_fn_(ctx);
      const gpu_stream_t cuda_stream = stream[i].get();
      const ::carbon::detail::Gpu& gpu = grid_.system().gpu_group().devices()[i];
      launch_errs[i] = ::carbon::detail::render::launch_pixel_kernel(
          gpu, frame_w, frame_h, cuda_stream, std::move(pixel_fn));
    }
    for (size_t i = 0; i < n; ++i) {
      if (launch_errs[i] != gpu_success) {
        log::throw_with_log<std::runtime_error>(
            "make_ray_march_kernel::launch: per-GPU launch failed for relative_gpu_id={}: {} "
            "(gpu error {})",
            i, gpu_error_string(launch_errs[i]), static_cast<int>(launch_errs[i]));
      }
    }
#elif CARBON_BACKEND_IS_SYCL
    const size_t n = grid_.gpu_count();
    if (n == 0U) {
      return;
    }
    if (stream.size() != n) {
      log::throw_with_log<std::logic_error>(
          "make_ray_march_kernel::launch: stream group size {} must equal gpu_count() {}",
          stream.size(), n);
    }

    const int frame_w = image_->width();
    const int frame_h = image_->height();
    for (size_t i = 0; i < n; ++i) {
      KernelLoader ctx{i, KernelTarget::Device, KernelLoaderPhase::Launch};
      auto pixel_fn = loading_fn_(ctx);
      ::carbon::detail::render::launch_pixel_kernel(grid_.system().gpu_group().devices()[i],
                                                    frame_w, frame_h, stream[i].queue(),
                                                    std::move(pixel_fn));
    }
#else
    static_assert(ray_march_launch_requires_cudacc_<LoadingFn>,
                  "make_ray_march_kernel::launch requires a device translation unit.");
    (void)stream;
#endif
  }

private:
  template <typename>
  static inline constexpr bool ray_march_launch_requires_cudacc_ = false;

  Grid grid_{};
  RayMarchImage* image_{nullptr};
  Camera camera_{};
  RayMarchOptions opts_{};
  LoadingFn loading_fn_{};
};

}  // namespace carbon::detail
