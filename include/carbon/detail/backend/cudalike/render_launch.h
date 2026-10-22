#pragma once

#include <carbon/cuda_annotation.h>
#include <carbon/detail/backend/cuda_like.h>
#include <carbon/detail/gpu.h>
#include <carbon/logging.h>

#include <cstddef>
#include <stdexcept>

namespace carbon::detail::render {

template <typename>
inline constexpr bool pixel_launch_requires_cudacc_ = false;

#if CARBON_CUDALIKE_DEVICE_TU

template <typename F>
__global__ auto
pixel_kernel(int width,
             int height,
             F fn) -> void {
  const int px = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int py = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (px >= width || py >= height) {
    return;
  }
  fn(px, py);
}

template <typename F>
[[nodiscard]] inline auto
launch_pixel_kernel(const ::carbon::detail::Gpu& gpu,
                    int width,
                    int height,
                    gpu_stream_t stream,
                    F fn) -> gpu_error_t {
  if (width <= 0 || height <= 0) {
    return gpu_success;
  }
  const dim3 block(16, 16, 1);
  const dim3 grid((static_cast<unsigned>(width) + block.x - 1U) / block.x,
                  (static_cast<unsigned>(height) + block.y - 1U) / block.y, 1U);
  return gpu.template typed_launch<&pixel_kernel<F>>(grid, block, 0U, stream, width, height, fn);
}

#else

template <typename F>
[[nodiscard]] inline auto
launch_pixel_kernel(const ::carbon::detail::Gpu&,
                    int,
                    int,
                    gpu_stream_t,
                    F) -> gpu_error_t {
  static_assert(pixel_launch_requires_cudacc_<F>,
                "carbon::detail::render::launch_pixel_kernel is only supported in translation "
                "units compiled with a cuda-like device compiler (CARBON_CUDALIKE_DEVICE_TU). "
                "Move the call site to a .cu file.");
  return gpu_error_not_supported;
}

#endif

}  // namespace carbon::detail::render
