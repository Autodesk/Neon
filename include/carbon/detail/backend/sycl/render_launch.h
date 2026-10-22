#pragma once

#include <carbon/detail/gpu.h>
#include <carbon/logging.h>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <stdexcept>

namespace carbon::detail::render {

template <typename F>
inline auto
launch_pixel_kernel(const ::carbon::detail::Gpu& /*gpu*/,
                    int width,
                    int height,
                    sycl::queue& queue,
                    F fn) -> void {
  if (width <= 0 || height <= 0) {
    return;
  }

  const sycl::range<2> local{16, 16};
  const sycl::range<2> global{
      static_cast<size_t>((width + 15) / 16) * 16,
      static_cast<size_t>((height + 15) / 16) * 16,
  };

  try {
    queue
        .parallel_for(sycl::nd_range<2>{global, local},
                      [=](sycl::nd_item<2> item) {
                        const int px = static_cast<int>(item.get_global_id(1));
                        const int py = static_cast<int>(item.get_global_id(0));
                        if (px >= width || py >= height) {
                          return;
                        }
                        fn(px, py);
                      })
        .wait_and_throw();
  } catch (const sycl::exception& ex) {
    ::carbon::log::throw_with_log<std::runtime_error>(
        "launch_pixel_kernel: SYCL parallel_for failed: {}", ex.what());
  }
}

}  // namespace carbon::detail::render
