#pragma once

// Single place that decides which sycl::device objects carbon treats as "compute devices".
//
// Default (GPU/accelerator builds): CPU devices are skipped -- carbon's "device" is the
// accelerator side and the host is reached through Cpu::buffer.
//
// CARBON_SYCL_CPU_DEVICE=1 (set by CMake for `-DCARBON_SYCL_TARGETS=omp`): the AdaptiveCpp
// OpenMP host device IS the compute device. AdaptiveCpp reports it with is_cpu() == true, so
// without this switch device_count() is 0 and every grid constructor throws.

#include <sycl/sycl.hpp>

namespace carbon::detail {

[[nodiscard]] inline auto
is_compute_device(const sycl::device& dev) -> bool {
#ifdef CARBON_SYCL_CPU_DEVICE
  (void)dev;
  return true;
#else
  return !dev.is_cpu();
#endif
}

}  // namespace carbon::detail
