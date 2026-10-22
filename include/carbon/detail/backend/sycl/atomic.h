#pragma once

#include <carbon/cuda_annotation.h>

#include <atomic>

#include <sycl/sycl.hpp>

namespace carbon {
namespace detail {

[[nodiscard]] CARBON_HD inline auto
atomic_add(float* address,
           float val) -> float {
#if defined(__SYCL_DEVICE_ONLY__)
  sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                   sycl::access::address_space::global_space>
      ref(*address);
  return ref.fetch_add(val);
#else
  // Reached for real on this backend, not just in host-side unit tests: with an omp target
  // AdaptiveCpp lowers kernels to OpenMP loops, __SYCL_DEVICE_ONLY__ is undefined, and every
  // thread of a per-cell accumulate lands here at once. A plain read-modify-write loses
  // updates. Relaxed matches the device branch above and the accumulator-only contract.
  std::atomic_ref<float> ref(*address);
  return ref.fetch_add(val, std::memory_order_relaxed);
#endif
}

[[nodiscard]] CARBON_HD inline auto
atomic_min(unsigned long long* address,
           unsigned long long val) -> unsigned long long {
#if defined(__SYCL_DEVICE_ONLY__)
  sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device,
                   sycl::access::address_space::global_space>
      ref(*address);
  return ref.fetch_min(val);
#else
  std::atomic_ref<unsigned long long> ref(*address);
  unsigned long long old = ref.load(std::memory_order_relaxed);
  while (val < old && !ref.compare_exchange_weak(old, val, std::memory_order_relaxed,
                                                 std::memory_order_relaxed)) {
  }
  return old;
#endif
}

}  // namespace detail
}  // namespace carbon
