#pragma once

#include <carbon/cuda_annotation.h>

#include <atomic>

namespace carbon {
namespace detail {

[[nodiscard]] CARBON_HD inline auto
atomic_add(float* address,
           float val) -> float {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return atomicAdd(address, val);
#else
  // The host branch is not dead code. It compiles into every host TU that includes this
  // header, and on a backend whose kernels lower to OpenMP loops it is what a per-cell
  // accumulate calls concurrently from every thread — so a plain read-modify-write here is
  // a lost-update race, not just a stylistic gap next to atomic_min below.
  //
  // Relaxed is the right order: every caller uses this purely as an accumulator and
  // publishes the buffer afterwards through a separate stream or host fence. It never
  // guards other writes, so no acquire/release edge is needed.
  std::atomic_ref<float> ref(*address);
  return ref.fetch_add(val, std::memory_order_relaxed);
#endif
}

[[nodiscard]] CARBON_HD inline auto
atomic_min(unsigned long long* address,
           unsigned long long val) -> unsigned long long {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return atomicMin(address, val);
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
