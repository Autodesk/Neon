#pragma once

#include <carbon/cuda_annotation.h>
#include <carbon/detail/backend/cuda_like.h>

#include <atomic>
#include <cstdint>

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
atomic_add(int32_t* address,
           int32_t val) -> int32_t {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return static_cast<int32_t>(atomicAdd(reinterpret_cast<int*>(address), static_cast<int>(val)));
#else
  std::atomic_ref<int32_t> ref(*address);
  return ref.fetch_add(val, std::memory_order_relaxed);
#endif
}

[[nodiscard]] CARBON_HD inline auto
atomic_add(double* address,
           double val) -> double {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
  return atomicAdd(address, val);
#elif defined(__CUDA_ARCH__)
  // CUDA before sm_60 has no native double atomicAdd. Emulate it with a CAS loop on the
  // 64-bit representation; only one thread per block reaches this helper in BlockAtomic.
  auto* bits = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *bits;
  while (true) {
    const unsigned long long assumed = old;
    const double current = __longlong_as_double(static_cast<long long>(assumed));
    const unsigned long long next =
        static_cast<unsigned long long>(__double_as_longlong(current + val));
    old = atomicCAS(bits, assumed, next);
    if (old == assumed) {
      return current;
    }
  }
#elif defined(__HIP_DEVICE_COMPILE__)
  // HIP defines __HIP_DEVICE_COMPILE__ but not __CUDA_ARCH__ on AMD. HIP's atomicAdd
  // accepts double and lowers to native atomics or a CAS loop depending on the target.
  return atomicAdd(address, val);
#else
  std::atomic_ref<double> ref(*address);
  return ref.fetch_add(val, std::memory_order_relaxed);
#endif
}

[[nodiscard]] CARBON_HD inline auto
atomic_add(int64_t* address,
           int64_t val) -> int64_t {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
  auto* bits = reinterpret_cast<unsigned long long*>(address);
  return static_cast<int64_t>(atomicAdd(bits, static_cast<unsigned long long>(val)));
#elif defined(__CUDA_ARCH__)
  // CUDA before sm_60 has no native 64-bit atomicAdd. Emulate it with a CAS loop; signed
  // addition is performed on the unsigned representation (two's-complement wraparound).
  auto* bits = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *bits;
  while (true) {
    const unsigned long long assumed = old;
    old = atomicCAS(bits, assumed, assumed + static_cast<unsigned long long>(val));
    if (old == assumed) {
      return static_cast<int64_t>(assumed);
    }
  }
#elif defined(__HIP_DEVICE_COMPILE__)
  auto* bits = reinterpret_cast<unsigned long long*>(address);
  return static_cast<int64_t>(atomicAdd(bits, static_cast<unsigned long long>(val)));
#else
  std::atomic_ref<int64_t> ref(*address);
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
