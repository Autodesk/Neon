#pragma once

#include <carbon/detail/allocation.h>
#include <carbon/detail/backend/select.h>
#include <carbon/export.h>
#include <carbon/logging.h>

#include <cstddef>
#include <cstdlib>
#include <functional>
#include <new>

#if CARBON_BACKEND_IS_CUDALIKE
#include <carbon/detail/backend/cuda_like.h>
#elif CARBON_BACKEND_IS_SYCL
#include <sycl/sycl.hpp>
#endif

namespace carbon {

/** Host-side allocator for Allocation objects (CPU memory). */
class CARBON_API Cpu {
public:
  /**
   * Allocate host memory for @p count elements of type T. gpu_id and relative_gpu_id on the
   * allocation are -1. @throws std::bad_alloc if allocation fails (count > 0).
   */
  template <typename T>
  [[nodiscard]] static auto
  buffer(size_t count) -> detail::Allocation<T> {
    const size_t size_bytes = count * sizeof(T);
    unsigned char* ptr = nullptr;
    if (size_bytes > 0U) {
      ptr = new (std::nothrow) unsigned char[size_bytes];
      if (ptr == nullptr) {
        log::throw_with_log<std::bad_alloc>(
            "Cpu::buffer: std::bad_alloc (failed to allocate {} bytes for host buffer)",
            size_bytes);
      }
    }

    std::function<void(void*)> free_fn = [](void* p) {
      if (p != nullptr) {
        delete[] static_cast<unsigned char*>(p);
      }
    };

    return detail::Allocation<T>(detail::AllocationLocation::Cpu, count, -1, -1, ptr, free_fn);
  }

  /**
   * Page-locked host memory suitable for async @c cudaMemcpyAsync D→H/H→D.
   * Falls back to @ref buffer when no pinned allocator is available.
   */
  template <typename T>
  [[nodiscard]] static auto
  pinned_buffer(size_t count) -> detail::Allocation<T> {
    const size_t size_bytes = count * sizeof(T);
    unsigned char* ptr = nullptr;
    if (size_bytes > 0U) {
#if CARBON_BACKEND_IS_CUDALIKE
      const detail::gpu_error_t err = detail::gpu_host_alloc(
          reinterpret_cast<void**>(&ptr), size_bytes, detail::gpu_host_alloc_default);
      if (err != detail::gpu_success || ptr == nullptr) {
        log::throw_with_log<std::bad_alloc>(
            "Cpu::pinned_buffer: gpu_host_alloc failed ({} bytes)", size_bytes);
      }
#elif CARBON_BACKEND_IS_SYCL
      try {
        ptr = static_cast<unsigned char*>(sycl::malloc_host(size_bytes, sycl::default_context()));
      } catch (const sycl::exception& ex) {
        log::throw_with_log<std::bad_alloc>("Cpu::pinned_buffer: sycl::malloc_host failed: {}",
                                            ex.what());
      }
      if (ptr == nullptr) {
        log::throw_with_log<std::bad_alloc>("Cpu::pinned_buffer: sycl::malloc_host returned null");
      }
#else
      return buffer<T>(count);
#endif
    }

    std::function<void(void*)> free_fn = [](void* p) {
      if (p == nullptr) {
        return;
      }
#if CARBON_BACKEND_IS_CUDALIKE
      (void)detail::gpu_host_free(p);
#elif CARBON_BACKEND_IS_SYCL
      sycl::free(p, sycl::default_context());
#endif
    };

    return detail::Allocation<T>(detail::AllocationLocation::Cpu, count, -1, -1, ptr, free_fn);
  }
};

}  // namespace carbon
