#pragma once

#include <carbon/detail/allocation.h>
#include <carbon/detail/event.h>
#include <carbon/detail/gpu_allocation_tracking.h>
#include <carbon/detail/stream.h>
#include <carbon/logging.h>

#include <carbon/detail/backend/cuda_like.h>

#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace carbon {
namespace detail {

inline auto
Gpu::stream() const -> CudaStream {
  set_device();
  gpu_stream_t s = nullptr;
  const gpu_error_t err = gpu_stream_create(&s);
  if (err != gpu_success) {
    ::carbon::log::throw_with_log<std::runtime_error>("gpu_stream_create: {}",
                                                      gpu_error_string(err));
  }
  const int dev = device_id_;
  auto deleter = [dev](void* p) {
    if (p == nullptr) {
      return;
    }
    // Best-effort context switch: destroy even if it fails. Bailing out on a failed
    // gpu_set_device leaked the stream outright, and the switch is only a courtesy here —
    // a stream handle carries its own context, so gpu_stream_destroy does not require its
    // device to be current. Deleters cannot throw, and logging is not safe either: these can
    // run during static destruction, after carbon's logger is gone.
    (void)gpu_set_device(dev);
    (void)gpu_stream_destroy(static_cast<gpu_stream_t>(p));
  };
  return CudaStream(std::shared_ptr<void>(static_cast<void*>(s), std::move(deleter)));
}

inline auto
Gpu::stream_with_priority(int greatest_priority) const -> CudaStream {
  set_device();
  gpu_stream_t s = nullptr;
  int least = 0;
  int greatest = 0;
  (void)gpu_stream_priority_range(&least, &greatest);
  const gpu_error_t err = gpu_stream_create_with_priority(&s, least, greatest_priority);
  if (err != gpu_success) {
    ::carbon::log::throw_with_log<std::runtime_error>("gpu_stream_create_with_priority: {}",
                                                      gpu_error_string(err));
  }
  const int dev = device_id_;
  auto deleter = [dev](void* p) {
    if (p == nullptr) {
      return;
    }
    (void)gpu_set_device(dev);
    (void)gpu_stream_destroy(static_cast<gpu_stream_t>(p));
  };
  return CudaStream(std::shared_ptr<void>(static_cast<void*>(s), std::move(deleter)));
}

inline auto
Gpu::event() const -> Event {
  set_device();
  gpu_event_t e = nullptr;
  const gpu_error_t err = gpu_event_create(&e);
  if (err != gpu_success) {
    ::carbon::log::throw_with_log<std::runtime_error>("gpu_event_create: {}",
                                                      gpu_error_string(err));
  }
  const int dev = device_id_;
  auto deleter = [dev](void* p) {
    if (p == nullptr) {
      return;
    }
    // Same reasoning as the stream deleter above: attempt the destroy regardless, since an
    // event handle carries its context and skipping the call leaks it for good.
    (void)gpu_set_device(dev);
    (void)gpu_event_destroy(static_cast<gpu_event_t>(p));
  };
  return Event(std::shared_ptr<void>(static_cast<void*>(e), std::move(deleter)));
}

template <typename T>
auto
Gpu::buffer(size_t count,
            int relative_gpu_id) const -> Allocation<T> {
  set_device();
  // Guard the element-count * element-size product: a silent size_t wrap would
  // hand gpu_malloc a tiny byte count and return a buffer far smaller than the
  // caller's element count implies, corrupting every subsequent indexed access.
  if (count != 0U && sizeof(T) > (std::numeric_limits<size_t>::max() / count)) {
    ::carbon::log::throw_with_log<std::runtime_error>(
        "Gpu::buffer: allocation size overflow ({} elements * {} bytes)", count, sizeof(T));
  }
  const size_t size_bytes = count * sizeof(T);
  void* ptr = nullptr;
  if (size_bytes > 0U) {
    const gpu_error_t err = gpu_malloc(&ptr, size_bytes);
    if (err != gpu_success) {
      ::carbon::log::throw_with_log<std::runtime_error>("gpu_malloc: {}", gpu_error_string(err));
    }
    gpu_track_allocation(device_id_, size_bytes);
  }

  const int dev = device_id_;
  std::function<void(void*)> free_fn = [dev, size_bytes](void* p) {
    if (p == nullptr) {
      return;
    }
    if (size_bytes > 0U) {
      gpu_track_deallocation(dev, size_bytes);
    }
    // Unlike streams and events, gpu_free does need the owning device current, so a failed
    // switch here genuinely cannot free the block. Still make the call rather than returning
    // early: it costs an error return in the case that was already lost, and it keeps this
    // deleter's shape identical to the two above. The accounting was already reversed, so
    // gpu_memory_in_use stays consistent with what carbon believes it holds either way.
    (void)gpu_set_device(dev);
    (void)gpu_free(p);
  };

  return Allocation<T>(AllocationLocation::Cuda, count, device_id_, relative_gpu_id, ptr, free_fn);
}

#if CARBON_CUDALIKE_DEVICE_TU
namespace {

template <typename... Args>
auto
gpu_launch_kernel(int device_id,
                  const void* kernel,
                  dim3 grid,
                  dim3 block,
                  size_t shared_mem,
                  gpu_stream_t stream,
                  Args... args) -> gpu_error_t {
  const gpu_error_t sd = gpu_set_device(device_id);
  if (sd != gpu_success) {
    return sd;
  }
  if (kernel == nullptr) {
    return gpu_error_invalid_value;
  }
  gpu_error_t err{};
  if constexpr (sizeof...(Args) == 0U) {
    err = gpu_runtime_launch_kernel(kernel, grid, block, nullptr, shared_mem, stream);
  } else {
    void* params[] = {const_cast<void*>(static_cast<const void*>(std::addressof(args)))...};
    err = gpu_runtime_launch_kernel(kernel, grid, block, params, shared_mem, stream);
  }
  if (err != gpu_success) {
    return err;
  }
  return gpu_get_last_error();
}

}  // namespace

template <typename... Args>
inline auto
Gpu::execute(const void* kernel,
             dim3 grid,
             dim3 block,
             size_t shared_mem,
             gpu_stream_t stream,
             Args... args) const -> gpu_error_t {
  return gpu_launch_kernel(device_id_, kernel, grid, block, shared_mem, stream, args...);
}

template <auto* Kernel,
          typename... Args>
inline auto
Gpu::execute(dim3 grid,
             dim3 block,
             size_t shared_mem,
             gpu_stream_t stream,
             Args... args) const -> gpu_error_t {
  return execute(reinterpret_cast<const void*>(Kernel), grid, block, shared_mem, stream, args...);
}

template <typename... Args>
inline auto
Gpu::typed_launch(dim3 grid,
                  dim3 block,
                  size_t shared_mem,
                  gpu_stream_t stream,
                  CudaGlobalKernelPtr<Args...> kernel,
                  Args... args) const -> gpu_error_t {
  return execute(reinterpret_cast<const void*>(kernel), grid, block, shared_mem, stream, args...);
}

template <auto* Kernel,
          typename... Args>
inline auto
Gpu::typed_launch(dim3 grid,
                  dim3 block,
                  size_t shared_mem,
                  gpu_stream_t stream,
                  Args... args) const -> gpu_error_t {
  static_assert(std::is_same_v<std::decay_t<decltype(Kernel)>, CudaGlobalKernelPtr<Args...>>,
                "typed_launch: kernel pointer type must be CudaGlobalKernelPtr<Args...>");
  return execute<Kernel>(grid, block, shared_mem, stream, args...);
}
#else
template <typename... Args>
inline auto
Gpu::execute(const void*,
             dim3,
             dim3,
             size_t,
             gpu_stream_t,
             Args...) const -> gpu_error_t {
  return gpu_error_not_supported;
}

template <auto*,
          typename... Args>
inline auto
Gpu::execute(dim3,
             dim3,
             size_t,
             gpu_stream_t,
             Args...) const -> gpu_error_t {
  return gpu_error_not_supported;
}

template <typename... Args>
inline auto
Gpu::typed_launch(dim3,
                  dim3,
                  size_t,
                  gpu_stream_t,
                  CudaGlobalKernelPtr<Args...>,
                  Args...) const -> gpu_error_t {
  return gpu_error_not_supported;
}

template <auto*,
          typename... Args>
inline auto
Gpu::typed_launch(dim3,
                  dim3,
                  size_t,
                  gpu_stream_t,
                  Args...) const -> gpu_error_t {
  return gpu_error_not_supported;
}
#endif

}  // namespace detail
}  // namespace carbon
