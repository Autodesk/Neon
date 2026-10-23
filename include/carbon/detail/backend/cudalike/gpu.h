#pragma once

#include <carbon/detail/allocation.h>
#include <carbon/export.h>
#include <carbon/logging.h>

#include <carbon/detail/backend/cuda_like.h>

#include <cstddef>
#include <exception>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace carbon {

class Report;

namespace detail {

class CudaStream;
class Event;

/** Snapshot of common CUDA device attributes (host API; scalar fields only). */
struct Properties {
  std::string name;
  int compute_major{};
  int compute_minor{};
  size_t total_global_mem_bytes{};
  int multiprocessor_count{};
  int max_threads_per_block{};
  int warp_size{};
  int max_threads_per_multiprocessor{};
  int clock_rate_khz{};
  int memory_clock_rate_khz{};
  int memory_bus_width_bits{};
  int l2_cache_size_bytes{};
  size_t total_const_mem_bytes{};
  size_t shared_mem_per_block_bytes{};
  int regs_per_block{};
};

/**
 * Host-side function pointer type for a @c __global__ kernel @c void kern(Args...).
 * Used by @c Gpu::typed_launch. The internal @c Gpu::execute overload uses @c auto* for the kernel
 * because @c void(*Kernel)(Args...) cannot follow @c typename... Args in a template parameter list
 * (Kernel would be non-deducible).
 */
template <typename... Args>
using CudaGlobalKernelPtr = void (*)(Args...);

/** Thin wrapper around a CUDA device ordinal (Carbon always builds against CUDA). */
class CARBON_API Gpu {
public:
  explicit Gpu(int device_id);

  [[nodiscard]] auto
  device_id() const noexcept -> int {
    return device_id_;
  }

  /** Human-readable device name (same as Properties::name). */
  [[nodiscard]] auto
  name() const noexcept -> const std::string& {
    return props_.name;
  }

  /** Full property snapshot queried at construction. */
  [[nodiscard]] auto
  properties() const noexcept -> const Properties& {
    return props_;
  }

  /** Number of visible CUDA devices (0 if the driver reports none). */
  [[nodiscard]] static auto
  device_count() noexcept -> int;

  /**
   * Append a JSON object "carbon" on @p report with gpu_count (1) and this device under
   * "gpu_<device_id>" (all fields from properties() plus device_id and allocated_bytes).
   */
  auto
  report(Report& report) const -> void;

  /** Same as report(carbon::global_report()). Requires carbon::report::init. */
  auto
  report() const -> void;

  /** Make this device the active CUDA device for the calling host thread. */
  auto
  set_device() const -> void;

  /**
   * Launch a minimal built-in kernel on this device (used for smoke tests and driver/runtime
   * checks). For your own kernels, use @ref typed_launch from a @c .cu translation unit (or other TU
   * compiled with nvcc).
   */
  auto
  launch_smoke_kernel() const -> void;

  /**
   * Typed kernel launch. Order: @p grid, @p block, @p shared_mem, @p stream, then kernel arguments
   * (variadic pack last). @p kernel must be @c CudaGlobalKernelPtr<Args...>. Returns
   * gpu_get_last_error() after launch configuration.
   */
  template <typename... Args>
  [[nodiscard]] auto
  typed_launch(dim3 grid,
               dim3 block,
               size_t shared_mem,
               gpu_stream_t stream,
               CudaGlobalKernelPtr<Args...> kernel,
               Args... args) const -> gpu_error_t;

  /**
   * Same as the pointer @ref typed_launch but @p Kernel is the address of a @c __global__ function
   * (non-type template parameter). Fails to compile unless
   * @c std::decay_t<decltype(Kernel)> is @c CudaGlobalKernelPtr<Args...>. Call:
   * @c typed_launch<&my_kernel>(grid, block, shared_mem, stream, args...).
   */
  template <auto* Kernel,
            typename... Args>
  [[nodiscard]] auto
  typed_launch(dim3 grid,
               dim3 block,
               size_t shared_mem,
               gpu_stream_t stream,
               Args... args) const -> gpu_error_t;

  /**
   * Allocates @p count * sizeof(T) bytes on this device with gpu_malloc and returns an
   * Allocation<T>. Sets the current device; @p relative_gpu_id is stored on the allocation (e.g.
   * index in detail::GpuGroup::devices()). Uses a refcounted deleter (gpu_set_device + gpu_free).
   * Updates allocated_bytes() for this device; the count decreases when the last Allocation
   * sharing the storage is destroyed.
   */
  template <typename T>
  [[nodiscard]] auto
  buffer(size_t count,
         int relative_gpu_id) const -> Allocation<T>;

  /** Creates a non-default CUDA stream on this device (refcounted; copies share the handle). */
  [[nodiscard]] auto
  stream() const -> CudaStream;

  /** Stream with explicit driver priority (@p greatest_priority wins over lower values). */
  [[nodiscard]] auto
  stream_with_priority(int greatest_priority) const -> CudaStream;

  /** Creates a CUDA event on this device (refcounted; copies share the handle). */
  [[nodiscard]] auto
  event() const -> Event;

  /**
   * Bytes of device memory allocated through Gpu::buffer on this device that are still live
   * (refcount not yet zero). Does not include gpu_malloc outside this API.
   */
  [[nodiscard]] auto
  allocated_bytes() const noexcept -> size_t;

private:
  /** Unchecked launch (@c gpu_runtime_launch_kernel); used by @ref typed_launch. */
  template <typename... Args>
  [[nodiscard]] auto
  execute(const void* kernel,
          dim3 grid,
          dim3 block,
          size_t shared_mem,
          gpu_stream_t stream,
          Args... args) const -> gpu_error_t;

  template <auto* Kernel,
            typename... Args>
  [[nodiscard]] auto
  execute(dim3 grid,
          dim3 block,
          size_t shared_mem,
          gpu_stream_t stream,
          Args... args) const -> gpu_error_t;

  int device_id_{};
  Properties props_{};
};

}  // namespace detail

}  // namespace carbon

#include "carbon/detail/gpu.inl"
