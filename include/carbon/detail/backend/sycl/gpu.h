#pragma once

/**
 * SYCL-backend per-device handle. Selected by the dispatcher in
 * include/carbon/detail/gpu.h when CARBON_BACKEND_IS_SYCL.
 *
 * @par Surface differences vs the CUDA wrapper
 * - The class name @c Gpu is preserved so detail-layer call sites that handle a
 *   @c std::vector<Gpu> stay unchanged across backends.
 * - The @c Properties struct is ABI-compatible with the CUDA version (same field names,
 *   same types, same docstrings); fields not exposed by SYCL device queries (e.g.
 *   @c clock_rate_khz, @c memory_bus_width_bits) are populated with zero with a comment
 *   noting the gap. The non-zero fields cover everything the existing carbon test suite
 *   asserts on (name, max_threads_per_block, multiprocessor_count, total_global_mem).
 * - @c typed_launch / @c execute do NOT exist under SYCL. SYCL kernels are launched via
 *   @c sycl::queue::parallel_for at the call site (see
 *   detail/backend/sycl/dense_launch.h); there is no host-visible kernel pointer to hand
 *   to a launcher.
 *
 * @par Internal storage
 * Each @c Gpu owns a @c sycl::device plus a (lazily initialised) @c sycl::context, both
 * by value (SYCL handles are refcounted internally). Streams (@c sycl::queue) and events
 * (@c sycl::event) are constructed against this device + context on demand; ownership
 * is then promoted to a @c std::shared_ptr inside the @ref CudaStream / @ref Event
 * wrapper for shallow-copy semantics.
 */

#include <carbon/detail/allocation.h>
#include <carbon/export.h>
#include <carbon/logging.h>

#include <sycl/sycl.hpp>

#include <cstddef>
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

/**
 * Snapshot of common device attributes; ABI-compatible with the CUDA backend's
 * @c Properties struct so the rest of carbon can read it without a backend fork. Fields
 * SYCL cannot answer (compute capability major/minor, clock rate, memory bus width,
 * register count) are populated with @c 0 — consumers that depend on those values are
 * responsible for handling the zero case (the existing @c report.cpp output simply
 * emits the zero, which is unambiguous).
 */
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
 * Per-device handle for the SYCL backend. Constructed from a host-visible "device id"
 * (an index into a stable, per-process ordering of @c sycl::platform::get_devices for
 * GPU + accelerator devices, see @c device_count() / @c device_at).
 *
 * The host code calling @c stream() / @c event() / @c buffer<T>() runs in the same
 * process as the SYCL runtime; concurrent @c stream() calls from multiple threads are
 * each safe because @c sycl::queue construction is internally synchronised.
 */
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

  /**
   * Number of SYCL devices visible to the AdaptiveCpp runtime that pass the GPU
   * selector filter. Identical semantics to the CUDA backend's @c device_count
   * (returns 0 when no GPU device is found).
   */
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

  /**
   * Pin the calling host thread to this device. SYCL has no global "current device"
   * concept (the device is fixed when each @c sycl::queue is constructed), so this is
   * a no-op under SYCL — kept for API symmetry with the CUDA backend so call sites
   * like @c GpuGroup::for_each_device(g.set_device(); f(g);) compile unchanged.
   */
  auto
  set_device() const -> void;

  /**
   * Launch a minimal built-in kernel on this device (used for smoke tests). Submits an
   * empty @c parallel_for(range_1d{1}) on a transient @c sycl::queue and waits.
   */
  auto
  launch_smoke_kernel() const -> void;

  /**
   * Allocate @p count * sizeof(T) bytes on this device using SYCL USM device allocation
   * and return an @ref Allocation<T>. The deleter holds a refcounted handle to this
   * device's @c sycl::context so the allocation can be freed even after the originating
   * @c Gpu has been destroyed (the same lifetime guarantee the CUDA backend offers).
   * Updates @c allocated_bytes() through the shared @c gpu_track_allocation hook.
   */
  template <typename T>
  [[nodiscard]] auto
  buffer(size_t count,
         int relative_gpu_id) const -> Allocation<T>;

  /** Construct a non-default @c sycl::queue on this device (refcounted; copies share). */
  [[nodiscard]] auto
  stream() const -> CudaStream;

  /** Construct an empty @c sycl::event sized to this device. */
  [[nodiscard]] auto
  event() const -> Event;

  /**
   * Bytes of device memory allocated through @c Gpu::buffer on this device that are
   * still live (refcount not yet zero). Same accounting as the CUDA backend; populated
   * via @c gpu_track_allocation in @c gpu_allocation_tracking.h.
   */
  [[nodiscard]] auto
  allocated_bytes() const noexcept -> size_t;

  /**
   * Internal accessors used by @c stream() / @c event() / @c buffer<T>() and the SYCL
   * launch helpers. Not stable; not part of the public detail/ surface. Returning by
   * value is fine because @c sycl::device / @c sycl::context are SYCL handles.
   */
  [[nodiscard]] auto
  sycl_device() const noexcept -> sycl::device {
    return device_;
  }
  [[nodiscard]] auto
  sycl_context() const noexcept -> sycl::context {
    return context_;
  }

private:
  int device_id_{};
  Properties props_{};
  sycl::device device_{};
  sycl::context context_{};
};

}  // namespace detail

}  // namespace carbon

#include <carbon/detail/gpu.inl>
