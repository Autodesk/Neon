#pragma once

#include <carbon/detail/buffer_group.h>
#include <carbon/detail/event_group.h>
#include <carbon/detail/gpu.h>
#include <carbon/detail/stream_group.h>

#include <cstddef>
#include <exception>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace carbon {

enum class StreamPriority : unsigned char {
  Normal = 0,
  High = 1,
};

class Report;

namespace detail {

/**
 * A set of detail::Gpu handles. Device enumeration is explicit and heavyweight
 * (one cudaGetDeviceProperties per device): construct with
 * GpuGroup(carbon::get_all_devices()) to hold one Gpu per visible CUDA device
 * (ordinals 0 .. Gpu::device_count()-1); that overload logs a short summary and
 * enables peer access between every P2P-capable device pair.
 */
class CARBON_API GpuGroup {
public:
  /** Default: an empty group with no devices (no device probe). */
  GpuGroup();

  /**
   * Takes ownership of @p gpus. Typical use: GpuGroup(carbon::get_all_devices()).
   */
  explicit GpuGroup(std::vector<Gpu> gpus);

  [[nodiscard]] auto
  size() const noexcept -> size_t;
  [[nodiscard]] auto
  empty() const noexcept -> bool {
    return gpus_.empty();
  }

  [[nodiscard]] auto
  devices() const noexcept -> const std::vector<Gpu>& {
    return gpus_;
  }

  /**
   * Append a JSON object "carbon" on @p report with gpu_count and each device as gpu_<id>
   * (same per-device layout as detail::Gpu::report, including allocated_bytes).
   */
  auto
  report(Report& report) const -> void;

  /** Same as report(carbon::global_report()). Requires carbon::report::init. */
  auto
  report() const -> void;

  /**
   * Set each device in turn as current and invoke @p f with the corresponding Gpu. Use this to
   * run host-side setup and kernel launches on every visible GPU.
   */
  template <typename F>
  auto
  for_each_device(F&& f) const -> void;

  /** Convenience overload when you prefer std::function over a lambda. */
  auto
  for_each_device(const std::function<void(const Gpu&)>& f) const -> void;

  /**
   * Like for_each_device, but runs @p f concurrently (OpenMP) with one iteration per GPU. Each
   * iteration calls set_device() for its Gpu before invoking @p f. If any invocation throws, one
   * exception is rethrown after the parallel region completes.
   */
  template <typename F>
  auto
  parallel_for_each_device(F&& f) const -> void;

  auto
  parallel_for_each_device(const std::function<void(const Gpu&)>& f) const -> void;

  /** Run launch_smoke_kernel() on every device (no-op if empty). */
  auto
  launch_smoke_kernel_on_each() const -> void;

  /**
   * Allocates one device buffer per GPU: for index @p i, allocates @p counts_per_gpu[i] elements of
   * type @p T on device @p i with relative_gpu_id @p i. @p counts_per_gpu.size() must equal
   * size(). @throws std::runtime_error if the lengths differ.
   */
  template <typename T>
  [[nodiscard]] auto
  buffer(const std::vector<size_t>& counts_per_gpu) const -> BufferGroup<T>;

  /** One CUDA stream per GPU (index @p i matches devices()[i]). */
  [[nodiscard]] auto
  stream() const -> StreamGroup;

  /** One stream per GPU with optional high driver priority on every device. */
  [[nodiscard]] auto
  stream(StreamPriority priority) const -> StreamGroup;

  /** One CUDA event per GPU (index @p i matches devices()[i]). */
  [[nodiscard]] auto
  event() const -> EventGroup;

private:
  std::vector<Gpu> gpus_{};
};

}  // namespace detail
}  // namespace carbon

#include "carbon/detail/gpu_group.inl"
