#pragma once

#include <carbon/export.h>
#include <carbon/logging.h>

#include <carbon/detail/backend/cuda_like.h>

#include <stdexcept>

#include <memory>

namespace carbon {
namespace detail {

class Gpu;

/**
 * Owning view of a CUDA event; shallow copies share the same underlying gpu_event_t via
 * std::shared_ptr. Created only through Gpu::event().
 */
class CARBON_API Event {
  friend class Gpu;

  Event() = default;
  explicit Event(std::shared_ptr<void> inner) noexcept;

public:
  Event(const Event&) = default;
  Event&
  operator=(const Event&) = default;
  Event(Event&&) noexcept = default;
  Event&
  operator=(Event&&) noexcept = default;

  /** Underlying event handle, or nullptr if empty. */
  [[nodiscard]] auto
  get() const noexcept -> gpu_event_t {
    return static_cast<gpu_event_t>(inner_.get());
  }

  [[nodiscard]] explicit
  operator bool() const noexcept {
    return static_cast<bool>(inner_);
  }

  /** Block until this event has been recorded and all prior work on its stream completed. */
  auto
  synchronize() const -> void;

  /**
   * @return @c true if the event is complete; @c false if still in flight.
   * @throws std::runtime_error on backend error other than "not ready".
   */
  [[nodiscard]] auto
  query() const -> bool;

private:
  std::shared_ptr<void> inner_{};
};

inline Event::Event(std::shared_ptr<void> inner) noexcept
    : inner_(std::move(inner)) {}

inline auto
Event::synchronize() const -> void {
  const gpu_event_t e = get();
  if (e == nullptr) {
    return;
  }
  const gpu_error_t err = gpu_event_synchronize(e);
  if (err != gpu_success) {
    ::carbon::log::throw_with_log<std::runtime_error>("Event::synchronize failed: {}",
                                                      gpu_error_string(err));
  }
}

inline auto
Event::query() const -> bool {
  const gpu_event_t e = get();
  if (e == nullptr) {
    return true;
  }
  const gpu_error_t err = gpu_event_query(e);
  if (err == gpu_success) {
    return true;
  }
#if CARBON_BACKEND_IS_HIP
  if (err == hipErrorNotReady) {
    return false;
  }
#else
  if (err == cudaErrorNotReady) {
    return false;
  }
#endif
  ::carbon::log::throw_with_log<std::runtime_error>("Event::query failed: {}",
                                                    gpu_error_string(err));
}

}  // namespace detail
}  // namespace carbon
