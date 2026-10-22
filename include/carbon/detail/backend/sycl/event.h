#pragma once

/**
 * SYCL-backend wrapper around a single @c sycl::event. Mirrors the CUDA wrapper's role
 * (per-GPU recordable handle for cross-stream synchronisation) using the same
 * shallow-shared-pointer ownership model.
 *
 * @par Public surface vs the CUDA wrapper
 * The CUDA wrapper exposes @c get() returning @c cudaEvent_t. The SYCL wrapper exposes
 * @c event() returning @c sycl::event&. Distinct accessor names keep fork-points
 * explicit at the call site (see the matching note in @c stream.h).
 *
 * @par Lifetime
 * @c sycl::event is itself a refcounted SYCL handle, so the @c shared_ptr layer is a
 * symmetry choice with the CUDA wrapper rather than a memory necessity. Default
 * construction yields an empty event; @c bool conversion reports populatedness.
 */

#include <carbon/export.h>
#include <carbon/logging.h>

#include <sycl/sycl.hpp>

#include <memory>
#include <stdexcept>
#include <utility>

namespace carbon {
namespace detail {

class Gpu;

class CARBON_API Event {
  friend class Gpu;

  Event() = default;
  explicit Event(std::shared_ptr<sycl::event> e) noexcept
      : event_(std::move(e)) {}

public:
  Event(const Event&) = default;
  Event&
  operator=(const Event&) = default;
  Event(Event&&) noexcept = default;
  Event&
  operator=(Event&&) noexcept = default;

  /** Underlying @c sycl::event. UB on a default-constructed / moved-from instance. */
  [[nodiscard]] auto
  event() const noexcept -> sycl::event& {
    return *event_;
  }

  [[nodiscard]] explicit
  operator bool() const noexcept {
    return static_cast<bool>(event_);
  }

  auto
  synchronize() const -> void;

  [[nodiscard]] auto
  query() const -> bool;

private:
  std::shared_ptr<sycl::event> event_{};
};

inline auto
Event::synchronize() const -> void {
  if (!event_) {
    return;
  }
  try {
    event_->wait_and_throw();
  } catch (const std::exception& ex) {
    ::carbon::log::throw_with_log<std::runtime_error>("Event::synchronize failed: {}", ex.what());
  }
}

inline auto
Event::query() const -> bool {
  if (!event_) {
    return true;
  }
  return event_->get_info<sycl::info::event::command_execution_status>() ==
         sycl::info::event_command_status::complete;
}

}  // namespace detail
}  // namespace carbon
