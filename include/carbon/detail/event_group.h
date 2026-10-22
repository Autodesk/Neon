#pragma once

#include <carbon/detail/event.h>
#include <carbon/detail/shared_vector_base.h>
#include <carbon/logging.h>

#include <cstddef>
#include <exception>
#include <stdexcept>

namespace carbon {

class StreamGroup;

namespace detail {
class GpuGroup;
}

/**
 * Contiguous sequence of detail::Event values in a shared std::vector.
 *
 * Inherits shallow-copy / move-keeps-valid semantics from @ref detail::SharedVectorBase.
 * Indexed access, @c size(), @c empty(), @c get_uid() and the @c bool conversion all come from
 * the base; @c get_uid() returns the address of the vector object as @c std::uint64_t (it is
 * not the addresses of the underlying events' handles).
 */
class EventGroup : public detail::SharedVectorBase<EventGroup, detail::Event> {
  using Base = detail::SharedVectorBase<EventGroup, detail::Event>;
  friend class detail::GpuGroup;

public:
  /** Default-construct an empty group (non-null vector with zero elements). */
  EventGroup() = default;

  /**
   * @brief Record one per-GPU event marking the current frontier of the matching per-GPU
   * stream.
   *
   * Backend-neutral wrapper around CUDA's @c cudaEventRecord and the equivalent SYCL
   * barrier event. After this call, each event @p i in the group fires once every prior
   * submission on @p streams[i] has completed. Both groups must have the same size (or
   * this group must be empty, which is a no-op). Throws @c std::runtime_error on size
   * mismatch or any underlying backend failure.
   *
   * @par CUDA backend
   * Each entry runs @c cudaEventRecord(events[i], streams[i]); the existing @c cudaEvent_t
   * handle is mutated in place, so prior shallow copies of this @c EventGroup (or @c Event
   * value handles obtained via indexed access) observe the new recording too.
   *
   * @par SYCL backend
   * Each per-GPU @c sycl::queue submits a no-op @c AdaptiveCpp_enqueue_custom_operation
   * whose returned @c sycl::event captures the queue's current frontier. The wrapped
   * @c sycl::event inside @c detail::Event is reassigned in place via the @c shared_ptr,
   * so all shallow copies see the new event.
   */
  inline auto
  record(const StreamGroup& streams) -> void;

  /** Block on the host until every event in the group has completed. */
  inline auto
  synchronize() const -> void;

  /** @c true when every event in the group has completed. */
  [[nodiscard]] inline auto
  query() const -> bool;

private:
  /** Adopting constructor used by @ref detail::GpuGroup. */
  using Base::Base;
};

static_assert(DataUid<EventGroup>);

}  // namespace carbon
