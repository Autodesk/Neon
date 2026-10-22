#pragma once

#include <carbon/detail/event_group.h>
#include <carbon/detail/shared_vector_base.h>
#include <carbon/detail/stream.h>
#include <carbon/logging.h>

#include <cstddef>
#include <exception>
#include <stdexcept>

namespace carbon {

namespace detail {
class GpuGroup;
}

/**
 * Contiguous sequence of @c detail::CudaStream values in a shared std::vector.
 *
 * Inherits shallow-copy / move-keeps-valid semantics from @ref detail::SharedVectorBase.
 * Indexed access, @c size(), @c empty(), @c get_uid() and the @c bool conversion all come from
 * the base; @c get_uid() returns the address of the vector object as @c std::uint64_t (it is
 * not the addresses of the underlying streams' handles).
 */
class StreamGroup : public detail::SharedVectorBase<StreamGroup, detail::CudaStream> {
  using Base = detail::SharedVectorBase<StreamGroup, detail::CudaStream>;
  friend class detail::GpuGroup;

public:
  /** Default-construct an empty group (non-null vector with zero elements). */
  StreamGroup() = default;

  /**
   * @brief Block until every per-GPU stream in the group has drained its outstanding work.
   *
   * Backend-neutral synchronisation point: equivalent to @c cudaStreamSynchronize on every
   * @c detail::CudaStream under the CUDA backend, and to @c sycl::queue::wait under the
   * SYCL backend. Replaces backend-specific sync call sites in user code so the public
   * API does not have to mention @c cudaStream_t / @c sycl::queue.
   *
   * Empty groups are a no-op. Throws @c std::runtime_error on any underlying failure;
   * later streams in the group are still attempted before the throw so a failure on
   * GPU 0 does not leak unsynchronised work on GPU 1.
   */
  auto
  wait() const -> void {
#if CARBON_BACKEND_IS_CUDA
    cudaError_t first_err = cudaSuccess;
    for (size_t i = 0; i < this->size(); ++i) {
      const cudaError_t err = cudaStreamSynchronize((*this)[i].get());
      if (err != cudaSuccess && first_err == cudaSuccess) {
        first_err = err;
      }
    }
    if (first_err != cudaSuccess) {
      ::carbon::log::throw_with_log<std::runtime_error>(
          "StreamGroup::wait: cudaStreamSynchronize failed: {} (cudaError {})",
          cudaGetErrorString(first_err), static_cast<int>(first_err));
    }
#elif CARBON_BACKEND_IS_HIP
    hipError_t first_err = hipSuccess;
    for (size_t i = 0; i < this->size(); ++i) {
      const hipError_t err = hipStreamSynchronize((*this)[i].get());
      if (err != hipSuccess && first_err == hipSuccess) {
        first_err = err;
      }
    }
    if (first_err != hipSuccess) {
      ::carbon::log::throw_with_log<std::runtime_error>(
          "StreamGroup::wait: hipStreamSynchronize failed: {} (hipError {})",
          hipGetErrorString(first_err), static_cast<int>(first_err));
    }
#elif CARBON_BACKEND_IS_SYCL
    // sycl::queue::wait() blocks until all enqueued work on this queue is complete; it
    // throws on async errors caught by the queue's exception handler. We let the first
    // failure propagate but still run wait() on every queue first so partial completion
    // is fully drained before the throw, matching the CUDA branch's "best-effort drain"
    // semantics.
    std::exception_ptr first_err{};
    for (size_t i = 0; i < this->size(); ++i) {
      try {
        (*this)[i].queue().wait_and_throw();
      } catch (...) {
        if (!first_err) {
          first_err = std::current_exception();
        }
      }
    }
    if (first_err) {
      try {
        std::rethrow_exception(first_err);
      } catch (const std::exception& ex) {
        ::carbon::log::throw_with_log<std::runtime_error>("StreamGroup::wait: SYCL: {}", ex.what());
      } catch (...) {
        ::carbon::log::throw_with_log<std::runtime_error>(
            "StreamGroup::wait: SYCL queue threw a non-std::exception");
      }
    }
#endif
  }

#if CARBON_BACKEND_IS_CUDALIKE
  /**
   * @brief Raw per-GPU stream handle for foreign-launch integrations (libcarbonpy Warp path).
   *
   * Returns the underlying @c cudaStream_t / @c hipStream_t for @p gpu . The caller must
   * issue work on the matching device context. Not available on the SYCL backend.
   */
  [[nodiscard]] auto
  native_at(size_t gpu) const -> detail::gpu_stream_t {
    return (*this)[gpu].get();
  }
#endif

  /**
   * @brief Per-GPU cross-stream ordering: make stream @c i wait until @p events at index @p i
   * have completed.
   *
   * Backend-neutral wrapper around CUDA's @c cudaStreamWaitEvent and the equivalent SYCL
   * barrier. The wait is on-device only — the host returns immediately; the per-GPU stream
   * just won't begin its next submission until the matching event has signalled. Both groups
   * must have the same size (or this group must be empty, which is a no-op). Throws
   * @c std::runtime_error if the sizes mismatch.
   *
   * @par CUDA backend
   * Each entry runs @c cudaStreamWaitEvent(streams[i], events[i], 0). Failures are
   * accumulated across GPUs and the first error is thrown after every GPU has been
   * attempted (matching the @c wait() best-effort drain semantics).
   *
   * @par SYCL backend
   * Each per-GPU @c sycl::queue submits a no-op @c AdaptiveCpp_enqueue_custom_operation that
   * declares @c events[i] as a dependency, ordering subsequent submissions on this queue
   * after the event. AdaptiveCpp's in-order queue semantics turn this into a barrier on the
   * receiving queue without needing a host round-trip.
   */
  inline auto
  wait_for(const EventGroup& events) const -> void;

  /**
   * @brief Each stream in this group waits for @b every event in @p events
   *        (full N×N fan-in), not just the per-GPU entry.
   *
   * The default @ref wait_for is a strict per-GPU barrier:
   * @c stream[i].wait_for(event[i]) . That's the right contract for
   * intra-GPU pipelines (one stream's compute fences against another
   * stream's prior compute @e on the same device), but it's @e not
   * enough for cross-GPU peer copies. `cudaMemcpyPeerAsync` issued on
   * @c stream[g] (destination side) only orders against operations on
   * @c stream[g] itself — it does @b not implicitly wait for prior
   * compute on @c stream[g'] (source side), where @c g' is the source
   * GPU whose memory is being read. The same gotcha is documented on
   * @c carbon::Halo and is the root cause of the cross-stream race
   * that @c OverlappedStencil used to expose (halo on its dedicated
   * @c halo_stream racing the previous iteration's compute on
   * @c compute_stream on the @e source GPU).
   *
   * Calling @c stream[g].wait_for(event[g']) for every @c g' = 0..N-1
   * pins the destination-side halo stream behind every source-side
   * compute stream, eliminating the race at the cost of @c N²
   * @c cudaStreamWaitEvent calls (cheap: each call is host-side only
   * and queues a single 8-byte payload on the per-stream command
   * buffer). For typical carbon device counts (1–8 visible GPUs) this
   * is a handful of nanoseconds per call.
   *
   * Both groups must have the same size (or this group must be empty,
   * which is a no-op). Throws @c std::runtime_error on size mismatch.
   *
   * @par CUDA backend
   * Two nested loops: outer over receiving streams @c i , inner over
   * source events @c j , both running @c cudaStreamWaitEvent(streams[i],
   * events[j], 0). Errors are accumulated and the first one is thrown
   * after every pair has been attempted.
   *
   * @par SYCL backend
   * Each per-GPU queue submits one @c AdaptiveCpp_enqueue_custom_operation
   * that declares every event in @p events as a dependency. Same fan-in
   * shape, same N×N cost.
   */
  inline auto
  wait_for_all(const EventGroup& events) const -> void;

private:
  /** Adopting constructor used by @ref detail::GpuGroup. */
  using Base::Base;
};

static_assert(DataUid<StreamGroup>);

// ---------------------------------------------------------------------------
// Out-of-class definitions for the cross-stream ordering primitives. Defined
// here (rather than inside the class bodies) so each side has a complete view
// of the other group type — `<carbon/detail/stream_group.h>` is the single
// translation unit users include to get both groups + the cross-ordering
// helpers.
// ---------------------------------------------------------------------------

inline auto
EventGroup::synchronize() const -> void {
  for (size_t i = 0; i < this->size(); ++i) {
    (*this)[i].synchronize();
  }
}

inline auto
EventGroup::query() const -> bool {
  for (size_t i = 0; i < this->size(); ++i) {
    if (!(*this)[i].query()) {
      return false;
    }
  }
  return true;
}

inline auto
EventGroup::record(const StreamGroup& streams) -> void {
  if (this->empty()) {
    return;
  }
  if (this->size() != streams.size()) {
    ::carbon::log::throw_with_log<std::runtime_error>(
        "EventGroup::record: size mismatch (events {} vs streams {})", this->size(),
        streams.size());
  }
#if CARBON_BACKEND_IS_CUDA
  cudaError_t first_err = cudaSuccess;
  for (size_t i = 0; i < this->size(); ++i) {
    const cudaError_t err = cudaEventRecord((*this)[i].get(), streams[i].get());
    if (err != cudaSuccess && first_err == cudaSuccess) {
      first_err = err;
    }
  }
  if (first_err != cudaSuccess) {
    ::carbon::log::throw_with_log<std::runtime_error>(
        "EventGroup::record: cudaEventRecord failed: {} (cudaError {})",
        cudaGetErrorString(first_err), static_cast<int>(first_err));
  }
#elif CARBON_BACKEND_IS_HIP
  hipError_t first_err = hipSuccess;
  for (size_t i = 0; i < this->size(); ++i) {
    const hipError_t err = hipEventRecord((*this)[i].get(), streams[i].get());
    if (err != hipSuccess && first_err == hipSuccess) {
      first_err = err;
    }
  }
  if (first_err != hipSuccess) {
    ::carbon::log::throw_with_log<std::runtime_error>(
        "EventGroup::record: hipEventRecord failed: {} (hipError {})", hipGetErrorString(first_err),
        static_cast<int>(first_err));
  }
#elif CARBON_BACKEND_IS_SYCL
  std::exception_ptr first_err{};
  for (size_t i = 0; i < this->size(); ++i) {
    try {
      (*this)[i].event() =
          streams[i].queue().AdaptiveCpp_enqueue_custom_operation([](sycl::interop_handle&) {
          });
    } catch (...) {
      if (!first_err) {
        first_err = std::current_exception();
      }
    }
  }
  if (first_err) {
    try {
      std::rethrow_exception(first_err);
    } catch (const std::exception& ex) {
      ::carbon::log::throw_with_log<std::runtime_error>("EventGroup::record: SYCL: {}", ex.what());
    } catch (...) {
      ::carbon::log::throw_with_log<std::runtime_error>(
          "EventGroup::record: SYCL queue threw a non-std::exception");
    }
  }
#endif
}

inline auto
StreamGroup::wait_for(const EventGroup& events) const -> void {
  if (this->empty()) {
    return;
  }
  if (this->size() != events.size()) {
    ::carbon::log::throw_with_log<std::runtime_error>(
        "StreamGroup::wait_for: size mismatch (streams {} vs events {})", this->size(),
        events.size());
  }
#if CARBON_BACKEND_IS_CUDA
  cudaError_t first_err = cudaSuccess;
  for (size_t i = 0; i < this->size(); ++i) {
    const cudaError_t err = cudaStreamWaitEvent((*this)[i].get(), events[i].get(), 0);
    if (err != cudaSuccess && first_err == cudaSuccess) {
      first_err = err;
    }
  }
  if (first_err != cudaSuccess) {
    ::carbon::log::throw_with_log<std::runtime_error>(
        "StreamGroup::wait_for: cudaStreamWaitEvent failed: {} (cudaError {})",
        cudaGetErrorString(first_err), static_cast<int>(first_err));
  }
#elif CARBON_BACKEND_IS_HIP
  hipError_t first_err = hipSuccess;
  for (size_t i = 0; i < this->size(); ++i) {
    const hipError_t err = hipStreamWaitEvent((*this)[i].get(), events[i].get(), 0);
    if (err != hipSuccess && first_err == hipSuccess) {
      first_err = err;
    }
  }
  if (first_err != hipSuccess) {
    ::carbon::log::throw_with_log<std::runtime_error>(
        "StreamGroup::wait_for: hipStreamWaitEvent failed: {} (hipError {})",
        hipGetErrorString(first_err), static_cast<int>(first_err));
  }
#elif CARBON_BACKEND_IS_SYCL
  std::exception_ptr first_err{};
  for (size_t i = 0; i < this->size(); ++i) {
    try {
      const sycl::event dep = events[i].event();
      (void)(*this)[i].queue().AdaptiveCpp_enqueue_custom_operation(
          [](sycl::interop_handle&) {
          },
          dep);
    } catch (...) {
      if (!first_err) {
        first_err = std::current_exception();
      }
    }
  }
  if (first_err) {
    try {
      std::rethrow_exception(first_err);
    } catch (const std::exception& ex) {
      ::carbon::log::throw_with_log<std::runtime_error>("StreamGroup::wait_for: SYCL: {}",
                                                        ex.what());
    } catch (...) {
      ::carbon::log::throw_with_log<std::runtime_error>(
          "StreamGroup::wait_for: SYCL queue threw a non-std::exception");
    }
  }
#endif
}

inline auto
StreamGroup::wait_for_all(const EventGroup& events) const -> void {
  if (this->empty()) {
    return;
  }
  if (this->size() != events.size()) {
    ::carbon::log::throw_with_log<std::runtime_error>(
        "StreamGroup::wait_for_all: size mismatch (streams {} vs events {})", this->size(),
        events.size());
  }
#if CARBON_BACKEND_IS_CUDA
  cudaError_t first_err = cudaSuccess;
  for (size_t i = 0; i < this->size(); ++i) {
    for (size_t j = 0; j < events.size(); ++j) {
      const cudaError_t err = cudaStreamWaitEvent((*this)[i].get(), events[j].get(), 0);
      if (err != cudaSuccess && first_err == cudaSuccess) {
        first_err = err;
      }
    }
  }
  if (first_err != cudaSuccess) {
    ::carbon::log::throw_with_log<std::runtime_error>(
        "StreamGroup::wait_for_all: cudaStreamWaitEvent failed: {} (cudaError {})",
        cudaGetErrorString(first_err), static_cast<int>(first_err));
  }
#elif CARBON_BACKEND_IS_HIP
  hipError_t first_err = hipSuccess;
  for (size_t i = 0; i < this->size(); ++i) {
    for (size_t j = 0; j < events.size(); ++j) {
      const hipError_t err = hipStreamWaitEvent((*this)[i].get(), events[j].get(), 0);
      if (err != hipSuccess && first_err == hipSuccess) {
        first_err = err;
      }
    }
  }
  if (first_err != hipSuccess) {
    ::carbon::log::throw_with_log<std::runtime_error>(
        "StreamGroup::wait_for_all: hipStreamWaitEvent failed: {} (hipError {})",
        hipGetErrorString(first_err), static_cast<int>(first_err));
  }
#elif CARBON_BACKEND_IS_SYCL
  std::exception_ptr first_err{};
  for (size_t i = 0; i < this->size(); ++i) {
    try {
      std::vector<sycl::event> deps;
      deps.reserve(events.size());
      for (size_t j = 0; j < events.size(); ++j) {
        deps.push_back(events[j].event());
      }
      (void)(*this)[i].queue().AdaptiveCpp_enqueue_custom_operation(
          [](sycl::interop_handle&) {
          },
          deps);
    } catch (...) {
      if (!first_err) {
        first_err = std::current_exception();
      }
    }
  }
  if (first_err) {
    try {
      std::rethrow_exception(first_err);
    } catch (const std::exception& ex) {
      ::carbon::log::throw_with_log<std::runtime_error>("StreamGroup::wait_for_all: SYCL: {}",
                                                        ex.what());
    } catch (...) {
      ::carbon::log::throw_with_log<std::runtime_error>(
          "StreamGroup::wait_for_all: SYCL queue threw a non-std::exception");
    }
  }
#endif
}

}  // namespace carbon
