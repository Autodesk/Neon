#pragma once

/**
 * @file graph_replay.h
 * @brief Host-callable CUDA/HIP graph capture + replay for one or many streams.
 *
 * A @ref carbon::Skeleton with a fixed compiled schedule replays the exact same
 * sequence of kernel launches every iteration. On a per-iteration @c run() that
 * means re-paying the host-side launch cost (one @c cudaLaunchKernel + argument
 * marshalling per kernel, plus any cross-stream event ops) on every step. CUDA
 * graphs collapse that: capture the sequence once into a @c cudaGraph_t ,
 * instantiate an executable @c cudaGraphExec_t , and replay it with a single
 * @c cudaGraphLaunch — the driver walks the pre-baked node list instead of the
 * host re-issuing each node.
 *
 * @ref GraphReplay is a thin RAII holder over one instantiated graph. It handles
 * two capture shapes through one @ref capture entry point that takes the list of
 * per-GPU streams backing a single @ref carbon::Skeleton stream:
 * - @b single-GPU (one stream): capture is driven with
 *   @c cudaStreamCaptureModeThreadLocal — only the calling thread's work on the
 *   stream is captured.
 * - @b multi-GPU intra-process (N streams, one per device): the classic
 *   @b fork/join origin-stream pattern. Capture begins on the origin stream
 *   (device 0) in @c cudaStreamCaptureModeGlobal so the OpenMP-fanned per-GPU
 *   kernel launches on every peer stream are captured too. The origin records a
 *   fork event that every peer stream waits on (pulling the peers into the same
 *   capture), the schedule is enqueued, then each peer records a join event the
 *   origin waits on. @c cudaStreamEndCapture on the origin yields one unified
 *   cross-device graph; a single @c cudaGraphLaunch on the origin stream replays
 *   every device's nodes, honouring the halo's cross-GPU peer copies + event
 *   fences that were captured as intra-graph dependencies.
 *
 * The distributed (cross-rank NCCL) case is intentionally @b not captured here;
 * @ref carbon::Skeleton only builds the stream list for the intra-process path
 * and otherwise falls back to eager launch.
 *
 * Under the SYCL backend the whole type degrades to a no-op stub whose
 * @ref supported() returns @c false, so callers can reference it unconditionally
 * and branch on @ref supported() at runtime.
 */

#include <carbon/detail/backend/cuda_like.h>
#include <carbon/detail/backend/select.h>

#include <utility>
#include <vector>

namespace carbon {
namespace detail {

/**
 * @brief RAII holder for one captured+instantiated device graph on one stream.
 *
 * Non-copyable (owns a device resource), movable. Destroying / resetting frees
 * the executable graph. All error paths clear the sticky runtime error and
 * return @c false so the caller can fall back to eager launch — a machine that
 * cannot capture a given pipeline is never left worse than the non-graph path.
 */
class GraphReplay {
public:
  // Native stream handle as spelled in the signatures below. The cuda-like alias where
  // it exists; an opaque pointer under SYCL, where every method is a stub that ignores
  // its stream arguments (detail/backend/cuda_like.h defines no gpu_* aliases there, so
  // naming gpu_stream_t in a signature — even an uninstantiated template's — would not
  // parse).
#if CARBON_BACKEND_IS_CUDALIKE
  using stream_handle = gpu_stream_t;
#else
  using stream_handle = void*;
#endif

  GraphReplay() = default;

  ~GraphReplay() {
    reset();
  }

  GraphReplay(const GraphReplay&) = delete;
  auto
  operator=(const GraphReplay&) -> GraphReplay& = delete;

  GraphReplay(GraphReplay&& other) noexcept {
    *this = std::move(other);
  }

  auto
  operator=(GraphReplay&& other) noexcept -> GraphReplay& {
    if (this != &other) {
      reset();
#if CARBON_BACKEND_IS_CUDALIKE
      exec_ = other.exec_;
      device_id_ = other.device_id_;
      other.exec_ = gpu_graph_exec_t{};
      other.device_id_ = -1;
#endif
      ready_ = other.ready_;
      other.ready_ = false;
    }
    return *this;
  }

  /** Whether the active backend can capture graphs at all (CUDA / HIP: yes; SYCL: no). */
  [[nodiscard]] static constexpr auto
  supported() noexcept -> bool {
#if CARBON_BACKEND_IS_CUDALIKE
    return true;
#else
    return false;
#endif
  }

  /** True once @ref capture has produced a replayable executable graph. */
  [[nodiscard]] auto
  ready() const noexcept -> bool {
    return ready_;
  }

  /**
   * @brief Capture the work @p enqueue issues on @p streams into a replayable graph.
   *
   * @p streams are the per-GPU streams backing a single @ref carbon::Skeleton
   * stream (@p device_ids[i] is the CUDA device index of @p streams[i]);
   * @p streams[0] is the @b origin. @p enqueue is invoked exactly once, between
   * begin/end capture, and must issue only capture-safe async work (kernel
   * launches, async copies, event ops) — no synchronous calls, no allocation.
   * The work is @b recorded, not executed; call @ref launch afterwards.
   *
   * A single-element @p streams captures with thread-local mode. A multi-element
   * @p streams uses the global-mode fork/join origin-stream pattern so the
   * OpenMP-fanned peer-stream launches and the halo's cross-GPU dependencies are
   * folded into one unified cross-device graph replayed from the origin.
   *
   * Returns @c false (leaving @ref ready() @c false, with the sticky runtime
   * error cleared) if the backend is SYCL, @p streams is empty / mismatched, any
   * capture step fails, or instantiation fails. On a thrown exception from
   * @p enqueue the in-flight capture is torn down before rethrowing so no stream
   * is left capturing. Every early-out path frees any fork/join events it
   * created, so a failed capture never leaks an event.
   */
  template <typename EnqueueFn>
  [[nodiscard]] auto
  capture(const std::vector<stream_handle>& streams,
          const std::vector<int>& device_ids,
          EnqueueFn&& enqueue) -> bool {
#if CARBON_BACKEND_IS_CUDALIKE
    reset();
    const size_t n = streams.size();
    if (n == 0U || device_ids.size() != n) {
      return false;
    }
    const gpu_stream_t origin = streams[0];
    const int origin_dev = device_ids[0];
    const bool multi = n > 1U;

    // Pre-create the fork/join synchronisation events (multi-GPU only) *before*
    // begin_capture so no resource allocation happens inside the capture window.
    // fork_event lives on the origin device; each peer's join event lives on the
    // peer device (an event must be recorded from its own device's stream).
    gpu_event_t fork_event{};
    std::vector<gpu_event_t> join_events;
    auto destroy_events = [&]() noexcept {
      if (multi) {
        (void)gpu_event_destroy(fork_event);
        for (gpu_event_t e : join_events) {
          (void)gpu_event_destroy(e);
        }
      }
    };
    if (multi) {
      if (gpu_set_device(origin_dev) != gpu_success ||
          gpu_event_create_with_flags(&fork_event, gpu_event_disable_timing) != gpu_success) {
        (void)gpu_get_last_error();
        return false;
      }
      join_events.reserve(n - 1U);
      for (size_t i = 1; i < n; ++i) {
        gpu_event_t e{};
        if (gpu_set_device(device_ids[i]) != gpu_success ||
            gpu_event_create_with_flags(&e, gpu_event_disable_timing) != gpu_success) {
          (void)gpu_get_last_error();
          (void)gpu_event_destroy(fork_event);
          for (gpu_event_t done : join_events) {
            (void)gpu_event_destroy(done);
          }
          return false;
        }
        join_events.push_back(e);
      }
    }

    const gpu_stream_capture_mode_t mode =
        multi ? gpu_stream_capture_mode_global : gpu_stream_capture_mode_thread_local;
    if (gpu_set_device(origin_dev) != gpu_success ||
        gpu_stream_begin_capture(origin, mode) != gpu_success) {
      (void)gpu_get_last_error();
      destroy_events();
      return false;
    }

    // Fork: origin records, every peer waits — this pulls the peer streams into
    // the origin's capture so their subsequently-enqueued work joins the graph.
    if (multi) {
      (void)gpu_event_record(fork_event, origin);
      for (size_t i = 1; i < n; ++i) {
        (void)gpu_set_device(device_ids[i]);
        (void)gpu_stream_wait_event(streams[i], fork_event, 0U);
      }
      (void)gpu_set_device(origin_dev);
    }

    gpu_graph_t graph{};
    try {
      std::forward<EnqueueFn>(enqueue)();
    } catch (...) {
      // Drain the capture so the origin returns to a usable (non-capturing)
      // state, then propagate. The half-built graph + events are discarded.
      (void)gpu_stream_end_capture(origin, &graph);
      (void)gpu_get_last_error();
      destroy_events();
      throw;
    }

    // Join: every peer records, origin waits — folds all peer work back so
    // end_capture on the origin sees a single unified cross-device graph.
    if (multi) {
      for (size_t i = 1; i < n; ++i) {
        (void)gpu_set_device(device_ids[i]);
        (void)gpu_event_record(join_events[i - 1U], streams[i]);
      }
      (void)gpu_set_device(origin_dev);
      for (gpu_event_t e : join_events) {
        (void)gpu_stream_wait_event(origin, e, 0U);
      }
    }

    const gpu_error_t end_err = gpu_stream_end_capture(origin, &graph);
    destroy_events();
    if (end_err != gpu_success) {
      (void)gpu_get_last_error();
      return false;
    }
    gpu_graph_exec_t exec{};
    const gpu_error_t inst_err = gpu_graph_instantiate(&exec, graph);
    (void)gpu_graph_destroy(graph);  // exec is self-contained; the template graph is not needed
    if (inst_err != gpu_success) {
      (void)gpu_get_last_error();
      return false;
    }
    exec_ = exec;
    device_id_ = origin_dev;
    ready_ = true;
    return true;
#else
    (void)streams;
    (void)device_ids;
    (void)enqueue;
    return false;
#endif
  }

  /**
   * @brief Replay the captured graph on @p stream. Precondition: @ref ready().
   * @return @c false on launch error (sticky error cleared) or if not ready.
   */
  [[nodiscard]] auto
  launch(stream_handle stream) const -> bool {
#if CARBON_BACKEND_IS_CUDALIKE
    if (!ready_) {
      return false;
    }
    if (gpu_set_device(device_id_) != gpu_success) {
      (void)gpu_get_last_error();
      return false;
    }
    if (gpu_graph_launch(exec_, stream) != gpu_success) {
      (void)gpu_get_last_error();
      return false;
    }
    return true;
#else
    (void)stream;
    return false;
#endif
  }

  /** Destroy the executable graph (if any) and return to the empty state. */
  auto
  reset() noexcept -> void {
#if CARBON_BACKEND_IS_CUDALIKE
    if (ready_) {
      (void)gpu_graph_exec_destroy(exec_);
      exec_ = gpu_graph_exec_t{};
    }
    device_id_ = -1;
#endif
    ready_ = false;
  }

private:
#if CARBON_BACKEND_IS_CUDALIKE
  gpu_graph_exec_t exec_{};
  int device_id_{-1};
#endif
  bool ready_{false};
};

}  // namespace detail
}  // namespace carbon
