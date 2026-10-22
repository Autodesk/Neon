#pragma once

/**
 * Stage-2 intra-process combine machinery for the multi-GPU reduce path (CUDA backend).
 *
 * Stage 2 takes one @c T per local GPU (carbon's @c ReductionResult::gpu_partial(g) ,
 * each on the matching GPU) and folds them through the user's @c Op into a single @c T
 * on the rank-local root GPU (@c ReductionResult::device_data(view) ). The fold is run
 * by a tiny one-thread @c __global__ over a contiguous staging buffer
 * ( @c ReductionResult::stage2_staging() , size @c gpu_count , on the root GPU); each
 * per-GPU partial is peer-copied into its slot before the fold launches.
 *
 * Two helpers live here:
 *
 * - @ref combine_partials_kernel — the one-thread @c __global__ that walks the staging
 *   buffer left-to-right, accumulating through @c Op . Callers always launch
 *   @c <<<1, 1>>> ; the loop is sequential because @c gpu_count is 1–8 in practice and
 *   serial @c Op is cheap relative to the peer copies that fed the buffer.
 *
 * - @ref sync_root_after_nonroot — fan-in cross-stream ordering: makes
 *   @c stream[root] wait until every other per-GPU stream has reached the same point.
 *   Carbon's public @c StreamGroup::wait_for is a per-i (zip) primitive, which doesn't
 *   model the gather pattern this stage needs, so the impl drops to the raw CUDA API
 *   inline. Uses scratch @c gpu_event_t s allocated and destroyed in place — the host
 *   side of the call is fast (microseconds) and the events do not survive the launch.
 *
 * Include from @c .cu translation units (or any TU compiled with nvcc).
 */

#include <carbon/cuda_annotation.h>
#include <carbon/detail/backend/cudalike/stream.h>
#include <carbon/logging.h>
#include <carbon/system.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#if CARBON_BACKEND_IS_CUDALIKE
#include <carbon/detail/backend/cuda_like.h>
#endif

namespace carbon::detail::dense {

#if CARBON_CUDALIKE_DEVICE_TU

/**
 * @brief Sequentially fold @p src [0..@p n -1] through @p op into @c *dst .
 *
 * Launched as @c <<<1, 1, 0, stream>>> . Only thread 0 runs the loop; every other
 * grid/block configuration would just race on @c *dst with no benefit since
 * @p n == @c gpu_count is small (1–8 in practice).
 *
 * @tparam T   Reduction value type.
 * @tparam Op  Binary reduction functor; same instance the user passed to
 *             @c make_reduce_kernel . Must be commutative-associative and provide
 *             @c identity() .
 *
 * @param src  Pointer to @p n contiguous @c T s on the same GPU as @p dst (the rank-local
 *             root GPU's stage-2 staging buffer).
 * @param dst  Pointer to one @c T on the same GPU as @p src ; receives the fold.
 * @param n    Number of partials to combine ( @c gpu_count ); must be >= 1.
 * @param op   Reduction operator (passed by value, captured into the kernel).
 */
template <typename T,
          typename Op>
__global__ void
combine_partials_kernel(const T* src,
                        T* dst,
                        int n,
                        Op op,
                        T init) {
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    T acc = init;
    for (int i = 0; i < n; ++i) {
      acc = op(acc, src[i]);
    }
    *dst = acc;
  }
}

/**
 * @brief Host-side launcher for @ref combine_partials_kernel .
 *
 * Always issues @c <<<1, 1, 0, stream>>> on @p device_id . Sets the current CUDA device
 * to @p device_id first because the @c <<<>>> launch syntax dispatches to the current
 * device, not to whatever device owns @p stream — passing a stream from a different
 * device would produce @c gpu_error_invalid_resource_handle . Returns immediately on
 * @p n == 0 (degenerate empty grid; nothing to combine). Throws @c std::runtime_error
 * on launch failure.
 */
template <typename T,
          typename Op>
inline auto
launch_combine_partials(int device_id,
                        gpu_stream_t stream,
                        const T* src,
                        T* dst,
                        int n,
                        Op op) -> void {
  if (n <= 0) {
    return;
  }
  const gpu_error_t set_err = gpu_set_device(device_id);
  if (set_err != gpu_success) {
    ::carbon::log::throw_with_log<std::runtime_error>(
        "launch_combine_partials: gpu_set_device({}) failed: {} (cudaError {})", device_id,
        gpu_error_string(set_err), static_cast<int>(set_err));
  }
  combine_partials_kernel<T, Op><<<1, 1, 0, stream>>>(src, dst, n, op, op.identity());
  const gpu_error_t err = gpu_get_last_error();
  if (err != gpu_success) {
    ::carbon::log::throw_with_log<std::runtime_error>(
        "launch_combine_partials: combine_partials_kernel launch failed on device {}: {} "
        "(cudaError {})",
        device_id, gpu_error_string(err), static_cast<int>(err));
  }
}

#endif  // CARBON_CUDALIKE_DEVICE_TU

#if CARBON_BACKEND_IS_CUDALIKE

/**
 * @brief Cross-stream fan-in: @c streams[root] waits until every other per-GPU stream
 *        in @p streams has reached the current submission point.
 *
 * For each non-root @p i , records a one-shot @c gpu_event_disable_timing event on
 * @c streams[i] then immediately calls @c gpu_stream_wait_event on @c streams[root] for
 * it. Events are destroyed before the function returns; the wait itself is on-device
 * (no host fence). Empty groups and groups of size <= 1 are no-ops.
 *
 * Carbon's public @c StreamGroup::wait_for is a per-i (zip) primitive — it pairs
 * @c streams[i] with @c events[i] — which is the natural shape for the halo exchange
 * but does not model this gather. The reduce path needs the fan-in for stage 2 and is
 * the only intra-detail caller, so the helper lives here rather than in the public
 * stream_group.h .
 *
 * @throws std::runtime_error on any underlying CUDA failure. Best-effort drain: every
 *         non-root stream is attempted before the throw so a failure on one GPU does
 *         not leave straggler events around.
 */
inline auto
sync_root_after_nonroot(const ::carbon::System& sys,
                        const ::carbon::Stream& streams,
                        size_t root) -> void {
  const size_t n = streams.size();
  if (n <= 1U) {
    return;
  }
  if (root >= n) {
    ::carbon::log::throw_with_log<std::runtime_error>(
        "sync_root_after_nonroot: root index {} out of range for {} streams", root, n);
  }
  // A CUDA event must be *created* on the same device that owns the stream it
  // is recorded on, otherwise gpu_event_record returns invalid-resource-handle.
  // The caller's current device is arbitrary here (e.g. the reduce stage-1
  // fan-out issues its per-GPU launches from an OpenMP team, so which device is
  // current on return is unspecified), so pin the device to each stream's GPU
  // before its create+record. Restore the caller's current device on exit.
  int prev_device = 0;
  const gpu_error_t get_err = gpu_get_device(&prev_device);
  std::vector<gpu_event_t> events;
  events.reserve(n);
  gpu_error_t first_err = get_err;

  for (size_t i = 0; i < n; ++i) {
    if (i == root) {
      events.push_back(nullptr);
      continue;
    }
    const gpu_error_t set_err = gpu_set_device(sys.gpu_group().devices()[i].device_id());
    if (set_err != gpu_success) {
      if (first_err == gpu_success) {
        first_err = set_err;
      }
      events.push_back(nullptr);
      continue;
    }
    gpu_event_t ev = nullptr;
    gpu_error_t err = gpu_event_create_with_flags(&ev, gpu_event_disable_timing);
    if (err != gpu_success) {
      if (first_err == gpu_success) {
        first_err = err;
      }
      events.push_back(nullptr);
      continue;
    }
    err = gpu_event_record(ev, streams[i].get());
    if (err != gpu_success && first_err == gpu_success) {
      first_err = err;
    }
    err = gpu_stream_wait_event(streams[root].get(), ev, 0);
    if (err != gpu_success && first_err == gpu_success) {
      first_err = err;
    }
    events.push_back(ev);
  }

  (void)gpu_set_device(prev_device);

  for (gpu_event_t ev : events) {
    if (ev != nullptr) {
      const gpu_error_t err = gpu_event_destroy(ev);
      if (err != gpu_success && first_err == gpu_success) {
        first_err = err;
      }
    }
  }

  if (first_err != gpu_success) {
    ::carbon::log::throw_with_log<std::runtime_error>(
        "sync_root_after_nonroot: CUDA error: {} (cudaError {})", gpu_error_string(first_err),
        static_cast<int>(first_err));
  }
}

#endif  // CARBON_BACKEND_IS_CUDALIKE

}  // namespace carbon::detail::dense
