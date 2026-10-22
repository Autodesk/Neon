#pragma once

/**
 * SYCL twin of @c detail/backend/cudalike/reduce_combine.h . Provides the stage-2
 * intra-process combine helpers used by the multi-GPU reduce path:
 *
 * - @ref launch_combine_partials — submits a one-thread @c sycl::single_task that walks
 *   the rank-local root GPU's stage-2 staging buffer left-to-right and folds it through
 *   @c Op into the per-view final scalar. Empty-fold (n == 0) is a no-op (matches the
 *   CUDA backend's degenerate path; the caller already wrote the identity into the
 *   per-view final via @c queue.fill ).
 *
 * - @ref sync_root_after_nonroot — fan-in cross-stream ordering. Records a barrier
 *   @c sycl::event on each non-root per-GPU queue (via
 *   @c AdaptiveCpp_enqueue_custom_operation that captures the queue's frontier) and
 *   declares it as a dependency on a no-op submission to the root queue. AdaptiveCpp's
 *   in-order queues turn the dependency into a barrier on the receiving queue, so the
 *   subsequent peer copies + combine launch on the root queue are ordered after every
 *   other per-GPU stream's stage-2 reduction.
 */

#include <carbon/detail/backend/sycl/stream.h>
#include <carbon/logging.h>
#include <carbon/system.h>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <vector>

namespace carbon::detail::dense {

template <typename T,
          typename Op>
inline auto
launch_combine_partials(sycl::queue& queue,
                        const T* src,
                        T* dst,
                        int n,
                        Op op) -> void {
  if (n <= 0) {
    return;
  }
  // Capture the identity by value so the device single_task does not have to call
  // op.identity() — for built-in operators like @c reduce::Min / @c reduce::Max
  // identity() lifts to @c std::numeric_limits<T>::{max,lowest} which is a host-only
  // constexpr in libstdc++. Mirrors the CUDA backend's @c init kernel-arg pattern.
  const T init = op.identity();
  try {
    queue.submit([&](sycl::handler& cgh) {
      cgh.single_task([=]() {
        T acc = init;
        for (int i = 0; i < n; ++i) {
          acc = op(acc, src[i]);
        }
        *dst = acc;
      });
    });
  } catch (const sycl::exception& ex) {
    ::carbon::log::throw_with_log<std::runtime_error>(
        "launch_combine_partials: SYCL submission failed: {}", ex.what());
  }
}

inline auto
sync_root_after_nonroot([[maybe_unused]] const ::carbon::System& sys,
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
  std::exception_ptr first_err{};
  std::vector<sycl::event> events;
  events.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    if (i == root) {
      events.emplace_back();
      continue;
    }
    try {
      sycl::event ev =
          streams[i].queue().AdaptiveCpp_enqueue_custom_operation([](sycl::interop_handle&) {
          });
      events.push_back(std::move(ev));
    } catch (...) {
      if (!first_err) {
        first_err = std::current_exception();
      }
      events.emplace_back();
    }
  }

  for (size_t i = 0; i < n; ++i) {
    if (i == root) {
      continue;
    }
    try {
      const sycl::event dep = events[i];
      (void)streams[root].queue().AdaptiveCpp_enqueue_custom_operation(
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
      ::carbon::log::throw_with_log<std::runtime_error>("sync_root_after_nonroot: SYCL: {}",
                                                        ex.what());
    } catch (...) {
      ::carbon::log::throw_with_log<std::runtime_error>(
          "sync_root_after_nonroot: SYCL queue threw a non-std::exception");
    }
  }
}

}  // namespace carbon::detail::dense
