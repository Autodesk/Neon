#pragma once

/**
 * SYCL launch for @c carbon::SubGrid compute kernels: a 1D @c nd_range over the lanes of one
 * @ref DataView . Twin of @c detail/backend/cudalike/sub_grid_launch.h .
 */

#include <carbon/data_view.h>
#include <carbon/logging.h>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace carbon::detail::sub_grid {

/** Work-items per work-group for SubGrid launches. */
inline constexpr size_t kLaneBlockThreads = 256U;

/**
 * Submit @p fn over the lanes of @p view on @p queue . An empty lane range is a no-op.
 * @throws std::runtime_error on dynamic shared memory (unsupported under SYCL, as for Dense)
 *         or a failed submission.
 */
template <typename IdxT,
          typename SpanT,
          typename F>
inline auto
launch_lanes(sycl::queue& queue,
             const SpanT& span,
             ::carbon::DataView view,
             size_t shared_mem,
             F fn) -> void {
  if (shared_mem != 0U) {
    ::carbon::log::throw_with_log<std::runtime_error>(
        "SubGrid kernel: dynamic shared memory ({} bytes) is not supported under the SYCL "
        "backend.",
        shared_mem);
  }
  const int64_t count = span.sub().lane_count(view);
  if (count <= 0) {
    return;
  }
  const int64_t begin = span.sub().lane_offset(view);
  const size_t groups = (static_cast<size_t>(count) + kLaneBlockThreads - 1U) / kLaneBlockThreads;
  const SpanT s = span;
  try {
    queue.parallel_for(sycl::nd_range<1>{groups * kLaneBlockThreads, kLaneBlockThreads},
                       [=](sycl::nd_item<1> it) {
                         const auto k = static_cast<int64_t>(it.get_global_id(0));
                         if (k >= count) {
                           return;
                         }
                         const int64_t lane = begin + k;
                         fn(s, IdxT(s.sub().parent_idx(lane), lane));
                       });
  } catch (const sycl::exception& ex) {
    ::carbon::log::throw_with_log<std::runtime_error>(
        "SubGrid kernel: SYCL parallel_for submission failed: {}", ex.what());
  }
}

}  // namespace carbon::detail::sub_grid
