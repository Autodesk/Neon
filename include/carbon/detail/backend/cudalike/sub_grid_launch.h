#pragma once

/**
 * CUDA / HIP launch for @c carbon::SubGrid compute kernels: a 1D grid over the lanes of one
 * @ref DataView . Include only from device translation units; @c SubGrid 's kernel impl is
 * the only caller.
 */

#include <carbon/data_view.h>
#include <carbon/detail/backend/select.h>
#include <carbon/detail/gpu.h>

#include <cstddef>
#include <cstdint>

#if CARBON_CUDALIKE_DEVICE_TU
#include <carbon/detail/backend/cuda_like.h>
#endif

namespace carbon::detail::sub_grid {

/** Threads per block for SubGrid launches (lanes are independent, so any warp multiple works). */
inline constexpr unsigned kLaneBlockThreads = 256U;

#if CARBON_CUDALIKE_DEVICE_TU

/**
 * One thread per lane of @c [lane_begin, lane_begin + lane_count) . Builds the SubGrid @c Idx
 * from the lane table, so the parent cell and the lane always agree.
 */
template <typename SpanT,
          typename IdxT,
          typename F>
__global__ void
sub_grid_kernel(SpanT span,
                int64_t lane_begin,
                int64_t lane_count,
                F fn) {
  const int64_t k = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
                    static_cast<int64_t>(threadIdx.x);
  if (k >= lane_count) {
    return;
  }
  const int64_t lane = lane_begin + k;
  fn(span, IdxT(span.sub().parent_idx(lane), lane));
}

/**
 * Launch @p fn over the lanes of @p view on @p gpu . An empty lane range is a no-op (CUDA
 * rejects a zero-sized grid).
 */
template <typename IdxT,
          typename SpanT,
          typename F>
[[nodiscard]] inline auto
launch_lanes(const ::carbon::detail::Gpu& gpu,
             const SpanT& span,
             ::carbon::DataView view,
             size_t shared_mem,
             gpu_stream_t stream,
             F fn) -> gpu_error_t {
  const int64_t count = span.sub().lane_count(view);
  if (count <= 0) {
    return gpu_success;
  }
  const int64_t begin = span.sub().lane_offset(view);
  const auto blocks = static_cast<unsigned>((count + kLaneBlockThreads - 1) / kLaneBlockThreads);
  return gpu.template typed_launch<&sub_grid_kernel<SpanT, IdxT, F>>(
      dim3(blocks), dim3(kLaneBlockThreads), shared_mem, stream, span, begin, count, fn);
}

#endif

}  // namespace carbon::detail::sub_grid
