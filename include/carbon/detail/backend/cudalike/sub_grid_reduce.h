#pragma once

/**
 * CUDA / HIP stage 1 of a reduction over a @c carbon::SubGrid : one thread per lane of a
 * @ref DataView , each block collapsed with @c gpucub::BlockReduce into a partial (or, for
 * @c ReduceFinalize::BlockAtomic , added atomically into the per-GPU scalar). Stages 2 / 3
 * are the grid-agnostic ones in @c reduce_kernel_impl.h ; @c ReduceLaunchTraits<SubGrid>
 * is the only caller.
 */

#include <carbon/detail/backend/select.h>
#include <carbon/detail/gpu.h>

#include <cstddef>
#include <cstdint>

#if CARBON_CUDALIKE_DEVICE_TU
#include <carbon/detail/backend/cuda_like.h>
#include <carbon/detail/backend/cudalike/reduce_atomic.h>
#endif

namespace carbon::detail::sub_grid {

/** Threads per block of the stage-1 reduce launch. */
inline constexpr int kReduceBlockThreads = 256;

#if CARBON_CUDALIKE_DEVICE_TU

template <typename IdxT,
          typename SpanT,
          typename F,
          typename T>
__device__ inline auto
lane_value(const SpanT& span,
           int64_t lane_begin,
           int64_t lane_count,
           F& fn,
           T init) -> T {
  const int64_t k =
      static_cast<int64_t>(blockIdx.x) * kReduceBlockThreads + static_cast<int64_t>(threadIdx.x);
  if (k >= lane_count) {
    return init;
  }
  const int64_t lane = lane_begin + k;
  return fn(span, IdxT(span.sub().parent_idx(lane), lane));
}

/** Fused map + block-reduce over @c [lane_begin, lane_begin + lane_count) ; one partial per block. */
template <typename SpanT,
          typename IdxT,
          typename F,
          typename T,
          typename Op>
__global__ void
sub_grid_reduce_kernel(SpanT span,
                       int64_t lane_begin,
                       int64_t lane_count,
                       F fn,
                       Op op,
                       T init,
                       T* partials) {
  using BlockReduceT = gpucub::BlockReduce<T, kReduceBlockThreads>;
  __shared__ typename BlockReduceT::TempStorage temp_storage;
  const T v = lane_value<IdxT>(span, lane_begin, lane_count, fn, init);
  const T block_val = BlockReduceT(temp_storage).Reduce(v, op);
  if (threadIdx.x == 0U) {
    partials[blockIdx.x] = block_val;
  }
}

/** Same, but each block sum is added into @p global_out ( @c reduce::Sum only). */
template <typename SpanT,
          typename IdxT,
          typename F,
          typename T,
          typename Op>
__global__ void
sub_grid_reduce_block_atomic_kernel(SpanT span,
                                    int64_t lane_begin,
                                    int64_t lane_count,
                                    F fn,
                                    Op op,
                                    T init,
                                    T* global_out) {
  using BlockReduceT = gpucub::BlockReduce<T, kReduceBlockThreads>;
  __shared__ typename BlockReduceT::TempStorage temp_storage;
  const T v = lane_value<IdxT>(span, lane_begin, lane_count, fn, init);
  const T block_val = BlockReduceT(temp_storage).Reduce(v, op);
  if (threadIdx.x == 0U) {
    reduce_atomic_accumulate(global_out, block_val);
  }
}

template <typename IdxT,
          bool BlockAtomic,
          typename SpanT,
          typename F,
          typename T,
          typename Op>
[[nodiscard]] inline auto
launch_lane_reduce(const ::carbon::detail::Gpu& gpu,
                   int64_t blocks,
                   size_t shared_mem,
                   gpu_stream_t stream,
                   const SpanT& span,
                   ::carbon::DataView view,
                   F& fn,
                   Op& op,
                   T init,
                   T* out) -> gpu_error_t {
  const int64_t begin = span.sub().lane_offset(view);
  const int64_t count = span.sub().lane_count(view);
  const dim3 grid(static_cast<unsigned>(blocks));
  const dim3 block(static_cast<unsigned>(kReduceBlockThreads));
  if constexpr (BlockAtomic) {
    return gpu.template typed_launch<&sub_grid_reduce_block_atomic_kernel<SpanT, IdxT, F, T, Op>>(
        grid, block, shared_mem, stream, span, begin, count, fn, op, init, out);
  } else {
    return gpu.template typed_launch<&sub_grid_reduce_kernel<SpanT, IdxT, F, T, Op>>(
        grid, block, shared_mem, stream, span, begin, count, fn, op, init, out);
  }
}

#endif

}  // namespace carbon::detail::sub_grid
