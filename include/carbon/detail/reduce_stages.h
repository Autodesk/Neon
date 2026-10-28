#pragma once

/**
 * Stage-2 (intra-process combine on the rank-local root GPU) and stage-3
 * (cross-rank @c ncclAllReduce ) helpers for grid-wide reductions. Both
 * stages are grid-agnostic — they only touch @c ReductionResult per-GPU
 * scalars and the @c System 's per-stream peer-copy + NCCL handles — so the
 * dense and block-sparse reduce impls share them as free functions instead
 * of duplicating the bodies. Stage 1 (fused map + block-reduce) is the only
 * piece that branches on grid layout.
 *
 * Include from impl TUs only (pulls in CUB / SYCL via
 * @c dense/reduce_combine.h ).
 */

#include <carbon/data_view.h>
#include <carbon/detail/allocation.h>
#include <carbon/detail/backend/select.h>
#include <carbon/detail/dense/reduce_combine.h>
#include <carbon/detail/gpu.h>
#include <carbon/detail/stream_group.h>
#include <carbon/logging.h>
#include <carbon/reduce.h>
#include <carbon/system.h>

#ifdef CARBON_HAVE_MPI_NCCL
#include <carbon/reduce_dist.h>
#endif

#include <cstddef>

namespace carbon::detail::reduce_stages {

/**
 * Stage 2 (intra-process combine, multi-GPU only): peer-copy each
 * @c dst.gpu_partial(view, g) into @c dst.stage2_staging(view)[g] on the
 * rank-local root GPU, fence the root stream behind every non-root stream,
 * then fold the staging buffer into @c dst.device_data(view) via @p op on the
 * root GPU.
 *
 * Caller must have already written every @c dst.gpu_partial(view, g) (stage 1 +
 * the per-GPU final cub::DeviceReduce::Reduce / sycl::reduction).
 *
 * No-op when @c sys.gpu_count() == 1 (single-GPU short-circuits stage 1
 * directly into @c dst.device_data(view) ).
 */
template <typename T,
          typename Op>
auto
run_stage2_combine(const ::carbon::System& sys,
                   const ::carbon::Stream& stream,
                   ::carbon::ReductionResult<T>& dst,
                   ::carbon::DataView view,
                   const Op& op) -> void {
  const size_t gpu_count = sys.gpu_group().size();
  const int root = dst.root_local_gpu();

  // Fan-in: root stream waits until every other per-GPU stream has finished
  // stage 1's per-GPU final cub::DeviceReduce::Reduce write into
  // @c dst.gpu_partial(view, g) .
  ::carbon::detail::dense::sync_root_after_nonroot(sys, stream, static_cast<size_t>(root));

  // Peer-copy each per-GPU partial into stage2_staging(view)[g] on the root GPU.
  // System::async_copy picks the destination's stream slot — i.e. the root
  // GPU's stream — exactly where the combine kernel runs next.
  const ::carbon::detail::Allocation<T>& staging = dst.stage2_staging(view);
  for (size_t g = 0; g < gpu_count; ++g) {
    ::carbon::detail::Allocation<T> staging_slot = ::carbon::detail::Allocation<T>::unmanaged(
        ::carbon::detail::AllocationLocation::Cuda, staging.data() + g, /*count=*/1U,
        staging.gpu_id(), staging.relative_gpu_id());
    sys.async_copy(dst.gpu_partial(view, g), staging_slot, sizeof(T), stream);
  }

  // Combine on the root GPU. CUDA's <<<>>> launch dispatches to the current
  // device, so the launcher needs the root GPU's device id and sets it
  // before the launch.
#if CARBON_BACKEND_IS_CUDALIKE && CARBON_CUDALIKE_DEVICE_TU
  const ::carbon::detail::Gpu& root_gpu = sys.gpu_group().devices()[static_cast<size_t>(root)];
  ::carbon::detail::dense::launch_combine_partials<T, Op>(
      root_gpu.device_id(), stream[static_cast<size_t>(root)].get(), staging.data(),
      dst.device_data(view), static_cast<int>(gpu_count), op);
#elif CARBON_BACKEND_IS_SYCL
  ::carbon::detail::dense::launch_combine_partials<T, Op>(stream[static_cast<size_t>(root)].queue(),
                                                          staging.data(), dst.device_data(view),
                                                          static_cast<int>(gpu_count), op);
#endif
}

/**
 * Stage 3 (cross-rank): in-place @c ncclAllReduce on
 * @c dst.device_data(view) when the @c System carries a @c Distributed
 * handle. No-op for single-process @c System or non-MPI builds. Uses the
 * carbon @c reduce::dist_allreduce_in_place free function so the helper
 * stays NCCL / RCCL-agnostic; the trait pair (@c reduce::op_kind ,
 * @c reduce::data_kind ) gates which @c (Op, T) pairs survive compilation
 * — anything without a specialisation throws at runtime with a pointed
 * diagnostic.
 */
template <typename T,
          typename Op>
auto
run_stage3_cross_rank(const ::carbon::System& sys,
                      const ::carbon::Stream& stream,
                      ::carbon::ReductionResult<T>& dst,
                      ::carbon::DataView view) -> void {
#ifdef CARBON_HAVE_MPI_NCCL
  if (sys.distributed() == nullptr) {
    return;
  }
  if constexpr (::carbon::reduce::has_op_kind<Op> && ::carbon::reduce::has_data_kind<T>) {
    ::carbon::reduce::dist_allreduce_in_place(sys, dst.device_data(view), /*count=*/1U,
                                              ::carbon::reduce::op_kind<Op>::value,
                                              ::carbon::reduce::data_kind<T>::value, stream);
  } else {
    log::throw_with_log<std::runtime_error>(
        "Kernel::reduce: cross-rank reduction requested but no NCCL mapping is registered "
        "for this (Op, T) pair (specialise carbon::reduce::op_kind / carbon::reduce::"
        "data_kind, or run on a single-process System)");
  }
#else
  (void)sys;
  (void)stream;
  (void)dst;
  (void)view;
#endif
}

}  // namespace carbon::detail::reduce_stages
