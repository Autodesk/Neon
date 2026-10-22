#pragma once

/**
 * Dual-target reduction kernel concrete impl behind @c Dense::make_universal_reduce_kernel .
 * Combines the device-side pipeline of @ref carbon::detail::ReduceKernelImpl (CUB
 * block-reduce + cub::DeviceReduce + async D→H copy under CUDA, sycl::reduction under
 * SYCL) with the host-side OpenMP per-thread partials + sequential combine of
 * @ref carbon::detail::ReduceHostKernelImpl , behind a single @ref KernelImpl handle.
 *
 * The constructor allocates the device-side scratch up-front (per-block partials and
 * CUB temp storage on CUDA, one set per local GPU; nothing extra under SYCL since
 * @c sycl::reduction owns its scratch). The host path needs no preallocation. The
 * polymorphic @ref launch defaults to @ref launch_on_device so existing call sites that
 * store the kernel in @c std::vector<Kernel> keep their old (device) behaviour; users
 * opt into the host path by calling @c Kernel::launch_on_host(stream) .
 *
 * @par Scope (matches the single-target reduce impls)
 * - Multi-GPU and multi-rank are both supported on both targets. Stage-2
 *   (intra-process combine on the rank-local root GPU) runs through
 *   @c detail/dense/reduce_combine.h ; stage-3 (cross-rank @c ncclAllReduce /
 *   @c MPI_Allreduce when @c System::distributed is set) runs through
 *   @c detail/reduce_stages.h .
 * - Any @ref DataView is accepted on @c launch_on_device and @c launch_on_host . Each
 *   launch uses @c span.iter_extent(view) to size the iteration; an Internal +
 *   Boundary pair recombines to the Standard total via
 *   @c ReductionResult::host_value_combined .
 */

#include <carbon/cuda_annotation.h>
#include <carbon/data_view.h>
#include <carbon/detail/allocation.h>
#include <carbon/detail/backend/select.h>
#include <carbon/detail/dense/reduce_combine.h>
#include <carbon/detail/dense/span.h>
#include <carbon/detail/gpu.h>
#include <carbon/detail/kernel_impl.h>
#include <carbon/detail/launch_traits.h>
#include <carbon/detail/reduce_scalar_tokens.h>
#include <carbon/detail/reduce_stages.h>
#include <carbon/integer3d.h>
#include <carbon/logging.h>
#include <carbon/reduce.h>
#include <carbon/reduce_dist.h>
#include <carbon/system.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// Pulls the cuda-like runtime types + (under a cuda-like device TU) cub/hipcub and the
// carbon::detail::gpucub namespace alias. Expands to nothing under the SYCL backend.
#include <carbon/detail/backend/cuda_like.h>

namespace carbon {

struct KernelLoader;

namespace detail {

template <typename>
inline constexpr bool universal_reduce_kernel_launch_requires_cudacc_ = false;

template <typename Grid, typename LoadingFn, typename T, typename Op>
class UniversalReduceKernelImpl final : public KernelImpl {
public:
  template <typename L>
  UniversalReduceKernelImpl(const Grid& g,
                            L&& loading,
                            ReductionResult<T>& dst,
                            Op op,
                            size_t shared_mem,
                            std::string name,
                            reduce::ReduceOptions options = {})
      : KernelImpl(std::move(name)),
        grid_(g),
        loading_fn_(std::forward<L>(loading)),
        op_(std::move(op)),
        dst_(&dst),
        shared_mem_(shared_mem),
        reduce_options_(options) {
    if (reduce_options_.host_copy == reduce::ReduceHostCopy::Deferred) {
#if CARBON_BACKEND_IS_SYCL
      log::throw_with_log<std::invalid_argument>(
          "Kernel::reduce: ReduceHostCopy::Deferred is supported only on the CUDA/HIP backend");
#endif
    }
    if (grid_.gpu_count() == 0U) {
      log::throw_with_log<std::invalid_argument>(
          "Kernel::reduce: grid has no GPUs (gpu_count == 0)");
    }
    if (dst_->gpu_count() != grid_.gpu_count()) {
      log::throw_with_log<std::invalid_argument>(
          "Kernel::reduce: ReductionResult was built on a System with {} GPU(s) but the "
          "Grid has {} — the destination's per-GPU partials must match the grid's slabs",
          dst_->gpu_count(), grid_.gpu_count());
    }

    const size_t gpu_count = grid_.gpu_count();
    total_cells_per_gpu_.assign(gpu_count, 0);
    block_count_per_gpu_.assign(gpu_count, 0);

    // ReduceLaunchTraits absorbs per-family sizing (Dense / DenseDisg : cells
    // via @c iter_extent.volume() / @c kReduceBlockThreads ; Block : retained
    // bricks via @c iter_block_count(Standard) ).
    using Traits = ReduceLaunchTraits<Grid>;
    for (size_t g_idx = 0; g_idx < gpu_count; ++g_idx) {
      total_cells_per_gpu_[g_idx] = Traits::iter_cell_count(grid_, g_idx, DataView::Standard);
      block_count_per_gpu_[g_idx] = Traits::stage1_unit_count(grid_, g_idx, DataView::Standard);
    }

#if CARBON_BACKEND_IS_CUDALIKE && CARBON_CUDALIKE_DEVICE_TU
    partials_.resize(gpu_count);
    cub_temp_storage_.resize(gpu_count);
    cub_temp_bytes_.assign(gpu_count, 0);
    for (size_t g_idx = 0; g_idx < gpu_count; ++g_idx) {
      const detail::Gpu& gpu = grid_.system().gpu_group().devices()[g_idx];
      const int64_t blocks = block_count_per_gpu_[g_idx];
      const size_t partial_count = static_cast<size_t>(blocks > 0 ? blocks : 1);
      partials_[g_idx] =
          gpu.template buffer<T>(partial_count, /*relative_gpu_id=*/static_cast<int>(g_idx));

      size_t temp_bytes = 0;
      const gpu_error_t err = gpucub::DeviceReduce::Reduce(
          /*d_temp_storage=*/nullptr, temp_bytes,
          /*d_in=*/partials_[g_idx].data(),
          /*d_out=*/static_cast<T*>(nullptr),
          /*num_items=*/static_cast<int>(blocks > 0 ? blocks : 1), op_, op_.identity(),
          /*stream=*/gpu_stream_t{0});
      if (err != gpu_success) {
        log::throw_with_log<std::runtime_error>(
            "Kernel::reduce: DeviceReduce::Reduce sizing query failed on local GPU {}: "
            "{} (gpu error {})",
            g_idx, gpu_error_string(err), static_cast<int>(err));
      }
      if (temp_bytes == 0U) {
        temp_bytes = 1U;
      }
      cub_temp_bytes_[g_idx] = temp_bytes;
      cub_temp_storage_[g_idx] =
          gpu.template buffer<std::byte>(temp_bytes, /*relative_gpu_id=*/static_cast<int>(g_idx));
    }
#elif CARBON_BACKEND_IS_SYCL
    // SYCL: nothing to allocate up front — sycl::reduction owns its own per-work-group
    // scratch.
#else
    static_assert(universal_reduce_kernel_launch_requires_cudacc_<LoadingFn>,
                  "Kernel::reduce constructor requires a translation unit compiled with nvcc "
                  "(__CUDACC__) under the CUDA backend. Move the call site to a .cu file (or "
                  "build that TU with nvcc) and include <carbon/kernel.h> from there.");
#endif

    discover_tokens_();
  }

  [[nodiscard]] auto
  kind() const noexcept -> KernelKind override {
    return KernelKind::UniversalReduction;
  }

  [[nodiscard]] auto
  tokens() const noexcept -> const std::vector<AccessToken>& override {
    return tokens_;
  }

  [[nodiscard]] auto
  runs_on_device() const noexcept -> bool override {
    return true;
  }
  [[nodiscard]] auto
  runs_on_host() const noexcept -> bool override {
    return true;
  }

  /**
   * @brief Default polymorphic launch — dispatches to @ref launch_on_device so a
   *        UniversalReduction kernel sitting in a heterogeneous std::vector<Kernel>
   *        keeps its old (device) behaviour.
   */
  auto
  launch(const ::carbon::Stream& stream,
         DataView view) -> void override {
    this->launch_on_device(stream, view);
  }

  auto
  launch_on_device(const ::carbon::Stream& stream,
                   DataView view) -> void override {
    if (stream.size() != grid_.gpu_count()) {
      log::throw_with_log<std::invalid_argument>(
          "Kernel::reduce::launch_on_device: stream-group size ({}) does not match the grid's "
          "GPU count ({})",
          stream.size(), grid_.gpu_count());
    }
    ReductionResult<T>& dst = *dst_;
    (void)dst.begin_launch_slot(view, stream);
    const size_t gpu_count = grid_.gpu_count();
    // Used only in the CUDALIKE device branch below; a host-compiled TU (or the
    // SYCL branch) leaves it unreferenced, so silence the unused-alias warning.
    using Traits [[maybe_unused]] = ReduceLaunchTraits<Grid>;

#if CARBON_BACKEND_IS_CUDALIKE && CARBON_CUDALIKE_DEVICE_TU
    for (size_t g_idx = 0; g_idx < gpu_count; ++g_idx) {
      const auto span = grid_.spans()[g_idx];
      const int64_t blocks = Traits::stage1_unit_count(grid_, g_idx, view);
      const int64_t iter_cells = Traits::iter_cell_count(grid_, g_idx, view);
      if (blocks > block_count_per_gpu_[g_idx]) {
        log::throw_with_log<std::runtime_error>(
            "Kernel::reduce::launch_on_device: view {} requires {} stage-1 units on local GPU "
            "{} but the ctor only sized {} (ReduceLaunchTraits::stage1_unit_count for this view "
            "exceeds Standard?)",
            ::carbon::to_string(view), blocks, g_idx, block_count_per_gpu_[g_idx]);
      }

      T* const stage2_out = gpu_count == 1U ? dst.device_data(view) : dst.gpu_partial(g_idx).data();
      const gpu_stream_t cuda_stream = stream[g_idx].get();

      // Pin the current CUDA device to this slab's GPU before any direct CUDA / CUB
      // call. typed_launch sets the device internally, but the empty-domain shortcut
      // and the stage-1.5 cub::DeviceReduce::Reduce both go through the runtime API
      // directly and would otherwise land on whatever device a prior iteration left
      // current — fine on single-GPU, fatal cross-GPU.
      const detail::Gpu& gpu = grid_.system().gpu_group().devices()[g_idx];
      const gpu_error_t set_err = gpu_set_device(gpu.device_id());
      if (set_err != gpu_success) {
        log::throw_with_log<std::runtime_error>(
            "Kernel::reduce::launch_on_device: gpu_set_device({}) on local GPU {} failed: {} "
            "(gpu error {})",
            gpu.device_id(), g_idx, gpu_error_string(set_err), static_cast<int>(set_err));
      }

      if (Traits::is_empty_domain(grid_, g_idx, view)) {
        size_t temp_bytes = cub_temp_bytes_[g_idx];
        const gpu_error_t err = gpucub::DeviceReduce::Reduce(
            cub_temp_storage_[g_idx].data(), temp_bytes, partials_[g_idx].data(), stage2_out,
            /*num_items=*/0, op_, op_.identity(), cuda_stream);
        if (err != gpu_success) {
          log::throw_with_log<std::runtime_error>(
              "Kernel::reduce::launch_on_device: empty-domain DeviceReduce::Reduce on "
              "local GPU {} failed: {} (gpu error {})",
              g_idx, gpu_error_string(err), static_cast<int>(err));
        }
        continue;
      }

      KernelLoader ctx{static_cast<unsigned>(g_idx), KernelTarget::Device,
                       KernelLoaderPhase::Launch};
      auto device_fn = loading_fn_(ctx);
      using F = decltype(device_fn);
      Traits::template assert_reduce_fn<F, T, Op>();

      const gpu_error_t launch_err = Traits::template cuda_stage1<F, T, Op>(
          gpu, blocks, shared_mem_, cuda_stream, span, view, device_fn, op_, op_.identity(),
          partials_[g_idx].data(), iter_cells);
      if (launch_err != gpu_success) {
        log::throw_with_log<std::runtime_error>(
            "Kernel::reduce::launch_on_device: stage-1 grid_reduce_kernel launch on local GPU "
            "{} failed: {} (gpu error {})",
            g_idx, gpu_error_string(launch_err), static_cast<int>(launch_err));
      }

      size_t temp_bytes = cub_temp_bytes_[g_idx];
      const gpu_error_t reduce_err = gpucub::DeviceReduce::Reduce(
          cub_temp_storage_[g_idx].data(), temp_bytes, partials_[g_idx].data(), stage2_out,
          static_cast<int>(blocks), op_, op_.identity(), cuda_stream);
      if (reduce_err != gpu_success) {
        log::throw_with_log<std::runtime_error>(
            "Kernel::reduce::launch_on_device: stage-1.5 DeviceReduce::Reduce on local "
            "GPU {} failed: {} (gpu error {})",
            g_idx, gpu_error_string(reduce_err), static_cast<int>(reduce_err));
      }
    }

    if (gpu_count > 1U) {
      reduce_stages::run_stage2_combine<T, Op>(grid_.system(), stream, dst, view, op_);
    }
    reduce_stages::run_stage3_cross_rank<T, Op>(grid_.system(), stream, dst, view);
    if (reduce_options_.host_copy == reduce::ReduceHostCopy::Inline) {
      const size_t slot = dst.active_slot();
      grid_.system().async_copy(dst.device_allocation(view, slot), dst.host_allocation(view, slot),
                                sizeof(T), stream);
      dst.record_readback_done(view, slot, stream);
      dst.mark_fresh(view);
    }
#elif CARBON_BACKEND_IS_SYCL
    for (size_t g_idx = 0; g_idx < gpu_count; ++g_idx) {
      const auto span = grid_.spans()[g_idx];
      const int64_t iter_cells = Traits::iter_cell_count(grid_, g_idx, view);

      T* const stage2_out = gpu_count == 1U ? dst.device_data(view) : dst.gpu_partial(g_idx).data();

      sycl::queue& q = stream[g_idx].queue();
      KernelLoader ctx{static_cast<unsigned>(g_idx), KernelTarget::Device,
                       KernelLoaderPhase::Launch};
      auto device_fn = loading_fn_(ctx);
      using F = decltype(device_fn);
      Traits::template sycl_singlepass<F, T, Op>(q, span, view, device_fn, op_, op_.identity(),
                                                 stage2_out, iter_cells);
    }

    if (gpu_count > 1U) {
      reduce_stages::run_stage2_combine<T, Op>(grid_.system(), stream, dst, view, op_);
    }
    reduce_stages::run_stage3_cross_rank<T, Op>(grid_.system(), stream, dst, view);
    if (reduce_options_.host_copy == reduce::ReduceHostCopy::Inline) {
      const size_t slot = dst.active_slot();
      grid_.system().async_copy(dst.device_allocation(view, slot), dst.host_allocation(view, slot),
                                sizeof(T), stream);
      dst.record_readback_done(view, slot, stream);
      dst.mark_fresh(view);
    }
#else
    static_assert(universal_reduce_kernel_launch_requires_cudacc_<LoadingFn>,
                  "Kernel::reduce::launch_on_device requires a translation unit compiled with "
                  "nvcc (__CUDACC__) under the CUDA backend.");
    (void)stream;
    (void)view;
#endif
  }

  auto
  launch_on_host(const ::carbon::Stream& stream,
                 DataView view) -> void override {
    if (stream.size() != grid_.gpu_count()) {
      log::throw_with_log<std::invalid_argument>(
          "Kernel::reduce::launch_on_host: stream-group size ({}) does not match the grid's "
          "GPU count ({})",
          stream.size(), grid_.gpu_count());
    }
    try {
      stream.wait();
    } catch (const std::exception& ex) {
      log::throw_with_log<std::runtime_error>(
          "Kernel::reduce::launch_on_host: stream.wait() before host loop failed: {}", ex.what());
    }

    ReductionResult<T>& dst = *dst_;
    const size_t gpu_count = grid_.gpu_count();
    using HostTraits = HostLaunchTraits<Grid>;
    const auto& host_spans = HostTraits::spans_for_host(grid_);
    if (host_spans.size() != gpu_count) {
      log::throw_with_log<std::logic_error>(
          "Kernel::reduce::launch_on_host: host_spans size {} must equal gpu_count() {}",
          host_spans.size(), gpu_count);
    }

    T result = op_.identity();
    for (size_t g_idx = 0; g_idx < gpu_count; ++g_idx) {
      KernelLoader ctx{g_idx, KernelTarget::Host, KernelLoaderPhase::Launch};
      auto host_fn = loading_fn_(ctx);
      const T slab_acc =
          HostTraits::template iterate_slab_reduce<std::remove_cvref_t<decltype(host_spans[g_idx])>,
                                                   decltype(host_fn), T, Op>(host_spans[g_idx],
                                                                             view, host_fn, op_);
      result = op_(result, slab_acc);
    }

    dst.host_data(view)[0] = result;
    dst.mark_fresh(view);
  }

  /** Owned-cell count summed across local GPUs for the @c Standard view. */
  [[nodiscard]] auto
  total_cells() const noexcept -> int64_t {
    int64_t sum = 0;
    for (const int64_t c : total_cells_per_gpu_) {
      sum += c;
    }
    return sum;
  }

  /** Number of CUDA / SYCL blocks for stage 1 of the device path; 0 for an empty grid. */
  [[nodiscard]] auto
  block_count() const noexcept -> int64_t {
    int64_t sum = 0;
    for (const int64_t b : block_count_per_gpu_) {
      sum += b;
    }
    return sum;
  }

private:
  // Stage 2 / 3 (multi-GPU combine + cross-rank NCCL allreduce) live in
  // detail::reduce_stages -- shared with ReduceKernelImpl and BlockReduceKernelImpl
  // so all three reducers route through one canonical implementation.

  /** One-shot token discovery — same shape as @ref DeviceKernelImpl::discover_tokens_ . */
  auto
  discover_tokens_() -> void {
    if (grid_.gpu_count() == 0U) {
      return;
    }
    KernelLoader ctx{0U, KernelTarget::Device};
    [[maybe_unused]] auto fn = loading_fn_(ctx);
    tokens_ = ctx.tokens();
    append_reduction_scalar_tokens(tokens_, *dst_, reduce_options_);
  }

  Grid grid_{};
  LoadingFn loading_fn_{};
  Op op_{};
  ReductionResult<T>* dst_{nullptr};
  size_t shared_mem_{};
  std::vector<int64_t> total_cells_per_gpu_{};
  std::vector<int64_t> block_count_per_gpu_{};
  std::vector<AccessToken> tokens_{};
  reduce::ReduceOptions reduce_options_{};

#if CARBON_BACKEND_IS_CUDALIKE && CARBON_CUDALIKE_DEVICE_TU
  std::vector<detail::Allocation<T>> partials_{};
  std::vector<detail::Allocation<std::byte>> cub_temp_storage_{};
  std::vector<size_t> cub_temp_bytes_{};
#endif
};

}  // namespace detail
}  // namespace carbon
