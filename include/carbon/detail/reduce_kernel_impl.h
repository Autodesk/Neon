#pragma once

/**
 * Grid-wide reduction concrete impl behind ::carbon::Kernel. The templated leaf
 * (parameterised on @c Grid, @c LoadingFn, @c T, @c Op) that previously lived as
 * @c carbon::ReduceKernel<Grid, LoadingFn, T, Op> is now a private
 * @c carbon::detail::ReduceKernelImpl; the public @c Kernel value type forwards
 * @c launch to it through virtual dispatch.
 *
 * Key API change vs. the old templated @c ReduceKernel: the destination
 * @c ReductionResult<T>& is captured at construction time so the polymorphic
 * @c launch(stream, view) signature stays uniform with the rest of the kernel kinds.
 *
 * @par Scope
 * - Multi-GPU and multi-rank are both supported. Stage-2 (intra-process combine on
 *   the rank-local root GPU) is shared via @c detail/dense/reduce_combine.h ;
 *   stage-3 (cross-rank @c ncclAllReduce / @c MPI_Allreduce when
 *   @c System::distributed is set) is shared via @c detail/reduce_stages.h .
 * - Any @ref DataView is accepted on @c launch_on_device . The per-launch iteration
 *   extent is read from @c span.iter_extent(view) , so @c Internal and @c Boundary
 *   passes index only their owned z-band (and an Internal+Boundary pair recombines to
 *   the Standard-view total via @c ReductionResult::host_value_combined ).
 *
 * Pipeline (per local GPU; the per-GPU loop is single-iteration on a single-GPU
 * @c System ):
 *   1. @c carbon::detail::dense::dense_grid_reduce_kernel — fused "map + block-reduce";
 *      each block produces one @c T partial via @c cub::BlockReduce . Writes into the
 *      per-GPU @c partials_[g] scratch (sized once in the ctor for the largest view).
 *   2. @c cub::DeviceReduce::Reduce — collapses the per-block partials to a single @c T
 *      written into @c dst.device_data(view) (single-GPU short-circuit) or into
 *      @c dst.gpu_partial(g) (multi-GPU; consumed by step 3's stage-2 combine).
 *   3. @c System::async_copy — copies the device scalar into @c dst.host_data(view) on
 *      the same stream so that @c dst.sync(stream); dst.host_value(view); is enough on
 *      the user side.
 *
 * Per-block scratch and CUB device-reduce temp storage are pre-allocated per local GPU
 * in the constructor, sized from the @c Standard -view iteration extent of each slab
 * (the largest of the three views — every Internal / Boundary launch uses
 * @c <= block_count_per_gpu_[g] blocks, so neither buffer ever needs to grow).
 *
 * Include from @c .cu translation units (or any TU compiled with nvcc) when the factory
 * @c carbon::Kernel::reduce / @c Dense::make_reduce_kernel is instantiated.
 */

#include <carbon/cuda_annotation.h>
#include <carbon/data_view.h>
#include <carbon/detail/allocation.h>
#include <carbon/detail/backend/select.h>
#include <carbon/detail/gpu.h>
#include <carbon/detail/kernel_impl.h>
#include <carbon/detail/launch_traits.h>
#include <carbon/detail/reduce_scalar_tokens.h>
#include <carbon/detail/reduce_stages.h>
#include <carbon/logging.h>
#include <carbon/reduce.h>
#include <carbon/reduce_dist.h>
#include <carbon/system.h>

// detail/dense/reduce_combine.h dispatches to the stage-2 helpers (one-thread combine
// kernel + fan-in cross-stream sync) for the multi-GPU path. Pulled in unconditionally
// — both backends expose the same launch_combine_partials / sync_root_after_nonroot
// surface and the call sites below switch the inner argument types on backend.
//
// All stage-1 per-family launch variation (which kernel, what block-dim, what arg
// shape) lives in @c detail::ReduceLaunchTraits<Grid> from launch_traits.h above.
#include <carbon/detail/dense/reduce_combine.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

// Pulls the cuda-like runtime types + (under a cuda-like device TU) cub/hipcub and the
// carbon::detail::gpucub namespace alias. Expands to nothing under the SYCL backend.
#include <carbon/detail/backend/cuda_like.h>

namespace carbon {

struct KernelLoader;

namespace detail {

/**
 * Dependent-false constant for static_assert inside ReduceKernelImpl::launch's non-CUDA
 * branch. Same trick as @ref device_kernel_launch_requires_cudacc_ but parameterised so
 * the assert never fires on plain header inclusion.
 */
template <typename>
inline constexpr bool reduce_kernel_launch_requires_cudacc_ = false;

/**
 * Concrete @ref KernelImpl that runs the user's per-cell reduction lambda over every
 * local GPU of @c Grid::system() , combines the per-GPU partials on the rank-local
 * root GPU (stage 2), optionally cross-rank-allreduces them (stage 3 when
 * @c System::distributed is set), and writes the resulting scalar @c T into a
 * captured @ref ReductionResult .
 *
 * The destination is captured by raw pointer (set from the @c ReductionResult& at
 * construction). Lifetime contract: the @c ReductionResult must outlive every
 * @c launch call on the owning @c Kernel .
 */
template <typename Grid, typename LoadingFn, typename T, typename Op>
class ReduceKernelImpl final : public KernelImpl {
public:
  /**
   * @brief Build the kernel and pre-allocate device scratch on every local GPU.
   *
   * @param g           Grid instance (e.g. @ref Dense ); copied into the kernel.
   * @param loading     Loading callable (Neon-style: takes @ref KernelLoader and returns
   *                    the per-cell device lambda).
   * @param dst         Destination @ref ReductionResult ; bound by reference for the
   *                    lifetime of this kernel.
   * @param op          Reduction operator instance.
   * @param shared_mem  Dynamic shared memory in bytes per launch (forwarded to the
   *                    fused map-and-block-reduce kernel).
   * @param name        Optional human-readable name.
   *
   * @throws std::invalid_argument when @c g.gpu_count() == 0 or when
   *                                @c dst.gpu_count() does not match
   *                                @c g.gpu_count() .
   * @throws std::runtime_error    on any CUDA error during scratch sizing or allocation.
   */
  template <typename L>
  ReduceKernelImpl(const Grid& g,
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
    if (reduce_options_.finalize == reduce::ReduceFinalize::BlockAtomic) {
      if (!std::is_same_v<Op, reduce::Sum<T>>) {
        log::throw_with_log<std::invalid_argument>(
            "Kernel::reduce: ReduceFinalize::BlockAtomic requires reduce::Sum<T>");
      }
      if constexpr (requires { typename Grid::DisgPolicy; }) {
        log::throw_with_log<std::invalid_argument>(
            "Kernel::reduce: ReduceFinalize::BlockAtomic is not supported on DenseDisg grids");
      }
      if constexpr (requires { typename Grid::SBlock; }) {
        log::throw_with_log<std::invalid_argument>(
            "Kernel::reduce: ReduceFinalize::BlockAtomic is not supported on Block grids");
      }
      const bool supported_atomic_t = std::is_same_v<T, float> || std::is_same_v<T, double> ||
                                      std::is_same_v<T, std::int32_t> ||
                                      std::is_same_v<T, std::int64_t>;
      if (!supported_atomic_t) {
        log::throw_with_log<std::invalid_argument>(
            "Kernel::reduce: ReduceFinalize::BlockAtomic requires T to be float, double, "
            "int32_t, or int64_t");
      }
#if CARBON_BACKEND_IS_SYCL
      log::throw_with_log<std::invalid_argument>(
          "Kernel::reduce: ReduceFinalize::BlockAtomic is supported only on the CUDA backend");
#endif
    }
    if (reduce_options_.host_copy == reduce::ReduceHostCopy::Deferred) {
#if CARBON_BACKEND_IS_SYCL
      log::throw_with_log<std::invalid_argument>(
          "Kernel::reduce: ReduceHostCopy::Deferred is supported only on the CUDA/HIP backend");
#endif
    }
    if (reduce_options_.stage1_threads != 0U || reduce_options_.stage1_threads_y != 0U) {
      if constexpr (requires { typename Grid::DisgPolicy; }) {
        log::throw_with_log<std::invalid_argument>(
            "Kernel::reduce: ReduceOptions::stage1_threads is supported only on plain Dense grids");
      } else if constexpr (requires { typename Grid::SBlock; }) {
        log::throw_with_log<std::invalid_argument>(
            "Kernel::reduce: ReduceOptions::stage1_threads is supported only on plain Dense grids");
      } else if (reduce_options_.stage1_threads_y != 0U) {
        const auto [bx, by] = reduce::normalize_dense_stage1_block_2d(
            reduce_options_.stage1_threads, reduce_options_.stage1_threads_y);
        dense_stage1_threads_ = bx;
        dense_stage1_threads_y_ = by;
      } else {
        dense_stage1_threads_ =
            reduce::normalize_dense_stage1_threads(reduce_options_.stage1_threads);
      }
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

    // The trait's stage1_unit_count returns the per-GPU "CUDA-block count"
    // the stage-1 reduce launch dispatches: Dense / DenseDisg derive it from
    // @c iter_extent.volume() / kReduceBlockThreads ; Block returns
    // @c iter_block_count(view) directly. Sized off Standard since that
    // view dominates Internal + Boundary (whose unit counts together cover
    // the same set of cells / bricks).
    using Traits = ReduceLaunchTraits<Grid>;
    for (size_t g_idx = 0; g_idx < gpu_count; ++g_idx) {
      total_cells_per_gpu_[g_idx] = Traits::iter_cell_count(grid_, g_idx, DataView::Standard);
      block_count_per_gpu_[g_idx] = Traits::stage1_unit_count(
          grid_, g_idx, DataView::Standard, dense_stage1_threads_, dense_stage1_threads_y_);
    }

#if CARBON_BACKEND_IS_CUDALIKE && CARBON_CUDALIKE_DEVICE_TU
    if (reduce_options_.finalize == reduce::ReduceFinalize::BlockAtomic) {
      discover_tokens_();  // includes scalar tokens
      return;
    }
    partials_.resize(gpu_count);
    cub_temp_storage_.resize(gpu_count);
    cub_temp_bytes_.assign(gpu_count, 0);
    for (size_t g_idx = 0; g_idx < gpu_count; ++g_idx) {
      const detail::Gpu& gpu = grid_.system().gpu_group().devices()[g_idx];
      const int64_t blocks = block_count_per_gpu_[g_idx];
      // Always allocate at least one element so partials_[g].data() is non-null and the
      // empty-domain branch in launch() can pass it as the d_in for cub::DeviceReduce
      // with num_items=0 (which writes init verbatim). Same trick the v1 code used.
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
        // CUB never returns 0 in practice but the contract allows it; allocate a single
        // byte so cub_temp_storage_.data() is non-null and the second call is unambiguous.
        temp_bytes = 1U;
      }
      cub_temp_bytes_[g_idx] = temp_bytes;
      cub_temp_storage_[g_idx] =
          gpu.template buffer<std::byte>(temp_bytes, /*relative_gpu_id=*/static_cast<int>(g_idx));
    }
    {
      const detail::Gpu& root = grid_.system().gpu_group().devices()[0U];
      merge_pair_scratch_ = root.template buffer<T>(2U, /*relative_gpu_id=*/0);
    }
#elif CARBON_BACKEND_IS_SYCL
    // SYCL: nothing to allocate up front — sycl::reduction owns its own per-work-group
    // scratch. The per-GPU total/block counts above are still computed for parity with
    // the CUDA accessors and to bound block counts inside launch_on_device().
#else
    static_assert(reduce_kernel_launch_requires_cudacc_<LoadingFn>,
                  "Kernel::reduce constructor requires a translation unit compiled with nvcc "
                  "(__CUDACC__). Move the call site to a .cu file (or build that TU with "
                  "nvcc) and include <carbon/kernel.h> from there.");
#endif

    discover_tokens_();
  }

  [[nodiscard]] auto
  kind() const noexcept -> KernelKind override {
    return KernelKind::Reduction;
  }

  [[nodiscard]] auto
  runs_on_device() const noexcept -> bool override {
    return true;
  }

  [[nodiscard]] auto
  tokens() const noexcept -> const std::vector<AccessToken>& override {
    return tokens_;
  }

  /**
   * @brief Polymorphic launch — device is the natural target for a device-only reduce
   *        kernel, so @c launch forwards to @ref launch_on_device . @c launch_on_host
   *        throws via the base class default.
   */
  auto
  launch(const ::carbon::Stream& stream,
         DataView view) -> void override {
    this->launch_on_device(stream, view);
  }

  auto
  launch_reduction_phase(const ::carbon::Stream& stream,
                         DataView view,
                         reduce::ReduceLaunchMode mode) -> void override {
    if (mode == reduce::ReduceLaunchMode::Partial) {
      launch_on_device_impl(stream, view, /*defer_host_copy=*/true);
      return;
    }
    if (mode == reduce::ReduceLaunchMode::Merge) {
      launch_merge_on_device(stream);
      return;
    }
    launch(stream, view);
  }

  /**
   * @brief Run stage 1 (fused map + block-reduce) and stage 2 (cub::DeviceReduce::Reduce)
   *        for the requested @p view on every local GPU, then async-D→H copy the
   *        per-view final into @c dst.host_data(view) . Asynchronous: the caller must
   *        @c dst.sync(stream) before reading @c dst.host_value(view) .
   *
   * @throws std::invalid_argument when @p stream 's per-GPU group size disagrees with
   *         the grid's GPU count, or when an internal block-count overflow trips.
   * @throws std::runtime_error on any kernel-launch or CUB error.
   */
  auto
  launch_on_device(const ::carbon::Stream& stream,
                   DataView view) -> void override {
    launch_on_device_impl(stream, view, /*defer_host_copy=*/false);
  }

  auto
  launch_on_device_impl(const ::carbon::Stream& stream,
                        DataView view,
                        bool defer_host_copy) -> void {
    if (stream.size() != grid_.gpu_count()) {
      log::throw_with_log<std::invalid_argument>(
          "Kernel::reduce::launch: stream-group size ({}) does not match the grid's GPU "
          "count ({})",
          stream.size(), grid_.gpu_count());
    }
    ReductionResult<T>& dst = *dst_;
    (void)dst.begin_launch_slot(view, stream);
    const size_t gpu_count = grid_.gpu_count();
    // Used only in the CUDALIKE device branch below; a host-compiled TU (or the
    // SYCL branch) leaves it unreferenced, so silence the unused-alias warning.
    using Traits [[maybe_unused]] = ReduceLaunchTraits<Grid>;

#if CARBON_BACKEND_IS_CUDALIKE && CARBON_CUDALIKE_DEVICE_TU
    const bool block_atomic = reduce_options_.finalize == reduce::ReduceFinalize::BlockAtomic;
    // Per-GPU stage-1 outcome captured inside the OpenMP region so exceptions
    // never cross the parallel boundary; the diagnostics are raised serially
    // after the fan-out. `stage` names the first failing step for the message.
    struct Stage1Outcome {
      gpu_error_t err{gpu_success};
      const char* stage{nullptr};
      int64_t sized{0};
      int64_t requested{0};
      int device_id{0};
    };
    std::vector<Stage1Outcome> outcomes(gpu_count);
    // Fan the per-GPU stage-1 launches (map + block-reduce + DeviceReduce) out
    // across one host thread per GPU — same rationale as the compute path:
    // each iteration pins its device and issues async work, so a serial loop
    // staircases the GPU start times. Single-GPU stays on the serial path.
#pragma omp parallel for num_threads(static_cast<int>(gpu_count)) if (gpu_count > 1)
    for (size_t g_idx = 0; g_idx < gpu_count; ++g_idx) {
      const auto span = grid_.spans()[g_idx];
      const int64_t blocks = Traits::stage1_unit_count(grid_, g_idx, view, dense_stage1_threads_,
                                                       dense_stage1_threads_y_);
      const int64_t iter_cells = Traits::iter_cell_count(grid_, g_idx, view);
      if (blocks > block_count_per_gpu_[g_idx]) {
        // Sanity check: Standard-view sizing in the ctor must dominate every other view's
        // launch; if this trips, the trait's @c stage1_unit_count for this view exceeds
        // Standard, which would invalidate the per-GPU partial scratch shape.
        outcomes[g_idx] = {gpu_success, "sizing", block_count_per_gpu_[g_idx], blocks, 0};
        continue;
      }

      // Single-GPU short-circuit: stage 2 writes the per-GPU result straight into the
      // destination's per-view final slot, since there are no other GPUs to combine
      // with. Multi-GPU writes into dst.gpu_partial(g_idx) and then the stage-2
      // combine kernel folds those into device_view_[view] on the root GPU.
      T* const stage2_out = gpu_count == 1U ? dst.device_data(view) : dst.gpu_partial(g_idx).data();

      const gpu_stream_t cuda_stream = stream[g_idx].get();

      // Pin the current CUDA device to this slab's GPU before any direct CUDA / CUB
      // call. typed_launch (used by stage 1 below) sets the device internally, but the
      // empty-domain shortcut and the stage-1.5 cub::DeviceReduce::Reduce after stage 1
      // both go through the runtime API directly and would land on whatever device a
      // prior iteration / launch left current — fine on single-GPU, fatal cross-GPU
      // (cudaErrorInvalidResourceHandle on streams from other devices). Under the
      // OpenMP fan-out each host thread owns one GPU, so the per-thread current
      // device set here is not clobbered by a sibling iteration.
      const detail::Gpu& gpu = grid_.system().gpu_group().devices()[g_idx];
      const gpu_error_t set_err = gpu_set_device(gpu.device_id());
      if (set_err != gpu_success) {
        outcomes[g_idx] = {set_err, "gpu_set_device", 0, 0, gpu.device_id()};
        continue;
      }

      const detail::Allocation<T>& stage2_alloc =
          gpu_count == 1U ? dst.device_allocation(view) : dst.gpu_partial(g_idx);

      if (Traits::is_empty_domain(grid_, g_idx, view)) {
        if (block_atomic) {
          try {
            grid_.system().async_memset(stage2_alloc, 0, sizeof(T), stream);
          } catch (const std::exception&) {
            outcomes[g_idx] = {gpu_success, "async_memset", 0, 0, 0};
          }
        } else {
          // Empty domain: write the identity into stage2_out. DeviceReduce::Reduce
          // with num_items=0 writes init verbatim.
          size_t temp_bytes = cub_temp_bytes_[g_idx];
          const gpu_error_t err = gpucub::DeviceReduce::Reduce(
              cub_temp_storage_[g_idx].data(), temp_bytes, partials_[g_idx].data(), stage2_out,
              /*num_items=*/0, op_, op_.identity(), cuda_stream);
          if (err != gpu_success) {
            outcomes[g_idx] = {err, "empty-domain DeviceReduce::Reduce", 0, 0, 0};
          }
        }
        continue;
      }

      // Stage 1: fused map + block-reduce. Trait absorbs the kernel-choice variation
      // (dense / disg / block) and the arg-shape variation (block has no iter_cells).
      KernelLoader ctx{static_cast<unsigned>(g_idx), KernelTarget::Device,
                       KernelLoaderPhase::Launch};
      auto device_fn = loading_fn_(ctx);
      using F = decltype(device_fn);
      Traits::template assert_reduce_fn<F, T, Op>();

      if (block_atomic) {
        try {
          grid_.system().async_memset(stage2_alloc, 0, sizeof(T), stream);
        } catch (const std::exception&) {
          outcomes[g_idx] = {gpu_success, "async_memset", 0, 0, 0};
          continue;
        }
        const gpu_error_t launch_err = Traits::template cuda_stage1_block_atomic<F, T, Op>(
            gpu, blocks, shared_mem_, cuda_stream, span, view, device_fn, op_, op_.identity(),
            stage2_out, iter_cells, dense_stage1_threads_, dense_stage1_threads_y_);
        if (launch_err != gpu_success) {
          outcomes[g_idx] = {launch_err, "stage-1 block-atomic reduce launch", 0, 0, 0};
        }
        continue;
      }

      const gpu_error_t launch_err = Traits::template cuda_stage1<F, T, Op>(
          gpu, blocks, shared_mem_, cuda_stream, span, view, device_fn, op_, op_.identity(),
          partials_[g_idx].data(), iter_cells, dense_stage1_threads_, dense_stage1_threads_y_);
      if (launch_err != gpu_success) {
        outcomes[g_idx] = {launch_err, "stage-1 grid_reduce_kernel launch", 0, 0, 0};
        continue;
      }

      // Stage 1.5: DeviceReduce over the per-block partials into stage2_out.
      size_t temp_bytes = cub_temp_bytes_[g_idx];
      const gpu_error_t reduce_err = gpucub::DeviceReduce::Reduce(
          cub_temp_storage_[g_idx].data(), temp_bytes, partials_[g_idx].data(), stage2_out,
          static_cast<int>(blocks), op_, op_.identity(), cuda_stream);
      if (reduce_err != gpu_success) {
        outcomes[g_idx] = {reduce_err, "stage-1.5 DeviceReduce::Reduce", 0, 0, 0};
      }
    }

    for (size_t g_idx = 0; g_idx < gpu_count; ++g_idx) {
      const Stage1Outcome& o = outcomes[g_idx];
      if (o.stage == nullptr) {
        continue;
      }
      using namespace std::string_view_literals;
      if (o.stage == "sizing"sv) {
        log::throw_with_log<std::runtime_error>(
            "Kernel::reduce::launch: view {} requires {} stage-1 units on local GPU {} but the "
            "ctor only sized {} (ReduceLaunchTraits::stage1_unit_count for this view exceeds "
            "Standard?)",
            ::carbon::to_string(view), o.requested, g_idx, o.sized);
      }
      if (o.stage == "gpu_set_device"sv) {
        log::throw_with_log<std::runtime_error>(
            "Kernel::reduce::launch: gpu_set_device({}) on local GPU {} failed: {} (gpu error "
            "{})",
            o.device_id, g_idx, gpu_error_string(o.err), static_cast<int>(o.err));
      }
      if (o.stage == "async_memset"sv) {
        log::throw_with_log<std::runtime_error>(
            "Kernel::reduce::launch: System::async_memset on local GPU {} failed", g_idx);
      }
      log::throw_with_log<std::runtime_error>(
          "Kernel::reduce::launch: {} on local GPU {} failed: {} (gpu error {})", o.stage, g_idx,
          gpu_error_string(o.err), static_cast<int>(o.err));
    }

    if (gpu_count > 1U) {
      reduce_stages::run_stage2_combine<T, Op>(grid_.system(), stream, dst, view, op_);
    }
    if (!defer_host_copy) {
      reduce_stages::run_stage3_cross_rank<T, Op>(grid_.system(), stream, dst, view);
      if (reduce_options_.host_copy == reduce::ReduceHostCopy::Inline) {
        const size_t slot = dst.active_slot();
        grid_.system().async_copy(dst.device_allocation(view, slot),
                                  dst.host_allocation(view, slot), sizeof(T), stream);
        dst.record_readback_done(view, slot, stream);
        dst.mark_fresh(view);
      }
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
    if (!defer_host_copy) {
      reduce_stages::run_stage3_cross_rank<T, Op>(grid_.system(), stream, dst, view);
      if (reduce_options_.host_copy == reduce::ReduceHostCopy::Inline) {
        const size_t slot = dst.active_slot();
        grid_.system().async_copy(dst.device_allocation(view, slot),
                                  dst.host_allocation(view, slot), sizeof(T), stream);
        dst.record_readback_done(view, slot, stream);
        dst.mark_fresh(view);
      }
    }
#else
    static_assert(reduce_kernel_launch_requires_cudacc_<LoadingFn>,
                  "Kernel::reduce::launch requires a translation unit compiled with nvcc "
                  "(__CUDACC__). Move the call site to a .cu file (or build that TU with "
                  "nvcc) and include <carbon/kernel.h> from there.");
    (void)stream;
    (void)view;
    (void)defer_host_copy;
#endif
  }

  auto
  launch_merge_on_device(const ::carbon::Stream& stream) -> void {
#if CARBON_BACKEND_IS_CUDALIKE && CARBON_CUDALIKE_DEVICE_TU
    if (stream.size() != grid_.gpu_count()) {
      log::throw_with_log<std::invalid_argument>(
          "Kernel::reduce::launch_merge: stream-group size ({}) != gpu_count ({})", stream.size(),
          grid_.gpu_count());
    }
    ReductionResult<T>& dst = *dst_;
    (void)dst.begin_launch_slot(DataView::Standard, stream);
    const int root_gpu_idx = 0;
    const detail::Gpu& root_gpu =
        grid_.system().gpu_group().devices()[static_cast<size_t>(root_gpu_idx)];
    const gpu_error_t set_err = gpu_set_device(root_gpu.device_id());
    if (set_err != gpu_success) {
      log::throw_with_log<std::runtime_error>(
          "Kernel::reduce::launch_merge: gpu_set_device failed");
    }
    T* staging = dst.gpu_count() > 1U ? dst.stage2_staging().data() : merge_pair_scratch_.data();
    T* const out = dst.device_data(DataView::Standard);
    const T* const internal = dst.device_data(DataView::Internal);
    const T* const boundary = dst.device_data(DataView::Boundary);
    const gpu_stream_t cuda_stream = stream[static_cast<size_t>(root_gpu_idx)].get();
    const gpu_error_t c0 = gpu_memcpy_async_default(staging, internal, sizeof(T), cuda_stream);
    const gpu_error_t c1 = gpu_memcpy_async_default(staging + 1, boundary, sizeof(T), cuda_stream);
    if (c0 != gpu_success || c1 != gpu_success) {
      log::throw_with_log<std::runtime_error>("Kernel::reduce::launch_merge: D2D copy failed");
    }
    dense::launch_combine_partials<T, Op>(root_gpu.device_id(), cuda_stream, staging, out, 2, op_);
    reduce_stages::run_stage3_cross_rank<T, Op>(grid_.system(), stream, dst, DataView::Standard);
    if (reduce_options_.host_copy == reduce::ReduceHostCopy::Inline) {
      const size_t slot = dst.active_slot();
      grid_.system().async_copy(dst.device_allocation(DataView::Standard, slot),
                                dst.host_allocation(DataView::Standard, slot), sizeof(T), stream);
      dst.record_readback_done(DataView::Standard, slot, stream);
      dst.mark_fresh(DataView::Standard);
    }
#else
    log::throw_with_log<std::logic_error>(
        "Kernel::reduce::launch_merge requires a CUDA/HIP device translation unit");
#endif
  }

  /**
   * Owned-cell count summed across local GPUs for the @c Standard view; the value the
   * existing tests assert on for single-GPU configurations. (For step-2 single-GPU this
   * is identical to the per-GPU count; the sum-form becomes meaningful in step 3 when
   * the gpu_count gate is lifted.)
   */
  [[nodiscard]] auto
  total_cells() const noexcept -> int64_t {
    int64_t sum = 0;
    for (const int64_t c : total_cells_per_gpu_) {
      sum += c;
    }
    return sum;
  }

  /**
   * Number of CUDA blocks launched by stage 1 for the @c Standard view, summed across
   * local GPUs (matches the legacy single-GPU contract — see @ref total_cells ).
   */
  [[nodiscard]] auto
  block_count() const noexcept -> int64_t {
    int64_t sum = 0;
    for (const int64_t b : block_count_per_gpu_) {
      sum += b;
    }
    return sum;
  }

private:
  // Stage 2 (multi-GPU intra-process combine) and stage 3 (cross-rank NCCL allreduce)
  // are grid-agnostic; this impl calls detail::reduce_stages::run_stage2_combine and
  // run_stage3_cross_rank directly from the launch body. The dense and block reducers
  // share the same two helpers.

  Grid grid_{};
  /**
   * @brief One-shot token discovery — twin of @ref DeviceKernelImpl::discover_tokens_ .
   *        A reduction kernel's loading lambda typically only reads its inputs, so
   *        users will often refine the recorded mode to @c AccessMode::Read by
   *        calling @c loader.read(field) — the conservative default ( @c ReadWrite )
   *        is still safe for the dependency analysis but blocks read-read
   *        concurrency the explicit annotation would expose.
   */
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

  LoadingFn loading_fn_{};
  Op op_{};
  ReductionResult<T>* dst_{nullptr};
  size_t shared_mem_{};
  std::vector<int64_t> total_cells_per_gpu_{};
  std::vector<int64_t> block_count_per_gpu_{};
  std::vector<AccessToken> tokens_{};
  reduce::ReduceOptions reduce_options_{};
  /** Plain @c Dense CUDA stage-1 block width; @c 0 → default @c 256 (1D) or @c 32 (2D). */
  int dense_stage1_threads_{0};
  /** @c 0 = 1D flat tiling; @c >0 = 2D block height ( @c z axis ). */
  int dense_stage1_threads_y_{0};
#if CARBON_BACKEND_IS_CUDALIKE && CARBON_CUDALIKE_DEVICE_TU
  std::vector<detail::Allocation<T>> partials_{};
  detail::Allocation<T> merge_pair_scratch_{};
  std::vector<size_t> cub_temp_bytes_{};
  std::vector<detail::Allocation<std::byte>> cub_temp_storage_{};
#endif
};

}  // namespace detail
}  // namespace carbon
