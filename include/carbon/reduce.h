#pragma once

/**
 * Public surface for grid-wide reductions: associative-commutative operators (with identity)
 * and the @ref ReductionResult handle that holds the post-reduction scalar on both the
 * device and the host.
 *
 * @ref ReductionResult is carbon's "PatternScalar" analogue: a per-@ref DataView, per-GPU
 * sink that is build-once / launch-many. It owns the device + host scratch the three-stage
 * reduction pipeline (per-GPU stage-1, intra-process combine, optional cross-rank
 * @c ncclAllReduce) writes into.
 *
 * Storage layout (one @c T scalar each, except the per-GPU partial vector):
 *
 * - One @c T per local GPU as the @b stage-1 output ( @ref gpu_partial ).
 * - One @c T per @ref DataView slot, on the rank-local root GPU, as the @b stage-2/3
 *   final ( @ref device_data ).
 * - One @c T per @ref DataView slot on the host, mirroring the device view ( @ref host_data ).
 * - One @c T-per-local-GPU @b stage-2 staging buffer on the rank-local root GPU; only
 *   allocated when @c gpu_count > 1 (the combine kernel reads it back into the per-view
 *   final slot).
 *
 * The no-arg accessors ( @c device_data() , @c host_data() , @c sync() , ...) all default
 * to @ref DataView::Standard , so call sites that only ever launch a Standard-view
 * reduction (the common case for plain @c Kernel::reduce ) never need to thread a
 * @c DataView argument through. The per-view slots themselves are always allocated
 * regardless of @c gpu_count or rank topology.
 *
 * Pipeline stages (executed by the @c carbon::Kernel returned by @c Kernel::reduce /
 * @c Dense::make_reduce_kernel — see @c carbon/kernel.h ):
 *
 * 1. @b Stage-1 (per-GPU map+reduce). Each local GPU runs the user's per-cell device
 *    lambda over @c span.iter_extent(view) and collapses to one @c T into
 *    @c gpu_partial(g) . The loading lambda may also perform stencil reads and field
 *    writes in the same launch (fused map/stencil + reduce). Backed by NVIDIA CUB
 *    (CUDA) or @c sycl::reduction (SYCL). With @ref ReduceFinalize::BlockAtomic , each
 *    CUDA block @c atomicAdd s into the per-GPU scalar and stage-1.5 @c DeviceReduce is
 *    skipped (floating-point sums may differ run-to-run vs the default path).
 * 2. @b Stage-2 (intra-process combine). Per-GPU partials are peer-copied into the
 *    rank-local root GPU's stage-2 staging buffer and combined via the user's @c Op into
 *    @c device_data(view) . Skipped when @c gpu_count == 1 (the per-GPU partial is
 *    transferred straight into the per-view final).
 * 3. @b Stage-3 (cross-rank allreduce). When @c System::distributed() is set, an in-place
 *    @c ncclAllReduce on @c device_data(view) replicates the result on every rank's root
 *    GPU. Gated on @c CARBON_HAVE_MPI_NCCL ; mapping the user's @c Op to the matching
 *    @c ncclRedOp_t is via @ref reduce::nccl_op (added in step 5; carbon today restricts
 *    the cross-rank case to operators that have a specialisation).
 * 4. @b D→H copy of @c device_data(view) into @c host_data(view) , so that
 *    @c result.sync(stream); result.host_value(view); is enough on the user side.
 *
 * Header-only (host code; no @c __device__ symbols here) so application TUs that only
 * declare @ref ReductionResult or pass a built-in @c reduce::Sum<T> do not need nvcc.
 */

#include <carbon/cpu.h>
#include <carbon/cuda_annotation.h>
#include <carbon/data_view.h>
#include <carbon/detail/allocation.h>
#include <carbon/detail/backend/select.h>
#if CARBON_BACKEND_IS_CUDALIKE
#include <carbon/detail/backend/cuda_like.h>
#elif CARBON_BACKEND_IS_SYCL
#include <sycl/sycl.hpp>
#endif
#include <carbon/detail/event.h>
#include <carbon/detail/event_group.h>
#include <carbon/detail/gpu.h>
#include <carbon/detail/gpu_group.h>
#include <carbon/detail/stream.h>
#include <carbon/logging.h>
#include <carbon/system.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace carbon {

namespace reduce {

/**
 * Sum reduction operator. Identity is the value-initialised @c T (zero for arithmetic
 * types). Operator is plain @c +. Trivially copyable so it can be passed by value into
 * @c __global__ kernels and into @c cub::DeviceReduce::Reduce; @c CARBON_HD on every
 * member so the same instance can be invoked from host code (cross-GPU / cross-rank
 * combine in later iterations) and device code.
 *
 * Mirrors the contract of @c cub::Sum but carries an explicit @ref identity so the same
 * type can drive both stages of the carbon pipeline (the per-block reset and the
 * @c cub::DeviceReduce initial value) without the caller having to spell the zero
 * separately.
 */
template <typename T>
struct Sum {
  [[nodiscard]] CARBON_HD constexpr auto
  identity() const noexcept -> T {
    return T{};
  }

  [[nodiscard]] CARBON_HD constexpr auto
  operator()(T a,
             T b) const noexcept -> T {
    return a + b;
  }
};

/**
 * Minimum reduction operator. Identity is @c std::numeric_limits<T>::max() (the largest
 * representable value for @p T ), chosen so that @c min(identity, x) == x for every
 * representable @p x — including negative values and floating-point @c nan / @c inf
 * (the standard @c < ordering between @c nan and @c max() is unordered, so kernels
 * using this operator on float types must avoid producing nans, same caveat as
 * @c cub::Min ).
 *
 * Mirrors the contract of @c cub::Min . Trivially copyable so it can be passed by
 * value into kernels and into @c cub::DeviceReduce::Reduce . @c operator() is
 * @c CARBON_HD because the per-thread combine in @c dense_grid_reduce_kernel calls it
 * directly on device; @c identity() is host-only because @c std::numeric_limits<T>::max()
 * is a host-only constexpr in libstdc++ and the carbon reduce pipeline only ever
 * invokes it from the host (the resulting @c T is then passed by value as the @c init
 * kernel argument).
 */
template <typename T>
struct Min {
  [[nodiscard]] constexpr auto
  identity() const noexcept -> T {
    return std::numeric_limits<T>::max();
  }

  [[nodiscard]] CARBON_HD constexpr auto
  operator()(T a,
             T b) const noexcept -> T {
    return (b < a) ? b : a;
  }
};

/**
 * Maximum reduction operator. Identity is @c std::numeric_limits<T>::lowest() (the
 * algebraically smallest representable value, including the negative end of float
 * types — @b not @c min() , which would return the smallest positive value for
 * floats and break the @c max(identity, x) == x invariant on negatives).
 *
 * Mirrors the contract of @c cub::Max . Same host-only @c identity() rationale as
 * @ref Min ; @c operator() is @c CARBON_HD .
 */
template <typename T>
struct Max {
  [[nodiscard]] constexpr auto
  identity() const noexcept -> T {
    return std::numeric_limits<T>::lowest();
  }

  [[nodiscard]] CARBON_HD constexpr auto
  operator()(T a,
             T b) const noexcept -> T {
    return (a < b) ? b : a;
  }
};

/**
 * Product reduction operator. Identity is @c T{1} (the multiplicative neutral element
 * for arithmetic types). Beware overflow on signed integer products over a sizeable
 * grid — the same caveat applies as to any product reduction; use floating-point
 * @p T or a saturating user @c Op when the value range warrants it.
 */
template <typename T>
struct Prod {
  [[nodiscard]] CARBON_HD constexpr auto
  identity() const noexcept -> T {
    return T{1};
  }

  [[nodiscard]] CARBON_HD constexpr auto
  operator()(T a,
             T b) const noexcept -> T {
    return a * b;
  }
};

/**
 * How stage-1 block partials are folded into the per-GPU scalar written before
 * stage-2 / D→H copy.
 */
enum class ReduceFinalize : unsigned char {
  /** Block partials array + @c cub::DeviceReduce::Reduce (default). */
  DeviceReduce = 0,
  /**
   * Each CUDA block @c atomicAdd s its block sum into the per-GPU output scalar
   * (NVIDIA @c jacobi_kernel style). Skips @c DeviceReduce on the partials buffer.
   * Only supported with @ref Sum and @c T in @c {float, double, int32_t, int64_t} on
   * plain @c Dense CUDA grids. Floating-point sums are not bitwise-deterministic across
   * launches (atomic add order varies).
   */
  BlockAtomic = 1,
};

/**
 * Where the async D→H copy of the final scalar runs relative to the reduce launch.
 */
enum class ReduceHostCopy : unsigned char {
  /** D→H in the same reduce launch (default; prior behaviour). */
  Inline = 0,
  /** Reduce ends on the device scalar; host copy is a separate @c ScalarReadback kernel. */
  Deferred = 1,
};

/** Skeleton OCC split: full launch, view partial (no D→H), or merge Internal+Boundary. */
enum class ReduceLaunchMode : unsigned char {
  Full = 0,
  Partial = 1,
  Merge = 2,
};

/**
 * Optional knobs on @c Kernel::reduce / @c Dense::make_reduce_kernel .
 */
struct ReduceOptions {
  ReduceFinalize finalize = ReduceFinalize::DeviceReduce;
  ReduceHostCopy host_copy = ReduceHostCopy::Inline;
  /**
   * CUDA @c Dense stage-1 threads per block. @c 0 selects the library default (@c 256 for
   * 1D flat tiling, or @c 32 when @c stage1_threads_y is set for 2D grid tiling).
   * Supported 1D values: @c 32 … @c 1024 (powers of two). With @c stage1_threads_y @c > 0,
   * @c stage1_threads is the X dimension and @c stage1_threads_y is Y (NVIDIA Jacobi-style
   * @c 32x32). Ignored on SYCL; unsupported on @c DenseDisg and @c Block grids.
   */
  unsigned stage1_threads = 0U;
  /** @c 0 = 1D flat reduce; @c >0 = 2D CUDA block (@c stage1_threads × this). */
  unsigned stage1_threads_y = 0U;
};

[[nodiscard]] inline auto
is_supported_dense_reduce_thread_count(unsigned n) -> bool {
  switch (n) {
    case 32U:
    case 64U:
    case 128U:
    case 256U:
    case 512U:
    case 1024U:
      return true;
    default:
      return false;
  }
}

/** Validates @p requested for @c ReduceOptions::stage1_threads ( @c 0 → default 256 ). */
[[nodiscard]] inline auto
normalize_dense_stage1_threads(unsigned requested) -> int {
  if (requested == 0U) {
    return 256;
  }
  if (!is_supported_dense_reduce_thread_count(requested)) {
    throw std::invalid_argument(
        "reduce::normalize_dense_stage1_threads: unsupported stage1_threads (use 0 or a "
        "power-of-two 32–1024)");
  }
  return static_cast<int>(requested);
}

/** Validates a 2D block (@c bx × @c by , product ≤ 1024). */
[[nodiscard]] inline auto
normalize_dense_stage1_block_2d(unsigned bx,
                                unsigned by) -> std::pair<int,
                                                          int> {
  if (bx == 0U) {
    bx = 32U;
  }
  if (!is_supported_dense_reduce_thread_count(bx) || !is_supported_dense_reduce_thread_count(by)) {
    throw std::invalid_argument(
        "reduce::normalize_dense_stage1_block_2d: bx/by must be powers-of-two in 32–1024");
  }
  if (static_cast<unsigned>(bx) * static_cast<unsigned>(by) > 1024U) {
    throw std::invalid_argument("reduce::normalize_dense_stage1_block_2d: bx*by must be <= 1024");
  }
  return {static_cast<int>(bx), static_cast<int>(by)};
}

/**
 * Slot ring size for @ref ReductionResult . Use @c slots = 2 with
 * @c ReduceHostCopy::Deferred and @ref ReductionResult::wait_value lag @c 1 for
 * pipelined host consumption (Neon @c PatternScalar style).
 */
struct ReductionResultOptions {
  size_t slots = 1U;
};

}  // namespace reduce

/**
 * @par PatternScalar hazard protocol (device + host slots)
 * Device-scalar and host-scalar resources participate in the @ref Skeleton conflict
 * scan via @ref AccessToken::ResourceKind . A deferred reduce exposes a
 * @c scalar_readback_factory so the skeleton inserts @c Kernel::scalar_readback on a
 * detached copy stream; the reduce→readback RAW edge is a cross-stream event. Before
 * reusing a ring slot, @c begin_launch_slot waits @c slotDone on the compute stream
 * (WAR across runs, including two skeletons sharing one @ref ReductionResult ). Host
 * reads use @ref ReductionResult::wait_value / @ref ReductionResult::ready , which
 * wait on @c slotDone only (not a full pipeline drain). With @c slots > 1 , the host
 * must consume values at least @c slots launches behind the writer or debug builds
 * may throw from @c record_readback_done .
 */

/**
 * Per-@ref DataView, per-GPU reduction destination. See the header-level docstring for
 * the allocation layout and the three-stage pipeline that consumes it.
 *
 * Lifetime model: build once, launch many times. A @c carbon::Kernel created via
 * @c Dense::make_reduce_kernel captures this destination by raw pointer and writes into
 * its slots on every @c launch ; the @ref ReductionResult must outlive every such
 * @c launch . Each launch overwrites the slot for the @ref DataView it was given (no
 * accumulation across launches); @ref mark_fresh records which slots are valid so
 * @ref host_value_combined can fold the right partials.
 *
 * @par Storage layout vs. kernel kinds
 * The per-view + per-GPU storage is allocated unconditionally so the same destination
 * type backs every reducer kind. @c ReduceKernelImpl (returned by @c Kernel::reduce
 * / @c Dense::make_reduce_kernel ) launches over @c DataView::Standard and uses all
 * three pipeline stages (per-GPU stage 1, intra-process combine in stage 2,
 * cross-rank @c ncclAllReduce / @c MPI_Allreduce in stage 3 when
 * @c System::distributed is set). @c UniversalReduceKernelImpl (returned by
 * @c Kernel::reduce_universal ) writes into the slot matching the @c DataView the
 * launch was given. Multi-GPU and multi-rank are both supported by both kernel
 * kinds today; the only runtime checks the ctors apply are @c gpu_count > 0 and
 * @c dst.gpu_count() @c == @c grid.gpu_count() .
 */
template <typename T>
class ReductionResult {
public:
  static constexpr size_t kViewCount = 3U;

private:
  struct SharedState {
    size_t slot_count{1U};
    /** Number of reduce launches completed (monotonic); mod @c slot_count for ring indices. */
    size_t launch_cursor{0U};
    /** Last host slot index consumed by @ref wait_value (debug WAR check). */
    size_t last_host_consumed_launch{0U};
    std::uint64_t device_uid{0U};
    std::uint64_t host_uid{0U};
    std::vector<detail::Allocation<T>> gpu_partials{};
    detail::Allocation<T> stage2_staging_{};
    std::array<std::vector<detail::Allocation<T>>, kViewCount> device_slots_{};
    std::array<std::vector<detail::Allocation<T>>, kViewCount> host_slots_{};
    std::array<std::vector<detail::Event>, kViewCount> slot_done_{};
    std::array<bool, kViewCount> fresh_{};
    int root_local_gpu{0};
    const System* sys{nullptr};
  };

public:
  /**
   * Allocate the per-GPU partials ( @c gpu_count scalars, one on each local GPU), the
   * per-view final slots ( @c 3 @c × @c slot_count scalars on the rank-local root GPU),
   * pinned host mirrors per slot, and (when @c gpu_count > 1 ) the stage-2 staging buffer.
   *
   * @throws std::runtime_error when @c sys.gpu_group() is empty.
   */
  explicit ReductionResult(const System& sys)
      : ReductionResult(sys,
                        reduce::ReductionResultOptions{}) {}

  explicit ReductionResult(const System& sys,
                           reduce::ReductionResultOptions options) {
    const auto& gpu_group = sys.gpu_group();
    const size_t gpu_count = gpu_group.size();
    if (gpu_count == 0U) {
      log::throw_with_log<std::runtime_error>(
          "ReductionResult: System has no GPUs (gpu_group().size() == 0)");
    }
    if (options.slots == 0U) {
      log::throw_with_log<std::invalid_argument>("ReductionResult: slots must be >= 1");
    }

    auto state = std::make_shared<SharedState>();
    state->slot_count = options.slots;
    state->device_uid = reinterpret_cast<std::uint64_t>(&state->device_uid);
    state->host_uid = reinterpret_cast<std::uint64_t>(&state->host_uid);

    state->gpu_partials.reserve(gpu_count);
    for (size_t g = 0; g < gpu_count; ++g) {
      const detail::Gpu& gpu = gpu_group.devices()[g];
      state->gpu_partials.push_back(
          gpu.template buffer<T>(1U, /*relative_gpu_id=*/static_cast<int>(g)));
    }

    const detail::Gpu& root = gpu_group.devices()[state->root_local_gpu];
    for (size_t v = 0; v < kViewCount; ++v) {
      state->device_slots_[v].reserve(state->slot_count);
      state->host_slots_[v].reserve(state->slot_count);
      for (size_t s = 0; s < state->slot_count; ++s) {
        state->device_slots_[v].push_back(
            root.template buffer<T>(1U, /*relative_gpu_id=*/state->root_local_gpu));
        state->host_slots_[v].push_back(Cpu::pinned_buffer<T>(1U));
        state->slot_done_[v].push_back(root.event());
      }
      state->fresh_[v] = false;
    }

    if (gpu_count > 1U) {
      state->stage2_staging_ =
          root.template buffer<T>(gpu_count, /*relative_gpu_id=*/state->root_local_gpu);
    }
    state->sys = &sys;
    state_ = std::move(state);
  }

  /** Completed reduce launches (monotonic); used by readback to pick the active slot. */
  [[nodiscard]] auto
  launch_cursor() const noexcept -> size_t {
    return state_->launch_cursor;
  }

  [[nodiscard]] auto
  system() const noexcept -> const System& {
    return *state_->sys;
  }

  // ---- Public scalar accessors ---------------------------------------------------------
  //
  // Default DataView::Standard so existing code that only ever reduces over the full
  // owned domain stays source-compatible. The allocations they return are still 1
  // element of T regardless of view.

  /** Stable uid for device-scalar dependency tokens (shared across copies of this handle). */
  [[nodiscard]] auto
  device_uid() const noexcept -> std::uint64_t {
    return state_->device_uid;
  }

  /** Stable uid for host-scalar dependency tokens. */
  [[nodiscard]] auto
  host_uid() const noexcept -> std::uint64_t {
    return state_->host_uid;
  }

  [[nodiscard]] auto
  slot_count() const noexcept -> size_t {
    return state_->slot_count;
  }

  /**
   * Ring slot index for the @em current reduce launch (set by @ref begin_launch_slot ).
   * With @c slots == 1 this is always @c 0 .
   */
  [[nodiscard]] auto
  active_slot() const noexcept -> size_t {
    if (state_->launch_cursor == 0U) {
      return 0U;
    }
    return (state_->launch_cursor - 1U) % state_->slot_count;
  }

  /** Pointer to the device scalar for @p view in the active slot (stage 2/3 target). */
  [[nodiscard]] auto
  device_data(DataView view = DataView::Standard) const noexcept -> T* {
    return state_->device_slots_[index_(view)][active_slot()].data();
  }

  /** Host mirror for @p view in @p slot (default: active slot). */
  [[nodiscard]] auto
  host_data(DataView view = DataView::Standard,
            size_t slot = static_cast<size_t>(-1)) const noexcept -> T* {
    const size_t s = (slot == static_cast<size_t>(-1)) ? active_slot() : slot;
    return state_->host_slots_[index_(view)][s].data();
  }

  /** Device allocation for @p view in the active slot. */
  [[nodiscard]] auto
  device_allocation(DataView view = DataView::Standard) const noexcept
      -> const detail::Allocation<T>& {
    return state_->device_slots_[index_(view)][active_slot()];
  }

  [[nodiscard]] auto
  device_allocation(DataView view,
                    size_t slot) const noexcept -> const detail::Allocation<T>& {
    return state_->device_slots_[index_(view)][slot];
  }

  /** Host allocation for @p view in @p slot (default: active slot). */
  [[nodiscard]] auto
  host_allocation(DataView view = DataView::Standard,
                  size_t slot = static_cast<size_t>(-1)) const noexcept
      -> const detail::Allocation<T>& {
    const size_t s = (slot == static_cast<size_t>(-1)) ? active_slot() : slot;
    return state_->host_slots_[index_(view)][s];
  }

  [[nodiscard]] auto
  slot_done_event(DataView view,
                  size_t slot) const noexcept -> const detail::Event& {
    return state_->slot_done_[index_(view)][slot];
  }

  /**
   * Read the most recently synced reduction result for @p view. Only meaningful after a
   * @c launch into that view followed by @ref sync ; calling this before either is not
   * undefined behaviour but returns whatever @c new[] gave the host slot.
   */
  [[nodiscard]] auto
  host_value(DataView view = DataView::Standard) const -> T {
    return *host_data(view);
  }

  /**
   * Block until the pinned host slot for launch @c (cursor - 1 - lag) is ready, then return
   * its value. @p lag = 0 is the most recent launch; @p lag = 1 is one launch behind
   * (pipelined convergence checks).
   */
  [[nodiscard]] auto
  wait_value(DataView view = DataView::Standard,
             size_t lag = 0U) const -> T {
    if (state_->launch_cursor == 0U) {
      log::throw_with_log<std::runtime_error>(
          "ReductionResult::wait_value: no reduction launch has started yet");
    }
    if (lag >= state_->launch_cursor) {
      log::throw_with_log<std::out_of_range>(
          "ReductionResult::wait_value: lag {} >= completed launches {}", lag,
          state_->launch_cursor);
    }
    const size_t launch_idx = state_->launch_cursor - 1U - lag;
    const size_t slot = launch_idx % state_->slot_count;
    state_->slot_done_[index_(view)][slot].synchronize();  // host wait on readback
    state_->last_host_consumed_launch = launch_idx + 1U;
    return *state_->host_slots_[index_(view)][slot].data();
  }

  /** Non-blocking: @c true when @ref wait_value would not block for the same @p lag . */
  [[nodiscard]] auto
  ready(DataView view = DataView::Standard,
        size_t lag = 0U) const -> bool {
    if (state_->launch_cursor == 0U || lag >= state_->launch_cursor) {
      return false;
    }
    const size_t launch_idx = state_->launch_cursor - 1U - lag;
    const size_t slot = launch_idx % state_->slot_count;
    return state_->slot_done_[index_(view)][slot].query();
  }

  /**
   * Combine every fresh per-view host slot through @p op and return the result. Useful
   * after launching @c Internal and @c Boundary separately (and mirroring Neon's
   * @c std == int + bd reduction split) so the user does not have to remember which
   * slots were written.
   *
   * If only one slot is fresh, returns that slot's value verbatim. If no slot is fresh,
   * returns @c op.identity() . Order of combination is Standard, Internal, Boundary; the
   * @p op is required to be associative + commutative so the order does not change the
   * result.
   */
  template <typename Op = reduce::Sum<T>>
  [[nodiscard]] auto
  host_value_combined(Op op = {}) const -> T {
    T acc = op.identity();
    for (size_t v = 0; v < kViewCount; ++v) {
      if (state_->fresh_[v]) {
        acc = op(acc, *host_data(static_cast<DataView>(v)));
      }
    }
    return acc;
  }

  // ---- Stream sync ---------------------------------------------------------------------

  /**
   * Block until every operation queued on @p stream has completed, then return. After
   * @ref sync , @ref host_value reflects the most recent reduction launch into the
   * matching slot.
   *
   * Accepts any stream-group size: in single-GPU / one-rank-per-GPU mode the group has
   * one stream; in multi-GPU mode it has one stream per local GPU and the wait drains
   * every per-GPU stream before returning.
   *
   * @throws std::runtime_error when the underlying backend sync fails.
   */
  auto
  sync(const Stream& stream) const -> void {
    stream.wait();
  }

  // ---- Slot freshness ------------------------------------------------------------------

  /** True if the slot for @p view has been written by a launch since construction. */
  [[nodiscard]] auto
  fresh(DataView view) const noexcept -> bool {
    return state_->fresh_[index_(view)];
  }

  /**
   * Mark the slot for @p view as written. Called by the reduce @c KernelImpl at the end
   * of each successful launch; user code typically does not call this directly.
   */
  auto
  mark_fresh(DataView view) noexcept -> void {
    state_->fresh_[index_(view)] = true;
  }

  /**
   * @brief Reserve the next ring slot at reduce launch entry; waits @p stream on the
   *        prior readback to that slot when reusing it.
   *
   * Called from reduce @c KernelImpl launch paths only.
   */
  auto
  begin_launch_slot(DataView view,
                    const Stream& stream) -> size_t {
    const size_t slot = state_->launch_cursor % state_->slot_count;
    if (state_->launch_cursor >= state_->slot_count) {
      wait_slot_done_on_stream_(view, slot, stream);
    }
    state_->launch_cursor += 1U;
    return slot;
  }

  auto
  record_readback_done(DataView view,
                       size_t slot,
                       const Stream& stream) -> void {
#ifndef NDEBUG
    if (state_->launch_cursor > 0U && state_->slot_count > 1U) {
      const size_t launch_idx = state_->launch_cursor - 1U;
      const size_t min_consumed =
          launch_idx + 1U >= state_->slot_count ? launch_idx + 1U - state_->slot_count : 0U;
      if (min_consumed > 0U && state_->last_host_consumed_launch < min_consumed) {
        log::throw_with_log<std::logic_error>(
            "ReductionResult: readback would overwrite pinned host slot {} before the host "
            "consumed launch {} (last consumed launch index {}); use wait_value with lag or "
            "increase ReductionResultOptions::slots",
            slot, launch_idx, state_->last_host_consumed_launch);
      }
    }
#else
    (void)view;
    (void)slot;
#endif
    record_slot_done_on_stream_(view, slot, stream);
  }

  // ---- Internal accessors for the reduce KernelImpl -----------------------------------
  //
  // These are public because the impls live in a separate translation unit (and are
  // templates that need to instantiate the body inline) but they are not part of the
  // user-facing surface. Treat as detail::-grade API.

  /**
   * Per-GPU partial scalar; each local GPU's stage-1 output lands here. Indexed by
   * @c relative_gpu_id (0..gpu_count-1).
   */
  [[nodiscard]] auto
  gpu_partial(size_t local_gpu) const -> const detail::Allocation<T>& {
    if (local_gpu >= state_->gpu_partials.size()) {
      log::throw_with_log<std::out_of_range>(
          "ReductionResult::gpu_partial: local_gpu {} >= gpu_count {}", local_gpu,
          state_->gpu_partials.size());
    }
    return state_->gpu_partials[local_gpu];
  }

  /**
   * Stage-2 staging buffer on the rank-local root GPU; size @c gpu_count . Empty (no
   * underlying allocation) when @c gpu_count == 1 because no peer copies are needed.
   */
  [[nodiscard]] auto
  stage2_staging() const noexcept -> const detail::Allocation<T>& {
    return state_->stage2_staging_;
  }

  /** Number of local GPUs this destination spans (== @c sys.gpu_group().size() ). */
  [[nodiscard]] auto
  gpu_count() const noexcept -> size_t {
    return state_->gpu_partials.size();
  }

  /** Logical root GPU for stage-2/3 writes (currently always 0). */
  [[nodiscard]] auto
  root_local_gpu() const noexcept -> int {
    return state_->root_local_gpu;
  }

  /** Device scalar for @p view at @p slot (readback source). */
  [[nodiscard]] auto
  device_data_at(DataView view,
                 size_t slot) const noexcept -> T* {
    return state_->device_slots_[index_(view)][slot].data();
  }

private:
  [[nodiscard]] static auto
  index_(DataView v) noexcept -> size_t {
    return static_cast<size_t>(v);
  }

  std::shared_ptr<SharedState> state_{};

  auto
  wait_slot_done_on_stream_(DataView view,
                            size_t slot,
                            const Stream& stream) const -> void {
    const size_t root = static_cast<size_t>(state_->root_local_gpu);
    if (root >= stream.size()) {
      log::throw_with_log<std::invalid_argument>(
          "ReductionResult::wait_slot_done: root gpu {} >= stream size {}", root, stream.size());
    }
    const detail::Event& evt = state_->slot_done_[index_(view)][slot];
#if CARBON_BACKEND_IS_CUDALIKE
    const detail::gpu_error_t err =
        detail::gpu_stream_wait_event(stream[root].get(), evt.get(), /*flags=*/0U);
    if (err != detail::gpu_success) {
      log::throw_with_log<std::runtime_error>("ReductionResult: gpu_stream_wait_event failed: {}",
                                              detail::gpu_error_string(err));
    }
#elif CARBON_BACKEND_IS_SYCL
    sycl::queue& q = stream[root].queue();
    q.submit([&](sycl::handler& h) {
      h.depends_on(evt.event());
    });
#else
    (void)view;
    (void)slot;
    (void)stream;
#endif
  }

  auto
  record_slot_done_on_stream_(DataView view,
                              size_t slot,
                              const Stream& stream) const -> void {
    const size_t root = static_cast<size_t>(state_->root_local_gpu);
    if (root >= stream.size()) {
      log::throw_with_log<std::invalid_argument>(
          "ReductionResult::record_readback_done: root gpu {} >= stream size {}", root,
          stream.size());
    }
    const detail::Event& evt = state_->slot_done_[index_(view)][slot];
#if CARBON_BACKEND_IS_CUDALIKE
    const detail::gpu_error_t err = detail::gpu_event_record(evt.get(), stream[root].get());
    if (err != detail::gpu_success) {
      log::throw_with_log<std::runtime_error>("ReductionResult: gpu_event_record failed: {}",
                                              detail::gpu_error_string(err));
    }
#elif CARBON_BACKEND_IS_SYCL
    sycl::queue& q = stream[root].queue();
    auto e = q.submit([](sycl::handler& h) {
    });
    *const_cast<detail::Event*>(&evt) = detail::Event(std::make_shared<sycl::event>(e));
#else
    (void)view;
    (void)slot;
    (void)stream;
#endif
  }
};

}  // namespace carbon
