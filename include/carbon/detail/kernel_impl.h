#pragma once

/**
 * Type-erasure base for the public ::carbon::Kernel value type. Mirrors
 * Neon::set::internal::ContainerAPI: a polymorphic abstract class held through
 * std::shared_ptr by the public wrapper, with concrete subclasses (compute, reduction,
 * halo, anchor) supplying the actual @ref launch behaviour.
 *
 * The base class itself is host-only — concrete impls that need nvcc
 * (DeviceKernelImpl, ReduceKernelImpl) gate their bodies on @c __CUDACC__ in the same
 * way the previous templated public types did.
 */

#include <carbon/access_token.h>
#include <carbon/data_view.h>
#include <carbon/logging.h>
#include <carbon/system.h>

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace carbon {

/**
 * Coarse classification of a Kernel for skeleton/graph-layer use. The future skeleton
 * (when it lands) will use this to short-circuit dependency analysis (e.g. an Anchor
 * carries no data dependencies, a Halo is pure transfer, etc.) without needing RTTI on
 * the concrete @c detail::*KernelImpl subclass.
 */
enum class KernelKind : int {
  /** Device-only compute kernel over a dense grid (@c detail::DeviceKernelImpl). */
  Compute = 0,
  /** Device-only grid-wide reduction (@c detail::ReduceKernelImpl). */
  Reduction = 1,
  /** Halo exchange (@c detail::HaloKernelImpl). Device + host (intra-process std::memcpy). */
  Halo = 2,
  /**
   * Graph synchronisation placeholder (@c detail::AnchorKernelImpl); @c launch is a no-op.
   * Anchors run trivially on either side, so @c runs_on_device and @c runs_on_host are
   * both true.
   */
  Anchor = 3,
  /**
   * Host-only (CPU / OpenMP) compute kernel over a dense grid
   * (@c detail::DenseHostKernelImpl). Same loading-lambda shape and per-cell signature as
   * @ref Compute (so a @c CARBON_HOST_DEVICE lambda can be reused as-is), but the launch
   * path is an OpenMP @c parallel for over the iter extent instead of a CUDA / SYCL
   * submit. The @c Stream argument is waited-on first (so a host kernel sitting in a
   * heterogeneous std::vector<Kernel> after a device kernel sees the device's writes
   * flushed to the CPU mirror) and then ignored.
   */
  ComputeHost = 4,
  /**
   * Host-only reduction over a dense grid (@c detail::ReduceHostKernelImpl). OpenMP
   * thread-local partials combined sequentially at the end through the same @c Op the
   * device path uses, so custom reducers (not just @c reduce::Sum) work without an
   * @c omp declare reduction. Result lands directly in @c ReductionResult::host_data;
   * @c sync is a no-op, but @c host_value() can be read straight after @c launch returns.
   */
  ReductionHost = 5,
  /**
   * Dual-target compute kernel over a dense grid (@c detail::UniversalKernelImpl). Both
   * @c launch_on_device (CUDA / SYCL) and @c launch_on_host (OpenMP) are supported on
   * the same handle from a single loading-lambda definition; the default @c launch
   * dispatches to the device side. Returned by @c Dense::make_universal_kernel; the
   * explicit single-target factories @c Dense::make_kernel (device, default) and
   * @c Dense::make_host_kernel produce @ref Compute and @ref ComputeHost respectively.
   */
  Universal = 6,
  /**
   * Dual-target grid-wide reduction (@c detail::UniversalReduceKernelImpl). Same idea
   * as @ref Universal: one factory call (@c Dense::make_universal_reduce_kernel)
   * compiles both the CUB / sycl::reduction device path AND the OpenMP per-thread-
   * partials host path, and the user picks at launch via @c Kernel::launch_on_device /
   * @c Kernel::launch_on_host (or relies on the default device dispatch through
   * @c Kernel::launch).
   */
  UniversalReduction = 7,
  /** Async D→H of a @ref ReductionResult device scalar (@c detail::ScalarReadbackKernelImpl). */
  ScalarReadback = 8,
};

namespace detail {

/**
 * Abstract base behind ::carbon::Kernel. Every concrete kernel kind (compute, reduction,
 * halo, anchor) inherits from this and overrides @ref launch. The public @c Kernel
 * wrapper holds a @c std::shared_ptr<KernelImpl> and forwards @c launch through virtual
 * dispatch.
 *
 * @par Lifetime
 * Always heap-allocated under a @c shared_ptr (constructed via @c std::make_shared inside
 * the @c Kernel factories). Copy and move are deleted to make it explicit that ownership
 * goes through the @c shared_ptr; the public @c Kernel wrapper carries value semantics
 * by way of that shared pointer.
 *
 * @par Naming
 * @c name() is informational (used by logs and the future skeleton's DAG node labels).
 * Default is empty.
 */
class KernelImpl {
public:
  virtual ~KernelImpl() = default;

  KernelImpl(const KernelImpl&) = delete;
  KernelImpl(KernelImpl&&) = delete;
  auto
  operator=(const KernelImpl&) -> KernelImpl& = delete;
  auto
  operator=(KernelImpl&&) -> KernelImpl& = delete;

  /**
   * @brief Run this kernel on @p stream using the kernel's *natural* target.
   *
   * Each concrete impl picks a sensible default:
   * - @ref DeviceKernelImpl, @ref ReduceKernelImpl, @ref HaloKernelImpl: device.
   * - @ref DenseHostKernelImpl, @ref ReduceHostKernelImpl: host.
   * - @ref AnchorKernelImpl: either (it's a no-op).
   * - @ref UniversalKernelImpl, @ref UniversalReduceKernelImpl: device by default
   *   (so a Universal kernel sitting in a @c std::vector<Kernel> iterated via
   *   @c launch behaves like a device kernel; users opt into host via
   *   @ref launch_on_host).
   *
   * @c view is honoured by Compute kernels (Standard / Internal / Boundary); Reduction
   * is Standard-only in v1; Halo / Anchor ignore @c view. Concrete impls throw
   * @c std::invalid_argument or @c std::runtime_error on misuse.
   */
  virtual auto
  launch(const ::carbon::Stream& stream,
         DataView view) -> void = 0;

  /**
   * @brief Run this kernel on the device side via @p stream.
   *
   * Default implementation throws @c std::logic_error — overridden by every kind that
   * supports device execution (@ref DeviceKernelImpl, @ref ReduceKernelImpl,
   * @ref HaloKernelImpl, @ref AnchorKernelImpl, @ref UniversalKernelImpl,
   * @ref UniversalReduceKernelImpl).
   *
   * @throws std::logic_error when this kernel kind has no device launch (e.g. a host-
   *         only kernel from @c Dense::make_host_kernel).
   */
  virtual auto
  launch_on_device(const ::carbon::Stream& /*stream*/,
                   DataView /*view*/) -> void {
    log::throw_with_log<std::logic_error>(
        "KernelImpl::launch_on_device: this kernel ('{}') does not support device execution. "
        "Either call launch_on_host (for a host-only kernel) or rebuild it via "
        "Dense::make_kernel / make_universal_kernel.",
        name_);
  }

  /**
   * @brief Run this kernel on the host (CPU / OpenMP) via @p stream.
   *
   * Default implementation throws @c std::logic_error — overridden by every kind that
   * supports host execution (@ref DenseHostKernelImpl, @ref ReduceHostKernelImpl,
   * @ref AnchorKernelImpl, @ref UniversalKernelImpl, @ref UniversalReduceKernelImpl).
   *
   * @throws std::logic_error when this kernel kind has no host launch (e.g. a device-
   *         only kernel from @c Dense::make_kernel or a halo exchange).
   */
  virtual auto
  launch_on_host(const ::carbon::Stream& /*stream*/,
                 DataView /*view*/) -> void {
    log::throw_with_log<std::logic_error>(
        "KernelImpl::launch_on_host: this kernel ('{}') does not support host execution. "
        "Either call launch_on_device (for a device-only kernel) or rebuild it via "
        "Dense::make_universal_kernel / make_host_kernel.",
        name_);
  }

  /**
   * @brief Whether @ref launch_on_device is supported by this concrete impl. Default
   *        @c false — overridden by every device-capable kind. Cheap (no virtual call
   *        cost on hot paths since this is a once-per-launch query at most).
   */
  [[nodiscard]] virtual auto
  runs_on_device() const noexcept -> bool {
    return false;
  }

  /**
   * @brief Whether @ref launch_on_host is supported by this concrete impl. Default
   *        @c false — overridden by every host-capable kind.
   */
  [[nodiscard]] virtual auto
  runs_on_host() const noexcept -> bool {
    return false;
  }

  /** Coarse kind tag; see @ref KernelKind. */
  [[nodiscard]] virtual auto
  kind() const noexcept -> KernelKind = 0;

  /**
   * @brief Field accesses this kernel performs, in the order they appear in the user's
   *        loading lambda. Drives @ref Skeleton 's dependency-graph construction.
   *
   * Default implementation returns an empty vector — fine for kinds that touch no
   * fields (e.g. @ref AnchorKernelImpl ) and for impls that have not been migrated
   * to token discovery yet. Compute / Reduction / Halo overrides populate this from
   * a one-shot loading-lambda invocation at construction (see
   * @c KernelLoader::tokens ).
   *
   * Returned by @c const& so the @c Skeleton can iterate without copying; the
   * vector is owned by the impl and lives for its lifetime.
   */
  [[nodiscard]] virtual auto
  tokens() const noexcept -> const std::vector<AccessToken>& {
    static const std::vector<AccessToken> empty;
    return empty;
  }

  /** Optional human-readable name; empty by default. */
  [[nodiscard]] auto
  name() const noexcept -> std::string_view {
    return name_;
  }

protected:
  explicit KernelImpl(std::string name) noexcept
      : name_(std::move(name)) {}

  std::string name_;
};

/**
 * No-op kernel used as a graph synchronisation placeholder (begin / end markers, fence
 * boundaries, etc.). @ref launch is a no-op; only @ref name and @ref kind are
 * meaningful. The skeleton layer (when it lands) will use it as graph-anchor nodes.
 */
class AnchorKernelImpl final : public KernelImpl {
public:
  explicit AnchorKernelImpl(std::string name)
      : KernelImpl(std::move(name)) {}

  auto
  launch(const ::carbon::Stream& /*stream*/,
         DataView /*view*/) -> void override {}

  // Anchors are pure synchronisation placeholders, so they trivially support either
  // launch surface. Both methods are no-ops; the @c runs_on_* queries return @c true
  // so an Anchor sitting in a heterogeneous std::vector<Kernel> is reported as both
  // device-capable and host-capable.
  auto
  launch_on_device(const ::carbon::Stream& /*stream*/,
                   DataView /*view*/) -> void override {}
  auto
  launch_on_host(const ::carbon::Stream& /*stream*/,
                 DataView /*view*/) -> void override {}

  [[nodiscard]] auto
  runs_on_device() const noexcept -> bool override {
    return true;
  }
  [[nodiscard]] auto
  runs_on_host() const noexcept -> bool override {
    return true;
  }

  [[nodiscard]] auto
  kind() const noexcept -> KernelKind override {
    return KernelKind::Anchor;
  }
};

}  // namespace detail
}  // namespace carbon
