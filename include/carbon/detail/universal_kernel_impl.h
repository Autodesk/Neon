#pragma once

/**
 * Dual-target compute kernel concrete impl behind @c Dense::make_universal_kernel —
 * the kernel the user gets when they want a single handle that can launch on either
 * device (CUDA / SYCL) or host (OpenMP) from the same loading lambda.
 *
 * Carries both launch bodies in one place:
 *
 *   - @ref launch_on_device performs the same per-GPU CUDA / SYCL submit
 *     @ref carbon::detail::DeviceKernelImpl does (gated on @c __CUDACC__ for CUDA,
 *     unconditional for SYCL).
 *   - @ref launch_on_host performs the same OpenMP nested-for traversal
 *     @ref carbon::detail::DenseHostKernelImpl does, after a @c stream.wait so prior
 *     device writes flushed to the CPU mirror are visible.
 *
 * The polymorphic @ref launch defaults to @ref launch_on_device so existing call sites
 * that store a Universal kernel in @c std::vector<Kernel> and iterate
 * @c kernel.launch(stream) keep their old (device) behaviour. Users opt into the host
 * path by calling @c Kernel::launch_on_host(stream) on the same handle.
 *
 * @par Why one class and not multiple inheritance from DeviceKernelImpl + HostKernelImpl
 * The two existing impls each store the @c Grid and @c LoadingFn by value (so they own
 * the user's lambda outright), and @c LoadingFn is captured by closure in nvcc's
 * extended @c __host__ __device__ form — multiply-inheriting both impls would force the
 * lambda to be copy-constructible across both bases and would double the storage. A
 * single class keeps state slim (one @c Grid, one @c LoadingFn) and makes the dual
 * launch bodies reachable through a single vtable dispatch.
 */

#include <carbon/data_view.h>
#include <carbon/detail/backend/select.h>
#include <carbon/detail/dense/kernel.h>
#include <carbon/detail/dense/span.h>
#include <carbon/detail/kernel_impl.h>
#include <carbon/detail/launch_traits.h>
#include <carbon/integer3d.h>
#include <carbon/launch_shape.h>
#include <carbon/logging.h>
#include <carbon/system.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <carbon/detail/backend/cuda_like.h>

namespace carbon {

struct KernelLoader;

namespace detail {

template <typename>
inline constexpr bool universal_kernel_launch_requires_cudacc_ = false;

/**
 * Concrete @ref KernelImpl that supports both a device launch and a host launch from
 * a single user loading lambda.
 *
 * @tparam Grid       Layout type stored by value (e.g. @ref Dense). Must satisfy the
 *                    union of @ref DeviceKernelImpl and @ref DenseHostKernelImpl
 *                    requirements (gpu_count, gpu_grid(view), gpu_block(view), spans()).
 * @tparam LoadingFn  Callable @c (KernelLoader&) -> F where @c F is invocable as
 *                    @c (Span, Idx) -> void on both targets — practically a
 *                    @c CARBON_HOST_DEVICE-annotated closure.
 */
template <typename Grid, typename LoadingFn>
class UniversalKernelImpl final : public KernelImpl {
public:
  template <typename L>
  UniversalKernelImpl(const Grid& g,
                      L&& loading,
                      size_t shared_mem,
                      std::string name)
      : KernelImpl(std::move(name)),
        grid_(g),
        loading_fn_(std::forward<L>(loading)),
        shared_mem_(shared_mem) {
    discover_tokens_();
  }

  [[nodiscard]] auto
  kind() const noexcept -> KernelKind override {
    return KernelKind::Universal;
  }

  [[nodiscard]] auto
  runs_on_device() const noexcept -> bool override {
    return true;
  }
  [[nodiscard]] auto
  runs_on_host() const noexcept -> bool override {
    return true;
  }

  [[nodiscard]] auto
  tokens() const noexcept -> const std::vector<AccessToken>& override {
    return tokens_;
  }

  /**
   * @brief Default polymorphic launch — dispatches to @ref launch_on_device so a
   *        Universal kernel sitting in @c std::vector<Kernel> behaves the same as the
   *        old @c Compute kernel did when iterated through @c launch(stream).
   */
  auto
  launch(const ::carbon::Stream& stream,
         DataView view) -> void override {
    this->launch_on_device(stream, view);
  }

  /**
   * @brief Device launch — same body as @ref carbon::detail::DeviceKernelImpl::launch.
   * @throws std::logic_error / std::runtime_error from the underlying CUDA / SYCL
   *         submit.
   * @throws via @c static_assert when instantiated outside an nvcc TU under CUDA.
   */
  auto
  launch_on_device(const ::carbon::Stream& stream,
                   DataView view) -> void override {
#if CARBON_BACKEND_IS_CUDALIKE && CARBON_CUDALIKE_DEVICE_TU
    using Traits = DeviceLaunchTraits<Grid>;
    const size_t n = Traits::preflight(grid_, view);
    if (n > 0 && stream.size() != n) {
      log::throw_with_log<std::logic_error>(
          "Kernel::launch_on_device: stream group size {} must equal gpu_count() {}", stream.size(),
          n);
    }
    for (size_t i = 0; i < n; ++i) {
      KernelLoader ctx{i, KernelTarget::Device, KernelLoaderPhase::Launch};
      auto device_fn = loading_fn_(ctx);
      const gpu_stream_t cuda_stream = stream[i].get();
      const gpu_error_t err =
          Traits::cuda_launch(grid_, i, shared_mem_, cuda_stream, view, device_fn);
      if (err != gpu_success) {
        log::throw_with_log<std::runtime_error>(
            "Kernel::launch_on_device: per-GPU launch failed for relative_gpu_id={}: {} "
            "(gpu error {})",
            i, gpu_error_string(err), static_cast<int>(err));
      }
    }
#elif CARBON_BACKEND_IS_SYCL
    using Traits = DeviceLaunchTraits<Grid>;
    const size_t n = Traits::preflight(grid_, view);
    if (n > 0 && stream.size() != n) {
      log::throw_with_log<std::logic_error>(
          "Kernel::launch_on_device: stream group size {} must equal gpu_count() {}", stream.size(),
          n);
    }
    for (size_t i = 0; i < n; ++i) {
      KernelLoader ctx{i, KernelTarget::Device, KernelLoaderPhase::Launch};
      auto device_fn = loading_fn_(ctx);
      Traits::sycl_launch(grid_, i, shared_mem_, stream[i].queue(), view, device_fn);
    }
#else
    static_assert(universal_kernel_launch_requires_cudacc_<LoadingFn>,
                  "Kernel::launch_on_device on a Universal kernel requires a translation unit "
                  "compiled with nvcc (__CUDACC__) under the CUDA backend. If you want a kernel "
                  "that only ever runs on host, switch to Dense::make_host_kernel; if you only "
                  "ever need the device path, Dense::make_kernel is the cheaper factory; "
                  "otherwise move the call site to a .cu file.");
    (void)stream;
    (void)view;
#endif
  }

  /**
   * @brief Host launch — same body as @ref carbon::detail::DenseHostKernelImpl::launch:
   *        @c stream.wait first, then a 3-deep OpenMP parallel-for over each slab's
   *        @c iter_extent(view) calling @c set_and_validate + the per-cell host
   *        callable.
   */
  auto
  launch_on_host(const ::carbon::Stream& stream,
                 DataView view) -> void override {
    try {
      stream.wait();
    } catch (const std::exception& ex) {
      log::throw_with_log<std::runtime_error>(
          "Kernel::launch_on_host: stream.wait() before host loop failed: {}", ex.what());
    }

    const size_t n = grid_.gpu_count();
    if (n == 0U) {
      return;
    }

    using HostTraits = HostLaunchTraits<Grid>;
    const auto& spans = HostTraits::spans_for_host(grid_);
    if (spans.size() != n) {
      log::throw_with_log<std::logic_error>(
          "Kernel::launch_on_host: spans size {} must equal gpu_count() {}", spans.size(), n);
    }

    for (size_t g = 0; g < n; ++g) {
      KernelLoader ctx{g, KernelTarget::Host, KernelLoaderPhase::Launch};
      auto host_fn = loading_fn_(ctx);
      HostTraits::iterate_slab(spans[g], view, host_fn);
    }
  }

  /** Grid layout (e.g. @ref Dense), copied at construction. */
  [[nodiscard]] auto
  grid() const noexcept -> const Grid& {
    return grid_;
  }

  /** User loading callable. */
  [[nodiscard]] auto
  loading_fn() const noexcept -> const LoadingFn& {
    return loading_fn_;
  }

  /** Dynamic shared memory in bytes; only honoured by the device path. */
  [[nodiscard]] auto
  shared_mem() const noexcept -> size_t {
    return shared_mem_;
  }

private:
  /** One-shot token discovery; same shape as @ref DeviceKernelImpl::discover_tokens_ . */
  auto
  discover_tokens_() -> void {
    if (grid_.gpu_count() == 0U) {
      return;
    }
    KernelLoader ctx{0U, KernelTarget::Device};
    [[maybe_unused]] auto fn = loading_fn_(ctx);
    tokens_ = ctx.tokens();
  }

  Grid grid_{};
  LoadingFn loading_fn_{};
  size_t shared_mem_{};
  std::vector<AccessToken> tokens_{};
};

}  // namespace detail
}  // namespace carbon
