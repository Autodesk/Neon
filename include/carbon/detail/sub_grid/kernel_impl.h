#pragma once

/**
 * Compute kernels over a @c carbon::SubGrid : the @c KernelImpl leaves behind
 * @c SubGrid::make_kernel (device, @ref KernelKind::Compute ) and @c SubGrid::make_host_kernel
 * (OpenMP, @ref KernelKind::ComputeHost ).
 *
 * Same loading-lambda contract as Dense kernels, with three SubGrid rules applied to every
 * loader before the user's lambda runs:
 * - the kernel-wide region is the SubGrid's mask, so parent-field accesses carry it as their
 *   footprint (the user may still narrow it with @c loader.region );
 * - the loader is bound to the SubGrid, so loading a field of another SubGrid throws at
 *   construction;
 * - launches are 1D over the lanes of the requested @ref DataView .
 *
 * Being @c Compute , a device SubGrid kernel that stencil-reads a parent field after a halo is
 * OCC-split by the @ref Skeleton like any parent kernel; the internal-first lane order makes
 * each half one contiguous lane range.
 */

#include <carbon/data_view.h>
#include <carbon/detail/backend/select.h>
#include <carbon/detail/kernel_impl.h>
#include <carbon/logging.h>
#include <carbon/system.h>

#if CARBON_BACKEND_IS_CUDALIKE
#include <carbon/detail/backend/cudalike/sub_grid_launch.h>
#elif CARBON_BACKEND_IS_SYCL
#include <carbon/detail/backend/sycl/sub_grid_launch.h>
#endif

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if CARBON_BACKEND_IS_CUDALIKE && CARBON_CUDALIKE_DEVICE_TU
#include <carbon/detail/backend/cuda_like.h>
#endif

namespace carbon {

struct KernelLoader;

namespace detail::sub_grid {

template <typename>
inline constexpr bool sub_grid_launch_requires_device_tu_ = false;

/** Applies the SubGrid loader rules, then runs the user's loading lambda. */
template <typename Grid,
          typename LoadingFn>
auto
load(const Grid& grid,
     LoadingFn& loading_fn,
     ::carbon::KernelLoader& loader) {
  loader.region(grid.mask());
  loader.bind_sub_grid(grid.uid());
  return loading_fn(loader);
}

/** Device compute kernel over a SubGrid (@ref KernelKind::Compute ). */
template <typename Grid, typename LoadingFn>
class DeviceKernelImpl final : public ::carbon::detail::KernelImpl {
public:
  template <typename L>
  DeviceKernelImpl(const Grid& grid,
                   L&& loading,
                   size_t shared_mem,
                   std::string name)
      : KernelImpl(std::move(name)),
        grid_(grid),
        loading_fn_(std::forward<L>(loading)),
        shared_mem_(shared_mem) {
    static_assert(std::is_invocable_v<LoadingFn&, ::carbon::KernelLoader&>,
                  "SubGrid::make_kernel: the loading callable must be invocable as "
                  "`loader(KernelLoader&) -> F`, where F is the per-cell "
                  "`CARBON_HD (SubGrid::Span, SubGrid::Idx)` callable.");
    discover_tokens_();
  }

  [[nodiscard]] auto
  kind() const noexcept -> KernelKind override {
    return KernelKind::Compute;
  }

  [[nodiscard]] auto
  tokens() const noexcept -> const std::vector<AccessToken>& override {
    return tokens_;
  }

  [[nodiscard]] auto
  runs_on_device() const noexcept -> bool override {
    return true;
  }

  auto
  launch(const ::carbon::Stream& stream,
         DataView view) -> void override {
    launch_on_device(stream, view);
  }

  auto
  launch_on_device(const ::carbon::Stream& stream,
                   DataView view) -> void override {
    const size_t n = grid_.gpu_count();
    if (n > 0U && stream.size() != n) {
      log::throw_with_log<std::logic_error>(
          "SubGrid kernel '{}': stream group size {} must equal gpu_count() {}", name(),
          stream.size(), n);
    }
#if CARBON_BACKEND_IS_CUDALIKE && CARBON_CUDALIKE_DEVICE_TU
    using Idx = typename Grid::Idx;
    std::vector<gpu_error_t> errs(n, gpu_success);
#pragma omp parallel for num_threads(static_cast<int>(n)) if (n > 1)
    for (size_t i = 0; i < n; ++i) {
      KernelLoader ctx{i, KernelTarget::Device, KernelLoaderPhase::Launch};
      auto device_fn = load(grid_, loading_fn_, ctx);
      errs[i] = launch_lanes<Idx>(grid_.system().gpu_group().devices()[i], grid_.spans()[i], view,
                                  shared_mem_, stream[i].get(), device_fn);
    }
    for (size_t i = 0; i < n; ++i) {
      if (errs[i] != gpu_success) {
        log::throw_with_log<std::runtime_error>(
            "SubGrid kernel '{}': launch failed on relative_gpu_id={}: {} (gpu error {})", name(),
            i, gpu_error_string(errs[i]), static_cast<int>(errs[i]));
      }
    }
#elif CARBON_BACKEND_IS_SYCL
    using Idx = typename Grid::Idx;
    for (size_t i = 0; i < n; ++i) {
      KernelLoader ctx{i, KernelTarget::Device, KernelLoaderPhase::Launch};
      auto device_fn = load(grid_, loading_fn_, ctx);
      launch_lanes<Idx>(stream[i].queue(), grid_.spans()[i], view, shared_mem_, device_fn);
    }
#else
    static_assert(sub_grid_launch_requires_device_tu_<LoadingFn>,
                  "SubGrid::make_kernel launches require a device translation unit (a .cu file "
                  "compiled with nvcc / hipcc). Only the call site needs to be device code.");
    (void)view;
#endif
  }

private:
  auto
  discover_tokens_() -> void {
    if (grid_.gpu_count() == 0U) {
      return;
    }
    KernelLoader ctx{0U, KernelTarget::Device};
    [[maybe_unused]] auto fn = load(grid_, loading_fn_, ctx);
    tokens_ = ctx.tokens();
  }

  Grid grid_{};
  LoadingFn loading_fn_;
  size_t shared_mem_{};
  std::vector<AccessToken> tokens_{};
};

/**
 * Host (OpenMP) compute kernel over a SubGrid (@ref KernelKind::ComputeHost ). Waits on the
 * stream, then runs the lanes of the requested view on each GPU's host lane table.
 * Every field it loads needs a host mirror.
 */
template <typename Grid, typename LoadingFn>
class HostKernelImpl final : public ::carbon::detail::KernelImpl {
public:
  template <typename L>
  HostKernelImpl(const Grid& grid,
                 L&& loading,
                 std::string name)
      : KernelImpl(std::move(name)),
        grid_(grid),
        loading_fn_(std::forward<L>(loading)) {
    static_assert(std::is_invocable_v<LoadingFn&, ::carbon::KernelLoader&>,
                  "SubGrid::make_host_kernel: the loading callable must be invocable as "
                  "`loader(KernelLoader&) -> F`, where F is the per-cell "
                  "`(SubGrid::Span, SubGrid::Idx)` callable.");
    discover_tokens_();
  }

  [[nodiscard]] auto
  kind() const noexcept -> KernelKind override {
    return KernelKind::ComputeHost;
  }

  [[nodiscard]] auto
  tokens() const noexcept -> const std::vector<AccessToken>& override {
    return tokens_;
  }

  [[nodiscard]] auto
  runs_on_host() const noexcept -> bool override {
    return true;
  }

  auto
  launch(const ::carbon::Stream& stream,
         DataView view) -> void override {
    launch_on_host(stream, view);
  }

  auto
  launch_on_host(const ::carbon::Stream& stream,
                 DataView view) -> void override {
    try {
      stream.wait();
    } catch (const std::exception& ex) {
      log::throw_with_log<std::runtime_error>(
          "SubGrid host kernel '{}': stream.wait() before the host loop failed: {}", name(),
          ex.what());
    }
    using Idx = typename Grid::Idx;
    const size_t n = grid_.gpu_count();
    for (size_t g = 0; g < n; ++g) {
      KernelLoader ctx{g, KernelTarget::Host, KernelLoaderPhase::Launch};
      auto host_fn = load(grid_, loading_fn_, ctx);
      const auto& span = grid_.host_spans()[g];
      const int64_t begin = span.sub().lane_offset(view);
      const int64_t count = span.sub().lane_count(view);
#pragma omp parallel for
      for (int64_t k = 0; k < count; ++k) {
        const int64_t lane = begin + k;
        host_fn(span, Idx(span.sub().parent_idx(lane), lane));
      }
    }
  }

private:
  auto
  discover_tokens_() -> void {
    if (grid_.gpu_count() == 0U) {
      return;
    }
    KernelLoader ctx{0U, KernelTarget::Device};
    [[maybe_unused]] auto fn = load(grid_, loading_fn_, ctx);
    tokens_ = ctx.tokens();
  }

  Grid grid_{};
  LoadingFn loading_fn_;
  std::vector<AccessToken> tokens_{};
};

}  // namespace detail::sub_grid
}  // namespace carbon
