#include <carbonpy/warp_kernel.h>

#include <carbon/data_view.h>
#include <carbon/detail/kernel_impl.h>
#include <carbon/logging.h>
#include <carbon/system.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace carbonpy {
namespace {

class WarpDenseKernelImpl final : public carbon::detail::KernelImpl {
public:
  explicit WarpDenseKernelImpl(void* container,
                               std::vector<carbon::AccessToken> tokens,
                               std::string name)
      : KernelImpl(std::move(name)),
        container_(container),
        tokens_(std::move(tokens)) {}

  auto
  launch(const carbon::Stream& stream,
         carbon::DataView view) -> void override {
    launch_on_device(stream, view);
  }

  auto
  launch_on_device(const carbon::Stream& stream,
                   carbon::DataView view) -> void override {
    if (container_ == nullptr) {
      carbon::log::throw_with_log<std::runtime_error>(
          "WarpDenseKernelImpl::launch_on_device: container handle is null");
    }
    const size_t n = stream.size();
    for (size_t g = 0; g < n; ++g) {
      const int rc = detail::launch_dense_container_on_carbon_stream(
          container_, g, stream.native_at(g), view);
      if (rc != 0) {
        carbon::log::throw_with_log<std::runtime_error>(
            "WarpDenseKernelImpl::launch_on_device: launch failed on gpu {} (rc={})",
            g, rc);
      }
    }
  }

  [[nodiscard]] auto
  runs_on_device() const noexcept -> bool override {
    return true;
  }

  [[nodiscard]] auto
  kind() const noexcept -> carbon::KernelKind override {
    return carbon::KernelKind::Compute;
  }

  [[nodiscard]] auto
  tokens() const noexcept -> const std::vector<carbon::AccessToken>& override {
    return tokens_;
  }

private:
  void* container_{};
  std::vector<carbon::AccessToken> tokens_;
};

class WarpBlockKernelImpl final : public carbon::detail::KernelImpl {
public:
  explicit WarpBlockKernelImpl(void* container,
                               std::vector<carbon::AccessToken> tokens,
                               std::string name)
      : KernelImpl(std::move(name)),
        container_(container),
        tokens_(std::move(tokens)) {}

  auto
  launch(const carbon::Stream& stream,
         carbon::DataView view) -> void override {
    launch_on_device(stream, view);
  }

  auto
  launch_on_device(const carbon::Stream& stream,
                   carbon::DataView view) -> void override {
    if (container_ == nullptr) {
      carbon::log::throw_with_log<std::runtime_error>(
          "WarpBlockKernelImpl::launch_on_device: container handle is null");
    }
    const size_t n = stream.size();
    for (size_t g = 0; g < n; ++g) {
      const int rc = detail::launch_block_container_on_carbon_stream(
          container_, g, stream.native_at(g), view);
      if (rc != 0) {
        carbon::log::throw_with_log<std::runtime_error>(
            "WarpBlockKernelImpl::launch_on_device: launch failed on gpu {} (rc={})",
            g, rc);
      }
    }
  }

  [[nodiscard]] auto
  runs_on_device() const noexcept -> bool override {
    return true;
  }

  [[nodiscard]] auto
  kind() const noexcept -> carbon::KernelKind override {
    return carbon::KernelKind::Compute;
  }

  [[nodiscard]] auto
  tokens() const noexcept -> const std::vector<carbon::AccessToken>& override {
    return tokens_;
  }

private:
  void* container_{};
  std::vector<carbon::AccessToken> tokens_;
};

class WarpMgridKernelImpl final : public carbon::detail::KernelImpl {
public:
  explicit WarpMgridKernelImpl(void* container,
                               std::vector<carbon::AccessToken> tokens,
                               std::string name)
      : KernelImpl(std::move(name)),
        container_(container),
        tokens_(std::move(tokens)) {}

  auto
  launch(const carbon::Stream& stream,
         carbon::DataView view) -> void override {
    launch_on_device(stream, view);
  }

  auto
  launch_on_device(const carbon::Stream& stream,
                   carbon::DataView view) -> void override {
    if (container_ == nullptr) {
      carbon::log::throw_with_log<std::runtime_error>(
          "WarpMgridKernelImpl::launch_on_device: container handle is null");
    }
    const size_t n = stream.size();
    for (size_t g = 0; g < n; ++g) {
      const int rc = detail::launch_mgrid_container_on_carbon_stream(
          container_, g, stream.native_at(g), view);
      if (rc != 0) {
        carbon::log::throw_with_log<std::runtime_error>(
            "WarpMgridKernelImpl::launch_on_device: launch failed on gpu {} (rc={})",
            g, rc);
      }
    }
  }

  [[nodiscard]] auto
  runs_on_device() const noexcept -> bool override {
    return true;
  }

  [[nodiscard]] auto
  kind() const noexcept -> carbon::KernelKind override {
    return carbon::KernelKind::Compute;
  }

  [[nodiscard]] auto
  tokens() const noexcept -> const std::vector<carbon::AccessToken>& override {
    return tokens_;
  }

private:
  void* container_{};
  std::vector<carbon::AccessToken> tokens_;
};

}  // namespace

auto
make_warp_dense_kernel(void* dense_container_handle) -> carbon::Kernel {
  const auto& tokens = detail::dense_container_tokens(dense_container_handle);
  auto impl = std::make_shared<WarpDenseKernelImpl>(dense_container_handle,
                                                    tokens, "warp_dense_kernel");
  return carbon::Kernel::from_impl(std::move(impl));
}

auto
make_warp_block_kernel(void* block_container_handle) -> carbon::Kernel {
  const auto& tokens = detail::block_container_tokens(block_container_handle);
  auto impl = std::make_shared<WarpBlockKernelImpl>(block_container_handle,
                                                    tokens, "warp_block_kernel");
  return carbon::Kernel::from_impl(std::move(impl));
}

auto
make_warp_mgrid_kernel(void* mgrid_container_handle) -> carbon::Kernel {
  const auto& tokens = detail::mgrid_container_tokens(mgrid_container_handle);
  auto impl = std::make_shared<WarpMgridKernelImpl>(mgrid_container_handle,
                                                    tokens, "warp_mgrid_kernel");
  return carbon::Kernel::from_impl(std::move(impl));
}

}  // namespace carbonpy
