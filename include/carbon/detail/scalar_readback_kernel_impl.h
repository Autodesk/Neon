#pragma once

#include <carbon/access_token.h>
#include <carbon/data_view.h>
#include <carbon/detail/kernel_impl.h>
#include <carbon/logging.h>
#include <carbon/reduce.h>
#include <carbon/system.h>

#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace carbon {
namespace detail {

template <typename T>
class ScalarReadbackKernelImpl final : public KernelImpl {
public:
  ScalarReadbackKernelImpl(ReductionResult<T>& dst,
                           std::string name = {})
      : KernelImpl(std::move(name)),
        dst_(&dst) {
    tokens_.push_back(AccessToken{dst.device_uid(),
                                  ResourceKind::ScalarDevice,
                                  AccessMode::Read,
                                  AccessPattern::Map,
                                  "scalar_device",
                                  {},
                                  {}});
    tokens_.push_back(AccessToken{dst.host_uid(),
                                  ResourceKind::ScalarHost,
                                  AccessMode::Write,
                                  AccessPattern::Map,
                                  "scalar_host",
                                  {},
                                  {}});
  }

  [[nodiscard]] auto
  kind() const noexcept -> KernelKind override {
    return KernelKind::ScalarReadback;
  }

  [[nodiscard]] auto
  runs_on_device() const noexcept -> bool override {
    return true;
  }

  [[nodiscard]] auto
  tokens() const noexcept -> const std::vector<AccessToken>& override {
    return tokens_;
  }

  auto
  launch(const Stream& stream,
         DataView view) -> void override {
    launch_on_device(stream, view);
  }

  auto
  launch_on_device(const Stream& stream,
                   DataView view) -> void override {
    ReductionResult<T>& dst = *dst_;
    if (dst.launch_cursor() == 0U) {
      log::throw_with_log<std::runtime_error>(
          "Kernel::scalar_readback: no preceding reduction launch wrote the device scalar");
    }
    const size_t slot = dst.active_slot();
    dst.system().async_copy(dst.device_allocation(view, slot), dst.host_allocation(view, slot),
                            sizeof(T), stream);
    dst.record_readback_done(view, slot, stream);
    dst.mark_fresh(view);
  }

private:
  ReductionResult<T>* dst_{nullptr};
  std::vector<AccessToken> tokens_{};
};

}  // namespace detail
}  // namespace carbon
