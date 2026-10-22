#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#if defined(CARBON_HAVE_NVTX) && CARBON_HAVE_NVTX

#include <nvtx3/nvToolsExt.h>

namespace carbon::detail {

/**
 * RAII NVTX thread range for host-side kernel scheduling. Zero overhead when no
 * profiler is attached. Default colour is yellow (ARGB 0xFFFFFF00).
 */
class NvtxPushPopScope {
public:
  explicit NvtxPushPopScope(std::string_view label,
                            uint32_t argb_color = 0xFFFFFF00U) {
    name_.assign(label.begin(), label.end());
    if (name_.empty()) {
      name_ = "carbon::Kernel";
    }
    nvtxEventAttributes_t attr{};
    attr.version = NVTX_VERSION;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.color = argb_color;
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name_.c_str();
    nvtxRangePushEx(&attr);
  }

  ~NvtxPushPopScope() {
    nvtxRangePop();
  }

  NvtxPushPopScope(const NvtxPushPopScope&) = delete;
  NvtxPushPopScope(NvtxPushPopScope&&) = delete;
  auto
  operator=(const NvtxPushPopScope&) -> NvtxPushPopScope& = delete;
  auto
  operator=(NvtxPushPopScope&&) -> NvtxPushPopScope& = delete;

private:
  std::string name_;
};

}  // namespace carbon::detail

#else

namespace carbon::detail {

class NvtxPushPopScope {
public:
  explicit NvtxPushPopScope(std::string_view /*label*/,
                            uint32_t /*argb_color*/ = 0xFFFFFF00U) {}
};

}  // namespace carbon::detail

#endif
