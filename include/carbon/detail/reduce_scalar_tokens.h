#pragma once

#include <carbon/access_token.h>
#include <carbon/detail/scalar_readback_factory.h>
#include <carbon/reduce.h>

#include <memory>
#include <vector>

namespace carbon {
namespace detail {

template <typename T>
inline auto
append_reduction_scalar_tokens(std::vector<AccessToken>& tokens,
                               ReductionResult<T>& dst,
                               reduce::ReduceOptions options) -> void {
  auto factory = std::make_shared<TypedScalarReadbackFactory<T>>(dst);
  AccessToken device_tok{};
  device_tok.field_uid = dst.device_uid();
  device_tok.resource = ResourceKind::ScalarDevice;
  device_tok.mode = AccessMode::Write;
  device_tok.field_name = "scalar_device";
  if (options.host_copy == reduce::ReduceHostCopy::Deferred) {
    device_tok.scalar_readback_factory = factory;
  }
  tokens.push_back(device_tok);
  if (options.host_copy == reduce::ReduceHostCopy::Inline) {
    AccessToken host_tok{};
    host_tok.field_uid = dst.host_uid();
    host_tok.resource = ResourceKind::ScalarHost;
    host_tok.mode = AccessMode::Write;
    host_tok.field_name = "scalar_host";
    tokens.push_back(host_tok);
  }
}

}  // namespace detail
}  // namespace carbon
