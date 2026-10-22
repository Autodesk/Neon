#pragma once

// Bridge header used only inside src/ and from .cu translation units that need raw access to
// the underlying ncclComm_t. Not installed.

#include <carbon/detail/communicator.h>

#include "gpu_runtime_compat.h"  // pulls in <nccl.h> or <rccl/rccl.h> based on CARBON_USE_RCCL

namespace carbon::detail {

// Definition of the friend bridge forward-declared in <carbon/detail/communicator.h>. Users
// cannot reach @c Communicator::raw_handle through this bridge because the definition lives in
// this header, which is not installed; in-tree code that needs the raw @c ncclComm_t includes
// this header explicitly.
struct nccl_bridge {
  [[nodiscard]] static auto
  comm(const Communicator& c) noexcept -> ncclComm_t {
    return reinterpret_cast<ncclComm_t>(c.raw_handle());
  }
};

/** Returns the underlying @c ncclComm_t held by @p comm. */
[[nodiscard]] inline auto
communicator_handle(const Communicator& comm) noexcept -> ncclComm_t {
  return nccl_bridge::comm(comm);
}

}  // namespace carbon::detail
