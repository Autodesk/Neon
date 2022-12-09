#pragma once

namespace Neon::set::internal {

enum struct HostManagedSyncType
{
    intraGPU = 0,
    multiGPU = 1,
    none = 2
};

}  // namespace Neon::set::internal
