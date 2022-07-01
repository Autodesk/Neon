#pragma once
#include "Neon/core/core.h"

namespace Neon::domain::internal::eGrid {

NEON_CUDA_HOST_DEVICE inline eCell::eCell(Location location)
{
    mLocation = location;
}

NEON_CUDA_HOST_DEVICE inline auto eCell::set() -> Location&
{
    return mLocation;
}
NEON_CUDA_HOST_DEVICE inline auto eCell::get() -> const Location&
{
    return mLocation;
}

}  // namespace Neon::domain::internal::eGrid