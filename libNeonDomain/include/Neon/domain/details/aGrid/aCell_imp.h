#pragma once
#include "Neon/core/core.h"

namespace Neon::domain::details::aGrid {

NEON_CUDA_HOST_DEVICE inline aCell::aCell(Location location)
{
    mLocation = location;
}

NEON_CUDA_HOST_DEVICE inline auto aCell::set() -> Location&
{
    return mLocation;
}
NEON_CUDA_HOST_DEVICE inline auto aCell::get() const -> const Location&
{
    return mLocation;
}

}  // namespace Neon::domain::array