#pragma once
#include "Neon/core/core.h"

namespace Neon::domain::details::ncclGrid {

NEON_CUDA_HOST_DEVICE inline ncclIndex::ncclIndex(const Location& location)
{
    mLocation = location;
}

NEON_CUDA_HOST_DEVICE inline ncclIndex::ncclIndex(const Location::Integer& x,
                                                  const Location::Integer& y,
                                                  const Location::Integer& z)
{
    mLocation.x = x;
    mLocation.y = y;
    mLocation.z = z;
}

NEON_CUDA_HOST_DEVICE inline auto ncclIndex::setLocation() -> Location&
{
    return mLocation;
}
NEON_CUDA_HOST_DEVICE inline auto ncclIndex::getLocation() const -> const Location&
{
    return mLocation;
}

}  // namespace Neon::domain::details::ncclGrid