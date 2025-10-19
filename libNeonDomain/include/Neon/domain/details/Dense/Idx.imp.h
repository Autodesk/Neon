#pragma once
#include "Neon/core/core.h"

namespace Neon::domain::details::Dense {

NEON_CUDA_HOST_DEVICE inline Idx::Idx(const Location& location)
{
    mLocation = location;
}

NEON_CUDA_HOST_DEVICE inline Idx::Idx(const Location::Integer &x,
                                            const Location::Integer &y,
                                            const Location::Integer &z){
    mLocation.x = x;
    mLocation.y = y;
    mLocation.z = z;
}

NEON_CUDA_HOST_DEVICE inline auto Idx::setLocation() -> Location&
{
    return mLocation;
}
NEON_CUDA_HOST_DEVICE inline auto Idx::getLocation() const -> const Location&
{
    return mLocation;
}

}  // namespace Neon::domain::details::Dense