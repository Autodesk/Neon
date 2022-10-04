#pragma once
#include "Neon/core/core.h"

namespace Neon::domain::internal::dGrid {

NEON_CUDA_HOST_DEVICE inline dCell::dCell(const Location& location)
{
    mLocation = location;
}

NEON_CUDA_HOST_DEVICE inline dCell::dCell(const Location::Integer &x,
                                            const Location::Integer &y,
                                            const Location::Integer &z){
    mLocation.x = x;
    mLocation.y = y;
    mLocation.z = z;
}

NEON_CUDA_HOST_DEVICE inline auto dCell::set() -> Location&
{
    return mLocation;
}
NEON_CUDA_HOST_DEVICE inline auto dCell::get() const -> const Location&
{
    return mLocation;
}

}  // namespace Neon::domain::dense