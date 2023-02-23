#pragma once
#include "Neon/core/core.h"

namespace Neon::domain::internal::exp::dGrid {

NEON_CUDA_HOST_DEVICE inline dIndex::dIndex(const Location& location)
{
    mLocation = location;
}

NEON_CUDA_HOST_DEVICE inline dIndex::dIndex(const Location::Integer &x,
                                            const Location::Integer &y,
                                            const Location::Integer &z){
    mLocation.x = x;
    mLocation.y = y;
    mLocation.z = z;
}

NEON_CUDA_HOST_DEVICE inline auto dIndex::set() -> Location&
{
    return mLocation;
}
NEON_CUDA_HOST_DEVICE inline auto dIndex::get() const -> const Location&
{
    return mLocation;
}

}  // namespace Neon::domain::dense