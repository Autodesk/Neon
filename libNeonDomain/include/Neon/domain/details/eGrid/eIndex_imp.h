#pragma once
#include "Neon/core/core.h"

namespace Neon::domain::details::eGrid {

NEON_CUDA_HOST_DEVICE inline eIndex::dIndex(const Idx& location)
{
    mLocation = location;
}

NEON_CUDA_HOST_DEVICE inline eIndex::dIndex(const Idx::Integer &x,
                                            const Idx::Integer &y,
                                            const Idx::Integer &z){
    mLocation.x = x;
    mLocation.y = y;
    mLocation.z = z;
}

NEON_CUDA_HOST_DEVICE inline auto eIndex::set() -> Idx&
{
    return mLocation;
}
NEON_CUDA_HOST_DEVICE inline auto eIndex::get() const -> const Idx&
{
    return mLocation;
}

}  // namespace Neon::domain::dense