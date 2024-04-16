#pragma once
#include "Neon/core/core.h"

namespace Neon::domain::details::disaggregated::dGrid {

// NEON_CUDA_HOST_DEVICE inline dIndex::dIndex(const Location& location)
//{
//     mLocation = location;
// }

// NEON_CUDA_HOST_DEVICE inline dIndex::dIndex(const Location::Integer &x,
//                                             const Location::Integer &y,
//                                             const Location::Integer &z){
//     mLocation.x = x;
//     mLocation.y = y;
//     mLocation.z = z;
// }

NEON_CUDA_HOST_DEVICE inline auto dIndex::setLocation() -> Location&
{
    return mLocation;
}
NEON_CUDA_HOST_DEVICE inline auto dIndex::getLocation() const -> const Location&
{
    return mLocation;
}

NEON_CUDA_HOST_DEVICE inline auto dIndex::getOffsetLocalNoCard() const -> size_t
{
    return mOffsetLocalNoCard;
}

NEON_CUDA_HOST_DEVICE inline auto dIndex::setOffsetLocalNoCard(size_t xyzOffset) -> void
{
    mOffsetLocalNoCard = xyzOffset;
}

NEON_CUDA_HOST_DEVICE inline auto dIndex::getRegionFirstZ() const -> int32_t
{
    return mRegionFirstZ;
}

NEON_CUDA_HOST_DEVICE inline auto dIndex::setRegionFirstZ(int32_t regionFirstZ) -> void
{
    mRegionFirstZ = regionFirstZ;
}

NEON_CUDA_HOST_DEVICE inline auto dIndex::getRegionZDim() const -> int32_t
{
    return mRegionZDim;
}

NEON_CUDA_HOST_DEVICE inline auto dIndex::setRegionZDim(int32_t regionZDim) -> void
{
    mRegionZDim = regionZDim;
}

}  // namespace Neon::domain::details::disaggregated::dGrid