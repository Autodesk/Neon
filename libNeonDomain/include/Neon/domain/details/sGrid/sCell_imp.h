#pragma once
#include "Neon/core/core.h"
#include "Neon/domain/details/sGrid/sIndex.h"

namespace Neon::domain::details::sGrid {

NEON_CUDA_HOST_DEVICE inline sIndex::sIndex(Location location)
{
    mLocation = location;
}

NEON_CUDA_HOST_DEVICE inline auto sIndex::get() -> Location&
{
    return mLocation;
}

NEON_CUDA_HOST_DEVICE inline auto sIndex::get() const -> const Location&
{
    return mLocation;
}
}  // namespace Neon::domain::details::sGrid