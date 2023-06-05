#pragma once
#include "Neon/core/core.h"

namespace Neon::domain::details::aGrid {

NEON_CUDA_HOST_DEVICE inline aIndex::aIndex(Location location)
{
    mLocation = location;
}

NEON_CUDA_HOST_DEVICE inline auto aIndex::set() -> Location&
{
    return mLocation;
}

NEON_CUDA_HOST_DEVICE inline auto aIndex::get() const -> const Location&
{
    return mLocation;
}

}  // namespace Neon::domain::details::aGrid