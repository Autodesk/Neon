#pragma once
#include "Neon/core/core.h"

namespace Neon::domain::details::dissagragated::dGrid {

NEON_CUDA_HOST_DEVICE inline dIndex::
    dIndex(const Location& location,
              Offset const&   offset)
{
    mLocation = location;
    mOffset = offset;
}

NEON_CUDA_HOST_DEVICE inline dIndex::
    dIndex(const Location::Integer& x,
              const Location::Integer& y,
              const Location::Integer& z,
              Offset const&            offset)
{
    mLocation.x = x;
    mLocation.y = y;
    mLocation.z = z;
    mOffset = offset;
}

NEON_CUDA_HOST_DEVICE inline auto dIndex::
    setLocation() -> Location&
{
    return mLocation;
}

NEON_CUDA_HOST_DEVICE inline auto dIndex::
    setOffset() -> Offset&
{
    return mOffset;
}

NEON_CUDA_HOST_DEVICE inline auto dIndex::
    getLocation() const -> const Location&
{
    return mLocation;
}

NEON_CUDA_HOST_DEVICE inline auto dIndex::
    getOffset()
        const -> const Offset&
{
    return mOffset;
}
}  // namespace Neon::domain::details::dGridSoA