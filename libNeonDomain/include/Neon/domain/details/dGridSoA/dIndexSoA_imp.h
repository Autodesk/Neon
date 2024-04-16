#pragma once
#include "Neon/core/core.h"

namespace Neon::domain::details::dGridSoA {

NEON_CUDA_HOST_DEVICE inline dIndexSoA::
    dIndexSoA(const Location& location,
              Offset const&   offset)
{
    mLocation = location;
    mOffset = offset;
}

NEON_CUDA_HOST_DEVICE inline dIndexSoA::
    dIndexSoA(const Location::Integer& x,
              const Location::Integer& y,
              const Location::Integer& z,
              Offset const&            offset)
{
    mLocation.x = x;
    mLocation.y = y;
    mLocation.z = z;
    mOffset = offset;
}

NEON_CUDA_HOST_DEVICE inline auto dIndexSoA::
    setLocation() -> Location&
{
    return mLocation;
}

NEON_CUDA_HOST_DEVICE inline auto dIndexSoA::
    setOffset() -> Offset&
{
    return mOffset;
}

NEON_CUDA_HOST_DEVICE inline auto dIndexSoA::
    getLocation() const -> const Location&
{
    return mLocation;
}

NEON_CUDA_HOST_DEVICE inline auto dIndexSoA::
    getOffset()
        const -> const Offset&
{
    return mOffset;
}
}  // namespace Neon::domain::details::dGridSoA