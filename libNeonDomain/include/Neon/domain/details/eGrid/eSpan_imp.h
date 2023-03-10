#pragma once
#include "eSpan.h"

namespace Neon::domain::details::eGrid {

NEON_CUDA_HOST_DEVICE
inline auto eSpan::setAndValidate(
    Idx&            Idx,
    const uint32_t& x)
    const
    -> bool
{
    Idx.set() = Idx::InternalIdx(x);


    bool isValid = false;

    if (Idx.get() < mCount) {
        isValid = true;
        Idx.mIdx += mFirstIndexOffset;
    }

    return isValid;
}


}  // namespace Neon::domain::details::eGrid