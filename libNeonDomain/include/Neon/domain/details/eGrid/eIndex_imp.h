#pragma once
#include "Neon/core/core.h"
#include "Neon/domain/details/eGrid/eIndex.h"

namespace Neon::domain::details::eGrid {

NEON_CUDA_HOST_DEVICE  eIndex::eIndex(const eIndex::InternalIdx& idx)
{
    mIdx = idx;
}

NEON_CUDA_HOST_DEVICE inline auto eIndex::set() -> InternalIdx&
{
    return mIdx;
}
NEON_CUDA_HOST_DEVICE inline auto eIndex::get() const -> const InternalIdx&
{
    return mIdx;
}


}  // namespace Neon::domain::dense