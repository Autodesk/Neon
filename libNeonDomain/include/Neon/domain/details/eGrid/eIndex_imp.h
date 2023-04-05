#pragma once
#include "Neon/core/core.h"
#include "Neon/domain/details/eGrid/eIndex.h"

namespace Neon::domain::details::eGrid {

NEON_CUDA_HOST_DEVICE  eIndex::eIndex(const eIndex::InternalIdx& idx)
{
    mIdx = idx;
}

NEON_CUDA_HOST_DEVICE inline auto eIndex::helpSet() -> InternalIdx&
{
    return mIdx;
}
NEON_CUDA_HOST_DEVICE inline auto eIndex::helpGet() const -> const InternalIdx&
{
    return mIdx;
}
NEON_CUDA_HOST_DEVICE inline auto  eIndex::manualSet(eIndex::InternalIdx idx) -> void
{
    mIdx = idx;
}


}  // namespace Neon::domain::dense