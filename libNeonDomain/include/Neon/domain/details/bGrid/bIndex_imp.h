#pragma once
#include "Neon/domain/details/bGrid/bIndex.h"

namespace Neon::domain::details::bGrid {

NEON_CUDA_HOST_DEVICE inline bIndex::bIndex(const DataBlockIdx&            blockIdx,
                                            const InDataBlockIdx::Integer& x,
                                            const InDataBlockIdx::Integer& y,
                                            const InDataBlockIdx::Integer& z)
{
    mDataBlockIdx = blockIdx;
    mInDataBlockIdx.x = x;
    mInDataBlockIdx.y = y;
    mInDataBlockIdx.z = z;
}

}  // namespace Neon::domain::details::bGrid