#pragma once
#include "Neon/domain/details/bGrid/bIndex.h"

namespace Neon::domain::details::bGrid {

template <int8_t dataBlockSizeX, int8_t dataBlockSizeY, int8_t dataBlockSizeZ>
NEON_CUDA_HOST_DEVICE inline bIndex<dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ>::
    bIndex(const DataBlockIdx&            blockIdx,
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