#pragma once
#include "Neon/domain/details/bGrid/bIndex.h"

namespace Neon::domain::details::bGrid {

template <typename SBlock>
NEON_CUDA_HOST_DEVICE inline bIndex<SBlock>::
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


template <typename SBlock>
NEON_CUDA_HOST_DEVICE inline auto bIndex<SBlock>::getMicroIndex() -> MicroIndex
{


    TrayIdx const exBlockOffset = mDataBlockIdx * (SBlock::blockRatioX * SBlock::blockRatioY * SBlock::blockRatioZ);
    TrayIdx const exTrayOffset = [&] {
        TrayIdx const trayBlockIdxX = mInDataBlockIdx.x / SBlock::userBlockSizeX;
        TrayIdx const trayBlockIdxY = mInDataBlockIdx.y / SBlock::userBlockSizeY;
        TrayIdx const trayBlockIdxZ = mInDataBlockIdx.z / SBlock::userBlockSizeZ;

        TrayIdx const res = trayBlockIdxX + trayBlockIdxY * SBlock::blockRatioX +
                            trayBlockIdxZ * (SBlock::blockRatioX * SBlock::blockRatioY);
        return res;
    }();
    MicroIndex res;
    res.setTrayBlockIdx(exBlockOffset + exTrayOffset);
    res.setInTrayBlockIdx({static_cast<InTrayIdx::Integer>(mInDataBlockIdx.x % SBlock::userBlockSizeX),
                           static_cast<InTrayIdx::Integer>(mInDataBlockIdx.y % SBlock::userBlockSizeY),
                           static_cast<InTrayIdx::Integer>(mInDataBlockIdx.z % SBlock::userBlockSizeZ)});
    return res;
}


template <typename SBlock>
NEON_CUDA_HOST_DEVICE inline auto bIndex<SBlock>::init(MicroIndex const& microIndex) -> void
{
    constexpr uint32_t memBlockSize = SBlock::memBlockSizeX * SBlock::memBlockSizeY * SBlock::memBlockSizeZ;
    constexpr uint32_t userBlockSize = SBlock::userBlockSizeX * SBlock::userBlockSizeY * SBlock::userBlockSizeZ;
    constexpr uint32_t blockRatioSize = memBlockSize / userBlockSize;

    constexpr uint32_t blockRatioX = SBlock::memBlockSizeX / SBlock::userBlockSizeX;
    constexpr uint32_t blockRatioY = SBlock::memBlockSizeY / SBlock::userBlockSizeY;

    mDataBlockIdx = microIndex.getTrayBlockIdx() / (blockRatioSize);

    uint32_t reminder = microIndex.getTrayBlockIdx() % (blockRatioSize);

    const uint32_t reminderInZ = reminder / (blockRatioX * blockRatioY);
    mInDataBlockIdx.z = static_cast<InDataBlockIdx::Integer>(microIndex.getInTrayBlockIdx().z + reminderInZ * SBlock::userBlockSizeZ);
    reminder = reminder % (blockRatioX * blockRatioY);
    const uint32_t reminderInY = reminder / (blockRatioX);
    mInDataBlockIdx.y = static_cast<InDataBlockIdx::Integer>(microIndex.getInTrayBlockIdx().y + reminderInY * SBlock::userBlockSizeY);
    const uint32_t reminderInX = reminder % blockRatioX;
    mInDataBlockIdx.x = static_cast<InDataBlockIdx::Integer>(microIndex.getInTrayBlockIdx().x + reminderInX * SBlock::userBlockSizeX);
}

}  // namespace Neon::domain::details::bGrid