#pragma once
#include "Neon/domain/details/bGrid/bIndex.h"

namespace Neon::domain::details::bGrid {

template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
NEON_CUDA_HOST_DEVICE inline bIndex<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::
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


template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
NEON_CUDA_HOST_DEVICE inline auto bIndex<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::getTrayIdx() -> TrayIdx
{

    TrayIdx const exBlockOffset = mDataBlockIdx * (userBlockSizeX * userBlockSizeY * userBlockSizeZ);
    TrayIdx const exTrayOffset = [&]() {
        int const trayBlockIdxX = mInDataBlockIdx.x / userBlockSizeX;
        int const trayBlockIdxY = mInDataBlockIdx.y / userBlockSizeY;
        int const trayBlockIdxZ = mInDataBlockIdx.z / userBlockSizeZ;

        constexpr int countMicroBlocksInTrayX = (memBlockSizeX / userBlockSizeX);
        constexpr int countMicroBlocksInTrayY = (memBlockSizeY / userBlockSizeY);

        int const res = trayBlockIdxX + trayBlockIdxY * countMicroBlocksInTrayX +
                        trayBlockIdxZ * (countMicroBlocksInTrayX * countMicroBlocksInTrayY);
        return res;
    };
    return exBlockOffset + exTrayOffset;
}


template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
NEON_CUDA_HOST_DEVICE inline auto bIndex<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::getInTrayIdx() -> InTrayIdx
{
    InTrayIdx inTrayIdx;
    inTrayIdx.x = mInDataBlockIdx.x % userBlockSizeX;
    inTrayIdx.y = mInDataBlockIdx.y % userBlockSizeY;
    inTrayIdx.z = mInDataBlockIdx.z % userBlockSizeZ;

    return inTrayIdx;
}

template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
NEON_CUDA_HOST_DEVICE inline auto bIndex<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::getMicroIndex() -> MicroIndex
{
    TrayIdx const exBlockOffset = mDataBlockIdx * (userBlockSizeX * userBlockSizeY * userBlockSizeZ);
    TrayIdx const exTrayOffset = [&]() {
        int const trayBlockIdxX = mInDataBlockIdx.x / userBlockSizeX;
        int const trayBlockIdxY = mInDataBlockIdx.y / userBlockSizeY;
        int const trayBlockIdxZ = mInDataBlockIdx.z / userBlockSizeZ;

        constexpr int countMicroBlocksInTrayX = (memBlockSizeX / userBlockSizeX);
        constexpr int countMicroBlocksInTrayY = (memBlockSizeY / userBlockSizeY);
        constexpr int countMicroBlocksInTrayZ = (memBlockSizeZ / userBlockSizeZ);

        int const res = trayBlockIdxX + trayBlockIdxY * countMicroBlocksInTrayX +
                        trayBlockIdxZ * (countMicroBlocksInTrayX * countMicroBlocksInTrayY);
        return res;
    };
    MicroIndex res;
    res.mDataBlockIdx = exBlockOffset + exTrayOffset;
    InTrayIdx inTrayIdx;
    res.mInDataBlockIdx.x = mInDataBlockIdx.x % userBlockSizeX;
    res.mInDataBlockIdx.y = mInDataBlockIdx.y % userBlockSizeY;
    res.mInDataBlockIdx.z = mInDataBlockIdx.z % userBlockSizeZ;

    return res;
}


template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
NEON_CUDA_HOST_DEVICE inline auto bIndex<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::init(MicroIndex const& microIndex) -> void
{
    uint32_t dataBlockSize = memBlockSizeX * memBlockSizeY * memBlockSizeZ;
    mDataBlockIdx = microIndex.mDataBlockIdx / dataBlockSize;

    uint32_t reminder = microIndex.mDataBlockIdx % dataBlockSize;
    mInDataBlockIdx.z = microIndex.mInDataBlockIdx.z + reminder / (userBlockSizeX * userBlockSizeY);
    reminder = reminder % (userBlockSizeX * userBlockSizeY);
    mInDataBlockIdx.y = microIndex.mInDataBlockIdx.y + reminder / userBlockSizeX;
    reminder = reminder % userBlockSizeX;
    mInDataBlockIdx.x = static_cast<uint8_t>(microIndex.mInDataBlockIdx.x + reminder);
}

}  // namespace Neon::domain::details::bGrid