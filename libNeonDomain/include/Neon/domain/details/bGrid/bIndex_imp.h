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

//
// template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
// NEON_CUDA_HOST_DEVICE inline auto bIndex<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::getTrayIdx() -> TrayIdx
//{
//
//    TrayIdx const exBlockOffset = mDataBlockIdx * (userBlockSizeX * userBlockSizeY * userBlockSizeZ);
//    TrayIdx const exTrayOffset = [&]() {
//        int const trayBlockIdxX = mInDataBlockIdx.x / userBlockSizeX;
//        int const trayBlockIdxY = mInDataBlockIdx.y / userBlockSizeY;
//        int const trayBlockIdxZ = mInDataBlockIdx.z / userBlockSizeZ;
//
//        constexpr int countMicroBlocksInTrayX = (memBlockSizeX / userBlockSizeX);
//        constexpr int countMicroBlocksInTrayY = (memBlockSizeY / userBlockSizeY);
//
//        int const res = trayBlockIdxX + trayBlockIdxY * countMicroBlocksInTrayX +
//                        trayBlockIdxZ * (countMicroBlocksInTrayX * countMicroBlocksInTrayY);
//        return res;
//    };
//    return exBlockOffset + exTrayOffset;
//}
//
//
// template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
// NEON_CUDA_HOST_DEVICE inline auto bIndex<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::getInTrayIdx() -> InTrayIdx
//{
//    InTrayIdx inTrayIdx;
//    inTrayIdx.x = mInDataBlockIdx.x % userBlockSizeX;
//    inTrayIdx.y = mInDataBlockIdx.y % userBlockSizeY;
//    inTrayIdx.z = mInDataBlockIdx.z % userBlockSizeZ;
//
//    return inTrayIdx;
//}

template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
NEON_CUDA_HOST_DEVICE inline auto bIndex<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::getMicroIndex() -> MicroIndex
{
    constexpr uint32_t blockRatioX = memBlockSizeX / userBlockSizeX;
    constexpr uint32_t blockRatioY = memBlockSizeY / userBlockSizeY;
    constexpr uint32_t blockRatioZ = memBlockSizeZ / userBlockSizeZ;

    TrayIdx const exBlockOffset = mDataBlockIdx * (blockRatioX * blockRatioY * blockRatioZ);
    TrayIdx const exTrayOffset = [&] {
        TrayIdx const trayBlockIdxX = mInDataBlockIdx.x / userBlockSizeX;
        TrayIdx const trayBlockIdxY = mInDataBlockIdx.y / userBlockSizeY;
        TrayIdx const trayBlockIdxZ = mInDataBlockIdx.z / userBlockSizeZ;

        TrayIdx const res = trayBlockIdxX + trayBlockIdxY * blockRatioX +
                            trayBlockIdxZ * (blockRatioX * blockRatioY);
        return res;
    }();
    MicroIndex res;
    res.setTrayBlockIdx(exBlockOffset + exTrayOffset);
    res.setInTrayBlockIdx({static_cast<InTrayIdx::Integer>(mInDataBlockIdx.x % userBlockSizeX),
                           static_cast<InTrayIdx::Integer>(mInDataBlockIdx.y % userBlockSizeY),
                           static_cast<InTrayIdx::Integer>(mInDataBlockIdx.z % userBlockSizeZ)});
    return res;
}


template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
NEON_CUDA_HOST_DEVICE inline auto bIndex<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::init(MicroIndex const& microIndex) -> void
{
    constexpr uint32_t memBlockSize = memBlockSizeX * memBlockSizeY * memBlockSizeZ;
    constexpr uint32_t userBlockSize = userBlockSizeX * userBlockSizeY * userBlockSizeZ;
    constexpr uint32_t blockRatioSize = memBlockSize / userBlockSize;

    constexpr uint32_t blockRatioX = memBlockSizeX / userBlockSizeX;
    constexpr uint32_t blockRatioY = memBlockSizeY / userBlockSizeY;

    mDataBlockIdx = microIndex.getTrayBlockIdx() / (blockRatioSize);

    uint32_t reminder = microIndex.getTrayBlockIdx() % (blockRatioSize);

    const uint32_t reminderInZ = reminder / (blockRatioX * blockRatioY);
    mInDataBlockIdx.z = microIndex.getInTrayBlockIdx().z + reminderInZ * userBlockSizeZ;
    reminder = reminder % (blockRatioX * blockRatioY);
    const uint32_t reminderInY = reminder / (blockRatioX);
    mInDataBlockIdx.y = microIndex.getInTrayBlockIdx().y + reminderInY * userBlockSizeY;
    const uint32_t reminderInX = reminder % blockRatioX;
    mInDataBlockIdx.x = static_cast<uint8_t>(microIndex.getInTrayBlockIdx().x + reminderInX * userBlockSizeX);
}

}  // namespace Neon::domain::details::bGrid