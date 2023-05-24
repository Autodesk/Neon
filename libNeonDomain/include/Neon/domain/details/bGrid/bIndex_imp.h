#pragma once
#include "Neon/domain/details/bGrid/bIndex.h"

namespace Neon::domain::details::bGrid {

template <int8_t dataBlockSizeX, int8_t dataBlockSizeY, int8_t dataBlockSizeZ, int8_t userBlockSizeX, int8_t userBlockSizeY, int8_t userBlockSizeZ>
NEON_CUDA_HOST_DEVICE inline bIndex<dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::
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


template <int8_t dataBlockSizeX, int8_t dataBlockSizeY, int8_t dataBlockSizeZ, int8_t userBlockSizeX, int8_t userBlockSizeY, int8_t userBlockSizeZ>
NEON_CUDA_HOST_DEVICE inline auto bIndex<dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::getTrayIdx() -> TrayIdx
{
    static_assert(dataBlockSizeX % userBlockSizeX == 0);
    static_assert(dataBlockSizeY % userBlockSizeY == 0);
    static_assert(dataBlockSizeZ % userBlockSizeZ == 0);

    constexpr uint32_t userBSX = static_cast<uint32_t>(userBlockSizeX);
    constexpr uint32_t userBSY = static_cast<uint32_t>(userBlockSizeY);
    constexpr uint32_t userBSZ = static_cast<uint32_t>(userBlockSizeZ);

    using Ops = Operations<dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>;

    TrayIdx const exBlockOffset = mDataBlockIdx * (userBSX * userBSY * userBSZ);
    TrayIdx const exTrayOffset = [&]() {
        int const trayBlockIdxX = Ops::divisionOp<userBSX>(mInDataBlockIdx.x);
        int const trayBlockIdxY = Ops::divisionOp<userBSY>(mInDataBlockIdx.y);
        int const trayBlockIdxZ = Ops::divisionOp<userBSZ>(mInDataBlockIdx.z);

        constexpr int countMicroBlocksInTrayX = (dataBlockSizeX / userBlockSizeX);
        constexpr int countMicroBlocksInTrayY = (dataBlockSizeY / userBlockSizeY);
        constexpr int countMicroBlocksInTrayZ = (dataBlockSizeZ / userBlockSizeZ);

        int const res = trayBlockIdxX + trayBlockIdxY * countMicroBlocksInTrayX +
                        trayBlockIdxZ * (countMicroBlocksInTrayX * countMicroBlocksInTrayY);
        return res;
    };
    return exBlockOffset + exTrayOffset;
}


template <int8_t dataBlockSizeX, int8_t dataBlockSizeY, int8_t dataBlockSizeZ, int8_t userBlockSizeX, int8_t userBlockSizeY, int8_t userBlockSizeZ>
NEON_CUDA_HOST_DEVICE inline auto bIndex<dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::getInTrayIdx() -> InTrayIdx
{
    static_assert(dataBlockSizeX % userBlockSizeX == 0);
    static_assert(dataBlockSizeY % userBlockSizeY == 0);
    static_assert(dataBlockSizeZ % userBlockSizeZ == 0);

    constexpr uint32_t userBSX = static_cast<uint32_t>(userBlockSizeX);
    constexpr uint32_t userBSY = static_cast<uint32_t>(userBlockSizeY);
    constexpr uint32_t userBSZ = static_cast<uint32_t>(userBlockSizeZ);

    using Ops = Operations<dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>;

    InTrayIdx inTrayIdx;
    inTrayIdx.x = Ops::template moduleOp<userBSX>(mInDataBlockIdx.x);
    inTrayIdx.y = Ops::template moduleOp<userBSY>(mInDataBlockIdx.y);
    inTrayIdx.z = Ops::template moduleOp<userBSZ>(mInDataBlockIdx.z);

    return inTrayIdx;
}

template <int8_t dataBlockSizeX, int8_t dataBlockSizeY, int8_t dataBlockSizeZ, int8_t userBlockSizeX, int8_t userBlockSizeY, int8_t userBlockSizeZ>
NEON_CUDA_HOST_DEVICE inline auto bIndex<dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::getMicroIndex() -> MicroIndex
{
    static_assert(dataBlockSizeX % userBlockSizeX == 0);
    static_assert(dataBlockSizeY % userBlockSizeY == 0);
    static_assert(dataBlockSizeZ % userBlockSizeZ == 0);

    constexpr uint32_t userBSX = static_cast<uint32_t>(userBlockSizeX);
    constexpr uint32_t userBSY = static_cast<uint32_t>(userBlockSizeY);
    constexpr uint32_t userBSZ = static_cast<uint32_t>(userBlockSizeZ);

    using Ops = Operations<dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>;

    TrayIdx const exBlockOffset = mDataBlockIdx * (userBSX * userBSY * userBSZ);
    TrayIdx const exTrayOffset = [&]() {
        int const trayBlockIdxX = Ops::divisionOp<userBSX>(mInDataBlockIdx.x);
        int const trayBlockIdxY = Ops::divisionOp<userBSY>(mInDataBlockIdx.y);
        int const trayBlockIdxZ = Ops::divisionOp<userBSZ>(mInDataBlockIdx.z);

        constexpr int countMicroBlocksInTrayX = (dataBlockSizeX / userBlockSizeX);
        constexpr int countMicroBlocksInTrayY = (dataBlockSizeY / userBlockSizeY);
        constexpr int countMicroBlocksInTrayZ = (dataBlockSizeZ / userBlockSizeZ);

        int const res = trayBlockIdxX + trayBlockIdxY * countMicroBlocksInTrayX +
                        trayBlockIdxZ * (countMicroBlocksInTrayX * countMicroBlocksInTrayY);
        return res;
    };
    MicroIndex res;
    res.mDataBlockIdx = exBlockOffset + exTrayOffset;
    InTrayIdx inTrayIdx;
    res.mInDataBlockIdx.x = Ops::template moduleOp<userBSX>(mInDataBlockIdx.x);
    res.mInDataBlockIdx.y = Ops::template moduleOp<userBSY>(mInDataBlockIdx.y);
    res.mInDataBlockIdx.z = Ops::template moduleOp<userBSZ>(mInDataBlockIdx.z);

    return res;
}


template <int8_t dataBlockSizeX, int8_t dataBlockSizeY, int8_t dataBlockSizeZ, int8_t userBlockSizeX, int8_t userBlockSizeY, int8_t userBlockSizeZ>
NEON_CUDA_HOST_DEVICE inline auto bIndex<dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::init(MicroIndex const& microIndex) -> void
{
    static_assert(dataBlockSizeX % userBlockSizeX == 0);
    static_assert(dataBlockSizeY % userBlockSizeY == 0);
    static_assert(dataBlockSizeZ % userBlockSizeZ == 0);

    constexpr uint32_t userBSX = static_cast<uint32_t>(userBlockSizeX);
    constexpr uint32_t userBSY = static_cast<uint32_t>(userBlockSizeY);
    constexpr uint32_t userBSZ = static_cast<uint32_t>(userBlockSizeZ);

    using Ops = Operations<dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>;

    uint32_t dataBlockSize = dataBlockSizeX * dataBlockSizeY * dataBlockSizeZ;
    mDataBlockIdx = Ops::template divisionOp<dataBlockSize>(microIndex.mDataBlockIdx);

    uint32_t reminder = Ops::template moduleOp<dataBlockSize>(microIndex.mDataBlockIdx);
    mInDataBlockIdx.z = microIndex.mInDataBlockIdx.z + Ops::template divisionOp<userBSX * userBSY>(reminder);
    reminder = Ops::template moduleOp<userBSX * userBSY>(reminder);
    mInDataBlockIdx.y = microIndex.mInDataBlockIdx.y + Ops::template divisionOp<userBSX>(reminder);
    reminder = Ops::template moduleOp<userBSX>(reminder);
    mInDataBlockIdx.x = microIndex.mInDataBlockIdx.x + reminder;
    return;
}

}  // namespace Neon::domain::details::bGrid