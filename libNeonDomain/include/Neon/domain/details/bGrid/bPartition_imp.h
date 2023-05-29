#pragma once

#include "Neon/domain/details/bGrid/bGrid.h"
#include "Neon/domain/details/bGrid/bSpan.h"

namespace Neon::domain::details::bGrid {

template <typename T, int C, uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
bPartition<T, C, memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::bPartition()
    : mCardinality(0),
      mMem(nullptr),
      mStencilNghIndex(),
      mBlockConnectivity(nullptr),
      mMask(nullptr),
      mOrigin(0),
      mSetIdx(0)
{
}

template <typename T, int C, uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
bPartition<T, C, memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::
    bPartition(int                    setIdx,
               int                    cardinality,
               T*                     mem,
               Idx::DataBlockIdx*     blockConnectivity,
               Span::BitMaskWordType* mask,
               Neon::int32_3d*        origin,
               NghIdx*                stencilNghIndex)
    : mCardinality(cardinality),
      mMem(mem),
      mStencilNghIndex(stencilNghIndex),
      mBlockConnectivity(blockConnectivity),
      mMask(mask),
      mOrigin(origin),
      mSetIdx(setIdx)
{
}

template <typename T, int C, uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C, memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::
    getGlobalIndex(const Idx& gidx)
        const -> Neon::index_3d
{
    auto location = mOrigin[gidx.mDataBlockIdx];
    location.x += gidx.mInDataBlockIdx.x;
    location.y += gidx.mInDataBlockIdx.y;
    location.z += gidx.mInDataBlockIdx.z;
    return location;
}

template <typename T, int C, uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
inline NEON_CUDA_HOST_DEVICE auto bPartition<T, C, memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::
    cardinality()
        const -> int
{
    return mCardinality;
}

template <typename T, int C, uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
inline NEON_CUDA_HOST_DEVICE auto bPartition<T, C, memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::
operator()(const Idx& cell,
           int        card) -> T&
{
    return mMem[helpGetPitch(cell, card)];
}

template <typename T, int C, uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
inline NEON_CUDA_HOST_DEVICE auto bPartition<T, C, memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::
operator()(const Idx& cell,
           int        card) const -> const T&
{
    return mMem[helpGetPitch(cell, card)];
}

template <typename T, int C, uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
inline NEON_CUDA_HOST_DEVICE auto bPartition<T, C, memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::
    helpGetPitch(const Idx& idx, int card)
        const -> uint32_t
{
    uint32_t const pitch = helpGetValidIdxPitchExplicit(idx, card);
    return pitch;
}


template <typename T, int C, uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
inline NEON_CUDA_HOST_DEVICE auto bPartition<T, C, memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::
    helpGetValidIdxPitchExplicit(const Idx& idx, int card)
        const -> uint32_t
{
    uint32_t const blockPitchByCard = memBlockSizeX * memBlockSizeY * memBlockSizeZ;
    uint32_t const inBlockInCardPitch = idx.mInDataBlockIdx.x +
                                        memBlockSizeX * idx.mInDataBlockIdx.y +
                                        (memBlockSizeX * memBlockSizeY) * idx.mInDataBlockIdx.z;
    uint32_t const blockAdnCardPitch = (idx.mDataBlockIdx * mCardinality + card) * blockPitchByCard;
    uint32_t const pitch = blockAdnCardPitch + inBlockInCardPitch;
    return pitch;
}

template <typename T, int C, uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
inline NEON_CUDA_HOST_DEVICE auto bPartition<T, C, memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::
    helpNghPitch(const Idx& nghIdx, int card)
        const -> std::tuple<bool, uint32_t>
{
    if (nghIdx.mDataBlockIdx == Span::getInvalidBlockId()) {
        return {false, 0};
    }

    bool isActive = Span::getActiveStatus(nghIdx.mDataBlockIdx,
                                          nghIdx.mInDataBlockIdx.x, nghIdx.mInDataBlockIdx.y, nghIdx.mInDataBlockIdx.z,
                                          mMask);

    if (!isActive) {
        return {false, 0};
    }
    auto const offset = helpGetValidIdxPitchExplicit(nghIdx, card);
    return {true, offset};
}

template <typename T, int C, uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C, memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::
    helpGetNghIdx(const Idx&    idx,
                  const NghIdx& offset)
        const -> Idx
{

    typename Idx::InDataBlockIdx ngh(idx.mInDataBlockIdx.x + offset.x,
                                     idx.mInDataBlockIdx.y + offset.y,
                                     idx.mInDataBlockIdx.z + offset.z);

    /**
     * 0 if no offset on the direction
     * 1 positive offset
     * -1 negative offset
     */
    const int xFlag = ngh.x < 0 ? -1 : (ngh.x >= memBlockSizeX ? +1 : 0);
    const int yFlag = ngh.y < 0 ? -1 : (ngh.y >= memBlockSizeX ? +1 : 0);
    const int zFlag = ngh.z < 0 ? -1 : (ngh.z >= memBlockSizeX ? +1 : 0);

    const bool isLocal = (xFlag | yFlag | zFlag) == 0;
    if (!(isLocal)) {
        typename Idx::InDataBlockIdx remoteInBlockOffset;
        /**
         * Example
         * - 8 block (1D case)
         * Case 1:
         * |0,1,2,3|0,1,2,3|0,1,2,3|
         *        ^     ^
         *       -3     starting point
         *
         * - idx.inBlock = 2
         * - offset = -1
         * - remote.x = (2-3) - ((-1) * 4) = -1 + 4 = 3
         * Case 2:
         * |0,1,2,3|0,1,2,3|0,1,2,3|
         *                ^     ^
         *  starting point      +3 from 3
         *
         * - idx.inBlock = 3
         * - offset = (+3,0)
         * - remote.x = (7+3) - ((+1) * 8) = 10 - 8 = 2
         *
         * |0,1,2,3|0,1,2,3|0,1,2,3|
         *  ^                   ^
         *  -3 from 0          +3 from 3
         *
         * NOTE: if in one direction the neighbour offet is zero, xFalg is 0;
         * */

        Idx remoteNghIdx;
        remoteNghIdx.mInDataBlockIdx.x = ngh.x - xFlag * memBlockSizeX;
        remoteNghIdx.mInDataBlockIdx.y = ngh.y - yFlag * memBlockSizeX;
        remoteNghIdx.mInDataBlockIdx.z = ngh.z - zFlag * memBlockSizeX;

        int connectivityJump = idx.mDataBlockIdx * 27 +
                               (xFlag + 1) +
                               (yFlag + 1) * 3 +
                               (zFlag + 1) * 9;
        remoteNghIdx.mDataBlockIdx = mBlockConnectivity[connectivityJump];

        return remoteNghIdx;
    } else {
        Idx localNghIdx;
        localNghIdx.mDataBlockIdx = idx.mDataBlockIdx;
        localNghIdx.mInDataBlockIdx = ngh;
        return localNghIdx;
    }
}

template <typename T, int C, uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C, memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::
    getNghData(const Idx& eId,
               uint8_t    nghID,
               int        card)
        const -> NghData
{
    NghIdx nghOffset = mStencilNghIndex[nghID];
    return getNghData(eId, nghOffset, card);
}

template <typename T, int C, uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C, memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::
    getNghData(const Idx&    idx,
               const NghIdx& offset,
               const int     card)
        const -> NghData
{
    NghData result;
    bIndex  nghIdx = helpGetNghIdx(idx, offset);
    auto [isValid, pitch] = helpNghPitch(nghIdx, card);
    if (!isValid) {
        result.invalidate();
        return result;
    }
    auto const value = mMem[pitch];
    result.set(value, true);
    return result;
}

}  // namespace Neon::domain::details::bGrid