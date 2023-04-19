#pragma once

#include "Neon/domain/details/bGrid/bGrid.h"
#include "Neon/domain/details/bGrid/bSpan.h"

namespace Neon::domain::details::bGrid {

template <typename T, int C>
bPartition<T, C>::bPartition()
    : mCardinality(0),
      mMem(nullptr),
      mStencilNghIndex(),
      mBlockSizeByPower(),
      mBlockConnectivity(nullptr),
      mMask(nullptr),
      mOrigin(0),
      mSetIdx(0)
{
}

template <typename T, int C>
bPartition<T, C>::
    bPartition(int                     setIdx,
               int                     cardinality,
               T*                      mem,
               uint32_3d               blockSize,
               bIndex::DataBlockIdx*   blockConnectivity,
               bSpan::BitMaskWordType* mask,
               Neon::int32_3d*         origin,
               NghIdx*                 stencilNghIndex)
    : mCardinality(cardinality),
      mMem(mem),
      mStencilNghIndex(stencilNghIndex),
      mBlockSizeByPower(blockSize),
      mBlockConnectivity(blockConnectivity),
      mMask(mask),
      mOrigin(origin),
      mSetIdx(setIdx)
{
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::
    getGlobalIndex(const Index& cell)
        const -> Neon::index_3d
{
#ifdef NEON_PLACE_CUDA_DEVICE
    auto location = mOrigin[cell.mDataBlockIdx];
    location.x += threadIdx.x;
    location.y += threadIdx.y;
    location.z += threadIdx.z;
    return location;
#else
    auto location = mOrigin[cell.mDataBlockIdx];
    location.x += cell.mInDataBlockIdx.x;
    location.y += cell.mInDataBlockIdx.y;
    location.z += cell.mInDataBlockIdx.z;
    return location;
#endif
}

template <typename T, int C>
inline NEON_CUDA_HOST_DEVICE auto bPartition<T, C>::
    cardinality()
        const -> int
{
    return mCardinality;
}

template <typename T, int C>
inline NEON_CUDA_HOST_DEVICE auto bPartition<T, C>::
operator()(const bIndex& cell,
           int           card) -> T&
{
    return mMem[helpGetPitch(cell, card)];
}

template <typename T, int C>
inline NEON_CUDA_HOST_DEVICE auto bPartition<T, C>::
operator()(const bIndex& cell,
           int           card) const -> const T&
{
    return mMem[helpGetPitch(cell, card)];
}

template <typename T, int C>
inline NEON_CUDA_HOST_DEVICE auto bPartition<T, C>::
    helpGetPitch(const Index& idx, int card)
        const -> uint32_t
{
#ifdef NEON_PLACE_CUDA_DEVICE
    uint32_t const blockPitchByCard = mBlockSizeByPower.v[2];
    uint32_t const inBlockInCardPitch = threadIdx.x +
                                        mBlockSizeByPower.v[0] * threadIdx.y +
                                        mBlockSizeByPower.v[1] * threadIdx.z;
    uint32_t const blockAdnCardPitch = (idx.mDataBlockIdx * mCardinality + card) * blockPitchByCard;
    uint32_t const pitch = blockAdnCardPitch + inBlockInCardPitch;
    return pitch;
#else
    uint32_t const pitch = helpGetValidIdxPitchExplicit(idx, card);
    return pitch;
#endif
}


template <typename T, int C>
inline NEON_CUDA_HOST_DEVICE auto bPartition<T, C>::
    helpGetValidIdxPitchExplicit(const Index& idx, int card)
        const -> uint32_t
{
    uint32_t const blockPitchByCard = mBlockSizeByPower.v[2];
    uint32_t const inBlockInCardPitch = idx.mInDataBlockIdx.x +
                                        mBlockSizeByPower.v[0] * idx.mInDataBlockIdx.y +
                                        mBlockSizeByPower.v[1] * idx.mInDataBlockIdx.z;
    uint32_t const blockAdnCardPitch = (idx.mDataBlockIdx * mCardinality + card) * blockPitchByCard;
    uint32_t const pitch = blockAdnCardPitch + inBlockInCardPitch;
    return pitch;
}

template <typename T, int C>
inline NEON_CUDA_HOST_DEVICE auto bPartition<T, C>::
    helpNghPitch(const Index& nghIdx, int card)
        const -> std::tuple<bool, uint32_t>
{
    if (nghIdx.mInDataBlockIdx == bSpan::getInvalidBlockId()) {
        return {false, 0};
    }

    bool isActive = bSpan::getActiveStatus(nghIdx.mDataBlockIdx,
                                           nghIdx.mInDataBlockIdx.x, nghIdx.mInDataBlockIdx.y, nghIdx.mInDataBlockIdx.z,
                                           mMask,
                                           mBlockSizeByPower.v[0]);

    if (!isActive) {
        return {false, 0};
    }

    helpGetValidIdxPitchExplicit(nghIdx, card);
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::
    helpGetNghIdx(const Index&  idx,
                  const NghIdx& offset)
        const -> bIndex
{
#ifdef NEON_PLACE_CUDA_DEVICE
#define NEON_BGRID_DATA_BLOCK_SIZE blockDim.x
#define NEON_BGRID_DATA_BLOCK_SIZE_POW2 blockDim.x* blockDim.x
#define NEON_BGRID_DATA_BLOCK_SIZE_POW3 blockDim.x* blockDim.x* blockDim.x
#define NEON_BGRID_DATA_BLOCK_BLOCKIDX_X idx.mDataBlockIdx
#define NEON_BGRID_DATA_BLOCK_THREADIDX_X threadIdx.x
#define NEON_BGRID_DATA_BLOCK_THREADIDX_Y threadIdx.y
#define NEON_BGRID_DATA_BLOCK_THREADIDX_Z threadIdx.z
#else
#define NEON_BGRID_DATA_BLOCK_SIZE mBlockSizeByPower.v[0]
#define NEON_BGRID_DATA_BLOCK_SIZE_POW2 mBlockSizeByPower.v[1]
#define NEON_BGRID_DATA_BLOCK_SIZE_POW3 mBlockSizeByPower.v[2]
#define NEON_BGRID_DATA_BLOCK_BLOCKIDX_X idx.mDataBlockIdx
#define NEON_BGRID_DATA_BLOCK_THREADIDX_X idx.mInDataBlockIdx.x
#define NEON_BGRID_DATA_BLOCK_THREADIDX_Y idx.mInDataBlockIdx.y
#define NEON_BGRID_DATA_BLOCK_THREADIDX_Z idx.mInDataBlockIdx.z
#endif
    bIndex::InDataBlockIdx ngh(NEON_BGRID_DATA_BLOCK_THREADIDX_X + offset.x,
                               NEON_BGRID_DATA_BLOCK_THREADIDX_Y + offset.y,
                               NEON_BGRID_DATA_BLOCK_THREADIDX_Z + offset.z);

    const int xInfo = ngh.x < 0 ? -1 : (ngh.x >= NEON_BGRID_DATA_BLOCK_SIZE ? +1 : 0);
    const int yInfo = ngh.y < 0 ? -1 : (ngh.y >= NEON_BGRID_DATA_BLOCK_SIZE ? +1 : 0);
    const int zInfo = ngh.z < 0 ? -1 : (ngh.z >= NEON_BGRID_DATA_BLOCK_SIZE ? +1 : 0);

    const bool isLocal = (xInfo | yInfo | zInfo) == 0;
    if (!(isLocal)) {
        bIndex::InDataBlockIdx remote;
        /**
         * Example
         * - 8 block (1D case)
         * Case 1:
         * |0,1,2,3|0,1,2,3|0,1,2,3|
         *  ^         ^
         *  -3        starting point
         *
         * - idx.inBlock = 0
         * - offset = -1
         * - remote.x = (0-3) - ((-1) * 4) = -3 + 4 = 1
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
         * */
        remote.x = ngh.x - xInfo * NEON_BGRID_DATA_BLOCK_SIZE;
        remote.y = ngh.y - yInfo * NEON_BGRID_DATA_BLOCK_SIZE;
        remote.z = ngh.z - zInfo * NEON_BGRID_DATA_BLOCK_SIZE;

        remote.x = xInfo == 0 ? ngh.x : remote.x;
        remote.y = yInfo == 0 ? ngh.y : remote.y;
        remote.z = zInfo == 0 ? ngh.z : remote.z;

        bIndex::DataBlockIdx remoteBlockIdx = mBlockConnectivity[(xInfo + 1) +
                                                                 (yInfo + 1) * 3 +
                                                                 (zInfo + 1) * 9];

        bIndex remoteNghIdx;
        remoteNghIdx.mDataBlockIdx = remoteBlockIdx;
        remoteNghIdx.mInDataBlockIdx = remote;
        return remoteNghIdx;
    } else {
        bIndex localNghIdx;
        localNghIdx.mDataBlockIdx = idx.mDataBlockIdx;
        localNghIdx.mInDataBlockIdx = ngh;
        return localNghIdx;
    }
#undef NEON_BGRID_DATA_BLOCK_SIZE
#undef NEON_BGRID_DATA_BLOCK_SIZE_POW2
#undef NEON_BGRID_DATA_BLOCK_SIZE_POW3
#undef NEON_BGRID_DATA_BLOCK_BLOCKIDX_X
#undef NEON_BGRID_DATA_BLOCK_THREADIDX_X
#undef NEON_BGRID_DATA_BLOCK_THREADIDX_Y
#undef NEON_BGRID_DATA_BLOCK_THREADIDX_Z
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::
    nghVal(const Index& eId,
           uint8_t      nghID,
           int          card,
           const T&     alternativeVal)
        const -> NghData
{
    NghIdx nghOffset = mStencilNghIndex[nghID];
    return nghVal(eId, nghOffset, card, alternativeVal);
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto bPartition<T, C>::
    nghVal(const Index&  idx,
           const NghIdx& offset,
           const int     card,
           const T       alternativeVal)
        const -> NghData
{
    NghData result;
    bIndex  nghIdx = helpGetNghIdx(idx, offset);
    auto [isValid, pitch] = helpNghPitch(nghIdx, card);
    if (!isValid) {
        result.set(alternativeVal, false);
        return result;
    }
    auto const value = mMem[pitch];
    result.set(value, true);
    return result;
}

}  // namespace Neon::domain::details::bGrid