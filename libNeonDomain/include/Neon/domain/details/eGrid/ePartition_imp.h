#pragma once

#include "Neon/domain/details//eGrid/ePartition.h"

namespace Neon::domain::details::eGrid {


template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE auto
ePartition<T, C>::prtID() const
    -> int
{
    return mPrtID;
}

template <typename T,
          int C>
template <int dummy_ta>
inline NEON_CUDA_HOST_DEVICE auto
ePartition<T, C>::cardinality() const
    -> std::enable_if_t<dummy_ta == 0, int>
{
    return mCardinality;
}

template <typename T,
          int C>
template <int dummy_ta>
constexpr inline NEON_CUDA_HOST_DEVICE auto
ePartition<T, C>::cardinality() const
    -> std::enable_if_t<dummy_ta != 0, int>
{
    return C;
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::operator()(eIndex eId, int cardinalityIdx) const
    -> T
{
    Offset jump = getOffset(eId, cardinalityIdx);
    return mMem[jump];
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::operator()(eIndex eId, int cardinalityIdx) -> T&
{
    Offset jump = getOffset(eId, cardinalityIdx);
    return mMem[jump];
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::getNghData(eIndex eId,
                             NghIdx nghIdx,
                             int    card)
    const -> NghData
{
    eIndex     eIdxNgh;
    const bool isValidNeighbour = isValidNgh(eId, nghIdx, eIdxNgh);
    if (isValidNeighbour) {
        T val = this->operator()(eIdxNgh, card);
        return NghData(val, isValidNeighbour);
    }
    return NghData(isValidNeighbour);
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::getNghData(eIndex               eId,
                             const Neon::int8_3d& ngh3dIdx,
                             int                  card)
    const -> NghData
{
    int tablePithc = (ngh3dIdx.x + mStencilRadius) +
                     (ngh3dIdx.y + mStencilRadius) * mStencilTableYPitch +
                     (ngh3dIdx.z + mStencilRadius) * mStencilTableYPitch * mStencilTableYPitch;
    NghIdx  nghIdx = mStencil3dTo1dOffset[tablePithc];
    NghData res = getNghData(eId, nghIdx, card);

    return res;
}

template <typename T,
          int C>
template <int xOff, int yOff, int zOff>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::getNghData(eIndex               eId,
                             int                  card)
    const -> NghData
{
    int tablePithc = (xOff + mStencilRadius) +
                     (yOff + mStencilRadius) * mStencilTableYPitch +
                     (zOff + mStencilRadius) * mStencilTableYPitch * mStencilTableYPitch;
    NghIdx  nghIdx = mStencil3dTo1dOffset[tablePithc];
    NghData res = getNghData(eId, nghIdx, card);

    return res;
}

template <typename T,
          int C>
template <int xOff, int yOff, int zOff>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::getNghData(eIndex               eId,
                             int                  card,
                             T defaultVal)
    const -> NghData
{
    int tablePithc = (xOff + mStencilRadius) +
                     (yOff + mStencilRadius) * mStencilTableYPitch +
                     (zOff + mStencilRadius) * mStencilTableYPitch * mStencilTableYPitch;
    NghIdx  nghIdx = mStencil3dTo1dOffset[tablePithc];
    NghData res = getNghData(eId, nghIdx, card);
    if (!res.isValid()) {
        res.set(defaultVal, false);
    }
    return res;
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::getNghIndex(eIndex               eId,
                              const Neon::int8_3d& ngh3dIdx,
                              eIndex&              eIdxNgh) const -> bool
{
    int tablePithc = (ngh3dIdx.x + mStencilRadius) +
                     (ngh3dIdx.y + mStencilRadius) * mStencilTableYPitch +
                     (ngh3dIdx.z + mStencilRadius) * mStencilTableYPitch * mStencilTableYPitch;
    NghIdx     nghIdx = mStencil3dTo1dOffset[tablePithc];
    eIndex     tmpEIdxNgh;
    const bool isValidNeighbour = isValidNgh(eId, nghIdx, tmpEIdxNgh);
    if (isValidNeighbour) {
        eIdxNgh = tmpEIdxNgh;
    }
    return isValidNeighbour;
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::isValidNgh(eIndex  eId,
                             NghIdx  nghIdx,
                             eIndex& neighbourIdx) const
    -> bool
{
    const eIndex::Offset connectivityJumo = mCountAllocated * nghIdx + eId.helpGet();
    neighbourIdx.helpSet() = NEON_CUDA_CONST_LOAD((mConnectivity + connectivityJumo));
    const bool isValidNeighbour = (neighbourIdx.mIdx > -1);
    //    printf("(prtId %d) getNghData id %d eIdxNgh %d connectivityJumo %d\n",
    //           mPrtID,
    //           eId.mIdx, neighbourIdx.mIdx, connectivityJumo);
    return isValidNeighbour;
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::getGlobalIndex(eIndex eIndex) const
    -> Neon::index_3d
{
    Neon::index_3d loc;
    loc = mOrigins[eIndex.helpGet()];
    return loc;
}

template <typename T,
          int C>
ePartition<T, C>::ePartition(int             prtId,
                             T*              mem,
                             ePitch          pitch,
                             int32_t         cardinality,
                             int32_t         countAllocated,
                             Offset*         connRaw,
                             Neon::index_3d* toGlobal,
                             int8_t*         stencil3dTo1dOffset,
                             int32_t         stencilRadius)
{
    mPrtID = prtId;
    mMem = mem;
    mPitch = pitch;
    mCardinality = cardinality;
    mCountAllocated = countAllocated;

    mConnectivity = connRaw;
    mOrigins = toGlobal;

    mStencil3dTo1dOffset = stencil3dTo1dOffset;
    mStencilTableYPitch = 2 * stencilRadius + 1;

    mStencilRadius = stencilRadius;
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE auto
ePartition<T, C>::pointer(eIndex eId, int cardinalityIdx) const
    -> const Type*
{
    Offset jump = getOffset(eId, cardinalityIdx);
    return mMem + jump;
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::getOffset(eIndex eId, int cardinalityIdx) const
    -> Offset
{
    return Offset(eId.helpGet() * mPitch.x + cardinalityIdx * mPitch.y);
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::mem()
    -> T*
{
    return mMem;
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::mem() const
    -> const T*
{
    return mMem;
}

}  // namespace Neon::domain::details::eGrid
