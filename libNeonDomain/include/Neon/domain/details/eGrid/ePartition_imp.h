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
ePartition<T, C>::operator()(eIndex gidx, int cardinalityIdx) const
    -> T
{
    Offset jump = getOffset(gidx, cardinalityIdx);
    return mMem[jump];
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::operator()(eIndex gidx, int cardinalityIdx) -> T&
{
    Offset jump = getOffset(gidx, cardinalityIdx);
    return mMem[jump];
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::getNghData(eIndex gidx,
                             NghIdx nghIdx,
                             int    card)
    const -> NghData
{
    eIndex     gidxxNgh;
    const bool isValidNeighbour = isValidNgh(gidx, nghIdx, gidxxNgh);
    if (isValidNeighbour) {
        T val = this->operator()(gidxxNgh, card);
        return NghData(val, isValidNeighbour);
    }
    return NghData(isValidNeighbour);
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::getNghData(eIndex               gidx,
                             const Neon::int8_3d& ngh3dIdx,
                             int                  card)
    const -> NghData
{
    int tablePithc = (ngh3dIdx.x + mStencilRadius) +
                     (ngh3dIdx.y + mStencilRadius) * mStencilTableYPitch +
                     (ngh3dIdx.z + mStencilRadius) * mStencilTableYPitch * mStencilTableYPitch;
    NghIdx  nghIdx = mStencil3dTo1dOffset[tablePithc];
    NghData res = getNghData(gidx, nghIdx, card);

    return res;
}

template <typename T,
          int C>
template <int xOff, int yOff, int zOff>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::getNghData(eIndex gidx,
                             int    card)
    const -> NghData
{
    int tablePithc = (xOff + mStencilRadius) +
                     (yOff + mStencilRadius) * mStencilTableYPitch +
                     (zOff + mStencilRadius) * mStencilTableYPitch * mStencilTableYPitch;
    NghIdx  nghIdx = mStencil3dTo1dOffset[tablePithc];
    NghData res = getNghData(gidx, nghIdx, card);

    return res;
}

template <typename T,
          int C>
template <int xOff, int yOff, int zOff>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::getNghData(eIndex gidx,
                             int    card,
                             T      defaultVal)
    const -> NghData
{
    int tablePithc = (xOff + mStencilRadius) +
                     (yOff + mStencilRadius) * mStencilTableYPitch +
                     (zOff + mStencilRadius) * mStencilTableYPitch * mStencilTableYPitch;
    NghIdx  nghIdx = mStencil3dTo1dOffset[tablePithc];
    NghData res = getNghData(gidx, nghIdx, card);
    if (!res.isValid()) {
        res.set(defaultVal, false);
    }
    return res;
}

template <typename T,
          int C>
template <int xOff,
          int yOff,
          int zOff,
          typename LambdaVALID,
          typename LambdaNOTValid>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::getNghData(const Idx&     gidx,
                             int            card,
                             LambdaVALID    funIfValid,
                             LambdaNOTValid funIfNOTValid)
    const -> std::enable_if_t<std::is_invocable_v<LambdaVALID, T> && (std::is_invocable_v<LambdaNOTValid, T> || std::is_same_v<LambdaNOTValid, void*>), void>
{
    int tablePithc = (xOff + mStencilRadius) +
                     (yOff + mStencilRadius) * mStencilTableYPitch +
                     (zOff + mStencilRadius) * mStencilTableYPitch * mStencilTableYPitch;
    NghIdx  nghIdx = mStencil3dTo1dOffset[tablePithc];
    NghData res = getNghData(gidx, nghIdx, card);
    if (res.isValid()) {
        funIfValid(res.getData());
        return;
    }
    if constexpr (!std::is_same_v<LambdaNOTValid, void*>) {
        funIfNOTValid();
    }
    return;
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::getNghIndex(eIndex               gidx,
                              const Neon::int8_3d& ngh3dIdx,
                              eIndex&              gidxxNgh) const -> bool
{
    int tablePithc = (ngh3dIdx.x + mStencilRadius) +
                     (ngh3dIdx.y + mStencilRadius) * mStencilTableYPitch +
                     (ngh3dIdx.z + mStencilRadius) * mStencilTableYPitch * mStencilTableYPitch;
    NghIdx     nghIdx = mStencil3dTo1dOffset[tablePithc];
    eIndex     tmpEIdxNgh;
    const bool isValidNeighbour = isValidNgh(gidx, nghIdx, tmpEIdxNgh);
    if (isValidNeighbour) {
        gidxxNgh = tmpEIdxNgh;
    }
    return isValidNeighbour;
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::isValidNgh(eIndex  gidx,
                             NghIdx  nghIdx,
                             eIndex& neighbourIdx) const
    -> bool
{
    const eIndex::Offset connectivityJumo = mCountAllocated * nghIdx + gidx.helpGet();
    neighbourIdx.helpSet() = NEON_CUDA_CONST_LOAD((mConnectivity + connectivityJumo));
    const bool isValidNeighbour = (neighbourIdx.mIdx > -1);
    //    printf("(prtId %d) getNghData id %d gidxxNgh %d connectivityJumo %d\n",
    //           mPrtID,
    //           gidx.mIdx, neighbourIdx.mIdx, connectivityJumo);
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
ePartition<T, C>::pointer(eIndex gidx, int cardinalityIdx) const
    -> const Type*
{
    Offset jump = getOffset(gidx, cardinalityIdx);
    return mMem + jump;
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::getOffset(eIndex gidx, int cardinalityIdx) const
    -> Offset
{
    return Offset(gidx.helpGet() * mPitch.x + cardinalityIdx * mPitch.y);
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
