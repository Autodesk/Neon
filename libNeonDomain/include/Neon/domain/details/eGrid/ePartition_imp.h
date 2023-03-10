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
template <int CardinalitySFINE>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::nghVal(eIndex      eId,
                         NghIdx      nghIdx,
                         int         card,
                         const Type& alternativeVal)
    const -> std::enable_if_t<CardinalitySFINE != 1, NghData<Type>>
{
    eIndex     eIdxNgh;
    const bool isValidNeighbour = isValidNgh(eId, nghIdx, eIdxNgh);
    T          val = (isValidNeighbour) ? this->operator()(eIdxNgh, card) : alternativeVal;
    return NghData<Type>(val, isValidNeighbour);
}


template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::isValidNgh(eIndex  eId,
                             NghIdx  nghIdx,
                             eIndex& neighbourIdx) const
    -> bool
{
    const eIndex::Offset connectivityJumo = mCountAllocated * nghIdx + eId.get();
    neighbourIdx.set() = NEON_CUDA_CONST_LOAD((mConnectivity + connectivityJumo));
    const bool isValidNeighbour = (neighbourIdx.mIdx > -1);
    return isValidNeighbour;
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::getGlobalIndex(eIndex eIndex) const
    -> Neon::index_3d
{
    Neon::index_3d loc;
    const auto     baseAddr = mOrigins + eIndex.get();
    loc = mOrigins[eIndex.get()];
    return loc;
}

template <typename T,
          int C>
ePartition<T, C>::ePartition(int                   prtId,
                             T*                    mem,
                             int32_t               cardinality,
                             int32_t               countAllocated,
                             Offset*               connRaw,
                             Neon::index_3d*       toGlobal)
{
    mPrtID = prtId;
    mMem = mem;
    mCardinality = cardinality;
    mCountAllocated = countAllocated;

    mConnectivity = connRaw;
    mOrigins = toGlobal;
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
    return Offset(eId.get() + cardinalityIdx * mCountAllocated);
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
