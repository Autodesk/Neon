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
ePartition<T, C>::cardinality() const -> std::enable_if_t<dummy_ta != 0, int>
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
    const bool isValidNeighbour = this->nghIdx(eId, nghIdx, eIdxNgh);
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
    const Offset         connJump = getOffset(eId);
    const eIndex::Offset connectivityPitch = mConnPitch.pMain * connJump +
                                             mConnPitch.pCardinality * nghIdx;
    neighbourIdx.set() = NEON_CUDA_CONST_LOAD((mConnRaw + connectivityPitch));
    const bool isValidNeighbour = (neighbourIdx.mIdx > -1);
    return isValidNeighbour;
}


template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::getPitch() const -> const ePitch&
{
    return mPitch;
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::getGlobalIndex(eIndex eIndex) const
    -> Neon::index_3d
{
    Neon::index_3d loc;
    const Count    pitch = this->nElements();
    const auto     baseAddr = mToGlobal + eIndex.get();
    loc.x = NEON_CUDA_CONST_LOAD((baseAddr));
    loc.y = NEON_CUDA_CONST_LOAD((baseAddr + pitch));
    loc.z = NEON_CUDA_CONST_LOAD((baseAddr + 2 * pitch));
    return loc;
}

template <typename T,
          int C>
ePartition<T, C>::ePartition(const Neon::DataView&                                                       dataView,
                             int                                                                         prtId,
                             T*                                                                          mem,
                             int                                                                         cardinality,
                             const ePitch&                                                               ePitch,
                             const std::array<Idx::Offset, ComDirectionUtils::toInt(ComDirection::NUM)>& bdrOff,
                             const std::array<Idx::Offset, ComDirectionUtils::toInt(ComDirection::NUM)>& ghostOff,
                             const std::array<Idx::Offset, ComDirectionUtils::toInt(ComDirection::NUM)>& bdrCount,
                             const std::array<Idx::Offset, ComDirectionUtils::toInt(ComDirection::NUM)>& ghostCount,
                             Offset*                                                                     connRaw,
                             const eIndex::ePitch&                                                       connPitch,
                             Neon::index_3d*                                                             inverseMapping)
{
    m_dataView = dataView;
    mPrtID = prtId;
    mMem = mem;
    mCardinality = cardinality;
    mPitch = ePitch;
    mBdrOff[ComDirectionUtils::toInt(ComDirection::DW)] = bdrOff[ComDirectionUtils::toInt(ComDirection::DW)];
    mBdrOff[ComDirectionUtils::toInt(ComDirection::UP)] = bdrOff[ComDirectionUtils::toInt(ComDirection::UP)];
    mGhostOff[ComDirectionUtils::toInt(ComDirection::DW)] = ghostOff[ComDirectionUtils::toInt(ComDirection::DW)];
    mGhostOff[ComDirectionUtils::toInt(ComDirection::UP)] = ghostOff[ComDirectionUtils::toInt(ComDirection::UP)];

    mBdrCount[ComDirectionUtils::toInt(ComDirection::DW)] = bdrCount[ComDirectionUtils::toInt(ComDirection::DW)];
    mBdrCount[ComDirectionUtils::toInt(ComDirection::UP)] = bdrCount[ComDirectionUtils::toInt(ComDirection::UP)];
    mGhostCount[ComDirectionUtils::toInt(ComDirection::DW)] = ghostCount[ComDirectionUtils::toInt(ComDirection::DW)];
    mGhostCount[ComDirectionUtils::toInt(ComDirection::UP)] = ghostCount[ComDirectionUtils::toInt(ComDirection::UP)];

    mConnRaw = connRaw;
    mConnPitch = connPitch;
    mToGlobal = inverseMapping;
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
ePartition<T, C>::getOffset(eIndex eId)
    const
    -> Offset
{
    return eId.get();
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::getOffset(eIndex eId, int cardinalityIdx) const
    -> Offset
{
    return Offset(getOffset(eId) * mPitch.pMain +
                  cardinalityIdx * mPitch.pCardinality);
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::mem()
    -> T*
{
    return &mMem[getOffset(eIndex(0))];
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::mem() const
    -> const T*
{
    return &mMem[getOffset(eIndex(0))];
}

}  // namespace Neon::domain::details::eGrid
