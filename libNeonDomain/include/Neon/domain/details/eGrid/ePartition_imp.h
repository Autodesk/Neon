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
    Offset jump = eJump(eId, cardinalityIdx);
    return mMem[jump];
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::operator()(eIndex eId, int cardinalityIdx) -> T&
{
    Offset jump = eJump(eId, cardinalityIdx);
    return mMem[jump];
}

template <typename T,
          int C>
template <bool enableLDG, int CardinalitySFINE>
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
ePartition<T, C>::nghIdx(eIndex  eId,
                         NghIdx  nghIdx,
                         eIndex& neighbourIdx) const
    -> bool
{
    const Offset         connJump = eJump(eId);
    const eIndex::Offset connectivityPitch = m_connPitch.pMain * connJump +
                                             m_connPitch.pCardinality * nghIdx;
    neighbourIdx.set() = NEON_CUDA_CONST_LOAD((m_connRaw + connectivityPitch));
    const bool isValidNeighbour = (neighbourIdx.mLocation > -1);
    return isValidNeighbour;
}


template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::ePitch() const -> const ePitch_t&
{
    return m_ePitch;
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::globalLocation(eIndex eIndex) const
    -> Neon::index_3d
{
    Neon::index_3d loc;
    const count_t  pitch = this->nElements();
    const auto     baseAddr = mToGlobal + eIndex.get();
    loc.x = NEON_CUDA_CONST_LOAD((baseAddr));
    loc.y = NEON_CUDA_CONST_LOAD((baseAddr + pitch));
    loc.z = NEON_CUDA_CONST_LOAD((baseAddr + 2 * pitch));
    return loc;
}

template <typename T,
          int C>
ePartition<T, C>::ePartition(const Neon::DataView&                                      dataView,
                             int                                                        prtId,
                             T*                                                         mem,
                             int                                                        cardinality,
                             const ePitch_t&                                            ePitch,
                             const std::array<eIndex::Offset, ComDirection_e::COM_NUM>& bdrOff,
                             const std::array<eIndex::Offset, ComDirection_e::COM_NUM>& ghostOff,
                             const std::array<eIndex::Offset, ComDirection_e::COM_NUM>& bdrCount,
                             const std::array<eIndex::Offset, ComDirection_e::COM_NUM>& ghostCount,
                             eIndex::Offset*                                            connRaw,
                             const ePitch_t&                                            connPitch,
                             index_t*                                                   inverseMapping)
{
    m_dataView = dataView;
    mPrtID = prtId;
    mMem = mem;
    mCardinality = cardinality;
    m_ePitch = ePitch;
    m_bdrOff[ComDirection_e::COM_DW] = bdrOff[ComDirection_e::COM_DW];
    m_bdrOff[ComDirection_e::COM_UP] = bdrOff[ComDirection_e::COM_UP];
    m_ghostOff[ComDirection_e::COM_DW] = ghostOff[ComDirection_e::COM_DW];
    m_ghostOff[ComDirection_e::COM_UP] = ghostOff[ComDirection_e::COM_UP];

    m_bdrCount[ComDirection_e::COM_DW] = bdrCount[ComDirection_e::COM_DW];
    m_bdrCount[ComDirection_e::COM_UP] = bdrCount[ComDirection_e::COM_UP];
    m_ghostCount[ComDirection_e::COM_DW] = ghostCount[ComDirection_e::COM_DW];
    m_ghostCount[ComDirection_e::COM_UP] = ghostCount[ComDirection_e::COM_UP];

    m_connRaw = connRaw;
    m_connPitch = connPitch;
    m_inverseMapping = inverseMapping;
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE auto
ePartition<T, C>::pointer(eIndex eId, int cardinalityIdx) const
    -> const Type*
{
    Offset jump = eJump(eId, cardinalityIdx);
    return mMem + jump;
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::eJump(eIndex eId)
    const
    -> Offset
{
    return eId.get();
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::eJump(eIndex eId, int cardinalityIdx) const
    -> Offset
{
    return Offset(eJump(eId) * m_ePitch.pMain +
                  cardinalityIdx * m_ePitch.pCardinality);
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::mem()
    -> T*
{
    return &mMem[eJump(eIndex(0))];
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::mem() const
    -> const T*
{
    return &mMem[eJump(eIndex(0))];
}

}  // namespace Neon::domain::details::eGrid

#include "Neon/domain/internal/eGrid/ePartition.h"

namespace Neon::domain::internal::eGrid {


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
    Offset jump = eJump(eId, cardinalityIdx);
    return mMem[jump];
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::operator()(eIndex eId, int cardinalityIdx) -> T&
{
    Offset jump = eJump(eId, cardinalityIdx);
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
ePartition<T, C>::nghIdx(eIndex  eId,
                         NghIdx  nghIdx,
                         eIndex& neighbourIdx) const
    -> bool
{
    const Offset         connJump = eJump(eId);
    const eIndex::Offset connectivityPitch = m_connPitch.pMain * connJump +
                                             m_connPitch.pCardinality * nghIdx;
    neighbourIdx.set() = NEON_CUDA_CONST_LOAD((m_connRaw + connectivityPitch));
    const bool isValidNeighbour = (neighbourIdx.mLocation > -1);
    return isValidNeighbour;
}


template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::ePitch() const -> const ePitch_t&
{
    return m_ePitch;
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::globalLocation(eIndex eIndex) const
    -> Neon::index_3d
{
    Neon::index_3d loc;
    const count_t  pitch = this->nElements();
    const auto     baseAddr = m_inverseMapping + eIndex.get();
    loc.x = NEON_CUDA_CONST_LOAD((baseAddr));
    loc.y = NEON_CUDA_CONST_LOAD((baseAddr + pitch));
    loc.z = NEON_CUDA_CONST_LOAD((baseAddr + 2 * pitch));
    return loc;
}

template <typename T,
          int C>
ePartition<T, C>::ePartition(const Neon::DataView&                                      dataView,
                             int                                                        prtId,
                             T*                                                         mem,
                             int                                                        cardinality,
                             const ePitch_t&                                            ePitch,
                             const std::array<eIndex::Offset, ComDirection_e::COM_NUM>& bdrOff,
                             const std::array<eIndex::Offset, ComDirection_e::COM_NUM>& ghostOff,
                             const std::array<eIndex::Offset, ComDirection_e::COM_NUM>& bdrCount,
                             const std::array<eIndex::Offset, ComDirection_e::COM_NUM>& ghostCount,
                             eIndex::Offset*                                            connRaw,
                             const ePitch_t&                                            connPitch,
                             index_t*                                                   inverseMapping)
{
    m_dataView = dataView;
    mPrtID = prtId;
    mMem = mem;
    mCardinality = cardinality;
    m_ePitch = ePitch;
    m_bdrOff[ComDirection_e::COM_DW] = bdrOff[ComDirection_e::COM_DW];
    m_bdrOff[ComDirection_e::COM_UP] = bdrOff[ComDirection_e::COM_UP];
    m_ghostOff[ComDirection_e::COM_DW] = ghostOff[ComDirection_e::COM_DW];
    m_ghostOff[ComDirection_e::COM_UP] = ghostOff[ComDirection_e::COM_UP];

    m_bdrCount[ComDirection_e::COM_DW] = bdrCount[ComDirection_e::COM_DW];
    m_bdrCount[ComDirection_e::COM_UP] = bdrCount[ComDirection_e::COM_UP];
    m_ghostCount[ComDirection_e::COM_DW] = ghostCount[ComDirection_e::COM_DW];
    m_ghostCount[ComDirection_e::COM_UP] = ghostCount[ComDirection_e::COM_UP];

    m_connRaw = connRaw;
    m_connPitch = connPitch;
    m_inverseMapping = inverseMapping;
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE auto
ePartition<T, C>::pointer(eIndex eId, int cardinalityIdx) const
    -> const Type*
{
    Offset jump = eJump(eId, cardinalityIdx);
    return mMem + jump;
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::eJump(eIndex eId)
    const
    -> Offset
{
    return eId.get();
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::eJump(eIndex eId, int cardinalityIdx) const
    -> Offset
{
    return Offset(eJump(eId) * m_ePitch.pMain +
                  cardinalityIdx * m_ePitch.pCardinality);
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::mem()
    -> T*
{
    return &mMem[eJump(eIndex(0))];
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::mem() const
    -> const T*
{
    return &mMem[eJump(eIndex(0))];
}

}  // namespace Neon::domain::internal::eGrid