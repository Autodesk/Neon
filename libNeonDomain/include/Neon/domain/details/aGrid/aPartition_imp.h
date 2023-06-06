#pragma once

#include "aPartition.h"

namespace Neon::domain::details::aGrid {

template <typename T, int C>
aPartition<T, C>::aPartition(const prt_idx& prtId,
                             Type*          mem,
                             const pitch_t& pitch,
                             const count_t& nElements,
                             const int      cardinality)
{
    m_prtID = prtId;
    m_mem = mem;
    m_pitch = pitch;
    m_nElements = nElements;
    m_cardinality = cardinality;
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto
aPartition<T, C>::prtID() const -> const int&
{
    return m_prtID;
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto
aPartition<T, C>::eJump(const Cell& cell,
                        const int&  cardinalityIdx) const
    -> eJump_t
{
    return static_cast<eJump_t>(cell.get() * m_pitch.pMain + cardinalityIdx * m_pitch.pCardinality);
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto
aPartition<T, C>::mem()
    -> T*
{
    return m_mem;
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto
aPartition<T, C>::mem() const
    -> const T*
{
    return m_mem;
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto
aPartition<T, C>::operator()(Cell eId, int cardinalityIdx)
    -> T&
{
    auto pitch = eJump(eId, cardinalityIdx);
    return m_mem[pitch];
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto
aPartition<T, C>::operator()(Cell eId, int cardinalityIdx)
    const -> const T&
{
    const auto elementPitch = eJump(eId, cardinalityIdx);
    return m_mem[elementPitch];
}


template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto
aPartition<T, C>::nElements()
    const
    -> count_t
{
    return m_nElements;
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto
aPartition<T, C>::pitch() const
    -> const pitch_t&
{
    return m_pitch;
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto
aPartition<T, C>::cardinality() const
    -> int
{
    return m_cardinality;
}

}  // namespace Neon::domain::array