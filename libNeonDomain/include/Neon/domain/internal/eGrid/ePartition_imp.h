#pragma once

#include "Neon/domain/internal/eGrid/ePartition.h"
#include "cuda_fp16.h"

namespace Neon::domain::internal::eGrid {


template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE auto
ePartition<T, C>::prtID() const
    -> int
{
    return m_prtID;
}

template <typename T,
          int C>
template <int dummy_ta>
inline NEON_CUDA_HOST_DEVICE auto
ePartition<T, C>::cardinality() const
    -> std::enable_if_t<dummy_ta == 0, int>
{
    return m_cardinality;
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
ePartition<T, C>::operator()(Cell eId, int cardinalityIdx) const
    -> T
{
    eJump_t jump = eJump(eId, cardinalityIdx);
    return m_mem[jump];
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::operator()(Cell eId, int cardinalityIdx) -> T&
{
    eJump_t jump = eJump(eId, cardinalityIdx);
    return m_mem[jump];
}

template <typename T, int C>
template <typename ComputeType>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::castRead(Cell eId,
                           int  cardinalityIdx) const -> ComputeType
{
    Type value = this->operator()(eId, cardinalityIdx);
    if constexpr (std::is_same_v<__half, Type>) {

        if constexpr (std::is_same_v<float, ComputeType>) {
            return __half2float(value);
        }
        if constexpr (std::is_same_v<double, ComputeType>) {
            return static_cast<double>(__half2float(value));
        }
    } else {
        return static_cast<ComputeType>(value);
    }
}

template <typename T, int C>
template <typename ComputeType>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::castWrite(Cell               eId,
                            int                cardinalityIdx,
                            const ComputeType& value) -> void
{

    if constexpr (std::is_same_v<__half, Type>) {
        if constexpr (std::is_same_v<float, ComputeType>) {
            this->operator()(eId, cardinalityIdx) = __float2half(value);
        }
        if constexpr (std::is_same_v<double, ComputeType>) {
            this->operator()(eId, cardinalityIdx) = __double2half(value);
        }
    } else {
        this->operator()(eId, cardinalityIdx) = static_cast<Type>(value);
    }
}

template <typename T,
          int C>
template <bool enableLDG, int shadowCardinality_ta>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::nghVal(Cell        eId,
                         nghIdx_t    nghIdx,
                         int         card,
                         const Type& alternativeVal)
    const -> std::enable_if_t<shadowCardinality_ta != 1, NghInfo<Type>>
{
    Cell       eIdxNgh;
    const bool isValidNeighbour = this->nghIdx(eId, nghIdx, eIdxNgh);
    T          val = (isValidNeighbour) ? this->operator()(eIdxNgh, card) : alternativeVal;
    return NghInfo<Type>(val, isValidNeighbour);
}


template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::nghIdx(Cell     eId,
                         nghIdx_t nghIdx,
                         Cell&    neighbourIdx) const
    -> bool
{
    const eJump_t      connJump = eJump(eId);
    const Cell::Offset connectivityPitch = m_connPitch.pMain * connJump +
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
ePartition<T, C>::globalLocation(Cell cell) const
    -> Neon::index_3d
{
    Neon::index_3d loc;
    const count_t  pitch = this->nElements();
    const auto     baseAddr = m_inverseMapping + cell.get();
    loc.x = NEON_CUDA_CONST_LOAD((baseAddr));
    loc.y = NEON_CUDA_CONST_LOAD((baseAddr + pitch));
    loc.z = NEON_CUDA_CONST_LOAD((baseAddr + 2 * pitch));
    return loc;
}

template <typename T,
          int C>
ePartition<T, C>::ePartition(const Neon::DataView&                                    dataView,
                             int                                                      prtId,
                             T*                                                       mem,
                             int                                                      cardinality,
                             const ePitch_t&                                          ePitch,
                             const std::array<Cell::Offset, ComDirection_e::COM_NUM>& bdrOff,
                             const std::array<Cell::Offset, ComDirection_e::COM_NUM>& ghostOff,
                             const std::array<Cell::Offset, ComDirection_e::COM_NUM>& bdrCount,
                             const std::array<Cell::Offset, ComDirection_e::COM_NUM>& ghostCount,
                             Cell::Offset*                                            connRaw,
                             const ePitch_t&                                          connPitch,
                             index_t*                                                 inverseMapping)
{
    m_dataView = dataView;
    m_prtID = prtId;
    m_mem = mem;
    m_cardinality = cardinality;
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
ePartition<T, C>::pointer(Cell eId, int cardinalityIdx) const
    -> const Type*
{
    eJump_t jump = eJump(eId, cardinalityIdx);
    return m_mem + jump;
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::eJump(Cell eId)
    const
    -> eJump_t
{
    return eId.get();
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::eJump(Cell eId, int cardinalityIdx) const
    -> eJump_t
{
    return eJump_t(eJump(eId) * m_ePitch.pMain +
                   cardinalityIdx * m_ePitch.pCardinality);
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::mem()
    -> T*
{
    return &m_mem[eJump(Cell(0))];
}

template <typename T,
          int C>
NEON_CUDA_HOST_DEVICE inline auto
ePartition<T, C>::mem() const
    -> const T*
{
    return &m_mem[eJump(Cell(0))];
}

}  // namespace Neon::domain::internal::eGrid
