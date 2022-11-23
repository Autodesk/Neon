#pragma once

#include "Neon/domain/internal/sGrid/sPartition.h"
#include "cuda_fp16.h"

namespace Neon::domain::internal::sGrid {


template <typename OuterGridT, typename T, int C>
sPartition<OuterGridT, T, C>::sPartition(const Neon::DataView&                      dataView,
                                         int                                        prtId,
                                         T*                                         mem,
                                         int                                        cardinality,
                                         const Pitch&                               ePitch,
                                         typename OuterGrid::Cell::OuterCell const* tableToOuterCell)
{
    mDataView = dataView;
    mPartitionId = prtId;
    mMemory = mem;
    mCardinality = cardinality;
    mPitch = ePitch;
    mTableToOuterCell = tableToOuterCell;
}

template <typename OuterGridT, typename T, int C>
NEON_CUDA_HOST_DEVICE auto sPartition<OuterGridT, T, C>::getPartitionId() const
    -> int
{
    return mPartitionId;
}

template <typename OuterGridT, typename T, int C>
template <int dummy_ta>
NEON_CUDA_HOST_DEVICE inline auto
sPartition<OuterGridT, T, C>::cardinality() const
    -> std::enable_if_t<dummy_ta == 0, int>
{
    return mCardinality;
}

template <typename OuterGridT, typename T, int C>
template <int dummy_ta>
NEON_CUDA_HOST_DEVICE constexpr inline auto
sPartition<OuterGridT, T, C>::cardinality() const
    -> std::enable_if_t<dummy_ta != 0, int>
{
    return C;
}

template <typename OuterGridT, typename T, int C>
NEON_CUDA_HOST_DEVICE auto
sPartition<OuterGridT, T, C>::helpGetJump(Cell const& eId) const
    -> Jump
{
    return eId.get();
}

template <typename OuterGridT, typename T, int C>
NEON_CUDA_HOST_DEVICE auto
sPartition<OuterGridT, T, C>::helpGetJump(Cell const& eId,
                                          int         cardinalityIdx) const
    -> Jump
{
    return Jump(helpGetJump(eId) * mPitch.pMain +
                cardinalityIdx * mPitch.pCardinality);
}

template <typename OuterGridT, typename T, int C>
NEON_CUDA_HOST_DEVICE auto
sPartition<OuterGridT, T, C>::operator()(Cell const& eId,
                                         int         cardinalityIdx) const
    -> T
{
    Jump jump = helpGetJump(eId, cardinalityIdx);
    return mMemory[jump];
}

template <typename OuterGridT, typename T, int C>
NEON_CUDA_HOST_DEVICE auto
sPartition<OuterGridT, T, C>::operator()(Cell const& eId,
                                         int         cardinalityIdx) -> T&
{
    Jump jump = helpGetJump(eId, cardinalityIdx);
    return mMemory[jump];
}

template <typename OuterGridT, typename T, int C>
template <typename ComputeType>
NEON_CUDA_HOST_DEVICE inline auto
sPartition<OuterGridT, T, C>::castRead(Cell eId,
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

template <typename OuterGridT, typename T, int C>
template <typename ComputeType>
NEON_CUDA_HOST_DEVICE inline auto
sPartition<OuterGridT, T, C>::castWrite(Cell               eId,
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

template <typename OuterGridT, typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto
sPartition<OuterGridT, T, C>::mapToOuterGrid(const sPartition::Cell& cell) const
    -> typename OuterGrid::Cell::OuterCell const&
{
    return mTableToOuterCell[cell.get()];
}

}  // namespace Neon::domain::internal::sGrid