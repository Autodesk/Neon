#pragma once
#include "Neon/set/DevSet.h"
#include "eIndex.h"
namespace Neon::domain::details::eGrid {

/**
 * Abstraction that represents the Cell space of a partition
 * This abstraction is used by the neon lambda executor to
 * run a containers on aGrid
 */
class eSpan
{
    friend class eGrid;

   public:
    using Idx = eIndex;
    static constexpr int SpaceDim = 1;

    NEON_CUDA_HOST_DEVICE
    inline auto setAndValidate(Idx&          idx,
                               const size_t& x)
        const
        -> bool;

    NEON_CUDA_HOST_DEVICE
    inline auto nElements() const -> int64_t;

    NEON_CUDA_HOST_DEVICE
    inline auto
    helpApplyDataViewShift(Idx& cell) const -> void;

    NEON_CUDA_HOST_DEVICE
    inline auto helpGetBoundaryOffset() -> Idx::Offset*;


    NEON_CUDA_HOST_DEVICE
    inline auto helpGetGhostOffset() -> Idx::Offset*;

    NEON_CUDA_HOST_DEVICE
    inline auto helpGetDataView() -> Neon::DataView&;


   private:
    Idx::Offset    m_ghostOff[ComDirectionUtils::toInt(ComDirection::NUM)] = {-1, -1};
    Idx::Offset    m_bdrOff[ComDirectionUtils::toInt(ComDirection::NUM)] = {-1, -1};
    Neon::DataView m_dataView;
};

}  // namespace Neon::domain::details::eGrid

#include "eSpan_imp.h"