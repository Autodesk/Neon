#pragma once

#include "Neon/core/core.h"
#include "eCell.h"
#include "eCommon.h"

namespace Neon::domain::internal::eGrid {

struct ePartitionIndexSpace
{
    friend class eGrid;

   public:
    using Cell = eCell;
    static constexpr int SpaceDim = 1;

    NEON_CUDA_HOST_DEVICE
    inline auto setAndValidate(Cell&                          cell,
                               const size_t&                  x,
                               [[maybe_unused]] const size_t& y,
                               [[maybe_unused]] const size_t& z)
        const
        -> bool;

    NEON_CUDA_HOST_DEVICE
    inline auto nElements() const -> int64_t;

    NEON_CUDA_HOST_DEVICE
    inline auto
    hApplyDataViewShift(Cell& cell) const -> void;

    NEON_CUDA_HOST_DEVICE
    inline auto hGetBoundaryOffset() -> Cell::Offset*;


    NEON_CUDA_HOST_DEVICE
    inline auto hgetGhostOffset() -> Cell::Offset*;

    NEON_CUDA_HOST_DEVICE
    inline auto hGetDataView() -> Neon::DataView&;


   private:
    Cell::Offset   m_ghostOff[ComDirection_e::COM_NUM] = {-1, -1};
    Cell::Offset   m_bdrOff[ComDirection_e::COM_NUM] = {-1, -1};
    Neon::DataView m_dataView;
};
}  // namespace Neon::domain::internal::eGrid

#include "ePartitionIndexSpace_imp.h"
