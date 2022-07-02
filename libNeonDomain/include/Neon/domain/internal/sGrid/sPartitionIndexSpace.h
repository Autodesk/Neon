#pragma once
#include "Neon/set/DevSet.h"

namespace Neon::domain::internal::sGrid {

/**
 * Abstraction that represents the Cell space of a partition
 * This abstraction is used by the neon lambda executor to
 * run a containers on aGrid
 */
class sPartitionIndexSpace
{
   public:

    template <typename OuterGrid>
    friend class sGrid;

    using Cell = sCell;
    static constexpr int SpaceDim = 1;

    NEON_CUDA_HOST_DEVICE
    inline auto setAndValidate(Cell&                          cell,
                               const size_t&                  x,
                               [[maybe_unused]] const size_t& y,
                               [[maybe_unused]] const size_t& z)const
        -> bool;

   private:

    NEON_CUDA_HOST_DEVICE
    inline auto nElements() const -> int64_t;

    NEON_CUDA_HOST_DEVICE
    inline auto
    helpApplyDataViewShift(Cell& cell) const
        -> void;

    NEON_CUDA_HOST_DEVICE
    inline auto helpGetBoundaryOffset() const
        -> Cell::Offset;


    NEON_CUDA_HOST_DEVICE
    inline auto helpGetGhostOffset() const
        -> Cell::Offset;

    NEON_CUDA_HOST_DEVICE
    inline auto helpGetDataView() const
        -> Neon::DataView;

    inline auto helpGetBoundaryOffset()
        -> Cell::Offset&;

    inline auto helpGetGhostOffset()
        -> Cell::Offset&;

    inline auto helpGetDataView()
        -> Neon::DataView&;


    Cell::Offset   mGhostOff = {-1};
    Cell::Offset   mBdrOff = {-1};
    Neon::DataView mDataView;
};

}  // namespace Neon::domain::internal::sGrid
