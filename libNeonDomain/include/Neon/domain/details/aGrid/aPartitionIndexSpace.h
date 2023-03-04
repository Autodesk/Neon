#pragma once
#include "Neon/set/DevSet.h"

namespace Neon::domain::internal::aGrid {

/**
 * Abstraction that represents the Cell space of a partition
 * This abstraction is used by the neon lambda executor to
 * run a containers on aGrid
 */
class aPartitionIndexSpace
{
   public:

    using Cell = aCell;
    static constexpr int SpaceDim = 1;

    inline aPartitionIndexSpace();
    inline aPartitionIndexSpace(int            nElements,
                                Neon::DataView dataView);


    NEON_CUDA_HOST_DEVICE
    inline auto setAndValidate(Cell&                          e,
                               const size_t&                  x,
                               [[maybe_unused]] const size_t& y,
                               [[maybe_unused]] const size_t& z) const -> bool;


   private:
    /**
     * Returns number of element in this partition
     * @return
     */
    NEON_CUDA_HOST_DEVICE
    inline auto numElements() const -> int;

    int            mNElements;
    Neon::DataView mDataView;
};

}  // namespace Neon::domain::array
