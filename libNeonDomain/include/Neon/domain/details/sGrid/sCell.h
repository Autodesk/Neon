#pragma once

#include "Neon/core/core.h"

namespace Neon::domain::details::sGrid {

/**
 * Cell abstraction for the sGrid
 */
struct sCell
{
    using OuterCell = sCell;

    friend struct sPartitionIndexSpace;

    template <typename OuterGridT,
              typename T,
              int Cardinality>
    friend struct sPartition;

    template <typename OuterGridT>
    friend class sGrid;

    // sGrid specific types
    using Offset = int32_t;
    using Location = int32_t;

    sCell() = default;

   private:
    Location mLocation = 0;

    /**
     * Private constructor
     * @param location
     */
    NEON_CUDA_HOST_DEVICE inline explicit sCell(Location location);

    /**
     * Method to access the information used to track a cell in the data structure.
     * For sGrid this is just a offset.
     * @return
     */
    NEON_CUDA_HOST_DEVICE inline auto get() -> Location&;

    /**
     * Method to access the information used to track a cell in the data structure
     * For sGrid this is just a offset.
     */
    NEON_CUDA_HOST_DEVICE inline auto get() const -> const Location&;
};
}  // namespace Neon::domain::details::sGrid

#include "Neon/domain/details/sGrid/sCell_imp.h"
