#pragma once

#include "Neon/core/core.h"

namespace Neon::domain::details::aGrid {

class aCell
{
   public:
    using OuterCell = aCell;

    template <typename T,
              int Cardinality>
    friend class aPartition;

    friend class aPartitionIndexSpace;
    friend class aGrid;

    // aGrid specific types
    using Location = int32_t;

    inline aCell() = default;

    NEON_CUDA_HOST_DEVICE inline explicit aCell(Location location);

   private:
    Location mLocation = 0;


    NEON_CUDA_HOST_DEVICE inline auto set() -> Location&;

    NEON_CUDA_HOST_DEVICE inline auto get() const -> const Location&;
};
}  // namespace Neon::domain::details::aGrid

#include "Neon/domain/details/aGrid/aCell_imp.h"
