#pragma once

#include "Neon/core/core.h"

namespace Neon::domain::internal::eGrid {

struct eCell
{
    using OuterCell = eCell;

    friend struct ePartitionIndexSpace;

    template <typename T,
              int Cardinality>
    friend struct ePartition;

    friend class eGrid;

    template <typename T,
              int Cardinality>
    friend class eFieldDevice_t;

    // eGrid specific types
    using Offset = int32_t;
    using Location = int32_t;
    using Count = int32_t;
    using ePitch_t = Neon::index64_2d;


    eCell() = default;

   private:
    Location mLocation = 0;

    NEON_CUDA_HOST_DEVICE inline explicit eCell(Location location);

    NEON_CUDA_HOST_DEVICE inline auto set() -> Location&;

    NEON_CUDA_HOST_DEVICE inline auto get() -> const Location&;
};


}  // namespace Neon::domain::internal::eGrid

#include "eCell_imp.h"
