#pragma once

#include "Neon/core/core.h"

namespace Neon::domain::details::dGrid {

// Common forward declarations
class dGrid;
class dSpan;
template <typename T, int C>
class dPartition;

struct dIndex
{
    using OuterCell = dIndex;

    template <typename T, int C>
    friend class dPartition;
    friend dSpan;
    friend dGrid;

    template <typename T,
              int Cardinality>
    friend class dField;

    // dGrid specific types
    using Offset = int32_t;
    using Location = index_3d;
    using Count = int32_t;

    dIndex() = default;
    Location mLocation = 0;

    NEON_CUDA_HOST_DEVICE inline explicit dIndex(const Location::Integer& x,
                                                 const Location::Integer& y,
                                                 const Location::Integer& z);

    NEON_CUDA_HOST_DEVICE inline explicit dIndex(const Location& location);

    NEON_CUDA_HOST_DEVICE inline auto set() -> Location&;

    NEON_CUDA_HOST_DEVICE inline auto get() const -> const Location&;
};

}  // namespace Neon::domain::details::dGrid

#include "dIndex_imp.h"
