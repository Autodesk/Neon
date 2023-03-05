#pragma once

#include "Neon/core/core.h"

namespace Neon::domain::details::dGrid {

// Common forward declarations
class eGrid;
class eSpan;
template <typename T, int C>
class ePartition;

struct eIndex
{
    using OuterCell = eIndex;

    template <typename T, int C>
    friend class ePartition;
    friend eSpan;
    friend eGrid;

    template <typename T,
              int Cardinality>
    friend class eField;

    // dGrid specific types
    using Offset = int32_t;
    using Location = index_3d;
    using Count = int32_t;

    eIndex() = default;
    Location mLocation = 0;

    NEON_CUDA_HOST_DEVICE inline explicit eIndex(const Location::Integer& x,
                                                 const Location::Integer& y,
                                                 const Location::Integer& z);

    NEON_CUDA_HOST_DEVICE inline explicit eIndex(const Location& location);

    NEON_CUDA_HOST_DEVICE inline auto set() -> Location&;

    NEON_CUDA_HOST_DEVICE inline auto get() const -> const Location&;
};

}  // namespace Neon::domain::details::dGrid

#include "eIndex_imp.h"
