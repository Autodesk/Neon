#pragma once

#include "Neon/core/core.h"

namespace Neon::domain::details::ncclGrid {

// Common forward declarations
class ncclGrid;
class ncclSpan;
template <typename T, int C>
class ncclPartition;

struct ncclIndex
{
    using OuterIdx = ncclIndex;

    template <typename T, int C>
    friend class ncclPartition;
    friend ncclSpan;
    friend ncclGrid;

    template <typename T,
              int Cardinality>
    friend class ncclField;

    // ncclGrid specific types
    using Offset = int32_t;
    using Location = index_3d;
    using Count = int32_t;

    ncclIndex() = default;
    Location mLocation = 0;

    NEON_CUDA_HOST_DEVICE inline explicit ncclIndex(const Location::Integer& x,
                                                    const Location::Integer& y,
                                                    const Location::Integer& z);

    NEON_CUDA_HOST_DEVICE inline explicit ncclIndex(const Location& location);

    NEON_CUDA_HOST_DEVICE inline auto setLocation() -> Location&;

    NEON_CUDA_HOST_DEVICE inline auto getLocation() const -> const Location&;
};

}  // namespace Neon::domain::details::ncclGrid

#include "ncclIndex_imp.h"
