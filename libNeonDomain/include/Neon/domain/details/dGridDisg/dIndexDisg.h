#pragma once

#include "Neon/core/core.h"

namespace Neon::domain::details::dissagragated::dGrid {

// Common forward declarations
class dSpan;
template <typename T, int C>
class dPartition;

struct dIndex
{
    using OuterIdx = dIndex;

    template <typename T, int C>
    friend class dPartition;
    friend dSpan;

    template <typename T,
              int Cardinality>
    friend class dField;

    // dGrid specific types
    using Offset = int32_t;
    using Location = index_3d;
    using Count = int32_t;

    dIndex() = default;
    Location mLocation = 0;
    Offset   mOffset = 0;

    NEON_CUDA_HOST_DEVICE inline explicit dIndex(Location const& location,
                                                 Offset const&   offset);

    NEON_CUDA_HOST_DEVICE inline explicit dIndex(Location::Integer const& x,
                                                 Location::Integer const& y,
                                                 Location::Integer const& z,
                                                 Offset const&            offset);

    NEON_CUDA_HOST_DEVICE inline auto setLocation() -> Location&;

    NEON_CUDA_HOST_DEVICE inline auto setOffset() -> Offset&;

    NEON_CUDA_HOST_DEVICE inline auto getLocation() const -> const Location&;

    NEON_CUDA_HOST_DEVICE inline auto getOffset() const -> const Offset&;
};

}  // namespace Neon::domain::details::dissagragated::dGrid

#include "dIndexDisg_imp.h"
