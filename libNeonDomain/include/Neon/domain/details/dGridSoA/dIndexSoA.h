#pragma once

#include "Neon/core/core.h"
#include "Neon/domain/details/dGridSoA/dIndexSoA.h"

namespace Neon::domain::details::dGridSoA {

// Common forward declarations
class dSpanSoA;
template <typename T, int C>
class dPartitionSoA;

struct dIndexSoA
{
    using OuterIdx = dIndexSoA;

    template <typename T, int C>
    friend class dPartition;
    friend dSpanSoA;

    template <typename T,
              int Cardinality>
    friend class dField;

    // dGrid specific types
    using Offset = int32_t;
    using Location = index_3d;
    using Count = int32_t;

    dIndexSoA() = default;
    Location mLocation = 0;
    Offset   mOffset = 0;

    NEON_CUDA_HOST_DEVICE inline explicit dIndexSoA(Location const& location,
                                                    Offset const&   offset);

    NEON_CUDA_HOST_DEVICE inline explicit dIndexSoA(Location::Integer const& x,
                                                 Location::Integer const& y,
                                                 Location::Integer const& z,
                                                 Offset const&            offset);

    NEON_CUDA_HOST_DEVICE inline auto setLocation() -> Location&;

    NEON_CUDA_HOST_DEVICE inline auto setOffset() -> Offset&;

    NEON_CUDA_HOST_DEVICE inline auto getLocation() const -> const Location&;

    NEON_CUDA_HOST_DEVICE inline auto getOffset() const -> const Offset&;
};

}  // namespace Neon::domain::details::dGridSoA

#include "dIndexSoA_imp.h"
