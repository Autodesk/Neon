#pragma once

#include "Neon/core/core.h"

namespace Neon::domain::details::disaggregated::dGrid {

// Common forward declarations
class dGrid;
class dSpan;
template <typename T, int C>
class dPartition;

enum class RegionId
{
    UpGhost,
    UpBoundary,
    Internal,
    DwBoundary,
    DwGhost,
};

struct dIndex
{
    using OuterIdx = dIndex;

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
    size_t   mOffsetLocalNoCard = 0;
    int32_t  mRegionFirstZ;
    int32_t  mRegionZDim;

    //    NEON_CUDA_HOST_DEVICE inline explicit dIndex(const Location::Integer& x,
    //                                                 const Location::Integer& y,
    //                                                 const Location::Integer& z);

    //    NEON_CUDA_HOST_DEVICE inline explicit dIndex(const Location& location);
    //
    NEON_CUDA_HOST_DEVICE inline auto setLocation() -> Location&;
    NEON_CUDA_HOST_DEVICE inline auto getLocation() const -> const Location&;

    NEON_CUDA_HOST_DEVICE inline auto getOffsetLocalNoCard() const -> size_t;
    NEON_CUDA_HOST_DEVICE inline auto setOffsetLocalNoCard(size_t xyzOffset) -> void;

    NEON_CUDA_HOST_DEVICE inline auto getRegionFirstZ() const -> int32_t;
    NEON_CUDA_HOST_DEVICE inline auto setRegionFirstZ(int32_t mRegionFirstZ) -> void;

    NEON_CUDA_HOST_DEVICE inline auto getRegionZDim() const -> int32_t;
    NEON_CUDA_HOST_DEVICE inline auto setRegionZDim(int32_t mRegionFirstZ) -> void;
};


}  // namespace Neon::domain::details::disaggregated::dGrid

#include "dIndex_imp.h"
