#pragma once
#include "Neon/set/DevSet.h"
#include "dIndex.h"
#include "Neon/set/ExecutionThreadSpan.h"

namespace Neon::domain::details::dGrid {

/**
 * Abstraction that represents the Cell space of a partition
 * This abstraction is used by the neon lambda executor to
 * run a containers on aGrid
 */
class dSpan
{
   public:
    using Idx = dIndex;
    friend class dGrid;

    static constexpr Neon::set::details::ExecutionThreadSpan executionThreadSpan = Neon::set::details::ExecutionThreadSpan::d3;
    using ExecutionThreadSpanIndexType = int32_t;


    NEON_CUDA_HOST_DEVICE inline auto
    setAndValidate(Idx&            idx,
                   const uint32_t& x,
                   const uint32_t& y,
                   const uint32_t& z) const
        -> bool;

    NEON_CUDA_HOST_DEVICE inline auto
    helpGetDataView()
        const -> Neon::DataView const&;

    NEON_CUDA_HOST_DEVICE inline auto
    helpGetZHaloRadius()
        const -> int const&;

    NEON_CUDA_HOST_DEVICE inline auto
    helpGetZBoundaryRadius()
        const -> int const&;

    NEON_CUDA_HOST_DEVICE inline auto
    helpGetDim()
        const -> Neon::index_3d const&;

   private:
    Neon::DataView mDataView;
    int            mZghostRadius;
    int            mZboundaryRadius;
    int            mMaxZInDomain;
    Neon::index_3d mSpanDim /** Dimension of the span, its values depends on the mDataView*/;
};

}  // namespace Neon::domain::deta  ils::dGrid

#include "dSpan_imp.h"