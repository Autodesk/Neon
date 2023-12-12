#pragma once
#include "Neon/set/DevSet.h"
#include "dIndexSoA.h"
#include "Neon/domain/details/dGrid/dSpan.h"

namespace Neon::domain::details::dGridSoA {

/**
 * Abstraction that represents the Cell space of a partition
 * This abstraction is used by the neon lambda executor to
 * run a containers on aGrid
 */
class dSpanSoA
{
   public:
    using Idx = dIndexSoA;

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

    NEON_CUDA_HOST_DEVICE inline auto
    helpInit(Neon::domain::details::dGrid::dSpan const&) ->void;

   private:
    Neon::DataView mDataView;
    int            mZHaloRadius;
    int            mZBoundaryRadius;
    Neon::index_3d mDim /** Dimension of the span, its values depends on the mDataView*/;
};

}  // namespace Neon::domain::details::dGrid

#include "dSpanSoA_imp.h"