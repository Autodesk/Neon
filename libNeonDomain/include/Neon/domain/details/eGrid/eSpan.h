#pragma once
#include "Neon/set/DevSet.h"
#include "eIndex.h"
namespace Neon::domain::details::dGrid {

/**
 * Abstraction that represents the Cell space of a partition
 * This abstraction is used by the neon lambda executor to
 * run a containers on aGrid
 */
class eSpan
{
   public:
    using Idx = eIndex;
    friend class eGrid;

    static constexpr int SpaceDim = 3;

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
    int            mZHaloRadius;
    int            mZBoundaryRadius;
    Neon::index_3d mDim /** Dimension of the span, its values depends on the mDataView*/;
};

}  // namespace Neon::domain::details::dGrid

#include "eSpan_imp.h"