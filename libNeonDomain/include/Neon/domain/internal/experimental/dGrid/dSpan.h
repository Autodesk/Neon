#pragma once
#include "Neon/set/DevSet.h"
#include "dVoxel.h"
namespace Neon::domain::internal::exp::dGrid {

/**
 * Abstraction that represents the Cell space of a partition
 * This abstraction is used by the neon lambda executor to
 * run a containers on aGrid
 */
class dSpan
{
   public:
    using Voxel = dVoxel;

    friend class dGrid;

    static constexpr int SpaceDim = 3;

    NEON_CUDA_HOST_DEVICE inline auto setAndValidate(Voxel&                         voxel,
                                                     const size_t&                  x,
                                                     [[maybe_unused]] const size_t& y,
                                                     [[maybe_unused]] const size_t& z) const
        -> bool;

   private:
    Neon::DataView mDataView;
    int            mZHaloRadius;
    int            mZBoundaryRadius;
    Neon::index_3d mDim;
};

}  // namespace Neon::domain::internal::exp::dGrid

#include "dSpan_imp.h"