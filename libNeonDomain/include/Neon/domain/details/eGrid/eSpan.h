#pragma once
#include "Neon/set/DevSet.h"
#include "eIndex.h"
namespace Neon::domain::details::eGrid {
/**
 * Abstraction that represents the Cell space of a partition
 * This abstraction is used by the neon lambda executor to
 * run a containers on aGrid
 */
class eSpan
{
    friend class eGrid;

   public:
    using Idx = eIndex;
    static constexpr int SpaceDim = 1;

    NEON_CUDA_HOST_DEVICE
    inline auto setAndValidate(Idx&            idx,
                               const uint32_t& x)
        const
        -> bool;

   private:
    Idx::Offset    mCount;
    Idx::Offset    mFirstIndexOffset;
    Neon::DataView mDataView;
};

}  // namespace Neon::domain::details::eGrid

#include "eSpan_imp.h"