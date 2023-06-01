#pragma once
#include <assert.h>
#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"
#include "Neon/domain/details/eGrid/eGrid.h"
#include "Neon/domain/details/eGrid/eIndex.h"
#include "Neon/domain/interface/NghData.h"
#include "Neon/set/DevSet.h"
#include "Neon/sys/memory/CudaIntrinsics.h"
#include "cuda_fp16.h"

namespace Neon::domain::details::bGrid {

template <typename T,
          int C = 1>
class BlockViewPartition : public Neon::domain::details::eGrid::ePartition<T, C>
{
   public:
    BlockViewPartition()
    {
    }
    BlockViewPartition(Neon::domain::details::eGrid::ePartition<T, C> ePartition)
        : Neon::domain::details::eGrid::ePartition<T, C>(ePartition)
    {
    }

    template <class BlockIdexType>
    static auto getInBlockIdx(typename Neon::domain::details::eGrid::ePartition<T, C>::Idx const& idx,
                                  uint8_3d const&                                            inBlockLocation) -> BlockIdexType
    {
        BlockIdexType blockIdx(idx.helpGet(), inBlockLocation);
        return inBlockLocation;
    }

    auto getCountAllocated() const -> int32_t;
};
template <typename T, int C>
auto BlockViewPartition<T, C>::getCountAllocated() const -> int32_t
{
    return this->mCountAllocated;
}
}  // namespace Neon::domain::details::bGrid
