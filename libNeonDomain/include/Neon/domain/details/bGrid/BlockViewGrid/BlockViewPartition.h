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
    BlockViewPartition(Neon::domain::details::eGrid::ePartition<T, C> ePartition)
        : Neon::domain::details::eGrid::ePartition<T, C>(ePartition)
    {
    }

};
}  // namespace Neon::domain::details::bGrid
