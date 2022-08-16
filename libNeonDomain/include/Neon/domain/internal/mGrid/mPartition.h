#pragma once

#include "Neon/domain/interface/NghInfo.h"
#include "Neon/domain/internal/mGrid/mCell.h"

namespace Neon::domain::internal::mGrid {

class mPartitionIndexSpace;

template <typename T, int C = 0>
class mPartition
{
   public:
    using PartitionIndexSpace = mPartitionIndexSpace;
    using Cell = mCell;
    using nghIdx_t = Cell::nghIdx_t;
    using Type = T;

   public:
    mPartition();

    ~mPartition() = default;

   private:
};
}  // namespace Neon::domain::internal::mGrid