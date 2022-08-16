#pragma once
#include "Neon/core/core.h"

namespace Neon::domain::internal::mGrid {
class mCell
{
   public:

    using nghIdx_t = int8_3d;
    template <typename T, int C>
    friend class mPartition;

    template <typename T, int C>
    friend class mField;

    friend class mPartitionIndexSpace;

    friend class mGrid;
};
}  // namespace Neon::domain::internal::mGrid