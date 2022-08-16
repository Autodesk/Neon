#pragma once

#include "Neon/domain/internal/mGrid/mCell.h"

namespace Neon::domain::internal::mGrid {

class mPartitionIndexSpace
{
   public:
    using Cell = mCell;

    friend class mGrid;

    static constexpr int SpaceDim = 1;

    mPartitionIndexSpace() = default;
    virtual ~mPartitionIndexSpace() = default;


    NEON_CUDA_HOST_DEVICE inline auto setAndValidate(Cell&         cell,
                                                     const size_t& x,
                                                     const size_t& y,
                                                     const size_t& z) const -> bool;

   private:
};

}  // namespace Neon::domain::internal::mGrid