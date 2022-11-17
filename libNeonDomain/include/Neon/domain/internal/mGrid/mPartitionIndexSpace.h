#pragma once

#include "Neon/domain/internal/bGrid/bCell.h"

namespace Neon::domain::internal::bGrid {
class bPartitionIndexSpace
{
   public:
    bPartitionIndexSpace() = default;
    virtual ~bPartitionIndexSpace() = default;

    using Cell = bCell;

    friend class bGrid;

    static constexpr int SpaceDim = 1;

    NEON_CUDA_HOST_DEVICE inline auto setAndValidate(bCell&        cell,
                                                     const size_t& x,
                                                     const size_t& y,
                                                     const size_t& z) const -> bool;

   private:
    NEON_CUDA_HOST_DEVICE inline auto setCell(bCell&                         cell,
                                              [[maybe_unused]] const size_t& x,
                                              [[maybe_unused]] const size_t& y,
                                              [[maybe_unused]] const size_t& z) const -> void;

    Neon::DataView  mDataView;
    Neon::int32_3d  mDomainSize;
    int             mBlockSize;
    int             mSpacing;
    uint32_t        mNumBlocks;
    uint32_t*       mHostActiveMask;
    uint32_t*       mDeviceActiveMask;
    Neon::int32_3d* mHostBlockOrigin;
    Neon::int32_3d* mDeviceBlockOrigin;
};
}  // namespace Neon::domain::internal::bGrid

#include "Neon/domain/internal/bGrid/bPartitionIndexSpace_imp.h"