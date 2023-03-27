#pragma once

#include "Neon/domain/details/bGrid/bCell.h"

namespace Neon::domain::details::bGrid {
class bSpan
{
   public:
    bSpan() = default;
    virtual ~bSpan() = default;

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
}  // namespace Neon::domain::details::bGrid

#include "Neon/domain/details/bGrid/bSpan_imp.h"