#pragma once
#include "Neon/core/core.h"
#include "Neon/domain/dGrid.h"

namespace Neon::domain::details::dissagragated::dGrid {

NEON_CUDA_HOST_DEVICE inline auto
dSpan::setAndValidate(Idx&            idx,
                         const uint32_t& x,
                         const uint32_t& y,
                         const uint32_t& z)
    const -> bool
{
    idx.setLocation().x = int(x);
    idx.setLocation().y = int(y);
    idx.setLocation().z = int(z);

    bool  isValid = idx.getLocation() < mDim;

    switch (mDataView) {
        case Neon::DataView::STANDARD: {
            idx.setLocation().z += mZHaloRadius;
            idx.setOffset() = idx.getLocation().x +
                              idx.getLocation().y * mDim.x +
                              idx.getLocation().z * mDim.x * mDim.y;
            break ;
        }
        case Neon::DataView::INTERNAL: {
            idx.setLocation().z += mZHaloRadius + mZBoundaryRadius;
            idx.setOffset() = idx.getLocation().x +
                              idx.getLocation().y * mDim.x +
                              idx.getLocation().z * mDim.x * mDim.y;
            break ;
        }
        case Neon::DataView::BOUNDARY: {
            idx.setLocation().z += idx.getLocation().z < mZBoundaryRadius
                                       ? 0
                                       : (mDim.z - 1) + (-1 * mZBoundaryRadius /* we remove zBoundaryRadius as the first zBoundaryRadius will manage the lower slices */);
            idx.setLocation().z += mZHaloRadius;
            idx.setOffset() = idx.getLocation().x +
                              idx.getLocation().y * mDim.x +
                              idx.getLocation().z * mDim.x * mDim.y;
            break ;
        }
        default: {
        }
    }
    return isValid;
}

NEON_CUDA_HOST_DEVICE inline auto
dSpan::helpGetDataView()
    const -> Neon::DataView const&
{
    return mDataView;
}

NEON_CUDA_HOST_DEVICE inline auto
dSpan::helpGetZHaloRadius()
    const -> int const&
{
    return mZHaloRadius;
}

NEON_CUDA_HOST_DEVICE inline auto
dSpan::helpGetZBoundaryRadius()
    const -> int const&
{
    return mZBoundaryRadius;
}

NEON_CUDA_HOST_DEVICE inline auto
dSpan::helpGetDim()
    const -> Neon::index_3d const&
{
    return mDim;
}

NEON_CUDA_HOST_DEVICE inline auto dSpan::helpInit(Neon::domain::details::dGrid::dSpan const& dspan) -> void
{
    mDataView = dspan.helpGetDataView();
    mZHaloRadius = dspan.helpGetZHaloRadius();
    mZBoundaryRadius = dspan.helpGetZBoundaryRadius();
    mDim = dspan.helpGetDim();
}


}  // namespace Neon::domain::details::dGridSoA