#pragma once

namespace Neon::domain::details::dGrid {

NEON_CUDA_HOST_DEVICE inline auto
dSpan::setAndValidate(Idx&            idx,
                      const uint32_t& x,
                      const uint32_t& y,
                      const uint32_t& z)
    const -> bool
{
    bool res = false;
    idx.setLocation().x = int(x);
    idx.setLocation().y = int(y);
    idx.setLocation().z = int(z);

    if (idx.getLocation() < mDim) {
        res = true;
    }

    switch (mDataView) {
        case Neon::DataView::STANDARD: {
            idx.setLocation().z += mZHaloRadius;
            return res;
        }
        case Neon::DataView::INTERNAL: {
            idx.setLocation().z += mZHaloRadius + mZBoundaryRadius;
            return res;
        }
        case Neon::DataView::BOUNDARY: {

            idx.setLocation().z += idx.getLocation().z < mZBoundaryRadius
                               ? 0
                               : (mDim.z - 1) + (-1 * mZBoundaryRadius /* we remove zBoundaryRadius as the first zBoundaryRadius will manage the lower slices */);
            idx.setLocation().z += mZHaloRadius;

            return res;
        }
        default: {
        }
    }
    return false;
}

NEON_CUDA_HOST_DEVICE inline auto dSpan::helpGetDataView()
    const -> Neon::DataView const&
{
    return mDataView;
}

NEON_CUDA_HOST_DEVICE inline auto dSpan::helpGetZHaloRadius()
    const -> int const&
{
    return mZHaloRadius;
}

NEON_CUDA_HOST_DEVICE inline auto dSpan::helpGetZBoundaryRadius()
    const -> int const&
{
    return mZBoundaryRadius;
}

NEON_CUDA_HOST_DEVICE inline auto dSpan::helpGetDim()
    const -> Neon::index_3d const&
{
    return mDim;
}

}  // namespace Neon::domain::details::dGrid