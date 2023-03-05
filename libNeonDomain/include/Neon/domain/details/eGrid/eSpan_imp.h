#pragma once

namespace Neon::domain::details::dGrid {

NEON_CUDA_HOST_DEVICE inline auto
eSpan::setAndValidate(Idx&            idx,
                      const uint32_t& x,
                      const uint32_t& y,
                      const uint32_t& z)
    const -> bool
{
    bool res = false;
    idx.set().x = int(x);
    idx.set().y = int(y);
    idx.set().z = int(z);

    if (idx.get() < mDim) {
        res = true;
    }

    switch (mDataView) {
        case Neon::DataView::STANDARD: {
            idx.set().z += mZHaloRadius;
            return res;
        }
        case Neon::DataView::INTERNAL: {
            idx.set().z += mZHaloRadius + mZBoundaryRadius;
            return res;
        }
        case Neon::DataView::BOUNDARY: {

            idx.set().z += idx.get().z < mZBoundaryRadius
                               ? 0
                               : (mDim.z - 1) + (-1 * mZBoundaryRadius /* we remove zBoundaryRadius as the first zBoundaryRadius will manage the lower slices */);
            idx.set().z += mZHaloRadius;

            return res;
        }
        default: {
        }
    }
    return false;
}

NEON_CUDA_HOST_DEVICE inline auto eSpan::helpGetDataView()
    const -> Neon::DataView const&
{
    return mDataView;
}

NEON_CUDA_HOST_DEVICE inline auto eSpan::helpGetZHaloRadius()
    const -> int const&
{
    return mZHaloRadius;
}

NEON_CUDA_HOST_DEVICE inline auto eSpan::helpGetZBoundaryRadius()
    const -> int const&
{
    return mZBoundaryRadius;
}

NEON_CUDA_HOST_DEVICE inline auto eSpan::helpGetDim()
    const -> Neon::index_3d const&
{
    return mDim;
}

}  // namespace Neon::domain::details::dGrid