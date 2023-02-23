#pragma once

namespace Neon::domain::internal::exp::dGrid {

NEON_CUDA_HOST_DEVICE inline auto
dSpan::setAndValidate(Idx&            idx,
                      const uint32_t& x,
                      const uint32_t& y,
                      const uint32_t& z)
    const
    -> bool
{
    bool res = false;
    idx.set().x = int(x);
    idx.set().y = int(y);
    idx.set().z = int(z);

    switch (mDataView) {
        case Neon::DataView::STANDARD: {
            if (idx.get() < mDim) {
                res = true;
            }
            idx.set().z += mZHaloRadius;
            return res;
        }
        case Neon::DataView::INTERNAL: {
            if (idx.get().x < (mDim.x) &&
                idx.get().y < (mDim.y) &&
                idx.get().z < (mDim.z - 2 * mZBoundaryRadius)) {
                res = true;
            }
            idx.set().z += mZHaloRadius + mZBoundaryRadius;

            return res;
        }
        case Neon::DataView::BOUNDARY: {
            if (idx.get().x < (mDim.x) &&
                idx.get().y < (mDim.y) &&
                idx.get().z < (mZBoundaryRadius * 2)) {
                res = true;
            }
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

NEON_CUDA_HOST_DEVICE inline auto dSpan::helpGetDataView()
    const -> Neon::DataView const&
{
    return mDataView;
}

NEON_CUDA_HOST_DEVICE inline auto dSpan::helpGetZHaloRadius()
    const -> int const&{
        return mZHaloRadius;
}

NEON_CUDA_HOST_DEVICE inline auto dSpan::helpGetZBoundaryRadius()
    const -> int const&{
        return mZBoundaryRadius;
}

NEON_CUDA_HOST_DEVICE inline auto dSpan::helpGetDim()
    const -> Neon::index_3d const&{
        return mDim;
}

}  // namespace Neon::domain::internal::exp::dGrid