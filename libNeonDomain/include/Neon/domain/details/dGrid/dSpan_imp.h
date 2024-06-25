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

#if !defined(NEON_WARP_COMPILATION)
    idx.setLocation().x = int(x);
    idx.setLocation().y = int(y);
    idx.setLocation().z = int(z);
#else
   idx.setLocation().x = x;
    idx.setLocation().y = y;
    idx.setLocation().z = z;
 #endif
    if (idx.getLocation() < mSpanDim) {
        res = true;
    }

    switch (mDataView) {
        case Neon::DataView::STANDARD: {
            idx.setLocation().z += mZghostRadius;
            return res;
        }
        case Neon::DataView::INTERNAL: {
            idx.setLocation().z += mZghostRadius + mZboundaryRadius;
            return res;
        }
        case Neon::DataView::BOUNDARY: {

            idx.setLocation().z += idx.getLocation().z < mZboundaryRadius
                                       ? 0
                               : (mMaxZInDomain - 1) + (-1 * mZboundaryRadius /* we remove zBoundaryRadius as the first zBoundaryRadius will manage the lower slices */);
            idx.setLocation().z += mZghostRadius;

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
    return mZghostRadius;
}

NEON_CUDA_HOST_DEVICE inline auto dSpan::helpGetZBoundaryRadius()
    const -> int const&
{
    return mZboundaryRadius;
}

NEON_CUDA_HOST_DEVICE inline auto dSpan::helpGetDim()
    const -> Neon::index_3d const&
{
    return mSpanDim;
}

#if !defined(NEON_WARP_COMPILATION)
inline std::vector<size_t> dSpan::getOffsets() {
    return {
        offsetof(dSpan, mDataView),
        offsetof(dSpan, mZghostRadius),
        offsetof(dSpan, mZboundaryRadius),
        offsetof(dSpan, mMaxZInDomain),
        offsetof(dSpan, mSpanDim)
    };
}
#endif

}  // namespace Neon::domain::details::dGrid