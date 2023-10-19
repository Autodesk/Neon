#pragma once

namespace Neon::domain::details::disaggregated::dGrid {

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

    if (idx.getLocation() < mSpanDim) {
        res = true;
    }

    switch (mDataView) {
        case Neon::DataView::STANDARD: {

            idx.setLocation().z += mZghostRadius;

            // Boundary DW up and down
            size_t regionFirstZ = idx.setLocation().z;
            int    regionZDim = 1;
            int    offsetLocalNoCard = size_t(x) + size_t(y) * mSpanDim.x;

            if (idx.getLocation().z > 1 && idx.getLocation().z < mSpanDim.z) {
                // Internal
                regionFirstZ = 2;
                regionZDim = mSpanDim.z - 2 * mZghostRadius;
                offsetLocalNoCard = size_t(x) +
                                    size_t(y) * mSpanDim.x +
                                    size_t(idx.setLocation().z - regionFirstZ) * mSpanDim.x * mSpanDim.y;
            }

            idx.setRegionFirstZ(regionFirstZ);
            idx.setOffsetLocalNoCard(offsetLocalNoCard);
            idx.setRegionZDim(regionZDim);

            return res;
        }
        case Neon::DataView::INTERNAL: {
            idx.setLocation().z += mZghostRadius + mZboundaryRadius;

            int regionFirstZ = 2;
            int regionZDim = mSpanDim.z - 2 * mZghostRadius;

            size_t offsetLocalNoCard = size_t(x) +
                                       size_t(y) * mSpanDim.x +
                                       size_t(z) * mSpanDim.x * mSpanDim.y;
            idx.setOffsetLocalNoCard(offsetLocalNoCard);
            idx.setRegionFirstZ(regionFirstZ);
            idx.setRegionZDim(regionZDim);

            return res;
        }
        case Neon::DataView::BOUNDARY: {

            idx.setLocation().z = idx.getLocation().z < mZboundaryRadius ? 1 : mSpanDim.z;
            int    regionFirstZ = 1;
            int    regionZDim = 1;
            size_t offsetLocalNoCard = size_t(x) +
                                       size_t(y) * mSpanDim.x;

            idx.setOffsetLocalNoCard(offsetLocalNoCard);
            idx.setRegionFirstZ(regionFirstZ);
            idx.setRegionZDim(regionZDim);
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


NEON_CUDA_HOST_DEVICE inline auto dSpan::helpGetDim()
    const -> Neon::index_3d const&
{
    return mSpanDim;
}

}  // namespace Neon::domain::details::disaggregated::dGrid