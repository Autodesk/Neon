#pragma once

namespace Neon::domain::details::dGrid
{
    NEON_CUDA_HOST_DEVICE inline auto
    dSpan::setAndValidate(Idx& idx,
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
        if (idx.getLocation() < mSpanDim)
        {
            res = true;
        }

        switch (mDataView)
        {
        case Neon::DataView::STANDARD:
            {
                idx.setLocation().z += mZghostRadius;
                return res;
            }
        case Neon::DataView::INTERNAL:
            {
                idx.setLocation().z += mZghostRadius + mZboundaryRadius;
                return res;
            }
        case Neon::DataView::BOUNDARY:
            {
                idx.setLocation().z += idx.getLocation().z < mZboundaryRadius
                                           ? 0
                                           : (mMaxZInDomain - 1) + (-1 * mZboundaryRadius
                                               /* we remove zBoundaryRadius as the first zBoundaryRadius will manage the lower slices */);
                idx.setLocation().z += mZghostRadius;

                return res;
            }
        default:
            {
            }
        }
        return false;
    }

    NEON_CUDA_HOST_DEVICE inline auto
    dSpan::setAndValidate_warp(Idx& idx,
                               const uint32_t& x,
                               const uint32_t& y,
                               const uint32_t& z)
    const -> void
    {
        printf("setAndValidate_warp: %d %d %d\n" ,x , y , z );

#if !defined(NEON_WARP_COMPILATION)
        idx.setLocation().x = int(x);
        idx.setLocation().y = int(y);
        idx.setLocation().z = int(z);
#else
    idx.setLocation().x = x;
    idx.setLocation().y = y;
    idx.setLocation().z = z;
#endif

        switch (mDataView)
        {
        case Neon::DataView::STANDARD:
            {
                idx.setLocation().z += mZghostRadius;
                return;
            }
        case Neon::DataView::INTERNAL:
            {
                idx.setLocation().z += mZghostRadius + mZboundaryRadius;
                return;
            }
        case Neon::DataView::BOUNDARY:
            {
                idx.setLocation().z += idx.getLocation().z < mZboundaryRadius
                                           ? 0
                                           : (mMaxZInDomain - 1) + (-1 * mZboundaryRadius
                                               /* we remove zBoundaryRadius as the first zBoundaryRadius will manage the lower slices */);
                idx.setLocation().z += mZghostRadius;

                return;
            }
        default:
            {
            }
        }
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
    inline void dSpan::getOffsets(size_t* offsets, size_t* length)
    {
        std::cout << "dGrid_dSpan cpp offsets: " << offsetof(dSpan, mDataView) << " " << offsetof(dSpan, mZghostRadius)
            << " " << offsetof(dSpan, mZboundaryRadius) << " " << offsetof(dSpan, mMaxZInDomain) << " " <<
            offsetof(dSpan, mSpanDim) << " " << std::endl;
        static std::vector<size_t> cpp_offsets = {
            offsetof(dSpan, mDataView),
            offsetof(dSpan, mZghostRadius),
            offsetof(dSpan, mZboundaryRadius),
            offsetof(dSpan, mMaxZInDomain),
            offsetof(dSpan, mSpanDim)
        };

        *length = cpp_offsets.size();
        for (size_t i = 0; i < cpp_offsets.size(); ++i)
        {
            offsets[i] = cpp_offsets[i];
        }
    }
#endif
} // namespace Neon::domain::details::dGrid
