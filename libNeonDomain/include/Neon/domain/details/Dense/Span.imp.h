#pragma once

namespace Neon::domain::details::Dense {

template <int Layout>
NEON_CUDA_HOST_DEVICE inline auto
Span<Layout>::setAndValidate(Idx&                                idx,
                             const ExecutionThreadSpanIndexType& x,
                             const ExecutionThreadSpanIndexType& y,
                             const ExecutionThreadSpanIndexType& z)
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
                                       : (mMaxZInDomain - 1) + (-1 * mZboundaryRadius
                                                                /* we remove zBoundaryRadius as the first zBoundaryRadius will manage the lower slices */);
            idx.setLocation().z += mZghostRadius;

            return res;
        }
        default: {
        }
    }
    return false;
}

template <int Layout>
NEON_CUDA_HOST_DEVICE inline auto
Span<Layout>::setAndValidate_warp(Idx&                                idx,
                                  const ExecutionThreadSpanIndexType& x,
                                  const ExecutionThreadSpanIndexType& y,
                                  const ExecutionThreadSpanIndexType& z)
    const -> void
{
#if !defined(NEON_WARP_COMPILATION)
    idx.setLocation().x = int(x);
    idx.setLocation().y = int(y);
    idx.setLocation().z = int(z);
#else
    idx.setLocation().x = x;
    idx.setLocation().y = y;
    idx.setLocation().z = z;
#endif

    switch (mDataView) {
        case Neon::DataView::STANDARD: {
            idx.setLocation().z += mZghostRadius;
            return;
        }
        case Neon::DataView::INTERNAL: {
            idx.setLocation().z += mZghostRadius + mZboundaryRadius;
            return;
        }
        case Neon::DataView::BOUNDARY: {
            idx.setLocation().z += idx.getLocation().z < mZboundaryRadius
                                       ? 0
                                       : (mMaxZInDomain - 1) + (-1 * mZboundaryRadius
                                                                /* we remove zBoundaryRadius as the first zBoundaryRadius will manage the lower slices */);
            idx.setLocation().z += mZghostRadius;

            return;
        }
        default: {
        }
    }
}

template <int Layout>
template<typename DataSetContainer>
NEON_CUDA_HOST_DEVICE inline auto
Span<Layout>::setAndValidate_warp(Idx& idx)
    const -> bool
{
    Idx  e;
    bool res = false;
    // printf("setAndValidate_warp\n");
#if !defined(NEON_COMPILER_CUDA)
    printf("setAndValidate_warp Error\n");
#else
#if !defined(NEON_WARP_COMPILATION)
    printf("setAndValidate_warp Error\n");
#else


    idx.setLocation().x = threadIdx.x + blockIdx.x * blockDim.x;
    idx.setLocation().y = threadIdx.y + blockIdx.y * blockDim.y;
    idx.setLocation().z = threadIdx.z + blockIdx.z * blockDim.z;

#endif
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
                                       : (mMaxZInDomain - 1) + (-1 * mZboundaryRadius
                                                                /* we remove zBoundaryRadius as the first zBoundaryRadius will manage the lower slices */);
            idx.setLocation().z += mZghostRadius;

            return res;
        }
        default: {
        }
    }
    return false;
}


template <int Layout>
NEON_CUDA_HOST_DEVICE inline auto Span<Layout>::helpGetDataView() const
    -> Neon::DataView const&
{
    return mDataView;
}

template <int Layout>
NEON_CUDA_HOST_DEVICE inline auto Span<Layout>::helpGetZHaloRadius() const
    -> int const&
{
    return mZghostRadius;
}

template <int Layout>
NEON_CUDA_HOST_DEVICE inline auto Span<Layout>::helpGetZBoundaryRadius() const
    -> int const&
{
    return mZboundaryRadius;
}

template <int Layout>
NEON_CUDA_HOST_DEVICE inline auto Span<Layout>::helpGetDim() const
    -> Neon::index_3d const&
{
    return mSpanDim;
}

}  // namespace Neon::domain::details::Dense