#include "Neon/domain/details/bGrid/bSpan.h"

namespace Neon::domain::details::bGrid {

template <typename SBlock>
NEON_CUDA_HOST_DEVICE inline auto
bSpan<SBlock>::setAndValidateGPUDevice([[maybe_unused]] Idx& bidx) const -> bool
{
#ifdef NEON_PLACE_CUDA_DEVICE
    bidx.mDataBlockIdx = blockIdx.x + mFirstDataBlockOffset;
    bidx.mInDataBlockIdx.x = threadIdx.x;
    bidx.mInDataBlockIdx.y = threadIdx.y;
    bidx.mInDataBlockIdx.z = threadIdx.z;

    const bool isActive = mActiveMask[bidx.mDataBlockIdx].isActive(bidx.mInDataBlockIdx.x, bidx.mInDataBlockIdx.y, bidx.mInDataBlockIdx.z);

    return isActive;
#else
    NEON_THROW_UNSUPPORTED_OPERATION("Operation supported only on GPU");
#endif
}

template <typename SBlock>
NEON_CUDA_HOST_DEVICE inline auto
bSpan<SBlock>::setAndValidateCPUDevice(Idx&            bidx,
                                       uint32_t const& dataBlockIdx,
                                       uint32_t const& x,
                                       uint32_t const& y,
                                       uint32_t const& z) const -> bool
{

    bidx.mDataBlockIdx = dataBlockIdx;
    bidx.mInDataBlockIdx.x = static_cast<typename Idx::InDataBlockIdx::Integer>(x);
    bidx.mInDataBlockIdx.y = static_cast<typename Idx::InDataBlockIdx::Integer>(y);
    bidx.mInDataBlockIdx.z = static_cast<typename Idx::InDataBlockIdx::Integer>(z);
    const bool isActive = mActiveMask[dataBlockIdx].isActive(bidx.mInDataBlockIdx.x, bidx.mInDataBlockIdx.y, bidx.mInDataBlockIdx.z);
    return isActive;
}

template <typename SBlock>
bSpan<SBlock>::bSpan(typename Idx::DataBlockCount                  firstDataBlockOffset,
                     typename SBlock::BitMask const* NEON_RESTRICT activeMask,
                     Neon::DataView                                dataView)
    : mFirstDataBlockOffset(firstDataBlockOffset),
      mActiveMask(activeMask),
      mDataView(dataView)
{
}


}  // namespace Neon::domain::details::bGrid