#pragma once

#include "./ClassSelector.h"

namespace Neon::domain::details::disaggregated::bGrid::cGrid {

// -------------------------------------
// | Internal     | BoundaryUP   | BoundaryDW    | Ghost UP        | Ghost Dw     |
// | alpha | beta | alpha | beta | alpha | beta  | alpha | beta    | alpha | beta |
// |              |              |               | Ghost setIdx    | Ghost setIdx |
//        ^               ^              ^               -                 -
// The span must manage two classes that are split into 3 sections: Internal, BoundaryUP, BoundaryDW
template <typename SBlock, typename FounderGrid, int classSelector>
class cSpan
{
   public:
    using Idx = typename FounderGrid::Idx;
    using bSpan = typename FounderGrid::Span;
    static constexpr int SpaceDim = 3;

    cSpan() = default;

    virtual ~cSpan() = default;

    NEON_CUDA_HOST_DEVICE inline static auto getInvalidBlockId()
        -> typename Idx::DataBlockIdx
    {
        return std::numeric_limits<uint32_t>::max();
    }

    inline cSpan(bSpan const&                 baseSpan,
                 typename Idx::DataBlockCount internalCountVirtualSingleClass,
                 typename Idx::DataBlockCount iandIupCountVirtualSingleClass,
                 typename Idx::DataBlockCount internalClassFirstMemoryOffset,
                 typename Idx::DataBlockCount bupClassFirstMemoryOffset,
                 typename Idx::DataBlockCount bdwClassFirstMemoryOffset,
                 Neon::DataView               dataView)

    {
        init(baseSpan,
             internalCountVirtualSingleClass,
             iandIupCountVirtualSingleClass,
             internalClassFirstMemoryOffset,
             bupClassFirstMemoryOffset,
             bdwClassFirstMemoryOffset);
    }

    inline auto init(bSpan const&                 baseSpan,
                     typename Idx::DataBlockCount internalCountVirtualSingleClass,
                     typename Idx::DataBlockCount iandIupCountVirtualSingleClass,
                     typename Idx::DataBlockCount internalClassFirstMemoryOffset,
                     typename Idx::DataBlockCount bupClassFirstMemoryOffset,
                     typename Idx::DataBlockCount bdwClassFirstMemoryOffset,
                     Neon::DataView) -> void

    {
        mActiveMask = baseSpan.mActiveMask;
        mIcountVirtualSingleClass = internalCountVirtualSingleClass;
        mIandIupCountVirtualSingleClass = iandIupCountVirtualSingleClass;
        mInternalClassFirstMemoryOffset = internalClassFirstMemoryOffset;
        mBupClassFirstMemoryOffset = bupClassFirstMemoryOffset;
        mBdwClassFirstMemoryOffset = bdwClassFirstMemoryOffset;
    }

    NEON_CUDA_HOST_DEVICE inline auto setAndValidateCPUDevice(
        Idx&            bidx,
        uint32_t const& blockIdx,
        uint32_t const& x,
        uint32_t const& y,
        uint32_t const& z) const -> bool
    {
        typename Idx::DataBlockIdx offset = blockIdx + mInternalClassFirstMemoryOffset;
        offset = blockIdx >= mIcountVirtualSingleClass
                     ? blockIdx + mBupClassFirstMemoryOffset
                     : offset;

        offset = blockIdx >= mIandIupCountVirtualSingleClass
                     ? blockIdx + mBdwClassFirstMemoryOffset
                     : offset;

        bidx = Idx(offset, x, y, z);
        const bool isActive = mActiveMask[bidx.mDataBlockIdx].isActive(bidx.mInDataBlockIdx.x,
                                                                       bidx.mInDataBlockIdx.y,
                                                                       bidx.mInDataBlockIdx.z);

        return isActive;
    }

    NEON_CUDA_HOST_DEVICE inline auto setAndValidateGPUDevice(
        Idx& bidx) const -> bool
    {
#ifdef NEON_PLACE_CUDA_DEVICE
        typename Idx::DataBlockIdx offset = blockIdx.x  + mInternalClassFirstMemoryOffset;
        offset = blockIdx.x >= mIcountVirtualSingleClass
                     ?  blockIdx.x + mBupClassFirstMemoryOffset
                     : offset;

        offset = blockIdx.x >= mIandIupCountVirtualSingleClass
                     ?  blockIdx.x + mBdwClassFirstMemoryOffset
                     : offset;

        bidx = Idx(offset, threadIdx.x, threadIdx.y, threadIdx.z);
        const bool isActive = mActiveMask[bidx.mDataBlockIdx].isActive(bidx.mInDataBlockIdx.x,
                                                                       bidx.mInDataBlockIdx.y,
                                                                       bidx.mInDataBlockIdx.z);

        return isActive;
#else
        NEON_THROW_UNSUPPORTED_OPERATION("Operation supported only on GPU");
#endif
    }

    typename SBlock::BitMask const* NEON_RESTRICT mActiveMask;
    typename Idx::DataBlockCount                  mIcountVirtualSingleClass;
    typename Idx::DataBlockCount                  mIandIupCountVirtualSingleClass;
    typename Idx::DataBlockCount                  mInternalClassFirstMemoryOffset;
    typename Idx::DataBlockCount                  mBupClassFirstMemoryOffset;
    typename Idx::DataBlockCount                  mBdwClassFirstMemoryOffset;
};
}  // namespace Neon::domain::details::disaggregated::bGrid::cGrid
