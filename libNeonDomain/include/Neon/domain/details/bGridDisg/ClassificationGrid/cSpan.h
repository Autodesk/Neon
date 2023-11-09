#pragma once

#include "./ClassSelector.h"
#include "Neon/domain/details/bGridDisg/ClassificationGrid/cIndex.h"
#include "Neon/domain/details/bGridDist"

namespace Neon::domain::details::disaggregated::bGrid::cGrid {

template <typename SBlock, ClassSelector classSelector>
class cSpan
{
   public:
    using Idx = cIndex<SBlock>;
    friend class cGrid<SBlock>;
    using bSpan = Neon::domain::details::disaggregated::bGrid::bSpan<SBlock>;

    static constexpr int SpaceDim = 3;

    cSpan() = default;
    virtual ~cSpan() = default;

    NEON_CUDA_HOST_DEVICE inline static auto getInvalidBlockId()
        -> typename Idx::DataBlockIdx
    {
        return std::numeric_limits<uint32_t>::max();
    }

    inline cSpan(typename Idx::DataBlockCount                  mFirstDataBlockOffset,
                 typename SBlock::BitMask const* NEON_RESTRICT mActiveMask,
                 typename Idx::DataBlockCount                  firstSectionClassSize,
                 typename Idx::DataBlockCount                  secondSectionClassBegin,
                 Neon::DataView                                mDataView)
        : mBaseSpan(mFirstDataBlockOffset, mActiveMask, mDataView),
          mFirstSectionClassSize(firstSectionClassSize),
          mSecondSectionClassBegin(secondSectionClassBegin)
    {
    }

    NEON_CUDA_HOST_DEVICE inline auto setAndValidateCPUDevice(
        Idx&            bidx,
        uint32_t const& dataBlockIdx,
        uint32_t const& x,
        uint32_t const& y,
        uint32_t const& z) const -> bool
    {

        bidx.mDataBlockIdx = dataBlockIdx < mFirstSectionClassSize ? dataBlockIdx + mBaseSpan.mFirstDataBlockOffset
                                                                   : dataBlockIdx + mSecondSectionClassBegin;

        bidx.mInDataBlockIdx.x = static_cast<typename Idx::InDataBlockIdx::Integer>(x);
        bidx.mInDataBlockIdx.y = static_cast<typename Idx::InDataBlockIdx::Integer>(y);
        bidx.mInDataBlockIdx.z = static_cast<typename Idx::InDataBlockIdx::Integer>(z);
        const bool isActive = mActiveMask[dataBlockIdx].isActive(bidx.mInDataBlockIdx.x, bidx.mInDataBlockIdx.y, bidx.mInDataBlockIdx.z);
        return isActive;
    }

    NEON_CUDA_HOST_DEVICE inline auto setAndValidateGPUDevice(
        Idx& bidx) const -> bool
    {

        bidx.mDataBlockIdx = blockIdx.x < mFirstSectionClassSize ? blockIdx.x + mBaseSpan.mFirstDataBlockOffset
                                                                 : blockIdx.x + mSecondSectionClassBegin;

        bidx.mInDataBlockIdx.x = threadIdx.x;
        bidx.mInDataBlockIdx.y = threadIdx.y;
        bidx.mInDataBlockIdx.z = threadIdx.z;

        const bool isActive = mActiveMask[bidx.mDataBlockIdx].isActive(bidx.mInDataBlockIdx.x, bidx.mInDataBlockIdx.y, bidx.mInDataBlockIdx.z);

        return isActive;
    }

    bSpan                        mBaseSpan;
    typename Idx::DataBlockCount mFirstSectionClassSize;
    typename Idx::DataBlockCount mSecondSectionClassBegin;
};
}  // namespace Neon::domain::details::disaggregated::cGrid::cGrid
