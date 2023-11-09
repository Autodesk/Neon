#pragma once

#include "./cIndex.h"
#include "./cSpan.h"
#include "Neon/domain/details/bGridDisg/bPartition.h"
#include "Neon/domain/interface/NghData.h"


namespace Neon::domain::details::disaggregated::bGrid {

template <typename SBlock, ClassSelector classSelector>
class cSpan;

template <typename T, int C, typename SBlock, ClassSelector classSelector>
class bPartition : public Neon::domain::details::disaggregated::bGrid::bPartition<T, C, SBlock>
{
   public:
    using Span = cSpan<SBlock, classSelector>;
    using Idx = cIndex<SBlock>;
    using NghIdx = typename Idx::NghIdx;
    using Type = T;
    using NghData = Neon::domain::NghData<T>;

    using BlockViewGrid = Neon::domain::tool::GridTransformer<details::GridTransformation>::Grid;
    using BlockViewGridIdx = BlockViewGrid::Idx;

   public:
    bPartition();

    ~bPartition() = default;

    explicit bPartition(int                                           setIdx,
                        int                                           mCardinality,
                        T*                                            mMem,
                        typename Idx::DataBlockIdx*                   mBlockConnectivity,
                        typename SBlock::BitMask const* NEON_RESTRICT mMask,
                        Neon::int32_3d*                               mOrigin,
                        NghIdx*                                       mStencilNghIndex,
                        Neon::int32_3d                                mDomainSize)
        : Neon::domain::details::disaggregated::bGrid::bPartition<T, C, SBlock>(setIdx,
                                                                                mCardinality,
                                                                                mMem,
                                                                                mBlockConnectivity,
                                                                                mMask,
                                                                                mOrigin,
                                                                                mStencilNghIndex,
                                                                                mDomainSize)
    {
    }
};

}  // namespace Neon::domain::details::disaggregated::bGrid
