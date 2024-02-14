#pragma once

#include "Neon/core/core.h"


namespace Neon::domain::details::disaggregated::bGridBlockMask {

// Common forward declarations
template <typename SBlock>
class bGridBlockMask;
template <typename SBlock>
class bMaskSpan;
template <typename T, int C, typename SBlock>
class bMaskPartition;

class MicroIndex
{
   public:
    using TrayIdx = int32_t;
    using InTrayIdx = int8_3d;

    NEON_CUDA_HOST_DEVICE inline explicit MicroIndex()
        : MicroIndex(0, 0, 0, 0)
    {
    }

    NEON_CUDA_HOST_DEVICE inline explicit MicroIndex(const TrayIdx&            blockIdx,
                                                     const InTrayIdx::Integer& x,
                                                     const InTrayIdx::Integer& y,
                                                     const InTrayIdx::Integer& z)
    {
        mTrayBlockIdx = blockIdx;
        mInTrayBlockIdx.x = x;
        mInTrayBlockIdx.y = y;
        mInTrayBlockIdx.z = z;
    }

    NEON_CUDA_HOST_DEVICE inline auto getInTrayBlockIdx() const -> InTrayIdx const&
    {
        return mInTrayBlockIdx;
    }

    NEON_CUDA_HOST_DEVICE inline auto getTrayBlockIdx() const -> TrayIdx const&
    {
        return mTrayBlockIdx;
    }

    NEON_CUDA_HOST_DEVICE inline auto setInTrayBlockIdx(InTrayIdx const& inTrayIdx) -> void
    {
        mInTrayBlockIdx = inTrayIdx;
    }

    NEON_CUDA_HOST_DEVICE inline auto setTrayBlockIdx(TrayIdx const& trayIdx) -> void
    {
        mTrayBlockIdx = trayIdx;
    }

    InTrayIdx mInTrayBlockIdx;
    TrayIdx   mTrayBlockIdx{};
};

template <typename SBlock>
class bMaskIndex
{
   public:
    template <typename SBlock_>
    friend class bMaskSpan;
    using OuterIdx = bMaskIndex<SBlock>;

    using NghIdx = int8_3d;
    template <typename T, int C, typename SBlock_>
    friend class bMaskPartition;

    template <typename T, int C, typename SBlock_>
    friend class bMaskField;

    template <typename SBlock_>
    friend class bMaskSpan;
    template <typename SBlock_>
    friend class bGridBlockMask;


    using TrayIdx = MicroIndex::TrayIdx;
    using InTrayIdx = MicroIndex::InTrayIdx;

    using DataBlockCount = std::make_unsigned_t<TrayIdx>;
    using DataBlockIdx = std::make_unsigned_t<TrayIdx>;
    using InDataBlockIdx = InTrayIdx;

    bMaskIndex() = default;
    ~bMaskIndex() = default;

    NEON_CUDA_HOST_DEVICE inline explicit bMaskIndex(const DataBlockIdx&            blockIdx,
                                                 const InDataBlockIdx::Integer& x,
                                                 const InDataBlockIdx::Integer& y,
                                                 const InDataBlockIdx::Integer& z);

    NEON_CUDA_HOST_DEVICE inline auto getMicroIndex() -> MicroIndex;
    NEON_CUDA_HOST_DEVICE inline auto init(MicroIndex const&) -> void;

    NEON_CUDA_HOST_DEVICE inline auto getInDataBlockIdx() const -> InDataBlockIdx const&;
    NEON_CUDA_HOST_DEVICE inline auto getDataBlockIdx() const -> DataBlockIdx const&;
    NEON_CUDA_HOST_DEVICE inline auto setInDataBlockIdx(InDataBlockIdx const&) -> void;
    NEON_CUDA_HOST_DEVICE inline auto setDataBlockIdx(DataBlockIdx const&) -> void;
    NEON_CUDA_HOST_DEVICE inline auto isActive() const -> bool;
    // the local index within the block
    InDataBlockIdx mInDataBlockIdx;
    DataBlockIdx   mDataBlockIdx{};
};

template <typename SBlock>
NEON_CUDA_HOST_DEVICE auto bMaskIndex<SBlock>::setDataBlockIdx(const bMaskIndex::DataBlockIdx& dataBlockIdx) -> void
{
    mDataBlockIdx = dataBlockIdx;
}

template <typename SBlock>
NEON_CUDA_HOST_DEVICE auto bMaskIndex<SBlock>::setInDataBlockIdx(const bMaskIndex::InDataBlockIdx& inDataBlockIdx) -> void
{
    mInDataBlockIdx = inDataBlockIdx;
}

template <typename SBlock>
NEON_CUDA_HOST_DEVICE auto bMaskIndex<SBlock>::getDataBlockIdx() const -> const bMaskIndex::DataBlockIdx&
{
    return mDataBlockIdx;
}
template <typename SBlock>
NEON_CUDA_HOST_DEVICE auto bMaskIndex<SBlock>::getInDataBlockIdx() const -> const bMaskIndex::InDataBlockIdx&
{
    return mInDataBlockIdx;
}

template <typename SBlock>
NEON_CUDA_HOST_DEVICE auto bMaskIndex<SBlock>::isActive() const -> bool
{
    return mDataBlockIdx != std::numeric_limits<typename bMaskIndex::DataBlockIdx>::max();
}

}  // namespace Neon::domain::details::disaggregated::bGrid

#include "Neon/domain/details/bGridDisgBlockMask/bIndex_imp.h"
