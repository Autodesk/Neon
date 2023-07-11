#pragma once

#include "Neon/core/core.h"


namespace Neon::domain::details::bGrid {

// Common forward declarations
template <typename SBlock>
class bGrid;
template <typename SBlock>
class bSpan;
template <typename T, int C, typename SBlock>
class bPartition;

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
class bIndex
{
   public:
    template <typename SBlock_>
    friend class bSpan;
    using OuterIdx = bIndex<SBlock>;

    using NghIdx = int8_3d;
    template <typename T, int C, typename SBlock_>
    friend class bPartition;

    template <typename T, int C, typename SBlock_>
    friend class bField;

    template <typename SBlock_>
    friend class bSpan;
    template <typename SBlock_>
    friend class bGrid;


    using TrayIdx = MicroIndex::TrayIdx;
    using InTrayIdx = MicroIndex::InTrayIdx;

    using DataBlockCount = std::make_unsigned_t<TrayIdx>;
    using DataBlockIdx = std::make_unsigned_t<TrayIdx>;
    using InDataBlockIdx = InTrayIdx;

    bIndex() = default;
    ~bIndex() = default;

    NEON_CUDA_HOST_DEVICE inline explicit bIndex(const DataBlockIdx&            blockIdx,
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
NEON_CUDA_HOST_DEVICE auto bIndex<SBlock>::setDataBlockIdx(const bIndex::DataBlockIdx& dataBlockIdx) -> void
{
    mDataBlockIdx = dataBlockIdx;
}

template <typename SBlock>
NEON_CUDA_HOST_DEVICE auto bIndex<SBlock>::setInDataBlockIdx(const bIndex::InDataBlockIdx& inDataBlockIdx) -> void
{
    mInDataBlockIdx = inDataBlockIdx;
}

template <typename SBlock>
NEON_CUDA_HOST_DEVICE auto bIndex<SBlock>::getDataBlockIdx() const -> const bIndex::DataBlockIdx&
{
    return mDataBlockIdx;
}
template <typename SBlock>
NEON_CUDA_HOST_DEVICE auto bIndex<SBlock>::getInDataBlockIdx() const -> const bIndex::InDataBlockIdx&
{
    return mInDataBlockIdx;
}

template <typename SBlock>
NEON_CUDA_HOST_DEVICE auto bIndex<SBlock>::isActive() const -> bool
{
    return mDataBlockIdx != std::numeric_limits<typename bIndex::DataBlockIdx>::max();
}

}  // namespace Neon::domain::details::bGrid

#include "Neon/domain/details/bGrid/bIndex_imp.h"
