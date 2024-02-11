#pragma once

#include "Neon/domain/details/bGridDisg/bDisgIndex.h"

namespace Neon::domain::details::disaggregated::bGridDisg {

template <typename SBlock>
class bDisgSpan
{
   public:
    // bit mask information
    using BitMaskWordType = uint64_t;

    static constexpr uint32_t           bitMaskStorageBitWidth = 64;
    static constexpr Neon::MemoryLayout activeMaskMemoryLayout = Neon::MemoryLayout::arrayOfStructs;
    static constexpr uint32_t           log2OfbitMaskWordSize = 6;

    using Idx = bDisgIndex<SBlock>;
    friend class bGridDisg<SBlock>;

    static constexpr int SpaceDim = 3;

    bDisgSpan() = default;
    virtual ~bDisgSpan() = default;

    NEON_CUDA_HOST_DEVICE inline static auto getInvalidBlockId()
        -> typename Idx::DataBlockIdx
    {
        return std::numeric_limits<uint32_t>::max();
    }

    inline bDisgSpan(
        typename Idx::DataBlockCount                  mFirstDataBlockOffset,
        typename SBlock::BitMask const* NEON_RESTRICT mActiveMask,
        Neon::DataView                                mDataView);

    NEON_CUDA_HOST_DEVICE inline auto setAndValidateCPUDevice(
        Idx&            bidx,
        uint32_t const& threadIdx,
        uint32_t const& x,
        uint32_t const& y,
        uint32_t const& z) const -> bool;

    NEON_CUDA_HOST_DEVICE inline auto setAndValidateGPUDevice(
        Idx& bidx) const -> bool;


    // We don't need to have a count on active blocks
    typename Idx::DataBlockCount                  mFirstDataBlockOffset;
    typename SBlock::BitMask const* NEON_RESTRICT mActiveMask;
    Neon::DataView                                mDataView;
};
}  // namespace Neon::domain::details::disaggregated::bGrid

#include "Neon/domain/details/bGridDisg/bSpan_imp.h"