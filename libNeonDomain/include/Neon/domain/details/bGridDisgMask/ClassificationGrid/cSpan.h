#pragma once

#include "Neon/domain/details/bGridDisgMask/bIndex.h"

namespace Neon::domain::details::disaggregated::bGridMask {
namespace details::cGrid {

template <typename SBlock, int classSelector>
class cSpan
{
   public:
    // bit mask information
    using BitMaskWordType = uint64_t;

    static constexpr uint32_t           bitMaskStorageBitWidth = 64;
    static constexpr Neon::MemoryLayout activeMaskMemoryLayout = Neon::MemoryLayout::arrayOfStructs;
    static constexpr uint32_t           log2OfbitMaskWordSize = 6;

    using Idx = bIndex<SBlock>;
    friend class bGrid<SBlock>;

    static constexpr int SpaceDim = 3;

    cSpan() = default;
    virtual ~cSpan() = default;

    NEON_CUDA_HOST_DEVICE inline static auto getInvalidBlockId()
        -> typename Idx::DataBlockIdx
    {
        return std::numeric_limits<uint32_t>::max();
    }

    inline cSpan(
        typename Idx::DataBlockCount                  mFirstDataBlockOffset,
        typename SBlock::BitMask const* NEON_RESTRICT mActiveMask,
        Neon::DataView                                mDataView,
        uint8_t const* NEON_RESTRICT                  mClassMask);

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
    uint8_t const* NEON_RESTRICT                  mClassMask;
};
}  // namespace Neon::domain::details::disaggregated::bGridMask
}
#include "Neon/domain/details/bGridDisgMask/ClassificationGrid/cSpan_imp.h"