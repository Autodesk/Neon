#pragma once

#include "Neon/domain/details/bGrid/bIndex.h"

namespace Neon::domain::details::bGrid {

template <typename SBlock>
class bSpan
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

    bSpan() = default;
    virtual ~bSpan() = default;

    NEON_CUDA_HOST_DEVICE inline static auto getInvalidBlockId() -> typename Idx::DataBlockIdx
    {
        return std::numeric_limits<uint32_t>::max();
    }

    inline bSpan(typename Idx::DataBlockCount     mFirstDataBlockOffset,
                 bSpan::BitMaskWordType* mActiveMask,
                 Neon::DataView          mDataView);

    NEON_CUDA_HOST_DEVICE inline auto setAndValidateCPUDevice(Idx&                   bidx,
                                                              uint32_t const&        threadIdx,
                                                              uint32_t const&        x,
                                                              uint32_t const&        y,
                                                              uint32_t const&        z) const -> bool;

    NEON_CUDA_HOST_DEVICE inline auto setAndValidateGPUDevice(
        Idx& bidx) const -> bool;

    static NEON_CUDA_HOST_DEVICE inline auto getRequiredWordsForBlockBitMask() -> uint32_t;

    static NEON_CUDA_HOST_DEVICE inline auto getActiveStatus(
        const typename Idx::DataBlockIdx& dataBlockIdx,
        int                      threadX,
        int                      threadY,
        int                      threadZ,
        bSpan::BitMaskWordType*  mActiveMask) -> bool;

    static inline auto getMaskAndWordIdforBlockBitMask(int              threadX,
                                                       int              threadY,
                                                       int              threadZ,
                                                       BitMaskWordType& mask,
                                                       uint32_t&        wordIdx) -> void;
    // We don't need to have a count on active blocks
    typename Idx::DataBlockCount     mFirstDataBlockOffset;
    bSpan::BitMaskWordType* mActiveMask;
    Neon::DataView          mDataView;
};
}  // namespace Neon::domain::details::bGrid

#include "Neon/domain/details/bGrid/bSpan_imp.h"