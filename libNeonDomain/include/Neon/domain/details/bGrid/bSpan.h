#pragma once

#include "Neon/domain/details/bGrid/bIndex.h"

namespace Neon::domain::details::bGrid {

class bSpan
{
   public:
    // bit mask information
    using bitMaskWordType = uint64_t;

    static constexpr uint32_t           bitMaskStorageBitWidth = 64;
    static constexpr Neon::MemoryLayout activeMaskMemoryLayout = Neon::MemoryLayout::arrayOfStructs;
    static constexpr uint32_t           log2OfbitMaskWordSize = 6;

    using Idx = bIndex;
    friend class bGrid;

    static constexpr int SpaceDim = 3;

    bSpan() = default;
    virtual ~bSpan() = default;

    NEON_CUDA_HOST_DEVICE inline static auto getInvalidBlockId() -> bIndex::DataBlockIdx
    {
        return std::numeric_limits<uint32_t>::max();
    }

    inline bSpan(Idx::DataBlockCount     mFirstDataBlockOffset,
                 uint32_t                mDataBlockSize,
                 bSpan::bitMaskWordType* mActiveMask,
                 Neon::DataView          mDataView);

    NEON_CUDA_HOST_DEVICE inline auto setAndValidateCPUDevice(bIndex&         bidx,
                                                              uint32_t const& threadIdx,
                                                              uint32_t const& threadBlockSize,
                                                              uint32_t const& x,
                                                              uint32_t const& y,
                                                              uint32_t const& z) const -> bool;

    NEON_CUDA_HOST_DEVICE inline auto setAndValidateGPUDevice(
        bIndex& bidx) const -> bool;

    static NEON_CUDA_HOST_DEVICE inline auto getRequiredWords(
        uint32_t const& blockSize) -> uint32_t;

    static NEON_CUDA_HOST_DEVICE inline auto getActiveStatus(
        const Idx::DataBlockIdx& dataBlockIdx,
        int                      threadX,
        int                      threadY,
        int                      threadZ,
        bSpan::bitMaskWordType*  mActiveMask,
        uint32_t const&          blockSize) -> bool;

    // We don't need to have a count on active blocks
    Idx::DataBlockCount     mFirstDataBlockOffset;
    uint32_t                mDataBlockSize;
    bSpan::bitMaskWordType* mActiveMask;
    Neon::DataView          mDataView;
};
}  // namespace Neon::domain::details::bGrid

#include "Neon/domain/details/bGrid/bSpan_imp.h"