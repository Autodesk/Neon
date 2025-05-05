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
    //
    bSpan() = default;
    // virtual ~bSpan() = default;

    NEON_CUDA_HOST_DEVICE inline static auto getInvalidBlockId()
        -> typename Idx::DataBlockIdx
    {
        return std::numeric_limits<uint32_t>::max();
    }

    bSpan(
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

    NEON_CUDA_HOST_DEVICE inline auto printLog([[maybe_unused]] bool masterOnly) const -> void
    {
#if defined(NEON_PLACE_CUDA_DEVICE)
        if ((!masterOnly) || (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0)) {
#else
        masterOnly = true;
        if (masterOnly) {
#endif

            printf("bSpan Log: BEGIN\n");
            printf("bSpan Log: mFirstDataBlockOffset %d\n", mFirstDataBlockOffset);
            printf("bSpan Log: mask %p\n", mActiveMask);
            printf("bSpan Log: mask Val %d\n", mActiveMask[0].isActive(0, 0, 0)? 1 : 0);
            printf("bSpan Log: mask word %d\n", mActiveMask[0].getFirstWord());
            printf("bSpan Log: data view %d\n", mDataView == Neon::DataView::STANDARD ? 1 : -1);
            printf("bSpan Log: END\n");

        }
    }


#if !defined(NEON_WARP_COMPILATION)
    // Function to get offsets of member variables
    static void getOffsets(size_t* offsets, size_t* length);
    // Function to get offsets of member variables
    static std::vector<size_t> getOffsets();
#endif

    // We don't need to have a count on active blocks
    typename Idx::DataBlockCount                  mFirstDataBlockOffset;
    typename SBlock::BitMask const* NEON_RESTRICT mActiveMask;
    Neon::DataView                                mDataView;
};
}  // namespace Neon::domain::details::bGrid

#include "Neon/domain/details/bGrid/bSpan_imp.h"