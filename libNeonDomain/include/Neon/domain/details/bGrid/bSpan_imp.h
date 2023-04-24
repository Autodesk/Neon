#include "Neon/domain/details/bGrid/bSpan.h"

namespace Neon::domain::details::bGrid {


NEON_CUDA_HOST_DEVICE inline auto
bSpan::setAndValidateGPUDevice([[maybe_unused]] bIndex& bidx) const -> bool
{
#ifdef NEON_PLACE_CUDA_DEVICE
    bidx.mDataBlockIdx = blockIdx.x + mFirstDataBlockOffset;
    bidx.mInDataBlockIdx.x =  threadIdx.x;
    bidx.mInDataBlockIdx.y =  threadIdx.y;
    bidx.mInDataBlockIdx.z =  threadIdx.z;

    assert(mDataBlockSize == blockDim.x);
    assert(mDataBlockSize == blockDim.y);
    assert(mDataBlockSize == blockDim.z);

    bidx.mInDataBlockIdx = blockIdx.x + mFirstDataBlockOffset;
    bool const isActive = getActiveStatus(blockIdx.x,
                                          threadIdx.x, threadIdx.y, threadIdx.z,
                                          mActiveMask, blockDim.x);
    return isActive;
#else
    NEON_THROW_UNSUPPORTED_OPERATION("Operation supported only on GPU");
#endif
}

NEON_CUDA_HOST_DEVICE inline auto
bSpan::setAndValidateCPUDevice(bIndex&                bidx,
                               uint32_t const&        dataBlockIdx,
                               Neon::uint32_3d const& blockSize,
                               uint32_t const&        x,
                               uint32_t const&        y,
                               uint32_t const&        z) const -> bool
{

    bidx.mDataBlockIdx = dataBlockIdx;
    bidx.mInDataBlockIdx.x = x;
    bidx.mInDataBlockIdx.y = y;
    bidx.mInDataBlockIdx.z = z;
    bool const isActive = getActiveStatus(bidx.mDataBlockIdx,
                                          bidx.mInDataBlockIdx.x, bidx.mInDataBlockIdx.y, bidx.mInDataBlockIdx.z,
                                          mActiveMask, blockSize.x);
    return isActive;
}

bSpan::bSpan(bIndex::DataBlockCount  firstDataBlockOffset,
             uint32_t                dataBlockSize,
             bSpan::BitMaskWordType* activeMask,
             Neon::DataView          dataView)
    : mFirstDataBlockOffset(firstDataBlockOffset),
      mDataBlockSize(dataBlockSize),
      mActiveMask(activeMask),
      mDataView(dataView)
{
}

NEON_CUDA_HOST_DEVICE inline auto bSpan::getRequiredWordsForBlockBitMask(uint32_t const& blockSize) -> uint32_t
{
    uint32_t requiredBits = blockSize * blockSize * blockSize;
    uint32_t requiredWords = ((requiredBits - 1) >> bSpan::log2OfbitMaskWordSize) + 1;
    return requiredWords;
}

inline auto bSpan::getMaskAndWordIdforBlockBitMask(int                       threadX,
                                                   int                       threadY,
                                                   int                       threadZ,
                                                   uint32_t const&           blockSize,
                                                   NEON_OUT BitMaskWordType& mask,
                                                   NEON_OUT uint32_t&        wordIdx) -> void
{
    if constexpr (activeMaskMemoryLayout == Neon::MemoryLayout::arrayOfStructs) {
        // 6 = log_2 64
        const uint32_t threadPitch = threadX + threadY * blockSize + threadZ * blockSize * blockSize;
        // threadPitch >> log2OfbitMaskWordSize
        // the same as: threadPitch / 2^{log2OfbitMaskWordSize}
        wordIdx = threadPitch >> log2OfbitMaskWordSize;
        // threadPitch & ((bitMaskWordType(bitMaskStorageBitWidth)) - 1);
        // same as threadPitch % 2^{log2OfbitMaskWordSize}
        const uint32_t offsetInWord = threadPitch & ((BitMaskWordType(bitMaskStorageBitWidth)) - 1);
        mask = BitMaskWordType(1) << offsetInWord;
    } else {
        assert(false);
    }
}


NEON_CUDA_HOST_DEVICE inline auto bSpan::getActiveStatus(
    const Idx::DataBlockIdx& dataBlockIdx,
    int                      threadX,
    int                      threadY,
    int                      threadZ,
    bSpan::BitMaskWordType*  mActiveMask,
    uint32_t const&          blockSize) -> bool
{
    if constexpr (activeMaskMemoryLayout == Neon::MemoryLayout::arrayOfStructs) {
        // 6 = log_2 64
        const uint32_t threadPitch = threadX + threadY * blockSize + threadZ * blockSize * blockSize;
        // threadPitch >> log2OfbitMaskWordSize
        // the same as: threadPitch / 2^{log2OfbitMaskWordSize}
        const uint32_t wordIdx = threadPitch >> log2OfbitMaskWordSize;
        // threadPitch & ((bitMaskWordType(bitMaskStorageBitWidth)) - 1);
        // same as threadPitch % 2^{log2OfbitMaskWordSize}
        const uint32_t  offsetInWord = threadPitch & ((BitMaskWordType(bitMaskStorageBitWidth)) - 1);
        BitMaskWordType mask = BitMaskWordType(1) << offsetInWord;

        uint32_t const  cardinality = getRequiredWordsForBlockBitMask(blockSize);
        uint32_t const  pitch = (cardinality * dataBlockIdx) + wordIdx;
        BitMaskWordType targetWord = mActiveMask[pitch];
        auto            masked = targetWord & mask;
        if (masked != 0) {
            return true;
        }
        return false;
    } else {
        assert(false);
    }
    //
}

}  // namespace Neon::domain::details::bGrid