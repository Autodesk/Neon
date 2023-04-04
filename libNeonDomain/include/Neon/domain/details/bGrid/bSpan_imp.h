#include "Neon/domain/details/bGrid/bSpan.h"

namespace Neon::domain::details::bGrid {


NEON_CUDA_HOST_DEVICE inline auto
bSpan::setAndValidateGPUDevice(bIndex& bidx) const -> bool
{
    assert(mDataBlockSize == blockDim.x);
    assert(mDataBlockSize == blockDim.y);
    assert(mDataBlockSize == blockDim.z);

    bool const isActive = getActiveStatus(blockIdx.x,
                                          threadIdx.x, threadIdx.y, threadIdx.z,
                                          mActiveMask, blockDim.x);
    return isActive;
}

NEON_CUDA_HOST_DEVICE inline auto
bSpan::setAndValidateCPUDevice(bIndex&         bidx,
                               uint32_t const& threadIdx,
                               uint32_t const& threadBlockSize,
                               uint32_t const& x,
                               uint32_t const& y,
                               uint32_t const& z) const -> bool
{

    bidx.mDataBlockIdx = threadIdx;
    bidx.mInDataBlockIdx.x = x;
    bidx.mInDataBlockIdx.y = y;
    bidx.mInDataBlockIdx.z = z;

    bool const isActive = getActiveStatus(bidx.mDataBlockIdx,
                                          bidx.mInDataBlockIdx.x, bidx.mInDataBlockIdx.y, bidx.mInDataBlockIdx.z,
                                          mActiveMask, threadBlockSize);
    return isActive;
}

bSpan::bSpan(bIndex::DataBlockCount  dataBlockCount,
             bIndex::DataBlockCount  firstDataBlockOffset,
             uint32_t                dataBlockSize,
             bSpan::bitMaskWordType* activeMask,
             Neon::DataView          dataView)
    : mDataBlocCount(dataBlockCount),
      mFirstDataBlockOffset(firstDataBlockOffset),
      mDataBlockSize(dataBlockSize),
      mActiveMask(activeMask),
      mDataView(dataView)
{
}

NEON_CUDA_HOST_DEVICE inline auto bSpan::getRequiredWords(uint32_t const& blockSize) -> uint32_t
{
    uint32_t requiredBits = blockSize * blockSize * blockSize;
    uint32_t requiredWords = ((requiredBits - 1) >> bSpan::log2OfbitMaskWordSize) + 1;
    return requiredWords;
}

NEON_CUDA_HOST_DEVICE inline auto bSpan::getActiveStatus(
    const Idx::DataBlockIdx& dataBlockIdx,
    int                      threadX,
    int                      threadY,
    int                      threadZ,
    bSpan::bitMaskWordType*  mActiveMask,
    uint32_t const&          blockSize) -> bool
{
    // 6 = log_2 64
    const uint32_t  threadPitch = threadX + threadY * blockSize + threadZ * blockSize * blockSize;
    const uint32_t  wordIdx = threadPitch >> log2OfbitMaskWordSize;
    const uint32_t  offsetInWork = threadPitch & ((bitMaskWordType(bitMaskStorageBitWidth)) - 1);
    bitMaskWordType mask = 1 << offsetInWork;

    if constexpr (activeMaskMemoryLayout == Neon::MemoryLayout::arrayOfStructs) {
        uint32_t const  cardinality = getRequiredWords(blockSize);
        uint32_t const  pitch = dataBlockIdx * cardinality;
        bitMaskWordType targetWord = mActiveMask[pitch];
        auto            masked = targetWord & mask;
        if (mask != 0) {
            return true;
        }
        return false;
    } else {
        assert(false);
    }
    //
}

}  // namespace Neon::domain::details::bGrid