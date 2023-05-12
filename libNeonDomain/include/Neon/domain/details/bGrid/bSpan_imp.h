#include "Neon/domain/details/bGrid/bSpan.h"

namespace Neon::domain::details::bGrid {

template <int8_t dataBlockSizeX, int8_t dataBlockSizeY, int8_t dataBlockSizeZ>
NEON_CUDA_HOST_DEVICE inline auto
bSpan<dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ>::setAndValidateGPUDevice([[maybe_unused]] Idx& bidx) const -> bool
{
#ifdef NEON_PLACE_CUDA_DEVICE
    bidx.mDataBlockIdx = blockIdx.x + mFirstDataBlockOffset;
    bidx.mInDataBlockIdx.x = threadIdx.x;
    bidx.mInDataBlockIdx.y = threadIdx.y;
    bidx.mInDataBlockIdx.z = threadIdx.z;

    bool const isActive = getActiveStatus(bidx.mDataBlockIdx,
                                          bidx.mInDataBlockIdx.x, bidx.mInDataBlockIdx.y, bidx.mInDataBlockIdx.z,
                                          mActiveMask);
    //  printf("%d %d %d is active %d\n",bidx.mInDataBlockIdx.x, bidx.mInDataBlockIdx.y, bidx.mInDataBlockIdx.z, (isActive?1:-1));
    return isActive;
#else
    NEON_THROW_UNSUPPORTED_OPERATION("Operation supported only on GPU");
#endif
}

template <int8_t dataBlockSizeX, int8_t dataBlockSizeY, int8_t dataBlockSizeZ>
NEON_CUDA_HOST_DEVICE inline auto
bSpan<dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ>::setAndValidateCPUDevice(Idx&                   bidx,
                                              uint32_t const&        dataBlockIdx,
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
                                          mActiveMask);
    return isActive;
}

template <int8_t dataBlockSizeX, int8_t dataBlockSizeY, int8_t dataBlockSizeZ>
bSpan<dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ>::bSpan(Idx::DataBlockCount firstDataBlockOffset,
                            BitMaskWordType*    activeMask,
                            Neon::DataView      dataView)
    : mFirstDataBlockOffset(firstDataBlockOffset),
      mActiveMask(activeMask),
      mDataView(dataView)
{
}

template <int8_t dataBlockSizeX, int8_t dataBlockSizeY, int8_t dataBlockSizeZ>
NEON_CUDA_HOST_DEVICE inline auto bSpan<dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ>::getRequiredWordsForBlockBitMask() -> uint32_t
{
    uint32_t requiredBits = dataBlockSizeX * dataBlockSizeY * dataBlockSizeZ;
    uint32_t requiredWords = ((requiredBits - 1) >> bSpan<dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ>::log2OfbitMaskWordSize) + 1;
    return requiredWords;
}

template <int8_t dataBlockSizeX, int8_t dataBlockSizeY, int8_t dataBlockSizeZ>
inline auto bSpan<dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ>::getMaskAndWordIdforBlockBitMask(int                       threadX,
                                                                  int                       threadY,
                                                                  int                       threadZ,
                                                                  NEON_OUT BitMaskWordType& mask,
                                                                  NEON_OUT uint32_t&        wordIdx) -> void
{
    if constexpr (activeMaskMemoryLayout == Neon::MemoryLayout::arrayOfStructs) {
        // 6 = log_2 64
        const uint32_t threadPitch = threadX + threadY * dataBlockSizeX + threadZ * dataBlockSizeX * dataBlockSizeY;
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


template <int8_t dataBlockSizeX, int8_t dataBlockSizeY, int8_t dataBlockSizeZ>
NEON_CUDA_HOST_DEVICE inline auto bSpan<dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ>::getActiveStatus(
    const Idx::DataBlockIdx& dataBlockIdx,
    int                      threadX,
    int                      threadY,
    int                      threadZ,
    BitMaskWordType*         mActiveMask) -> bool
{
    if constexpr (activeMaskMemoryLayout == Neon::MemoryLayout::arrayOfStructs) {
        // 6 = log_2 64
        const uint32_t threadPitch = threadX + threadY * dataBlockSizeX + threadZ * dataBlockSizeX * dataBlockSizeY;
        // threadPitch >> log2OfbitMaskWordSize
        // the same as: threadPitch / 2^{log2OfbitMaskWordSize}
        const uint32_t wordIdx = threadPitch >> log2OfbitMaskWordSize;
        // threadPitch & ((bitMaskWordType(bitMaskStorageBitWidth)) - 1);
        // same as threadPitch % 2^{log2OfbitMaskWordSize}
        const uint32_t  offsetInWord = threadPitch & ((BitMaskWordType(bitMaskStorageBitWidth)) - 1);
        BitMaskWordType mask = BitMaskWordType(1) << offsetInWord;

        uint32_t const  cardinality = getRequiredWordsForBlockBitMask();
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