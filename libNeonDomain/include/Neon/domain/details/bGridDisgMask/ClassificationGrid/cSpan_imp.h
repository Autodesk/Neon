#include <cstdint>
#include "Neon/domain/details/bGridDisgMask/ClassificationGrid/cSpan.h"
namespace Neon::domain::details::disaggregated::bGridMask {
namespace details::cGrid {

template <typename SBlock, int classSelector>
NEON_CUDA_HOST_DEVICE inline auto
cSpan<SBlock, classSelector>::setAndValidateGPUDevice([[maybe_unused]] Idx& bidx) const -> bool
{
#ifdef NEON_PLACE_CUDA_DEVICE
    bidx.mDataBlockIdx = blockIdx.x + mFirstDataBlockOffset;
    bidx.mInDataBlockIdx.x = threadIdx.x;
    bidx.mInDataBlockIdx.y = threadIdx.y;
    bidx.mInDataBlockIdx.z = threadIdx.z;

    const bool isActive = mActiveMask[bidx.mDataBlockIdx].isActive(bidx.mInDataBlockIdx.x, bidx.mInDataBlockIdx.y, bidx.mInDataBlockIdx.z);
    const bool isClass = mClassMask[bidx.mDataBlockIdx * SBlock::memBlockCountElements +
                                    bidx.mInDataBlockIdx.x +
                                    bidx.mInDataBlockIdx.y * SBlock::memBlockSizeX +
                                    bidx.mInDataBlockIdx.z * SBlock::memBlockSizeX * SBlock::memBlockSizeY] == classSelector;
    return isActive && isClass;
#else
    NEON_THROW_UNSUPPORTED_OPERATION("Operation supported only on GPU");
#endif
}

template <typename SBlock, int classSelector>
NEON_CUDA_HOST_DEVICE inline auto
cSpan<SBlock, classSelector>::setAndValidateCPUDevice(Idx&            bidx,
                                                      uint32_t const& dataBlockIdx,
                                                      uint32_t const& x,
                                                      uint32_t const& y,
                                                      uint32_t const& z) const -> bool
{

    bidx.mDataBlockIdx = dataBlockIdx + mFirstDataBlockOffset;
    ;
    bidx.mInDataBlockIdx.x = static_cast<typename Idx::InDataBlockIdx::Integer>(x);
    bidx.mInDataBlockIdx.y = static_cast<typename Idx::InDataBlockIdx::Integer>(y);
    bidx.mInDataBlockIdx.z = static_cast<typename Idx::InDataBlockIdx::Integer>(z);
    const bool isActive = mActiveMask[dataBlockIdx].isActive(bidx.mInDataBlockIdx.x, bidx.mInDataBlockIdx.y, bidx.mInDataBlockIdx.z);
    int const  voxelClass = mClassMask[bidx.mDataBlockIdx * SBlock::memBlockCountElements +
                                      bidx.mInDataBlockIdx.x +
                                      bidx.mInDataBlockIdx.y * SBlock::memBlockSizeX +
                                      bidx.mInDataBlockIdx.z * SBlock::memBlockSizeX * SBlock::memBlockSizeY];
    bool const isClass = voxelClass == classSelector;

    return isActive && isClass;
}

template <typename SBlock, int classSelector>
cSpan<SBlock, classSelector>::cSpan(typename Idx::DataBlockCount                  firstDataBlockOffset,
                                    typename SBlock::BitMask const* NEON_RESTRICT activeMask,
                                    Neon::DataView                                dataView,
                                    uint8_t const* NEON_RESTRICT                  ClassMask)
    : mFirstDataBlockOffset(firstDataBlockOffset),
      mActiveMask(activeMask),
      mDataView(dataView),
      mClassMask(ClassMask)
{
}

}  // namespace details::cGrid
}  // namespace Neon::domain::details::disaggregated::bGridMask