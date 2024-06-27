#include "Neon/domain/details/bGrid/bSpan.h"

namespace Neon::domain::details::bGrid {

template <typename SBlock>
NEON_CUDA_HOST_DEVICE inline auto
bSpan<SBlock>::setAndValidateGPUDevice([[maybe_unused]] Idx& bidx) const -> bool
{
#ifdef NEON_PLACE_CUDA_DEVICE
    bidx.mDataBlockIdx = blockIdx.x + mFirstDataBlockOffset;
    bidx.mInDataBlockIdx.x = threadIdx.x;
    bidx.mInDataBlockIdx.y = threadIdx.y;
    bidx.mInDataBlockIdx.z = threadIdx.z;

    const bool isActive = mActiveMask[bidx.mDataBlockIdx].isActive(bidx.mInDataBlockIdx.x, bidx.mInDataBlockIdx.y, bidx.mInDataBlockIdx.z);

    return isActive;
#else
    NEON_THROW_UNSUPPORTED_OPERATION("Operation supported only on GPU");
#endif
}

template <typename SBlock>
NEON_CUDA_HOST_DEVICE inline auto
bSpan<SBlock>::setAndValidateCPUDevice(Idx&            bidx,
                                       uint32_t const& dataBlockIdx,
                                       uint32_t const& x,
                                       uint32_t const& y,
                                       uint32_t const& z) const -> bool
{

    bidx.mDataBlockIdx = dataBlockIdx + mFirstDataBlockOffset;
    bidx.mInDataBlockIdx.x = static_cast<typename Idx::InDataBlockIdx::Integer>(x);
    bidx.mInDataBlockIdx.y = static_cast<typename Idx::InDataBlockIdx::Integer>(y);
    bidx.mInDataBlockIdx.z = static_cast<typename Idx::InDataBlockIdx::Integer>(z);
    const bool isActive = mActiveMask[dataBlockIdx].isActive(bidx.mInDataBlockIdx.x, bidx.mInDataBlockIdx.y, bidx.mInDataBlockIdx.z);
    return isActive;
}

template <typename SBlock>
bSpan<SBlock>::bSpan(typename Idx::DataBlockCount                  firstDataBlockOffset,
                     typename SBlock::BitMask const* NEON_RESTRICT activeMask,
                     Neon::DataView                                dataView)
    : mFirstDataBlockOffset(firstDataBlockOffset),
      mActiveMask(activeMask),
      mDataView(dataView)
{
}


// #if !defined(NEON_WARP_COMPILATION)
// template <typename SBlock>
// // std::vector<size_t> bSpan<SBlock>::getOffsets() {
// //     return {
// //         // can't use the `offsetof` macro here since bSpan is a template type
// //         0,
// //         sizeof(Idx::DataBlockCount),
// //         sizeof(Idx::DataBlockCount) + sizeof(SBlock::BitMask*)
// //     };
// // }
// std::vector<size_t> bSpan<SBlock>::getOffsets() {
//     std::vector<size_t> offsets;
//     bSpan temp(typename Idx::DataBlockCount(), nullptr, Neon::DataView());

//     offsets.push_back(reinterpret_cast<char*>(&(temp.mFirstDataBlockOffset)) - reinterpret_cast<char*>(&temp));
//     offsets.push_back(reinterpret_cast<char*>(&(temp.mActiveMask)) - reinterpret_cast<char*>(&temp));
//     offsets.push_back(reinterpret_cast<char*>(&(temp.mDataView)) - reinterpret_cast<char*>(&temp));

//     return offsets;
// }
// #endif

#if !defined(NEON_WARP_COMPILATION)
template <typename SBlock>
inline std::vector<size_t> bSpan<SBlock>::getOffsets() {
    bSpan temp({0}, nullptr, {});

    // Calculate offsets directly
    std::vector<size_t> offsets;
    offsets.push_back(reinterpret_cast<size_t>(&(temp.mFirstDataBlockOffset)) - reinterpret_cast<size_t>(&temp));
    offsets.push_back(reinterpret_cast<size_t>(&(temp.mActiveMask)) - reinterpret_cast<size_t>(&temp));
    offsets.push_back(reinterpret_cast<size_t>(&(temp.mDataView)) - reinterpret_cast<size_t>(&temp));
    
    return offsets;
}
#endif

}  // namespace Neon::domain::details::bGrid