#pragma once

#include "Neon/core/core.h"



namespace Neon::domain::details::bGrid {

// Common forward declarations
class bGrid;
class bSpan;
template <typename T, int C>
class bPartition;

class bIndex
{
   public:
    friend class bSpan;
    using OuterIdx = bIndex;

    using NghIdx = int8_3d;
    template <typename T, int C>
    friend class bPartition;

    template <typename T, int C>
    friend class bField;

    friend class bSpan;
    friend class bGrid;

    using DataBlockCount = uint32_t;
    using DataBlockIdx = uint32_t;
    using InDataBlockIdx = int8_3d;


    bIndex() = default;
    virtual ~bIndex() = default;

    NEON_CUDA_HOST_DEVICE inline explicit bIndex(const DataBlockIdx&            blockIdx,
                                                 const InDataBlockIdx::Integer& x,
                                                 const InDataBlockIdx::Integer& y,
                                                 const InDataBlockIdx::Integer& z);


    // the local index within the block
    InDataBlockIdx mInDataBlockIdx;
    DataBlockIdx   mDataBlockIdx;

    //    NEON_CUDA_HOST_DEVICE inline auto getLocal1DID() const -> Location::Integer;
    //
    //    NEON_CUDA_HOST_DEVICE inline auto getMaskLocalID() const -> int32_t;
    //
    //    NEON_CUDA_HOST_DEVICE inline auto getMaskBitPosition() const -> int32_t;
    //
    //    NEON_CUDA_HOST_DEVICE inline auto getBlockMaskStride() const -> int32_t;
    //
    //    NEON_CUDA_HOST_DEVICE inline auto computeIsActive(const uint32_t* activeMask) const -> bool;
    //
    //    static NEON_CUDA_HOST_DEVICE inline auto getNeighbourBlockID(const int16_3d& blockOffset) -> uint32_t;
    //
    //    NEON_CUDA_HOST_DEVICE inline auto pitch(int card) const -> Location::Integer;
};
}  // namespace Neon::domain::details::bGrid

#include "Neon/domain/details/bGrid/bIndex_imp.h"
