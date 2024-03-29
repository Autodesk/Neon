#pragma once

#include "Neon/core/core.h"

namespace Neon::domain::internal::bGrid {
class bCell
{
   public:
    friend class bPartitionIndexSpace;

    template <typename T, int C>
    friend class bPartition;

    template <typename T, int C>
    friend class bField;

    friend class bPartitionIndexSpace;

    friend class bGrid;

    using Location = int16_3d;
    using BlockSizeT = int8_t;
    using OuterCell = bCell;

    static constexpr BlockSizeT sBlockSizeX = 8;
    static constexpr BlockSizeT sBlockSizeY = 8;
    static constexpr BlockSizeT sBlockSizeZ = 8;
    static constexpr BlockSizeT sBlockSize[3]{sBlockSizeX, sBlockSizeY, sBlockSizeZ};

    //We use uint32_t data type to store the block mask and thus the mask size is 32
    //i.e., each entry in the mask array store the state of 32 voxels
    static constexpr uint32_t sMaskSize = 32;

    bCell() = default;
    virtual ~bCell() = default;

    NEON_CUDA_HOST_DEVICE inline auto isActive() const -> bool;

   private:
    //the local index within the block
    Location mLocation;
    uint32_t mBlockID;
    bool     mIsActive;

    NEON_CUDA_HOST_DEVICE inline explicit bCell(const Location::Integer& x,
                                                const Location::Integer& y,
                                                const Location::Integer& z);
    NEON_CUDA_HOST_DEVICE inline explicit bCell(const Location& location);

    NEON_CUDA_HOST_DEVICE inline auto set() -> Location&;

    NEON_CUDA_HOST_DEVICE inline auto get() const -> const Location&;

    NEON_CUDA_HOST_DEVICE inline auto getLocal1DID() const -> Location::Integer;

    NEON_CUDA_HOST_DEVICE inline auto getMaskLocalID() const -> int32_t;

    NEON_CUDA_HOST_DEVICE inline auto getMaskBitPosition() const -> int32_t;

    NEON_CUDA_HOST_DEVICE inline auto getBlockMaskStride() const -> int32_t;

    NEON_CUDA_HOST_DEVICE inline auto computeIsActive(const uint32_t* activeMask) const -> bool;

    static NEON_CUDA_HOST_DEVICE inline auto getNeighbourBlockID(const int16_3d& blockOffset) -> uint32_t;

    NEON_CUDA_HOST_DEVICE inline auto pitch(int card) const -> Location::Integer;
};
}  // namespace Neon::domain::internal::bGrid

#include "Neon/domain/internal/bGrid/bCell_imp.h"