#include "Neon/domain/internal/bGrid/bPartitionIndexSpace.h"

namespace Neon::domain::internal::bGrid {

NEON_CUDA_HOST_DEVICE inline auto bPartitionIndexSpace::setCell(
    bCell&                         cell,
    [[maybe_unused]] const size_t& x,
    [[maybe_unused]] const size_t& y,
    [[maybe_unused]] const size_t& z) const -> void
{
#ifdef NEON_PLACE_CUDA_DEVICE
    assert(bCell::sBlockAllocGranularity == blockDim.x);
    assert(bCell::sBlockAllocGranularity == blockDim.y);
    assert(bCell::sBlockAllocGranularity == blockDim.z);

    const uint32_t trayID = blockIdx.x;

    const uint32_t blocksPerTray = bCell::sBlockAllocGranularity / mBlockSize;

    const Neon::index_3d blockIDWithinTray(threadIdx.x / mBlockSize,
                                           threadIdx.y / mBlockSize,
                                           threadIdx.z / mBlockSize);

    //first term accounts for all blocks located in all trays before this one
    //second term linearized the blockID within the tray into 1D index
    cell.mBlockID = trayID * blocksPerTray * blocksPerTray * blocksPerTray +
                    blockIDWithinTray.mPitch(Neon::index_3d(blocksPerTray, blocksPerTray, blocksPerTray));


    cell.mLocation.x = threadIdx.x % mBlockSize;
    cell.mLocation.y = threadIdx.y % mBlockSize;
    cell.mLocation.z = threadIdx.z % mBlockSize;

#else
    cell.mBlockID = static_cast<uint32_t>(x) / (mBlockSize * mBlockSize * mBlockSize);
    Cell::Location::Integer reminder = static_cast<Cell::Location::Integer>(x % (mBlockSize * mBlockSize * mBlockSize));
    cell.set().z = reminder / (static_cast<Cell::Location::Integer>(mBlockSize * mBlockSize));
    reminder -= (cell.set().z * (static_cast<Cell::Location::Integer>(mBlockSize * mBlockSize)));
    cell.set().y = reminder / static_cast<Cell::Location::Integer>(mBlockSize);
    cell.set().x = reminder % static_cast<Cell::Location::Integer>(mBlockSize);
#endif
    cell.mBlockSize = mBlockSize;
    cell.mIsActive = true;
}

NEON_CUDA_HOST_DEVICE inline auto
bPartitionIndexSpace::setAndValidate(bCell&        cell,
                                     const size_t& x,
                                     const size_t& y,
                                     const size_t& z) const -> bool
{
    setCell(cell, x, y, z);

#ifdef NEON_PLACE_CUDA_DEVICE
    uint32_t*       activeMask = mDeviceActiveMask;
    Neon::int32_3d* blockOrigin = mDeviceBlockOrigin;
#else
    uint32_t*       activeMask = mHostActiveMask;
    Neon::int32_3d* blockOrigin = mHostBlockOrigin;
#endif


    if (blockOrigin[cell.mBlockID].x + cell.mLocation.x * mSpacing >= mDomainSize.x ||
        blockOrigin[cell.mBlockID].y + cell.mLocation.y * mSpacing >= mDomainSize.y ||
        blockOrigin[cell.mBlockID].z + cell.mLocation.z * mSpacing >= mDomainSize.z ||
        !cell.computeIsActive(activeMask)) {
        cell.mIsActive = false;
    }

    return cell.mIsActive;
}

}  // namespace Neon::domain::internal::bGrid