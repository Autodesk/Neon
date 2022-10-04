#include "Neon/domain/internal/bGrid/bPartitionIndexSpace.h"

namespace Neon::domain::internal::bGrid {

NEON_CUDA_HOST_DEVICE inline auto bPartitionIndexSpace::setCell(
    bCell&                         cell,
    [[maybe_unused]] const size_t& x,
    [[maybe_unused]] const size_t& y,
    [[maybe_unused]] const size_t& z) const -> void
{
#ifdef NEON_PLACE_CUDA_DEVICE
    assert(mBlockSize == blockDim.x);
    assert(mBlockSize == blockDim.y);
    assert(mBlockSize == blockDim.z);
    cell.mBlockID = blockIdx.x;
    cell.mBlockSize = mBlockSize;
    cell.mLocation.x = threadIdx.x;
    cell.mLocation.y = threadIdx.y;
    cell.mLocation.z = threadIdx.z;
#else
    cell.mBlockID = static_cast<uint32_t>(x) / (mBlockSize * mBlockSize * mBlockSize);
    Cell::Location::Integer reminder = static_cast<Cell::Location::Integer>(x % (mBlockSize * mBlockSize * mBlockSize));
    cell.set().z = reminder / (static_cast<Cell::Location::Integer>(mBlockSize * mBlockSize));
    reminder -= (cell.set().z * (static_cast<Cell::Location::Integer>(mBlockSize * mBlockSize)));
    cell.set().y = reminder / static_cast<Cell::Location::Integer>(mBlockSize);
    cell.set().x = reminder % static_cast<Cell::Location::Integer>(mBlockSize);
#endif
    cell.mIsActive = true;
}

NEON_CUDA_HOST_DEVICE inline auto
bPartitionIndexSpace::setAndValidate(bCell&                         cell,
                                     [[maybe_unused]] const size_t& x,
                                     [[maybe_unused]] const size_t& y,
                                     [[maybe_unused]] const size_t& z) const -> bool
{
    setCell(cell, x, y, z);

#ifdef NEON_PLACE_CUDA_DEVICE
    uint32_t*       activeMask = mDeviceActiveMask;
    Neon::int32_3d* blockOrigin = mDeviceBlockOrigin;
#else
    uint32_t*       activeMask = mHostActiveMask;
    Neon::int32_3d* blockOrigin = mHostBlockOrigin;
#endif

    if (cell.mBlockID >= mNumBlocks) {
        cell.mIsActive = false;
        return false;
    }

    if (blockOrigin[cell.mBlockID].x + cell.mLocation.x >= mDomainSize.x ||
        blockOrigin[cell.mBlockID].y + cell.mLocation.y >= mDomainSize.y ||
        blockOrigin[cell.mBlockID].z + cell.mLocation.z >= mDomainSize.z ||
        !cell.computeIsActive(activeMask)) {
        cell.mIsActive = false;
    }

    return true;
}

}  // namespace Neon::domain::internal::bGrid