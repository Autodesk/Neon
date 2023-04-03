#pragma once
#include "Neon/domain/details/bGrid/bIndex.h"

namespace Neon::domain::details::bGrid {

NEON_CUDA_HOST_DEVICE inline bIndex::bIndex(const DataBlockIdx&            blockIdx,
                                            const InDataBlockIdx::Integer& x,
                                            const InDataBlockIdx::Integer& y,
                                            const InDataBlockIdx::Integer& z)
{
    mDataBlockIdx = blockIdx;
    mInDataBlockIdx.x = x;
    mInDataBlockIdx.y = y;
    mInDataBlockIdx.z = z;
    mIsActive = true;
}

NEON_CUDA_HOST_DEVICE inline auto bIndex::set() -> Location&
{
    return mLocation;
}
NEON_CUDA_HOST_DEVICE inline auto bIndex::get() const -> const Location&
{
    return mLocation;
}

NEON_CUDA_HOST_DEVICE inline auto bIndex::getLocal1DID() const -> Location::Integer
{
    // On the GPU, the cell index is always the canonical index to preserve coalescing
    // On the CPU and in case of swirl index, we map the index to its swirl index
#ifdef NEON_PLACE_CUDA_DEVICE
    if constexpr (sUseSwirlIndex) {
        // the swirl index changes the xy coordinates only and keeps the z as it is
        Location::Integer xy = canonicalToSwirl(mLocation.x +
                                                mLocation.y * mBlockSize);

        return xy + mLocation.z * mBlockSize * mBlockSize;
    } else {
#endif
        return mLocation.x +
               mLocation.y * Location::Integer(mBlockSize) +
               mLocation.z * Location::Integer(mBlockSize) * Location::Integer(mBlockSize);
#ifdef NEON_PLACE_CUDA_DEVICE
    }
#endif
}

NEON_CUDA_HOST_DEVICE inline auto bIndex::getMaskLocalID() const -> int32_t
{
    return getLocal1DID() / sMaskSize;
}

NEON_CUDA_HOST_DEVICE inline auto bIndex::getMaskBitPosition() const -> int32_t
{
    return getLocal1DID() % sMaskSize;
}

NEON_CUDA_HOST_DEVICE inline auto bIndex::getBlockMaskStride() const -> int32_t
{
    return mBlockID * NEON_DIVIDE_UP(mBlockSize * mBlockSize * mBlockSize, sMaskSize);
}


NEON_CUDA_HOST_DEVICE inline auto bIndex::computeIsActive(const uint32_t* activeMask) const -> bool
{
    const uint32_t mask = activeMask[getBlockMaskStride() + getMaskLocalID()];
    return (mask & (1 << getMaskBitPosition()));
}


NEON_CUDA_HOST_DEVICE inline auto bIndex::isActive() const -> bool
{
    return mIsActive;
}

NEON_CUDA_HOST_DEVICE inline auto bIndex::getNeighbourBlockID(const int16_3d& blockOffset) -> uint32_t
{
    /* We only store the indices of the immediate neighbor blocks of each block. This method takes a 3d offset
     * from the block assigned to this cell and find 1d index of the neighbor block. This x, y, or z component of
     * this offset could only be -1, 0, or 1 (just the immediate neighbor) and hence the below assert(). The offset
     * can not be 0,0,0 since this is block itself.
     * Below is the map from the 3D offset to the 1d index. Note that we only have 26 neighbor only.
     * Note also that since the offset could be -1, we shift them by 1. After this shift, we should subtract 1
     * from block larger than 12 because, following the Cartesian indexing, index 13 does not exist because this is
     * the one corresponds to 0,0,0
     *
     *            z = -1               z = 0                z = 1
     *        -------------        -------------        -------------
     * Y = 1 | 6 | 7  | 8  |      | 14| 15 | 16 |      | 23| 24 | 25 |
     *        -------------        -------------        -------------
     * Y = 0 | 3 | 4  | 5  |      | 12| XX | 13 |      | 20| 21 | 22 |
     *        -------------        -------------        -------------
     * Y =-1 | 0 | 1  | 2  |      | 9 | 10 | 11 |      | 17| 18 | 19 |
     *        -------------        -------------        -------------
     *    X=   -1   0    1
     *
     */

    assert(blockOffset.x == 1 || blockOffset.x == 0 || blockOffset.x == -1);
    assert(blockOffset.y == 1 || blockOffset.y == 0 || blockOffset.y == -1);
    assert(blockOffset.z == 1 || blockOffset.z == 0 || blockOffset.z == -1);
    assert(!(blockOffset.x == 0 && blockOffset.y == 0 && blockOffset.z == 0));

    uint32_t id = (blockOffset.x + 1) +
                  (blockOffset.y + 1) * 3 +
                  (blockOffset.z + 1) * 3 * 3;
    if (id > 12) {
        id -= 1;
    }
    return id;
}

NEON_CUDA_HOST_DEVICE inline auto bIndex::pitch(int card) const -> Location::Integer
{
    return
        // stride across cardinalities before card within the block
        Location::Integer(mBlockSize) * Location::Integer(mBlockSize) * Location::Integer(mBlockSize) * Location::Integer(card) +
        // offset to this cell's data
        getLocal1DID();
}
}  // namespace Neon::domain::details::bGrid