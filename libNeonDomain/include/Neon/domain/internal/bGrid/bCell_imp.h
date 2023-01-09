#include "Neon/domain/internal/bGrid/bCell.h"

namespace Neon::domain::internal::bGrid {

NEON_CUDA_HOST_DEVICE inline bCell::bCell(const Location& location)
{
    mLocation = location;
    mIsActive = true;
}

NEON_CUDA_HOST_DEVICE inline bCell::bCell(const Location::Integer& x,
                                          const Location::Integer& y,
                                          const Location::Integer& z)
{
    mLocation.x = x;
    mLocation.y = y;
    mLocation.z = z;
    mIsActive = true;
}

NEON_CUDA_HOST_DEVICE inline auto bCell::set() -> Location&
{
    return mLocation;
}
NEON_CUDA_HOST_DEVICE inline auto bCell::get() const -> const Location&
{
    return mLocation;
}

NEON_CUDA_HOST_DEVICE inline auto bCell::getLocal1DID() const -> Location::Integer
{
    //On the GPU, the cell index is always the canonical index to preserve coalescing
    //On the CPU and in case of swirl index, we map the index to its swirl index
#ifdef NEON_PLACE_CUDA_DEVICE
    if constexpr (sUseSwirlIndex) {
        //the swirl index changes the xy coordinates only and keeps the z as it is
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

NEON_CUDA_HOST_DEVICE inline auto bCell::swirlToCanonical(const Location::Integer id) -> Location::Integer
{
    //from 0-7, no change
    if (id == 8) {
        return 15;
    } else if (id == 9) {
        return 23;
    } else if (id == 10) {
        return 31;
    } else if (id == 11) {
        return 39;
    } else if (id == 12) {
        return 47;
    } else if (id == 13) {
        return 55;
    } else if (id == 14) {
        return 63;
    } else if (id == 15) {
        return 62;
    } else if (id == 16) {
        return 61;
    } else if (id == 17) {
        return 60;
    } else if (id == 18) {
        return 59;
    } else if (id == 19) {
        return 58;
    } else if (id == 20) {
        return 57;
    } else if (id == 21) {
        return 56;
    } else if (id == 22) {
        return 48;
    } else if (id == 23) {
        return 40;
    } else if (id == 24) {
        return 32;
    } else if (id == 25) {
        return 24;
    } else if (id == 26) {
        return 16;
    } else if (id == 27) {
        return 8;
    } else if (id == 28) {
        return 9;
    } else if (id == 29) {
        return 10;
    } else if (id == 30) {
        return 11;
    } else if (id == 31) {
        return 12;
    } else if (id == 32) {
        return 13;
    } else if (id == 33) {
        return 14;
    } else if (id == 34) {
        return 22;
    } else if (id == 35) {
        return 30;
    } else if (id == 36) {
        return 38;
    } else if (id == 37) {
        return 46;
    } else if (id == 38) {
        return 54;
    } else if (id == 39) {
        return 53;
    } else if (id == 40) {
        return 52;
    } else if (id == 41) {
        return 51;
    } else if (id == 42) {
        return 50;
    } else if (id == 43) {
        return 49;
    } else if (id == 44) {
        return 41;
    } else if (id == 45) {
        return 33;
    } else if (id == 46) {
        return 25;
    } else if (id == 47) {
        return 17;
    } else if (id == 48) {
        return 18;
    } else if (id == 49) {
        return 19;
    } else if (id == 50) {
        return 20;
    } else if (id == 51) {
        return 21;
    } else if (id == 52) {
        return 29;
    } else if (id == 53) {
        return 37;
    } else if (id == 54) {
        return 45;
    } else if (id == 55) {
        return 44;
    } else if (id == 56) {
        return 43;
    } else if (id == 57) {
        return 42;
    } else if (id == 58) {
        return 34;
    } else if (id == 59) {
        return 26;
    } else if (id == 60) {
        return 27;
    } else if (id == 61) {
        return 28;
    } else if (id == 62) {
        return 36;
    } else if (id == 63) {
        return 35;
    } else {
        assert(1 != 1);
        return -1;
    }
}

NEON_CUDA_HOST_DEVICE inline auto bCell::canonicalToSwirl(const Location::Integer id) -> Location::Integer
{
    //The swirl indexing scheme which can only be used for 2 grid
    // +---------------------------------------+
    // | 21 | 20 | 19 | 18 | 17 | 16 | 15 | 14 |
    // +---------------------------------------+
    // | 22 | 43 | 42 | 41 | 40 | 39 | 38 | 13 |
    // +---------------------------------------+
    // | 23 | 44 | 57 | 56 | 55 | 54 | 37 | 12 |
    // +---------------------------------------+
    // | 24 | 45 | 58 | 63 | 62 | 53 | 36 | 11 |
    // +---------------------------------------+
    // | 25 | 46 | 59 | 60 | 61 | 52 | 35 | 10 |
    // +---------------------------------------+
    // | 26 | 47 | 48 | 49 | 50 | 51 | 34 | 9  |
    // +---------------------------------------+
    // | 27 | 28 | 29 | 30 | 31 | 32 | 33 | 8  |
    // +---------------------------------------+
    // | 0  | 1  | 2  | 3  | 4  | 5  | 6  | 7  |
    // +---------------------------------------+

    //from 0-7, no change
    if (id == 0) {
        return 0;
    } else if (id == 1) {
        return 1;
    } else if (id == 2) {
        return 2;
    } else if (id == 3) {
        return 3;
    } else if (id == 4) {
        return 4;
    } else if (id == 5) {
        return 5;
    } else if (id == 6) {
        return 6;
    } else if (id == 7) {
        return 7;
    } else if (id == 8) {
        return 27;
    } else if (id == 9) {
        return 28;
    } else if (id == 10) {
        return 29;
    } else if (id == 11) {
        return 30;
    } else if (id == 12) {
        return 31;
    } else if (id == 13) {
        return 32;
    } else if (id == 14) {
        return 33;
    } else if (id == 15) {
        return 8;
    } else if (id == 16) {
        return 26;
    } else if (id == 17) {
        return 47;
    } else if (id == 18) {
        return 48;
    } else if (id == 19) {
        return 49;
    } else if (id == 20) {
        return 50;
    } else if (id == 21) {
        return 51;
    } else if (id == 22) {
        return 34;
    } else if (id == 23) {
        return 9;
    } else if (id == 24) {
        return 25;
    } else if (id == 25) {
        return 46;
    } else if (id == 26) {
        return 59;
    } else if (id == 27) {
        return 60;
    } else if (id == 28) {
        return 61;
    } else if (id == 29) {
        return 52;
    } else if (id == 30) {
        return 35;
    } else if (id == 31) {
        return 10;
    } else if (id == 32) {
        return 24;
    } else if (id == 33) {
        return 45;
    } else if (id == 34) {
        return 58;
    } else if (id == 35) {
        return 63;
    } else if (id == 36) {
        return 62;
    } else if (id == 37) {
        return 53;
    } else if (id == 38) {
        return 36;
    } else if (id == 39) {
        return 11;
    } else if (id == 40) {
        return 23;
    } else if (id == 41) {
        return 44;
    } else if (id == 42) {
        return 57;
    } else if (id == 43) {
        return 56;
    } else if (id == 44) {
        return 55;
    } else if (id == 45) {
        return 54;
    } else if (id == 46) {
        return 37;
    } else if (id == 47) {
        return 12;
    } else if (id == 48) {
        return 22;
    } else if (id == 49) {
        return 43;
    } else if (id == 50) {
        return 42;
    } else if (id == 51) {
        return 41;
    } else if (id == 52) {
        return 40;
    } else if (id == 53) {
        return 39;
    } else if (id == 54) {
        return 38;
    } else if (id == 55) {
        return 13;
    } else if (id == 56) {
        return 21;
    } else if (id == 57) {
        return 20;
    } else if (id == 58) {
        return 19;
    } else if (id == 59) {
        return 18;
    } else if (id == 60) {
        return 17;
    } else if (id == 61) {
        return 16;
    } else if (id == 62) {
        return 15;
    } else if (id == 63) {
        return 14;
    } else {
        assert(1 != 1);
        return -1;
    }
}

NEON_CUDA_HOST_DEVICE inline auto bCell::toSwirl() const -> bCell
{

    Location::Integer xy = canonicalToSwirl(mLocation.x +
                                            mLocation.y * Location::Integer(mBlockSize));

    bCell ret;
    ret.mLocation.x = xy % Location::Integer(mBlockSize);
    ret.mLocation.y = xy / Location::Integer(mBlockSize);
    ret.mLocation.z = mLocation.z;
    ret.mBlockID = mBlockID;
    return ret;
}

NEON_CUDA_HOST_DEVICE inline auto bCell::getMaskLocalID() const -> int32_t
{
    return getLocal1DID() / sMaskSize;
}

NEON_CUDA_HOST_DEVICE inline auto bCell::getMaskBitPosition() const -> int32_t
{
    return getLocal1DID() % sMaskSize;
}

NEON_CUDA_HOST_DEVICE inline auto bCell::getBlockMaskStride() const -> int32_t
{
    return mBlockID * NEON_DIVIDE_UP(mBlockSize * mBlockSize * mBlockSize, sMaskSize);
}


NEON_CUDA_HOST_DEVICE inline auto bCell::computeIsActive(const uint32_t* activeMask) const -> bool
{
    const uint32_t mask = activeMask[getBlockMaskStride() + getMaskLocalID()];
    return (mask & (1 << getMaskBitPosition()));
}


NEON_CUDA_HOST_DEVICE inline auto bCell::isActive() const -> bool
{
    return mIsActive;
}

NEON_CUDA_HOST_DEVICE inline auto bCell::getNeighbourBlockID(const int16_3d& blockOffset) -> uint32_t
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

NEON_CUDA_HOST_DEVICE inline auto bCell::pitch(int card) const -> Location::Integer
{
    return
        //stride across cardinalities before card within the block
        Location::Integer(mBlockSize) * Location::Integer(mBlockSize) * Location::Integer(mBlockSize) * Location::Integer(card) +
        //offset to this cell's data
        getLocal1DID();
}
}  // namespace Neon::domain::internal::bGrid