#pragma once
#include "Neon/domain/details/mGrid/mPartition.h"

namespace Neon::domain::details::mGrid {

template <typename T, int C>
mPartition<T, C>::mPartition()
    : Neon::domain::details::bGrid::bPartition<T, C>(),
      mMemParent(nullptr),
      mMemChild(nullptr),
      mParentBlockID(nullptr),
      mParentLocalID(nullptr),
      mMaskLowerLevel(nullptr),
      mChildBlockID(nullptr),
      mRefFactors(nullptr),
      mSpacing(nullptr)
{
}

template <typename T, int C>
mPartition<T, C>::mPartition(Neon::DataView  dataView,
                             int             level,
                             T*              mem,
                             T*              memParent,
                             T*              memChild,
                             int             cardinality,
                             uint32_t*       neighbourBlocks,
                             Neon::int32_3d* origin,
                             uint32_t*       parentBlockID,
                             Idx::Location* parentLocalID,
                             uint32_t*       mask,
                             uint32_t*       maskLowerLevel,
                             uint32_t*       childBlockID,
                             T               outsideValue,
                             NghIdx*       stencilNghIndex,
                             int*            refFactors,
                             int*            spacing)
    : Neon::domain::details::bGrid::bPartition<T, C>(0, dataView, mem, cardinality, neighbourBlocks, origin, mask, outsideValue),
      mLevel(level),
      mMemParent(memParent),
      mMemChild(memChild),
      mParentBlockID(parentBlockID),
      mParentLocalID(parentLocalID),
      mMaskLowerLevel(maskLowerLevel),
      mChildBlockID(childBlockID),
      mRefFactors(refFactors),
      mSpacing(spacing)
{
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline Neon::index_3d mPartition<T, C>::mapToGlobal(const Idx& cell) const
{
    Neon::index_3d ret = this->mOrigin[cell.mBlockID];
#ifdef NEON_PLACE_CUDA_DEVICE
    if constexpr (Cell::sUseSwirlIndex) {
        auto swirl = cell.toSwirl();
        ret.x += swirl.mLocation.x;
        ret.y += swirl.mLocation.y;
        ret.z += swirl.mLocation.z;
    } else {
#endif
        const int sp = (mLevel == 0) ? 1 : mSpacing[mLevel - 1];
        ret.x += cell.mLocation.x * sp;
        ret.y += cell.mLocation.y * sp;
        ret.z += cell.mLocation.z * sp;
#ifdef NEON_PLACE_CUDA_DEVICE
    }
#endif
    return ret;
}


template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::getRefFactor(const int level) const -> int
{
    return mRefFactors[level];
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::getSpacing(const int level) const -> int
{
    return mSpacing[level];
}

template <typename T, int C>
inline NEON_CUDA_HOST_DEVICE auto mPartition<T, C>::childID(const Idx& cell) const -> uint32_t
{
    //return the child block id corresponding to this cell
    //the child block id lives at level mLevel-1

    const uint32_t childPitch =
        //stride across all block before cell's block
        cell.mBlockID *
            cell.mBlockSize * cell.mBlockSize * cell.mBlockSize +
        //stride within the block
        cell.mLocation.x +
        cell.mLocation.y * cell.mBlockSize +
        cell.mLocation.z * cell.mBlockSize * cell.mBlockSize;

    return mChildBlockID[childPitch];
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::hasParent(const Idx& cell) const -> bool
{
    if (mMemParent) {
        return true;
    }
    return false;
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::getChild(const Idx&   parent_cell,
                                                             Neon::int8_3d child) const -> Idx
{
    Idx childCell;
    childCell.mBlockID = childID(parent_cell);
    childCell.mBlockSize = mRefFactors[mLevel - 1];
    childCell.mLocation.x = child.x;
    childCell.mLocation.y = child.y;
    childCell.mLocation.z = child.z;
    childCell.mIsActive = childCell.computeIsActive(mMaskLowerLevel);
    return childCell;
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::childVal(const Idx& childCell,
                                                             int         card) -> T&
{
    return mMemChild[this->pitch(childCell, card)];
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::childVal(const Idx& childCell,
                                                             int         card) const -> const T&
{
    return mMemChild[this->pitch(childCell, card)];
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::childVal(const Idx&   parent_cell,
                                                             Neon::int8_3d child,
                                                             int           card,
                                                             const T&      alternativeVal) const -> NghData<T>
{
    NghData<T> ret;
    ret.value = alternativeVal;
    ret.mIsValid = false;
    if (!parent_cell.mIsActive || !hasChildren(parent_cell)) {
        return ret;
    }

    Idx child_cell = getChild(parent_cell, child);

    if (!child_cell.mIsActive) {
        return ret;
    }

    ret.mIsValid = true;
    ret.value = childVal(child_cell, card);

    return ret;
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::hasChildren(const Idx& cell) const -> bool
{
    if (mMemChild == nullptr || mMaskLowerLevel == nullptr || mLevel == 0) {
        return false;
    }
    if (childID(cell) == std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    return true;
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::parent(const Idx& eId,
                                                           int         card) -> T&
{
    if (mMemParent != nullptr) {
        Idx parentCell;
        parentCell.mBlockID = mParentBlockID[eId.mBlockID];
        parentCell.mLocation = mParentLocalID[eId.mBlockID];
        parentCell.mBlockSize = mRefFactors[mLevel + 1];
        return mMemParent[this->pitch(parentCell, card)];
    }
}


}  // namespace Neon::domain::details::mGrid