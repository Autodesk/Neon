#pragma once
#include "Neon/domain/internal/mGrid/mPartition.h"

namespace Neon::domain::internal::mGrid {

template <typename T, int C>
mPartition<T, C>::mPartition()
    : Neon::domain::internal::bGrid::bPartition<T, C>(),
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
                             Cell::Location* parentLocalID,
                             uint32_t*       mask,
                             uint32_t*       maskLowerLevel,
                             uint32_t*       childBlockID,
                             uint32_t*       parentNeighbourBlocks,
                             T               outsideValue,
                             nghIdx_t*       stencilNghIndex,
                             int*            refFactors,
                             int*            spacing)
    : Neon::domain::internal::bGrid::bPartition<T, C>(dataView, mem, cardinality, neighbourBlocks, origin, mask, outsideValue, stencilNghIndex),
      mLevel(level),
      mMemParent(memParent),
      mMemChild(memChild),
      mParentBlockID(parentBlockID),
      mParentLocalID(parentLocalID),
      mMaskLowerLevel(maskLowerLevel),
      mChildBlockID(childBlockID),
      mParentNeighbourBlocks(parentNeighbourBlocks),
      mRefFactors(refFactors),
      mSpacing(spacing)
{
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline Neon::index_3d mPartition<T, C>::mapToGlobal(const Cell& cell) const
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
        const int sp = getSpacing();
        ret.x += cell.mLocation.x * sp;
        ret.y += cell.mLocation.y * sp;
        ret.z += cell.mLocation.z * sp;
#ifdef NEON_PLACE_CUDA_DEVICE
    }
#endif
    return ret;
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::nghVal(const Cell& eId,
                                                           uint8_t     nghID,
                                                           int         card,
                                                           const T&    alternativeVal) const -> NghInfo<T>
{
    nghIdx_t nghOffset = mStencilNghIndex[nghID];
    return mPartition<T, C>::nghVal(eId, nghOffset, card, alternativeVal);
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::nghVal(const Cell&     cell,
                                                           const nghIdx_t& offset,
                                                           const int       card,
                                                           const T         alternativeVal) const -> NghInfo<T>
{
    NghInfo<T> ret;
    ret.value = alternativeVal;
    ret.isValid = false;
    if (!cell.mIsActive) {
        return ret;
    }

    Cell ngh_cell = getNghCell(cell, offset, getneighbourBlocksPtr(cell), getSpacing());
    if (ngh_cell.isActive()) {
        ret.isValid = ngh_cell.computeIsActive(mMask);
        if (ret.isValid) {
            if (mIsInSharedMem) {
                ngh_cell.mLocation.x = cell.mLocation.x + offset.x;
                ngh_cell.mLocation.y = cell.mLocation.y + offset.y;
                ngh_cell.mLocation.z = cell.mLocation.z + offset.z;
            }
            ret.value = this->operator()(ngh_cell, card);
        }
    }

    return ret;
}


template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::getRefFactor(const int level) const -> int
{
    return mRefFactors[level];
}

template <typename T, int C>
inline NEON_CUDA_HOST_DEVICE auto mPartition<T, C>::getSpacing() const -> int
{
    return getSpacing(mLevel);
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::getSpacing(const int level) const -> int
{
    return (level == 0) ? 1 : mSpacing[mLevel - 1];
}

template <typename T, int C>
inline NEON_CUDA_HOST_DEVICE auto mPartition<T, C>::childID(const Cell& cell) const -> uint32_t
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
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::getChild(const Cell&   parent_cell,
                                                             Neon::int8_3d child) const -> Cell
{
    Cell childCell;
    if (hasChildren(parent_cell)) {
        childCell.mBlockID = childID(parent_cell);
        childCell.mBlockSize = mRefFactors[mLevel - 1];
        childCell.mLocation.x = child.x;
        childCell.mLocation.y = child.y;
        childCell.mLocation.z = child.z;
        childCell.mIsActive = childCell.computeIsActive(mMaskLowerLevel);
    }
    return childCell;
}


template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::childVal(const Cell& childCell,
                                                             int         card) -> T&
{
    return mMemChild[this->pitch(childCell, card)];
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::childVal(const Cell& childCell,
                                                             int         card) const -> const T&
{
    return mMemChild[this->pitch(childCell, card)];
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::childVal(const Cell&   parent_cell,
                                                             Neon::int8_3d child,
                                                             int           card,
                                                             const T&      alternativeVal) const -> NghInfo<T>
{
    NghInfo<T> ret;
    ret.value = alternativeVal;
    ret.isValid = false;
    if (!parent_cell.mIsActive || !hasChildren(parent_cell)) {
        return ret;
    }

    Cell child_cell = getChild(parent_cell, child);

    if (!child_cell.mIsActive) {
        return ret;
    }

    ret.isValid = true;
    ret.value = childVal(child_cell, card);

    return ret;
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::hasChildren(const Cell& cell) const -> bool
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
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::hasChildren(const Cell& cell, const Neon::int8_3d nghDir) const -> bool
{
    if (mMemChild == nullptr || mMaskLowerLevel == nullptr || mLevel == 0) {
        return false;
    }

    Cell nghCell = this->getNghCell(cell, nghDir, this->getneighbourBlocksPtr(cell));
    if (!nghCell.isActive()) {
        return false;
    }
    return hasChildren(nghCell);
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::getParent(const Cell& cell) const -> Cell
{
    Cell parentCell;
    if (mMemParent != nullptr) {
        parentCell.mBlockID = mParentBlockID[cell.mBlockID];
        parentCell.mLocation = mParentLocalID[cell.mBlockID];
        parentCell.mBlockSize = mRefFactors[mLevel + 1];
        parentCell.mIsActive = true;
    }
    return parentCell;
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::parentVal(const Cell& eId,
                                                              int         card) -> T&
{
    auto parentCell = getParent(eId);
    if (parentCell.isActive()) {
        return mMemParent[this->pitch(parentCell, card)];
    }
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::parentVal(const Cell& eId,
                                                              int         card) const -> const T&
{
    auto parentCell = getParent(eId);
    if (parentCell.isActive()) {
        return mMemParent[this->pitch(parentCell, card)];
    }
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::hasParent(const Cell& cell) const -> bool
{
    if (mMemParent) {
        return true;
    }
    return false;
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::getUncle(const Cell&   cell,
                                                             Neon::int8_3d direction) const -> Cell
{
    Cell uncle = getParent(cell);
    if (uncle.isActive()) {
        uncle = this->getNghCell(uncle, direction, (mParentNeighbourBlocks + (26 * uncle.mBlockID)));
        uncle.mBlockSize = mRefFactors[mLevel + 1];
        uncle.mIsActive = uncle.mBlockID != std::numeric_limits<uint32_t>::max();
    }
    return uncle;
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::uncleVal(const Cell&   cell,
                                                             Neon::int8_3d direction,
                                                             int           card,
                                                             const T&      alternativeVal) const -> NghInfo<T>
{
    NghInfo<T> ret;
    ret.value = alternativeVal;
    ret.isValid = false;

    Cell uncle = getUncle(cell, direction);
    ret.isValid = uncle.isActive();
    if (ret.isValid) {
        ret.value = mMemParent[this->pitch(uncle, card)];
    }
    return ret;
}


}  // namespace Neon::domain::internal::mGrid