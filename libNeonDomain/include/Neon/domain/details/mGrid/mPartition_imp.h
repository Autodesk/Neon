#pragma once

namespace Neon::domain::details::mGrid {

template <typename T, int C>
mPartition<T, C>::mPartition()
    : Neon::domain::details::bGrid::bPartition<T, C, kStaticBlock>(),
      mMemParent(nullptr),
      mMemChild(nullptr),
      mParentBlockID(nullptr),
      mParentLocalID(nullptr),
      mMaskLowerLevel(nullptr),
      mMaskUpperLevel(nullptr),
      mChildBlockID(nullptr),
      mRefFactors(nullptr),
      mSpacing(nullptr)
{
}

template <typename T, int C>
mPartition<T, C>::mPartition(Neon::DataView /*dataView*/,
                             int                  level,
                             T*                   mem,
                             T*                   memParent,
                             T*                   memChild,
                             int                  cardinality,
                             Idx::DataBlockIdx*   neighbourBlocks,
                             Neon::int32_3d*      origin,
                             uint32_t*            parentBlockID,
                             Idx::InDataBlockIdx* parentLocalID,
                             MaskT*               mask,
                             MaskT*               maskLowerLevel,
                             MaskT*               maskUpperLevel,
                             uint32_t*            childBlockID,
                             uint32_t*            parentNeighbourBlocks,
                             T /*outsideValue*/,
                             NghIdx* stencilNghIndex,
                             int*    refFactors,
                             int*    spacing)
    : Neon::domain::details::bGrid::bPartition<T, C, kStaticBlock>(0, cardinality, mem, neighbourBlocks, mask, origin, stencilNghIndex),
      mLevel(level),
      mMemParent(memParent),
      mMemChild(memChild),
      mParentBlockID(parentBlockID),
      mParentLocalID(parentLocalID),
      mMaskLowerLevel(maskLowerLevel),
      mMaskUpperLevel(maskUpperLevel),
      mChildBlockID(childBlockID),
      mParentNeighbourBlocks(parentNeighbourBlocks),
      mRefFactors(refFactors),
      mSpacing(spacing)
{
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline Neon::index_3d mPartition<T, C>::mapToGlobal(const Idx& gidx) const
{
    Neon::index_3d ret = this->mOrigin[gidx.getDataBlockIdx()];
#ifdef NEON_PLACE_CUDA_DEVICE
    if constexpr (Cell::sUseSwirlIndex) {
        auto swirl = cell.toSwirl();
        ret.x += swirl.mLocation.x;
        ret.y += swirl.mLocation.y;
        ret.z += swirl.mLocation.z;
    } else {
#endif
        const int sp = (mLevel == 0) ? 1 : mSpacing[mLevel - 1];
        ret.x += gidx.getInDataBlockIdx().x * sp;
        ret.y += gidx.getInDataBlockIdx().y * sp;
        ret.z += gidx.getInDataBlockIdx().z * sp;
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
inline NEON_CUDA_HOST_DEVICE auto mPartition<T, C>::childID(const Idx& gidx) const -> uint32_t
{
    // return the child block id corresponding to this cell
    // the child block id lives at level mLevel-1

    const uint32_t childPitch =
        // stride across all block before cell's block
        gidx.getDataBlockIdx() *
            gidx.memBlock3DSize.x * gidx.memBlock3DSize * gidx.memBlock3DSize +
        // stride within the block
        gidx.getInDataBlockIdx().x +
        gidx.getInDataBlockIdx().y * gidx.memBlock3DSize.x +
        gidx.getInDataBlockIdx().z * gidx.memBlock3DSize.x * gidx.memBlock3DSize.y;

    return mChildBlockID[childPitch];
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::getChild(const Idx&    parent_cell,
                                                             Neon::int8_3d child) const -> Idx
{
    Idx childCell;
    if (hasChildren(parent_cell)) {
        childCell.getDataBlockIdx = childID(parent_cell);
        childCell.mBlockSize = mRefFactors[mLevel - 1];
        childCell.mLocation.x = child.x;
        childCell.mLocation.y = child.y;
        childCell.mLocation.z = child.z;
        childCell.mIsActive = childCell.computeIsActive(mMaskLowerLevel);
    }
    return childCell;
}


template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::childVal(const Idx& childCell,
                                                             int        card) -> T&
{
    return mMemChild[this->pitch(childCell, card)];
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::childVal(const Idx& childCell,
                                                             int        card) const -> const T&
{
    return mMemChild[this->pitch(childCell, card)];
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::childVal(const Idx&    parent_cell,
                                                             Neon::int8_3d child,
                                                             int           card,
                                                             const T&      alternativeVal) const -> NghData
{
    NghData ret;
    ret.value = alternativeVal;
    ret.isValid = false;
    if (!parent_cell.mIsActive || !hasChildren(parent_cell)) {
        return ret;
    }

    Idx child_cell = getChild(parent_cell, child);

    if (!child_cell.mIsActive) {
        return ret;
    }

    ret.isValid = true;
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
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::hasChildren(const Idx& cell, const Neon::int8_3d nghDir) const -> bool
{
    if (mMemChild == nullptr || mMaskLowerLevel == nullptr || mLevel == 0) {
        return false;
    }

    Idx nghCell = this->getNghCell(cell, nghDir, this->getneighbourBlocksPtr(cell));
    if (!nghCell.isActive()) {
        return false;
    }
    return hasChildren(nghCell);
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::getParent(const Idx& cell) const -> Idx
{
    Idx parentCell;
    if (mMemParent != nullptr) {
        parentCell.mBlockID = mParentBlockID[cell.mBlockID];
        parentCell.mLocation = mParentLocalID[cell.mBlockID];
        parentCell.mBlockSize = mRefFactors[mLevel + 1];
        parentCell.mIsActive = true;
    }
    return parentCell;
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::parentVal(const Idx& eId,
                                                              int        card) -> T&
{
    auto parentCell = getParent(eId);
    if (parentCell.isActive()) {
        return mMemParent[this->pitch(parentCell, card)];
    }
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::parentVal(const Idx& eId,
                                                              int        card) const -> const T&
{
    auto parentCell = getParent(eId);
    if (parentCell.isActive()) {
        return mMemParent[this->pitch(parentCell, card)];
    }
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
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::getUncle(const Idx&    cell,
                                                             Neon::int8_3d direction) const -> Idx
{
    Idx uncle = getParent(cell);
    if (uncle.isActive()) {
        uncle = this->getNghCell(uncle, direction, (mParentNeighbourBlocks + (26 * uncle.mBlockID)));
        uncle.mBlockSize = mRefFactors[mLevel + 1];
        uncle.mIsActive = uncle.mBlockID != std::numeric_limits<uint32_t>::max();
        if (uncle.mIsActive) {
            uncle.mIsActive = uncle.computeIsActive(mMaskUpperLevel);
        }
    }
    return uncle;
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::uncleVal(const Idx&    cell,
                                                             Neon::int8_3d direction,
                                                             int           card,
                                                             const T&      alternativeVal) const -> NghData
{
    NghData ret;
    ret.value = alternativeVal;
    ret.isValid = false;

    Idx uncle = getUncle(cell, direction);
    ret.isValid = uncle.isActive();
    if (ret.isValid) {
        ret.value = mMemParent[this->pitch(uncle, card)];
    }
    return ret;
}


template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::uncleVal(const Idx&    cell,
                                                             Neon::int8_3d direction,
                                                             int           card) const -> T&
{
    Idx uncle = getUncle(cell, direction);
    assert(uncle.isActive());
    return mMemParent[this->pitch(uncle, card)];
}

}  // namespace Neon::domain::details::mGrid