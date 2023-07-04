#pragma once

namespace Neon::domain::details::mGrid {

template <typename T, int C>
mPartition<T, C>::mPartition()
    : Neon::domain::details::bGrid::bPartition<T, C, kStaticBlock>(),
      mMemParent(nullptr),
      mMemChild(nullptr),
      mParentBlockID(nullptr),
      mMaskLowerLevel(nullptr),
      mMaskUpperLevel(nullptr),
      mChildBlockID(nullptr),
      mRefFactors(nullptr),
      mSpacing(nullptr)
{
}

template <typename T, int C>
mPartition<T, C>::mPartition(int                level,
                             T*                 mem,
                             T*                 memParent,
                             T*                 memChild,
                             int                cardinality,
                             Idx::DataBlockIdx* neighbourBlocks,
                             Neon::int32_3d*    origin,
                             uint32_t*          parentBlockID,
                             MaskT*             mask,
                             MaskT*             maskLowerLevel,
                             MaskT*             maskUpperLevel,
                             uint32_t*          childBlockID,
                             uint32_t*          parentNeighbourBlocks,
                             NghIdx*            stencilNghIndex,
                             int*               refFactors,
                             int*               spacing)
    : Neon::domain::details::bGrid::bPartition<T, C, kStaticBlock>(0, cardinality, mem, neighbourBlocks, mask, origin, stencilNghIndex),
      mLevel(level),
      mMemParent(memParent),
      mMemChild(memChild),
      mParentBlockID(parentBlockID),
      mMaskLowerLevel(maskLowerLevel),
      mMaskUpperLevel(maskUpperLevel),
      mChildBlockID(childBlockID),
      mParentNeighbourBlocks(parentNeighbourBlocks),
      mRefFactors(refFactors),
      mSpacing(spacing)
{
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline Neon::index_3d mPartition<T, C>::getGlobalIndex(Idx gidx) const
{
    const int sp = (mLevel == 0) ? 1 : mSpacing[mLevel - 1];

    Neon::index_3d ret = this->mOrigin[gidx.getDataBlockIdx()];
    ret.x += gidx.mInDataBlockIdx.x * sp;
    ret.y += gidx.mInDataBlockIdx.y * sp;
    ret.z += gidx.mInDataBlockIdx.z * sp;
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

    //const uint32_t childPitch =
    //    // stride across all block before cell's block
    //    gidx.getDataBlockIdx() *
    //        gidx.memBlock3DSize.x * gidx.memBlock3DSize * gidx.memBlock3DSize +
    //    // stride within the block
    //    gidx.getInDataBlockIdx().x +
    //    gidx.getInDataBlockIdx().y * gidx.memBlock3DSize.x +
    //    gidx.getInDataBlockIdx().z * gidx.memBlock3DSize.x * gidx.memBlock3DSize.y;
    //
    //return mChildBlockID[childPitch];
    return std::numeric_limits<uint32_t>::max();
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::getChild(const Idx& parent_cell,
                                                             NghIdx     child) const -> Idx
{
    Idx childCell;
    childCell.mDataBlockIdx = std::numeric_limits<typename Idx::DataBlockIdx>::max();

    //if (hasChildren(parent_cell)) {
    //    childCell.mDataBlockIdx = childID(parent_cell);
    //    childCell.mInDataBlockIdx.x = child.x;
    //    childCell.mInDataBlockIdx.y = child.y;
    //    childCell.mInDataBlockIdx.z = child.z;
    //}
    return childCell;
}


template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::childVal(const Idx& childCell,
                                                             int        card) -> T&
{
    return mMemChild[this->helpGetPitch(childCell, card)];
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::childVal(const Idx& childCell,
                                                             int        card) const -> const T&
{
    return mMemChild[this->helpGetPitch(childCell, card)];
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::childVal(const Idx&   parent_cell,
                                                             const NghIdx child,
                                                             int          card,
                                                             const T&     alternativeVal) const -> NghData
{
    NghData ret;
    ret.mData = alternativeVal;
    ret.mIsValid = false;
    //if (!parent_cell.mIsActive || !hasChildren(parent_cell)) {
    //    return ret;
    //}

    //Idx child_cell = getChild(parent_cell, child);

    //if (!child_cell.mIsActive) {
    //    return ret;
    //}

    //ret.mIsValid = true;
    //ret.mData = childVal(child_cell, card);

    return ret;
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::hasChildren(const Idx& cell) const -> bool
{
    if (mMemChild == nullptr || mMaskLowerLevel == nullptr || mLevel == 0) {
        return false;
    }
    if (childID(cell) == std::numeric_limits<typename Idx::DataBlockIdx>::max()) {
        return false;
    }
    return true;
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::hasChildren(const Idx& cell, const NghIdx nghDir) const -> bool
{
    if (mMemChild == nullptr || mMaskLowerLevel == nullptr || mLevel == 0) {
        return false;
    }

    Idx nghCell = this->getNghCell(cell, nghDir, this->getneighbourBlocksPtr(cell));
    if (!this->isActive(nghCell)) {
        return false;
    }
    return hasChildren(nghCell);
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::getParent(const Idx& cell) const -> Idx
{
    Idx parentCell;
    parentCell.mDataBlockIdx = std::numeric_limits<typename Idx::DataBlockIdx>::max();
    if (mMemParent != nullptr) {
        parentCell.mDataBlockIdx = mParentBlockID[cell.mDataBlockIdx];
        const Neon::index_3d g = this->getGlobalIndex(cell);
        const uint32_t       sp = getSpacing(mLevel);
        parentCell.mInDataBlockIdx.x = (g.x / sp) % kMemBlockSizeX;
        parentCell.mInDataBlockIdx.y = (g.y / sp) % kMemBlockSizeY;
        parentCell.mInDataBlockIdx.z = (g.z / sp) % kMemBlockSizeZ;
    }
    return parentCell;
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::parentVal(const Idx& eId,
                                                              int        card) -> T&
{
    auto parentCell = getParent(eId);
    if (parentCell.isActive()) {
        return mMemParent[this->helpGetPitch(parentCell, card)];
    }
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::parentVal(const Idx& eId,
                                                              int        card) const -> const T&
{
    auto parentCell = getParent(eId);
    if (parentCell.isActive()) {
        return mMemParent[this->helpGetPitch(parentCell, card)];
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
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::getUncle(const Idx&   cell,
                                                             const NghIdx direction) const -> Idx
{
    Idx uncle = getParent(cell);
    if (uncle.isActive()) {
        uncle = this->helpGetNghIdx(uncle, direction, mParentNeighbourBlocks);
        if (!this->isActive(uncle, mMaskUpperLevel)) {
            uncle.mDataBlockIdx = std::numeric_limits<typename Idx::DataBlockIdx>::max();
        }
    }
    return uncle;
}

template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::uncleVal(const Idx&   cell,
                                                             const NghIdx direction,
                                                             int          card,
                                                             const T&     alternativeVal) const -> NghData
{
    NghData ret;
    ret.mData = alternativeVal;
    ret.mIsValid = false;

    Idx uncle = getUncle(cell, direction);
    ret.mIsValid = uncle.isActive();
    if (ret.mIsValid) {
        ret.mData = mMemParent[this->helpGetPitch(uncle, card)];
    }
    return ret;
}


template <typename T, int C>
NEON_CUDA_HOST_DEVICE inline auto mPartition<T, C>::uncleVal(const Idx&   cell,
                                                             const NghIdx direction,
                                                             int          card) const -> T&
{
    Idx uncle = getUncle(cell, direction);
    assert(this->isActive(uncle, mMaskUpperLevel));
    return mMemParent[this->helpGetPitch(uncle, card)];
}

}  // namespace Neon::domain::details::mGrid