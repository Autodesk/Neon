#pragma once

#include "Neon/domain/details//bGrid/bIndex.h"
#include "Neon/domain/details//bGrid/bPartition.h"
#include "Neon/domain/interface/NghData.h"

#include "Neon/sys/memory/CUDASharedMemoryUtil.h"

namespace Neon::domain::details::mGrid {

class bPartitionIndexSpace;

template <typename T, int C = 0>
class mPartition : public Neon::bGrid::bGrid::Partition<T, C>
{
   public:
    using PartitionIndexSpace = Neon::bGrid::Span;
    using Idx = Neon::bGrid::Idx;
    using NghIdx = Idx::NghIdx;
    using Type = T;

   public:
    mPartition();

    ~mPartition() = default;

    explicit mPartition(Neon::DataView     dataView,
                        int                level,
                        T*                 mem,
                        T*                 memParent,
                        T*                 memChild,
                        int                cardinality,
                        uint32_t*          neighbourBlocks,
                        Neon::int32_3d*    origin,
                        uint32_t*          parent,
                        Idx::DataBlockIdx* parentLocalID,
                        uint32_t*          mask,
                        uint32_t*          maskLowerLevel,
                        uint32_t*          childBlockID,
                        uint32_t*          parentNeighbourBlocks,
                        T                  defaultValue,
                        NghIdx*            stencilNghIndex,
                        int*               refFactors,
                        int*               spacing);

    /**
     * get the child of a cell
     * @param parent_cell the parent at which the child is queried
     * @param child which child to return. A cell has number of children defined by the branching factor
     * at the level. This defines the 3d local index of the child
     * @param card which cardinality is desired from the child
     * @param alternativeVal in case the child requested is not present
     */
    NEON_CUDA_HOST_DEVICE inline auto childVal(const Idx&    parent_cell,
                                               Neon::int8_3d child,
                                               int           card,
                                               const T&      alternativeVal) const -> NghData<T>;

    /**
     * Get a cell that represents the child of a parent cell
     * @param parent_cell the parent cell that its child is requested
     * @param child the child 3d local index relative to the parent
     */
    NEON_CUDA_HOST_DEVICE inline auto getChild(const Idx&    parent_cell,
                                               Neon::int8_3d child) const -> Idx;


    /**
     * Given a child cell (as returned by getChild), return the value of this child
     * @param childIdx the child cell as returned by getChild
     * @param card the cardinality in case of vector-valued data
     */
    NEON_CUDA_HOST_DEVICE inline auto childVal(const Idx& childIdx,
                                               int        card = 0) -> T&;

    /**
     * Given a child cell (as returned by getChild), return the value of this child
     * @param childIdx the child cell as returned by getChild
     * @param card the cardinality in case of vector-valued data
     */
    NEON_CUDA_HOST_DEVICE inline auto childVal(const Idx& childIdx,
                                               int        card = 0) const -> const T&;

    /**
     * Check if the cell is refined i.e., has children
     * @param cell the cell i.e., parent at which the children are checked
     */
    NEON_CUDA_HOST_DEVICE inline auto hasChildren(const Idx& cell) const -> bool;

    /**
     * Check if a neighbor to 'cell' has children
     * @param cell the main cell
     * @param nghDir the direction relative to cell
     */
    NEON_CUDA_HOST_DEVICE inline auto hasChildren(const Idx& cell, const Neon::int8_3d nghDir) const -> bool;


    /**
     * Get a cell that represents the parent of a given cell
     * @param cell the child cell for which the parent is queried
     */
    NEON_CUDA_HOST_DEVICE inline auto getParent(const Idx& cell) const -> Idx;

    /**
     * Given a cell (child), return the value of the parent
     * @param eId the cell
     * @param card the cardinality in case of vector-valued data
     */
    NEON_CUDA_HOST_DEVICE inline auto parentVal(const Idx& eId,
                                                int        card) -> T&;

    /**
     * Given a cell (child), return the value of the parent
     * @param eId the cell
     * @param card the cardinality in case of vector-valued data
     */
    NEON_CUDA_HOST_DEVICE inline auto parentVal(const Idx& eId,
                                                int        card) const -> const T&;

    /**
     * check if the cell has a parent as defined by the user during the construction of the mGrid
     * @param cell the cell i.e., the child at which the parent is checked
     */
    NEON_CUDA_HOST_DEVICE inline auto hasParent(const Idx& cell) const -> bool;

    /**
     * The uncle of a cell at level L is a cell at level L+1 and is a neighbor to the cell's parent.
     * This function returns the uncle of a given cell in a certain direction w.r.t the cell's parent
     * @param cell the main cell at level L
     * @param direction the direction w.r.t the parent of cell
     */
    NEON_CUDA_HOST_DEVICE inline auto getUncle(const Idx&    cell,
                                               Neon::int8_3d direction) const -> Idx;

    /**
     * The uncle of a cell at level L is a cell at level L+1 and is a neighbor to the cell's parent.
     * This function returns the value of a give cell in a certain direction w.r.t the cell's parent along a certain cardinality.
     * @param cell the main cell at level L
     * @param direction the direction w.r.t the parent of cell
     * @param card the cardinality
     * @param alternativeVal alternative value in case the uncle does not exist.
     */
    NEON_CUDA_HOST_DEVICE inline auto uncleVal(const Idx&    cell,
                                               Neon::int8_3d direction,
                                               int           card,
                                               const T&      alternativeVal) const -> NghData<T>;

    /**
     * Get the refinement factor i.e., number of children at each dimension
     * @param level at which the refinement factor is queried
     */
    NEON_CUDA_HOST_DEVICE inline auto getRefFactor(const int level) const -> int;


    NEON_CUDA_HOST_DEVICE inline auto getSpacing(const int level) const -> int;

    /**
     * Map the cell to its global index as defined by the finest level of the grid (Level 0)
     * @param gidx which will be mapped to global index space
     */
    NEON_CUDA_HOST_DEVICE inline Neon::index_3d mapToGlobal(const Idx& gidx) const;


   private:
    inline NEON_CUDA_HOST_DEVICE auto childID(const Idx& gidx) const -> uint32_t;


    int               mLevel;
    T*                mMemParent;
    T*                mMemChild;
    uint32_t*         mParentBlockID;
    Idx::DataBlockIdx* mParentLocalID;
    uint32_t*         mMaskLowerLevel;
    uint32_t*         mChildBlockID;
    uint32_t*         mParentNeighbourBlocks;
    int*              mRefFactors;
    int*              mSpacing;
};
}  // namespace Neon::domain::details::mGrid

#include "Neon/domain/details/mGrid/mPartition_imp.h"