#pragma once


#include "Neon/domain/details/bGrid/bIndex.h"
#include "Neon/domain/details/bGrid/bPartition.h"
#include "Neon/domain/interface/NghData.h"

#include "Neon/domain/details/bGrid/StaticBlock.h"
#include "Neon/sys/memory/CUDASharedMemoryUtil.h"

namespace Neon::domain::details::mGrid {

constexpr uint32_t kMemBlockSizeX = 8;
constexpr uint32_t kMemBlockSizeY = 8;
constexpr uint32_t kMemBlockSizeZ = 8;
constexpr uint32_t kUserBlockSizeX = 2;
constexpr uint32_t kUserBlockSizeY = 2;
constexpr uint32_t kUserBlockSizeZ = 2;

constexpr uint32_t kNumUserBlockPerMemBlockX = kMemBlockSizeX / kUserBlockSizeX;
constexpr uint32_t kNumUserBlockPerMemBlockY = kMemBlockSizeY / kUserBlockSizeY;
constexpr uint32_t kNumUserBlockPerMemBlockZ = kMemBlockSizeZ / kUserBlockSizeZ;

using kStaticBlock = Neon::domain::details::bGrid::StaticBlock<kMemBlockSizeX, kMemBlockSizeY, kMemBlockSizeZ, kUserBlockSizeX, kUserBlockSizeY, kUserBlockSizeZ, true>;

template <typename T, int C = 0>
class mPartition : public Neon::domain::details::bGrid::bPartition<T, C, kStaticBlock>
{
   public:
    using Idx = Neon::domain::details::bGrid::bIndex<kStaticBlock>;
    using NghIdx = Idx::NghIdx;
    using NghData = Neon::domain::NghData<T>;
    using Type = T;
    using MaskT = typename kStaticBlock::BitMask;

   public:
    mPartition();

    ~mPartition() = default;

    explicit mPartition(int                level,
                        T*                 mem,
                        T*                 memParent,
                        T*                 memChild,
                        int                cardinality,
                        Idx::DataBlockIdx* neighbourBlocks,
                        Neon::int32_3d*    origin,
                        Idx::DataBlockIdx* parent,
                        MaskT*             mask,
                        MaskT*             maskLowerLevel,
                        MaskT*             maskUpperLevel,
                        Idx::DataBlockIdx* childBlockID,
                        Idx::DataBlockIdx* parentNeighbourBlocks,
                        NghIdx*            stencilNghIndex,
                        int*               refFactors,
                        int*               spacing);

    /**
     * get the child of a cell
     * @param parentCell the parent at which the child is queried
     * @param child which child to return. A cell has number of children defined by the branching factor
     * at the level. This defines the 3d local index of the child
     * @param card which cardinality is desired from the child
     * @param alternativeVal in case the child requested is not present
     */
    NEON_CUDA_HOST_DEVICE inline auto childVal(const Idx&   parentCell,
                                               const NghIdx child,
                                               int          card,
                                               const T&     alternativeVal) const -> NghData;

    /**
     * Get a cell that represents the child of a parent cell
     * @param parentCell the parent cell that its child is requested
     * @param child the child 3d local index relative to the parent
     */
    NEON_CUDA_HOST_DEVICE inline auto getChild(const Idx& parentCell,
                                               NghIdx     child) const -> Idx;


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
    NEON_CUDA_HOST_DEVICE inline auto hasChildren(const Idx& cell, const NghIdx nghDir) const -> bool;


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
    NEON_CUDA_HOST_DEVICE inline auto getUncle(const Idx&   cell,
                                               const NghIdx direction) const -> Idx;

    /**
     * The uncle of a cell at level L is a cell at level L+1 and is a neighbor to the cell's parent.
     * This function returns the value of a give cell in a certain direction w.r.t the cell's parent along a certain cardinality.
     * @param cell the main cell at level L
     * @param direction the direction w.r.t the parent of cell
     * @param card the cardinality
     * @param alternativeVal alternative value in case the uncle does not exist.
     */
    NEON_CUDA_HOST_DEVICE inline auto uncleVal(const Idx&   cell,
                                               const NghIdx direction,
                                               int          card,
                                               const T&     alternativeVal) const -> NghData;

    /**
     * @brief similar to the above uncleVal but returns a reference. Additionally, it is now
     * the user responsibility to check if the uncle is active (we only assert it)
     * @param cell the main cell at level L 
     * @param direction the direction w.r.t the parent of cell      
     * @param card the cardinality      
    */
    NEON_CUDA_HOST_DEVICE inline auto uncleVal(const Idx&   cell,
                                               const NghIdx direction,
                                               int          card) const -> T&;

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
    NEON_CUDA_HOST_DEVICE inline Neon::index_3d getGlobalIndex(Idx gidx) const;


   private:
    inline NEON_CUDA_HOST_DEVICE auto childID(const Idx& gidx) const -> uint32_t;


    int                mLevel;
    T*                 mMemParent;
    T*                 mMemChild;
    Idx::DataBlockIdx* mParentBlockID;
    MaskT*             mMaskLowerLevel;
    MaskT*             mMaskUpperLevel;
    Idx::DataBlockIdx* mChildBlockID;
    Idx::DataBlockIdx* mParentNeighbourBlocks;
    int*               mRefFactors;
    int*               mSpacing;
};
}  // namespace Neon::domain::details::mGrid

#include "Neon/domain/details/mGrid/mPartition_imp.h"