#pragma once

#include "Neon/domain/interface/NghInfo.h"
#include "Neon/domain/internal/bGrid/bCell.h"

#include "Neon/sys/memory/CUDASharedMemoryUtil.h"

namespace Neon::domain::internal::bGrid {

class bPartitionIndexSpace;

template <typename T, int C = 0>
class bPartition
{
   public:
    using PartitionIndexSpace = bPartitionIndexSpace;
    using Cell = bCell;
    using nghIdx_t = Cell::nghIdx_t;
    using Type = T;

   public:
    bPartition();

    ~bPartition() = default;

    explicit bPartition(Neon::DataView  dataView,
                        int             level,
                        T*              mem,
                        T*              memParent,
                        T*              memChild,
                        int             cardinality,
                        uint32_t*       neighbourBlocks,
                        Neon::int32_3d* origin,
                        uint32_t*       parent,
                        Cell::Location* parentLocalID,
                        uint32_t*       mask,
                        uint32_t*       maskLowerLevel,
                        uint32_t*       childBlockID,
                        T               defaultValue,
                        nghIdx_t*       stencilNghIndex,
                        int*            descriptor,
                        int*            spacing);

    inline NEON_CUDA_HOST_DEVICE auto cardinality() const -> int;

    inline NEON_CUDA_HOST_DEVICE auto operator()(const bCell& cell,
                                                 int          card)
        -> T&;

    inline NEON_CUDA_HOST_DEVICE auto operator()(const bCell& cell,
                                                 int          card) const -> const T&;

    NEON_CUDA_HOST_DEVICE inline auto nghVal(const Cell&     cell,
                                             const nghIdx_t& offset,
                                             const int       card,
                                             const T         alternativeVal) const -> NghInfo<T>;

    NEON_CUDA_HOST_DEVICE inline auto nghVal(const Cell& eId,
                                             uint8_t     nghID,
                                             int         card,
                                             const T&    alternativeVal) const -> NghInfo<T>;

    NEON_CUDA_HOST_DEVICE inline auto childVal(const Cell&   parent_cell,
                                               Neon::int8_3d child,
                                               int           card,
                                               const T&      alternativeVal) const -> NghInfo<T>;

    NEON_CUDA_HOST_DEVICE inline auto getChild(const Cell&   cell,
                                               Neon::int8_3d child) const -> Cell;

    NEON_CUDA_HOST_DEVICE inline auto childVal(const Cell& childCell,
                                               int         card) -> T&;

    NEON_CUDA_HOST_DEVICE inline auto childVal(const Cell& childCell,
                                               int         card) const -> const T&;


    NEON_CUDA_HOST_DEVICE inline auto parent(const Cell& eId,
                                             int         card) -> T&;

    NEON_CUDA_HOST_DEVICE inline void loadInSharedMemory(const Cell&                cell,
                                                         const nghIdx_t::Integer    stencilRadius,
                                                         Neon::sys::ShmemAllocator& shmemAlloc) const;

    NEON_CUDA_HOST_DEVICE inline void loadInSharedMemoryAsync(const Cell&                cell,
                                                              const nghIdx_t::Integer    stencilRadius,
                                                              Neon::sys::ShmemAllocator& shmemAlloc) const;

    NEON_CUDA_HOST_DEVICE inline auto mapToGlobal(const Cell& cell) const -> Neon::index_3d;

    NEON_CUDA_HOST_DEVICE inline auto hasParent(const Cell& cell) const -> bool;

    NEON_CUDA_HOST_DEVICE inline auto hasChildren(const Cell& cell) const -> bool;

    template <typename FuncT>
    NEON_CUDA_HOST_DEVICE inline auto forEachActiveChild(const Cell& local,
                                                         FuncT       op) const -> void
    {
        assert(mLevel > 0);
        int childRefLevel = mRefFactors[mLevel - 1];
        for (Cell::Location::Integer i = 0; i < childRefLevel; ++i) {
            for (Cell::Location::Integer j = 0; j < childRefLevel; ++j) {
                for (Cell::Location::Integer k = 0; k < childRefLevel; ++k) {
                    Cell child(i, j, k);
                    child.mBlockID;  //???
                    child.mBlockSize = mRefFactors[mLevel - 1];
                    //if (child.computeIsActive(mMaskLowerLevel)) {
                    //    op(child.mLocation, );
                    //}
                }
            }
        }
    }

   private:
    inline NEON_CUDA_HOST_DEVICE auto pitch(const Cell& cell, int card) const -> uint32_t;
    inline NEON_CUDA_HOST_DEVICE auto childID(const Cell& cell) const -> uint32_t;
    inline NEON_CUDA_HOST_DEVICE auto setNghCell(const Cell& cell, const nghIdx_t& offset) const -> Cell;
    inline NEON_CUDA_HOST_DEVICE auto shmemPitch(Cell cell, const int card) const -> Cell::Location::Integer;

    Neon::DataView            mDataView;
    int                       mLevel;
    T*                        mMem;
    T*                        mMemParent;
    T*                        mMemChild;
    int                       mCardinality;
    uint32_t*                 mNeighbourBlocks;
    Neon::int32_3d*           mOrigin;
    uint32_t*                 mParentBlockID;
    Cell::Location*           mParentLocalID;
    uint32_t*                 mMask;
    uint32_t*                 mMaskLowerLevel;
    uint32_t*                 mChildBlockID;
    T                         mOutsideValue;
    nghIdx_t*                 mStencilNghIndex;
    int*                      mRefFactors;
    int*                      mSpacing;
    mutable bool              mIsInSharedMem;
    mutable T*                mMemSharedMem;
    mutable uint32_t*         mSharedNeighbourBlocks;
    mutable nghIdx_t::Integer mStencilRadius;
};
}  // namespace Neon::domain::internal::bGrid

#include "Neon/domain/internal/bGrid/bPartition_imp.h"