#pragma once

#include "Neon/domain/interface/NghInfo.h"
#include "Neon/domain/internal/bGrid/bCell.h"
#include "Neon/domain/internal/bGrid/bPartition.h"

#include "Neon/sys/memory/CUDASharedMemoryUtil.h"

namespace Neon::domain::internal::mGrid {

class bPartitionIndexSpace;

template <typename T, int C = 0>
class mPartition : public Neon::domain::internal::bGrid::bPartition<T, C>
{
   public:
    using PartitionIndexSpace = Neon::domain::internal::bGrid::bPartitionIndexSpace;
    using Cell = Neon::domain::internal::bGrid::bCell;
    using nghIdx_t = Cell::nghIdx_t;
    using Type = T;

   public:
    mPartition();

    ~mPartition() = default;

    explicit mPartition(Neon::DataView  dataView,
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
                        int*            refFactors,
                        int*            spacing);


    NEON_CUDA_HOST_DEVICE inline auto childVal(const Cell&   parent_cell,
                                               Neon::int8_3d child,
                                               int           card,
                                               const T&      alternativeVal) const -> NghInfo<T>;

    NEON_CUDA_HOST_DEVICE inline auto getChild(const Cell&   cell,
                                               Neon::int8_3d child) const -> Cell;

    NEON_CUDA_HOST_DEVICE inline auto childVal(const Cell& childCell,
                                               int         card = 0) -> T&;

    NEON_CUDA_HOST_DEVICE inline auto childVal(const Cell& childCell,
                                               int         card = 0) const -> const T&;

    NEON_CUDA_HOST_DEVICE inline auto parent(const Cell& eId,
                                             int         card) -> T&;

    NEON_CUDA_HOST_DEVICE inline auto hasParent(const Cell& cell) const -> bool;

    NEON_CUDA_HOST_DEVICE inline auto hasChildren(const Cell& cell) const -> bool;

    NEON_CUDA_HOST_DEVICE inline auto getRefFactor(const int level) const -> int;

    NEON_CUDA_HOST_DEVICE inline auto getSpacing(const int level) const -> int;

    NEON_CUDA_HOST_DEVICE inline Neon::index_3d mapToGlobal(const Cell& cell) const;


   private:
    inline NEON_CUDA_HOST_DEVICE auto childID(const Cell& cell) const -> uint32_t;


    int             mLevel;
    T*              mMemParent;
    T*              mMemChild;
    uint32_t*       mParentBlockID;
    Cell::Location* mParentLocalID;
    uint32_t*       mMaskLowerLevel;
    uint32_t*       mChildBlockID;
    int*            mRefFactors;
    int*            mSpacing;
};
}  // namespace Neon::domain::internal::mGrid

#include "Neon/domain/internal/mGrid/mPartition_imp.h"