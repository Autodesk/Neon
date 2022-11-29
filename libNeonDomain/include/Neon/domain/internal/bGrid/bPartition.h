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
                        T*              mem,
                        int             cardinality,
                        uint32_t*       neighbourBlocks,
                        Neon::int32_3d* origin,
                        uint32_t*       mask,
                        T               outsideValue,
                        nghIdx_t*       stencilNghIndex);

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


    NEON_CUDA_HOST_DEVICE inline void loadInSharedMemory(const Cell&                cell,
                                                         const nghIdx_t::Integer    stencilRadius,
                                                         Neon::sys::ShmemAllocator& shmemAlloc) const;

    NEON_CUDA_HOST_DEVICE inline void loadInSharedMemoryAsync(const Cell&                cell,
                                                              const nghIdx_t::Integer    stencilRadius,
                                                              Neon::sys::ShmemAllocator& shmemAlloc) const;

    NEON_CUDA_HOST_DEVICE inline Neon::index_3d mapToGlobal(const Cell& cell) const;


   protected:
    inline NEON_CUDA_HOST_DEVICE auto pitch(const Cell& cell, int card) const -> uint32_t;
    inline NEON_CUDA_HOST_DEVICE auto setNghCell(const Cell& cell, const nghIdx_t& offset) const -> Cell;
    inline NEON_CUDA_HOST_DEVICE auto shmemPitch(Cell cell, const int card) const -> Cell::Location::Integer;

    Neon::DataView            mDataView;
    T*                        mMem;
    int                       mCardinality;
    uint32_t*                 mNeighbourBlocks;
    Neon::int32_3d*           mOrigin;
    uint32_t*                 mMask;
    T                         mOutsideValue;
    nghIdx_t*                 mStencilNghIndex;
    mutable bool              mIsInSharedMem;
    mutable T*                mMemSharedMem;
    mutable uint32_t*         mSharedNeighbourBlocks;
    mutable nghIdx_t::Integer mStencilRadius;
};
}  // namespace Neon::domain::internal::bGrid

#include "Neon/domain/internal/bGrid/bPartition_imp.h"