#pragma once

#include "Neon/domain/details/bGrid/bIndex.h"
#include "Neon/domain/details/bGrid/bSpan.h"

#include "Neon/domain/interface/NghData.h"

#include "Neon/sys/memory/CUDASharedMemoryUtil.h"

namespace Neon::domain::details::bGrid {

template <typename SBlock>
class bSpan;

template <typename T, int C, typename SBlock>
class bPartition
{
   public:
    using Span = bSpan<SBlock>;
    using Idx = bIndex<SBlock>;
    using NghIdx = typename Idx::NghIdx;
    using Type = T;
    using NghData = Neon::domain::NghData<T>;

    using BlockViewGrid = Neon::domain::tool::GridTransformer<details::GridTransformation>::Grid;
    using BlockViewGridIdx = BlockViewGrid::Idx;

   public:
    bPartition();

    ~bPartition() = default;

    explicit bPartition(int                                           setIdx,
                        int                                           mCardinality,
                        T*                                            mMem,
                        typename Idx::DataBlockIdx*                   mBlockConnectivity,
                        typename SBlock::BitMask const* NEON_RESTRICT mMask,
                        Neon::int32_3d*                               mOrigin,
                        NghIdx*                                       mStencilNghIndex);

    /**
     * Retrieve the cardinality of the field.
     */
    inline NEON_CUDA_HOST_DEVICE auto
    cardinality()
        const -> int;

    /**
     * Gets the field metadata at a cartesian point.
     */
    inline NEON_CUDA_HOST_DEVICE auto
    operator()(const Idx& cell,
               int        card)
        -> T&;

    /**
     * Gets the field metadata at a cartesian point.
     */
    inline NEON_CUDA_HOST_DEVICE auto
    operator()(const Idx& cell,
               int        card)
        const -> const T&;

    /**
     * Gets the field metadata at a neighbour cartesian point.
     */
    NEON_CUDA_HOST_DEVICE inline auto
    getNghData(const Idx&    cell,
               const NghIdx& offset,
               const int     card)
        const -> NghData;

    /**
     * Gets the field metadata at a neighbour cartesian point.
     */
    NEON_CUDA_HOST_DEVICE inline auto
    getNghData(const Idx& eId,
               uint8_t    nghID,
               int        card)
        const -> NghData;

    /**
     * Gets the field metadata at a neighbour cartesian point.
     */
    template <int xOff, int yOff, int zOff>
    NEON_CUDA_HOST_DEVICE inline auto
    getNghData(const Idx& eId,
               int        card)
        const -> NghData;

    /**
     * Gets the field metadata at a neighbour cartesian point.
     */
    template <int xOff, int yOff, int zOff>
    NEON_CUDA_HOST_DEVICE inline auto
    getNghData(const Idx& eId,
               int        card,
               T          defaultValue)
        const -> NghData;

    /**
     * Gets the global coordinates of the cartesian point.
     */
    NEON_CUDA_HOST_DEVICE inline auto
    getGlobalIndex(const Idx& cell)
        const -> Neon::index_3d;

    NEON_CUDA_HOST_DEVICE inline auto
    isActive(const Idx&                      cell,
             const typename SBlock::BitMask* mask = nullptr) const -> bool;

    /**
     * Gets the Idx for in the block view space.
     */
    NEON_CUDA_HOST_DEVICE inline auto
    getBlockViewIdx(const Idx& cell)
        const -> BlockViewGridIdx;

   
    NEON_CUDA_HOST_DEVICE inline auto
    helpGetPitch(const Idx& cell, int card)
        const -> uint32_t;

    NEON_CUDA_HOST_DEVICE inline auto
    helpGetValidIdxPitchExplicit(const Idx& idx, int card)
        const -> uint32_t;

    NEON_CUDA_HOST_DEVICE inline auto
    helpNghPitch(const Idx& nghIdx, int card)
        const -> std::tuple<bool, uint32_t>;

    NEON_CUDA_HOST_DEVICE inline auto
    helpGetNghIdx(const Idx& idx, const NghIdx& offset)
        const -> Idx;

    template <int xOff, int yOff, int zOff>
    NEON_CUDA_HOST_DEVICE inline auto
    helpGetNghIdx(const Idx& idx)
        const -> Idx;

    NEON_CUDA_HOST_DEVICE inline auto
    helpGetNghIdx(const Idx& idx, const NghIdx& offset, const typename Idx::DataBlockIdx* blockConnectivity)
        const -> Idx;

    template <int xOff, int yOff, int zOff>
    NEON_CUDA_HOST_DEVICE inline auto
    helpGetNghIdx(const Idx& idx, const typename Idx::DataBlockIdx* blockConnectivity)
        const -> Idx;

    int                                             mCardinality;
    T*                                              mMem;
    NghIdx const* NEON_RESTRICT                     mStencilNghIndex;
    typename Idx::DataBlockIdx const* NEON_RESTRICT mBlockConnectivity;
    typename SBlock::BitMask const* NEON_RESTRICT   mMask;
    Neon::int32_3d const* NEON_RESTRICT             mOrigin;
    int                                             mSetIdx;
};

}  // namespace Neon::domain::details::bGrid

#include "Neon/domain/details/bGrid/bPartition_imp.h"