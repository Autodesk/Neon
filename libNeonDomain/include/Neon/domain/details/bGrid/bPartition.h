#pragma once

#include "Neon/domain/details/bGrid/bIndex.h"
#include "Neon/domain/details/bGrid/bSpan.h"

#include "Neon/domain/interface/NghData.h"

#include "Neon/sys/memory/CUDASharedMemoryUtil.h"

namespace Neon::domain::details::bGrid {

template <int8_t memBlockSizeX, int8_t memBlockSizeY, int8_t memBlockSizeZ, int8_t userBlockSizeX, int8_t userBlockSizeY, int8_t userBlockSizeZ>
class bSpan;

template <typename T, int C, int8_t memBlockSizeX, int8_t memBlockSizeY, int8_t memBlockSizeZ, int8_t userBlockSizeX, int8_t userBlockSizeY, int8_t userBlockSizeZ>
class bPartition
{
   public:
    using Span = bSpan<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>;
    using Idx = bIndex<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>;
    using NghIdx = Idx::NghIdx;
    using Type = T;
    using NghData = Neon::domain::NghData<T>;

   public:
    bPartition();

    ~bPartition() = default;

    explicit bPartition(int                    setIdx,
                        int                    mCardinality,
                        T*                     mMem,
                        Idx::DataBlockIdx*     mBlockConnectivity,
                        Span::BitMaskWordType* mMask,
                        Neon::int32_3d*        mOrigin,
                        NghIdx*                mStencilNghIndex);

    inline NEON_CUDA_HOST_DEVICE auto
    cardinality()
        const -> int;

    inline NEON_CUDA_HOST_DEVICE auto
    operator()(const Idx& cell,
               int        card)
        -> T&;

    inline NEON_CUDA_HOST_DEVICE auto
    operator()(const Idx& cell,
               int        card)
        const -> const T&;

    NEON_CUDA_HOST_DEVICE inline auto
    getNghData(const Idx&    cell,
               const NghIdx& offset,
               const int     card)
        const -> NghData;

    NEON_CUDA_HOST_DEVICE inline auto
    getNghData(const Idx& eId,
               uint8_t    nghID,
               int        card)
        const -> NghData;

    NEON_CUDA_HOST_DEVICE inline auto
    getGlobalIndex(const Idx& cell)
        const -> Neon::index_3d;

   protected:
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


    int       mCardinality;
    T*        mMem;
    NghIdx*   mStencilNghIndex;
    Idx::DataBlockIdx*     mBlockConnectivity;
    Span::BitMaskWordType* mMask;
    Neon::int32_3d*        mOrigin;
    int                    mSetIdx;
};
}  // namespace Neon::domain::details::bGrid

#include "Neon/domain/details/bGrid/bPartition_imp.h"