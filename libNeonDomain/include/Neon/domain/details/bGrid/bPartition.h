#pragma once

#include "Neon/domain/details/bGrid/bIndex.h"
#include "Neon/domain/details/bGrid/bSpan.h"

#include "Neon/domain/interface/NghData.h"

#include "Neon/sys/memory/CUDASharedMemoryUtil.h"

namespace Neon::domain::details::bGrid {

class bSpan;

template <typename T, int C>
class bPartition
{
   public:
    using Span = bSpan;
    using Index = bIndex;
    using NghIdx = Index::NghIdx;
    using Type = T;

   public:
    bPartition();

    ~bPartition() = default;

    explicit bPartition(int                     mCardinality,
                        T*                      mMem,
                        uint32_3d               blockSize,
                        bIndex::DataBlockIdx*   mBlockConnectivity,
                        bSpan::bitMaskWordType* mMask,
                        Neon::int32_3d*         mOrigin,
                        NghIdx*                 mStencilNghIndex);

    inline NEON_CUDA_HOST_DEVICE auto cardinality() const -> int;

    inline NEON_CUDA_HOST_DEVICE auto operator()(const bIndex& cell,
                                                 int           card)
        -> T&;

    inline NEON_CUDA_HOST_DEVICE auto operator()(const bIndex& cell,
                                                 int           card) const -> const T&;

    NEON_CUDA_HOST_DEVICE inline auto nghVal(const Index&  cell,
                                             const NghIdx& offset,
                                             const int     card,
                                             const T       alternativeVal) const -> NghData<T>;

    NEON_CUDA_HOST_DEVICE inline auto nghVal(const Index& eId,
                                             uint8_t      nghID,
                                             int          card,
                                             const T&     alternativeVal) const -> NghData<T>;

    NEON_CUDA_HOST_DEVICE inline Neon::index_3d mapToGlobal(const Index& cell) const;

   protected:
    inline NEON_CUDA_HOST_DEVICE auto helpGetPitch(const Index& cell, int card) const -> uint32_t;

    auto helpGetValidIdxPitchExplicit(const Index& idx, int card) const -> uint32_t;
    auto helpNghPitch(const Index& nghIdx, int card) const -> std::tuple<bool, uint32_t>;
    auto helpGetNghIdx(const Index& idx, const NghIdx& offset) const -> bIndex;


    int       mCardinality;
    T*        mMem;
    NghIdx*   mStencilNghIndex;
    uint32_3d mBlockSizeByPower /**<< v[0] = blockDim.x,
                                 *   v[1] = blockDim.x * blockDim.y ,
                                 *   v[1] = blockDim.x * blockDim.y * blockDim.z*/
        ;
    bIndex::DataBlockIdx*   mBlockConnectivity;
    bSpan::bitMaskWordType* mMask;
    Neon::int32_3d*         mOrigin;
};
}  // namespace Neon::domain::details::bGrid

#include "Neon/domain/details/bGrid/bPartition_imp.h"