#pragma once

#include "Neon/domain/details/bGridDisg/bDisgIndex.h"
#include "Neon/domain/details/bGridDisg/bDisgSpan.h"

#include "Neon/domain/interface/NghData.h"

#include "Neon/sys/memory/CUDASharedMemoryUtil.h"

namespace Neon::domain::details::disaggregated::bGridDisg {

    template<typename SBlock>
    class bDisgSpan;

    template<typename T, int C, typename SBlock>
    class bDisgPartition {
    public:
        using Span = bDisgSpan<SBlock>;
        using Idx = bDisgIndex<SBlock>;
        using NghIdx = typename Idx::NghIdx;
        using Type = T;
        using NghData = Neon::domain::NghData<T>;

        using BlockViewGrid = Neon::domain::tool::GridTransformer<details::GridTransformation>::Grid;
        using BlockViewGridIdx = BlockViewGrid::Idx;

    public:
        bDisgPartition();

        ~bDisgPartition() = default;

        explicit bDisgPartition(int setIdx,
                            int mCardinality,
                            T *mMem,
                            typename Idx::DataBlockIdx *mBlockConnectivity,
                            typename SBlock::BitMask const *NEON_RESTRICT mMask,
                            Neon::int32_3d *mOrigin,
                            NghIdx *mStencilNghIndex,
                            Neon::int32_3d mDomainSize);

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
        operator()(const Idx &cell,
                   int card)
        -> T &;

        /**
         * Gets the field metadata at a cartesian point.
         */
        inline NEON_CUDA_HOST_DEVICE auto
        operator()(const Idx &cell,
                   int card)
        const -> const T &;

        /**
         * Gets the field metadata at a neighbour cartesian point.
         */
        NEON_CUDA_HOST_DEVICE inline auto
        getNghData(const Idx &cell,
                   const NghIdx &offset,
                   const int card)
        const -> NghData;

        /**
         * Gets the field metadata at a neighbour cartesian point.
         */
        NEON_CUDA_HOST_DEVICE inline auto
        getNghData(const Idx &eId,
                   uint8_t nghID,
                   int card)
        const -> NghData;

        /**
         * Gets the field metadata at a neighbour cartesian point.
         */
        template<int xOff, int yOff, int zOff>
        NEON_CUDA_HOST_DEVICE inline auto
        getNghData(const Idx &eId,
                   int card)
        const -> NghData;

        /**
         * Gets the field metadata at a neighbour cartesian point.
         */
        template<int xOff, int yOff, int zOff>
        NEON_CUDA_HOST_DEVICE inline auto
        getNghData(const Idx &eId,
                   int card,
                   T defaultValue)
        const -> NghData;

        template<int xOff,
                int yOff,
                int zOff,
                typename LambdaVALID,
                typename LambdaNOTValid = void *>
        NEON_CUDA_HOST_DEVICE inline auto
        getNghData(const Idx &gidx,
                   int card,
                   LambdaVALID funIfValid,
                   LambdaNOTValid funIfNOTValid = nullptr)
        const -> std::enable_if_t<std::is_invocable_v<LambdaVALID, T> && (std::is_invocable_v<LambdaNOTValid, T> ||
                                                                          std::is_same_v<LambdaNOTValid, void *>), void>;

        template<int xOff,
                int yOff,
                int zOff>
        NEON_CUDA_HOST_DEVICE inline auto
        writeNghData(const Idx &gidx,
                     int card,
                     T value)
        -> bool;

        /**
         * Gets the global coordinates of the cartesian point.
         */
        NEON_CUDA_HOST_DEVICE inline auto
        getGlobalIndex(const Idx &cell)
        const -> Neon::index_3d;

        NEON_CUDA_HOST_DEVICE inline auto
        isActive(const Idx &cell,
                 const typename SBlock::BitMask *mask = nullptr) const -> bool;


        NEON_CUDA_HOST_DEVICE inline auto
        getDomainSize()
        const -> Neon::index_3d;

        NEON_CUDA_HOST_DEVICE
        auto mem() const -> T const *;

        /**
         * Gets the Idx for in the block view space.
         */
        NEON_CUDA_HOST_DEVICE inline auto
        getBlockViewIdx(const Idx &cell)
        const -> BlockViewGridIdx;


        NEON_CUDA_HOST_DEVICE inline auto
        helpGetPitch(const Idx &cell, int card)
        const -> uint32_t;

        NEON_CUDA_HOST_DEVICE inline auto
        helpGetValidIdxPitchExplicit(const Idx &idx, int card)
        const -> uint32_t;

        NEON_CUDA_HOST_DEVICE inline auto
        helpNghPitch(const Idx &nghIdx, int card)
        const -> std::tuple<bool, uint32_t>;

        NEON_CUDA_HOST_DEVICE inline auto
        helpGetNghIdx(const Idx &idx, const NghIdx &offset)
        const -> Idx;

        template<int xOff, int yOff, int zOff>
        NEON_CUDA_HOST_DEVICE inline auto
        helpGetNghIdx(const Idx &idx)
        const -> Idx;

        NEON_CUDA_HOST_DEVICE inline auto
        helpGetNghIdx(const Idx &idx, const NghIdx &offset, const typename Idx::DataBlockIdx *blockConnectivity)
        const -> Idx;

        template<int xOff, int yOff, int zOff>
        NEON_CUDA_HOST_DEVICE inline auto
        helpGetNghIdx(const Idx &idx, const typename Idx::DataBlockIdx *blockConnectivity)
        const -> Idx;


        int mCardinality;
        T *mMem;
        NghIdx const *NEON_RESTRICT mStencilNghIndex;
        typename Idx::DataBlockIdx const *NEON_RESTRICT mBlockConnectivity;
        typename SBlock::BitMask const *NEON_RESTRICT mMask;
        Neon::int32_3d const *NEON_RESTRICT mOrigin;
        int mSetIdx;
        int mMultiResDiscreteIdxSpacing = 1;
        Neon::int32_3d mDomainSize;
    };

}  // namespace Neon::domain::details::disaggregated::bGrid

#include "Neon/domain/details/bGridDisg/bPartition_imp.h"