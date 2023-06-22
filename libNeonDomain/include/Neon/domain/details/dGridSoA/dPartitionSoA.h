#pragma once
#include <assert.h>
#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"
#include "Neon/domain/details/dGrid/dGrid.h"
#include "Neon/domain/interface/NghData.h"
#include "Neon/set/DevSet.h"
#include "Neon/sys/memory/CudaIntrinsics.h"
#include "cuda_fp16.h"
#include "dIndexSoA.h"

namespace Neon::domain::details::dGridSoA {

template <typename T,
          int C = 1>
class dPartitionSoA
{
   public:
    using Idx = dIndexSoA;
    using NghData = Neon::domain::NghData<T>;
    using Pitch = uint32_4d;

    dPartitionSoA()
    {
    }

    dPartitionSoA(Neon::domain::details::dGrid::dPartition<T, C> const& dPartitionOriginal)
    {
        mDataView = dPartitionOriginal.getDataView();
        mMem = dPartitionOriginal.mem();
        mDim = dPartitionOriginal.dim();
        mZHaloRadius = dPartitionOriginal.halo().z;
        mPitch = dPartitionOriginal.getPitchData().template newType<Pitch::Integer>();
        mPrtID = dPartitionOriginal.prtID();
        mOrigin = dPartitionOriginal.origin();
        mCardinality = dPartitionOriginal.cardinality();
        mFullGridSize = dPartitionOriginal.fullGridSize();
        NghIdx* mStencil = dPartitionOriginal.helpGetGlobalToLocalOffets();
    }

    inline NEON_CUDA_HOST_DEVICE auto
    prtID()
        const -> int
    {
        return mPrtID();
    }

    inline NEON_CUDA_HOST_DEVICE auto
    cardinality()
        const -> int
    {
        return mCardinality();
    }

    inline NEON_CUDA_HOST_DEVICE auto
    getPitchData()
        const -> const Pitch&
    {
        return mPitch;
    }

    inline NEON_CUDA_HOST_DEVICE auto
    getPitch(const Idx& idx,
             int        cardinality)
        -> Idx::Offset
    {
        return idx.getLocationOffset() + cardinality * mPitch.w;
    }

    inline NEON_CUDA_HOST_DEVICE auto
    dim()
        const -> const Neon::index_3d
    {
        return mDim();
    }

    inline NEON_CUDA_HOST_DEVICE auto
    halo()
        const -> const Neon::index_3d
    {
        return mDPartition.halo();
    }

    inline NEON_CUDA_HOST_DEVICE auto
    origin()
        const -> const Neon::index_3d
    {
        return m_ormDPartition.origin();
    }

    NEON_CUDA_HOST_DEVICE inline auto
    getNghData(const Idx& gidx,
               NghIdx     nghOffset,
               int        card,
               const T&   alternativeVal)
        const -> NghData
    {
        Idx        gidxNgh;
        const bool isValidNeighbour = nghIdx(gidx, nghOffset, gidxNgh);
        T          val = alternativeVal;
        if (isValidNeighbour) {
            val = operator()(gidxNgh, card);
        }
        return NghData(val, isValidNeighbour);
    }

    NEON_CUDA_HOST_DEVICE inline auto
    getNghData(const Idx& gidx,
               NghIdx     nghOffset,
               int        card)
        const -> NghData
    {
        Idx        gidxNgh;
        const bool isValidNeighbour = nghIdx(gidx, nghOffset, gidxNgh);
        T          val;
        if (isValidNeighbour) {
            val = operator()(gidxNgh, card);
        }
        return NghData(val, isValidNeighbour);
    }

    template <int xOff,
              int yOff,
              int zOff,
              typename LambdaVALID,
              typename LambdaNOTValid = void*>
    NEON_CUDA_HOST_DEVICE inline auto
    getNghData(const Idx&     gidx,
               int            card,
               LambdaVALID    funIfValid,
               LambdaNOTValid funIfNOTValid = nullptr)
        const -> std::enable_if_t<std::is_invocable_v<LambdaVALID, T>, void>
    {
        Idx        gidxNgh;
        const bool isValidNeighbour = nghIdx<xOff, yOff, zOff>(gidx, gidxNgh);
        if (isValidNeighbour) {
            T val = this->operator()(gidxNgh, card);
            funIfValid(val);
        }
        if constexpr (!std::is_same_v<LambdaNOTValid, void*>) {
            if (!isValidNeighbour) {
                funIfNOTValid();
            }
        }
    }

    template <int xOff, int yOff, int zOff>
    NEON_CUDA_HOST_DEVICE inline auto
    getNghData(const Idx& gidx,
               int        card)
        const -> NghData
    {
        NghData    res;
        Idx        gidxNgh;
        const bool isValidNeighbour = nghIdx<xOff, yOff, zOff>(gidx, gidxNgh);
        if (isValidNeighbour) {
            T val = operator()(gidxNgh, card);
            res.set(val, true);
        } else {
            res.invalidate();
        }
        return res;
    }

    template <int xOff, int yOff, int zOff>
    NEON_CUDA_HOST_DEVICE inline auto
    getNghData(const Idx& gidx,
               int        card,
               T const&   defaultValue)
        const -> NghData
    {
        NghData    res(defaultValue, false);
        Idx        gidxNgh;
        const bool isValidNeighbour = nghIdx<xOff, yOff, zOff>(gidx, gidxNgh);
        if (isValidNeighbour) {
            T val = operator()(gidxNgh, card);
            res.set(val, true);
        }
        return res;
    }

    NEON_CUDA_HOST_DEVICE inline auto
    nghVal(const Idx& gidx,
           uint8_t    nghID,
           int        card,
           const T&   alternativeVal)
        const -> NghData
    {
        NghIdx nghOffset = mStencil[nghID];
        return getNghData(gidx, nghOffset, card, alternativeVal);
    }

    /**
     * Get the index of the neighbor given the offset
     * @tparam dataView_ta
     * @param[in] gidx Index of the current element
     * @param[in] nghOffset Offset of the neighbor of interest from the current element
     * @param[in,out] neighbourIdx Index of the neighbor
     * @return Whether the neighbour is valid
     */
    NEON_CUDA_HOST_DEVICE inline auto
    nghIdx(const Idx&    gidx,
           const NghIdx& nghOffset,
           Idx&          neighbourIdx)
        const -> bool
    {
        Neon::index_3d cartesian(gidx.get().x + nghOffset.x,
                                 gidx.get().y + nghOffset.y,
                                 gidx.get().z + nghOffset.z);

        neighbourIdx = Idx(cartesian,
                           gidx.getOffset() + nghOffset.x * getPitchData().x +
                               nghOffset.y * getPitchData().y +
                               nghOffset.z * getPitchData().z);

        Idx::Location nghCartesianGlobal = getGlobalIndex(gidxNgh);

        bool isValidNeighbour = true;

        isValidNeighbour = (gidxNghGlobal.x >= 0) &&
                           (gidxNghGlobal.y >= 0) &&
                           (gidxNghGlobal.z >= 0);

        isValidNeighbour = (gidxNghGlobal.x < m_fullGridSize.x) &&
                           (gidxNghGlobal.y < m_fullGridSize.y) &&
                           (gidxNghGlobal.z < m_fullGridSize.z) &&
                           isValidNeighbour;

        return isValidNeighbour;
    }

    template <int xOff, int yOff, int zOff>
    NEON_CUDA_HOST_DEVICE inline auto
    helpGetNghIdx(const Idx& gidx,
                  Idx&       gidxNgh)
        const -> bool
    {
        Neon::index_3d cartesian(gidx.get().x + xOff,
                                 gidx.get().y + yOff,
                                 gidx.get().z + zOff);
        gidxNgh = Idx(cartesian,
                      gidx.getOffset() + xOff * getPitchData().x +
                          yOff * getPitchData().y +
                          zOff * getPitchData().z);

        Idx::Location nghCartesianGlobal(getGlobalIndex(gidxNgh));

        bool isValidNeighbour = true;
        if constexpr (xOff > 0) {
            isValidNeighbour = cellNgh.get().x < (m_dim.x) && isValidNeighbour;
            isValidNeighbour = nghCartesianGlobal.x <= mDPartition.m_fullGridSize.x && isValidNeighbour;
        }
        if constexpr (xOff < 0) {
            isValidNeighbour = nghCartesianGlobal.x >= 0 && isValidNeighbour;
        }
        if constexpr (yOff > 0) {
            isValidNeighbour = cellNgh.get().y < (m_dim.y) && isValidNeighbour;
            isValidNeighbour = nghCartesianGlobal.y <= mDPartition.m_fullGridSize.y && isValidNeighbour;
        }
        if constexpr (yOff < 0) {
            isValidNeighbour = nghCartesianGlobal.y >= 0 && isValidNeighbour;
        }
        if constexpr (zOff > 0) {
            isValidNeighbour = cellNgh.get().z < (m_dim.z + m_zHaloRadius * 2) && isValidNeighbour;
            isValidNeighbour = nghCartesianGlobal.z <= mDPartition.m_fullGridSize.z && isValidNeighbour;
        }
        if constexpr (zOff < 0) {
            isValidNeighbour = nghCartesianGlobal.z >= mDPartition.m_zHaloRadius && isValidNeighbour;
        }
        return isValidNeighbour;
    }

    NEON_CUDA_HOST_DEVICE inline auto
    mem()
        -> T*
    {
        return mDPartition.m_mem;
    }

    NEON_CUDA_HOST_DEVICE inline auto
    mem() const
        -> const T*
    {
        return mDPartition.m_mem;
    }

    NEON_CUDA_HOST_DEVICE inline auto
    mem(const Idx& cell,
        int        cardinalityIdx)
        -> T*
    {
        Idx::Offset p = getPitch(cell, cardinalityIdx);
        return mDPartition.m_mem[p];
    }

    NEON_CUDA_HOST_DEVICE inline auto
    operator()(const Idx& cell,
               int        cardinalityIdx)
        -> T&
    {
        Idx::Offset p = getPitch(cell, cardinalityIdx);
        return mDPartition.m_mem[p];
    }

    NEON_CUDA_HOST_DEVICE inline auto
    operator()(const Idx& cell,
               int        cardinalityIdx)
        const -> const T&
    {
        Idx::Offset p = getPitch(cell, cardinalityIdx);
        return mDPartition.m_mem[p];
    }

    NEON_CUDA_HOST_DEVICE inline auto getGlobalIndex(const Idx& local)
        const -> Neon::index_3d
    {
        Neon::index_3d result = local.mLocation + m_origin;
        result.z -= mDPartition.m_zHaloRadius;
        return result;
    }

    NEON_CUDA_HOST_DEVICE inline auto getDomainSize()
        const -> Neon::index_3d
    {
        return mDPartition.m_fullGridSize;
    }

    Neon::DataView        mDataView;
    T* NEON_RESTRICT      mMem;
    Neon::index_3d        mDim;
    int                   mZHaloRadius;
    int                   mZBoundaryRadius;
    Pitch                 mPitch;
    int                   mPrtID;
    Neon::index_3d        mOrigin;
    int                   mCardinality;
    Neon::index_3d        mFullGridSize;
    bool                  mPeriodicZ;
    NghIdx* NEON_RESTRICT mStencil;
};

}  // namespace Neon::domain::details::dGridSoA
