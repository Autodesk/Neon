#pragma once
#include <assert.h>
#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"
#include "Neon/domain/interface/NghData.h"
#include "Neon/set/DevSet.h"
#include "Neon/sys/memory/CudaIntrinsics.h"
#include "cuda_fp16.h"
#include "dIndex.h"
namespace Neon::domain::details::dGrid {

/**
 * Local representation for the dField for one device
 * works as a wrapper for the mem3d which represent the allocated memory on a
 * single device.
 **/

template <typename T, int C = 0>
class dPartition
{
   public:
    using PartitionIndexSpace = dSpan;
    using Span = dSpan;
    using Self = dPartition<T, C>;
    using Idx = dIndex;
    using NghIdx = int8_3d;
    using NghData = Neon::domain::NghData<T>;
    using Type = T;
    using Pitch = Neon::size_4d;

   public:
    dPartition() = default;

    ~dPartition() = default;

    explicit dPartition(Neon::DataView dataView,
                        T*             mem,
                        Neon::index_3d dim,
                        int            zHaloRadius,
                        int            zBoundaryRadius,
                        Pitch          pitch,
                        int            prtID,
                        Neon::index_3d origin,
                        int            cardinality,
                        Neon::index_3d fullGridSize,
                        NghIdx*        stencil = nullptr)
        : mDataView(dataView),
          mMem(mem),
          mDim(dim),
          mZHaloRadius(zHaloRadius),
          mZBoundaryRadius(zBoundaryRadius),
          mPitch(pitch),
          mPrtID(prtID),
          mOrigin(origin),
          mCardinality(cardinality),
          mFullGridSize(fullGridSize),
          mPeriodicZ(false),
          mStencil(stencil)
    {
    }

    inline NEON_CUDA_HOST_ONLY auto
    enablePeriodicAlongZ()
        -> void
    {
        mPeriodicZ = true;
    }

    inline NEON_CUDA_HOST_DEVICE auto
    prtID()
        const -> int
    {
        return mPrtID;
    }

    inline NEON_CUDA_HOST_DEVICE auto
    cardinality()
        const -> int
    {
        return mCardinality;
    }

    inline NEON_CUDA_HOST_DEVICE auto
    getPitchData()
        const -> const Pitch&
    {
        return mPitch;
    }

    inline NEON_CUDA_HOST_DEVICE auto
    getPitch(const Idx& idx,
             int        cardinalityIdx = 0)
        const -> int64_t
    {
        return idx.getLocation().x * int64_t(mPitch.x) +
               idx.getLocation().y * int64_t(mPitch.y) +
               idx.getLocation().z * int64_t(mPitch.z) +
               cardinalityIdx * int64_t(mPitch.w);
    }

    inline NEON_CUDA_HOST_DEVICE auto
    dim()
        const -> const Neon::index_3d
    {
        return mDim;
    }

    inline NEON_CUDA_HOST_DEVICE auto
    halo()
        const -> const Neon::index_3d
    {
        return Neon::index_3d(0, 0, mZHaloRadius);
    }

    inline NEON_CUDA_HOST_DEVICE auto
    origin()
        const -> const Neon::index_3d
    {
        return mOrigin;
    }

    NEON_CUDA_HOST_DEVICE inline auto
    getNghData(const Idx& gidx,
               NghIdx     nghOffset,
               int        card,
               const T&   alternativeVal)
        const -> NghData
    {
        Idx        gidxNgh;
        const bool isValidNeighbour = helpGetNghIdx(gidx, nghOffset, gidxNgh);
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
        const bool isValidNeighbour = helpGetNghIdx(gidx, nghOffset, gidxNgh);
        T          val;
        if (isValidNeighbour) {
            val = operator()(gidxNgh, card);
        }
        return NghData(val, isValidNeighbour);
    }

    template <int xOff, int yOff, int zOff, typename LambdaVALID, typename LambdaNOTValid = void*>
    NEON_CUDA_HOST_DEVICE inline auto
    getNghData(const Idx&     gidx,
               int            card,
               LambdaVALID    funIfValid,
               LambdaNOTValid funIfNOTValid = nullptr)
        const -> std::enable_if_t<std::is_invocable_v<LambdaVALID, T>, void>
    {
        Idx        gidxNgh;
        const bool isValidNeighbour = helpGetNghIdx<xOff, yOff, zOff>(gidx, gidxNgh);
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

    template <int xOff,
              int yOff,
              int zOff>
    NEON_CUDA_HOST_DEVICE inline auto
    getNghData(const Idx& gidx,
               int        card)
        const -> NghData
    {
        Idx        gidxNgh;
        const bool isValidNeighbour = helpGetNghIdx<xOff, yOff, zOff>(gidx, gidxNgh);
        T          val;
        if (isValidNeighbour) {
            val = operator()(gidxNgh, card);
        }
        return NghData(val, isValidNeighbour);
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
        const bool isValidNeighbour = helpGetNghIdx<xOff, yOff, zOff>(gidx, gidxNgh);
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
    helpGetNghIdx(const Idx&    gidx,
                  const NghIdx& nghOffset,
                  Idx&          neighbourIdx)
        const -> bool
    {
        Idx gidxNgh(gidx.getLocation().x + nghOffset.x,
                    gidx.getLocation().y + nghOffset.y,
                    gidx.getLocation().z + nghOffset.z);

        const auto gidxNghGlobal = getGlobalIndex(gidxNgh);

        bool isValidNeighbour = true;

        isValidNeighbour = (gidxNghGlobal.x >= 0) &&
                           (gidxNghGlobal.y >= 0) &&
                           (gidxNghGlobal.z >= 0);

        isValidNeighbour = (gidxNghGlobal.x < mFullGridSize.x) &&
                           (gidxNghGlobal.y < mFullGridSize.y) &&
                           (gidxNghGlobal.z < mFullGridSize.z) &&
                           isValidNeighbour;

        if (isValidNeighbour) {
            neighbourIdx = gidxNgh;
        }
        return isValidNeighbour;
    }

    template <int xOff, int yOff, int zOff>
    NEON_CUDA_HOST_DEVICE inline auto
    helpGetNghIdx(const Idx& gidx,
                  Idx&       gidxNgh)
        const -> bool
    {
        return helpGetNghIdx(gidx, NghIdx{xOff, yOff, zOff}, gidxNgh);
        //        gidxNgh = Idx(gidx.getLocation().x + xOff,
        //                      gidx.getLocation().y + yOff,
        //                      gidx.getLocation().z + zOff);
        //
        //        bool isValidNeighbour = true;
        //        if constexpr (xOff > 0) {
        //            int constexpr direction = Neon::index_3d::directionX;
        //            int const cartesianByDirection = getGlobalIndexByDirection<direction>(gidxNgh);
        //            isValidNeighbour = cartesianByDirection < mFullGridSize.v[direction] && isValidNeighbour;
        //        }
        //        if constexpr (xOff < 0) {
        //            int constexpr direction = Neon::index_3d::directionX;
        //            int const cartesianByDirection = getGlobalIndexByDirection<direction>(gidxNgh);
        //            isValidNeighbour = cartesianByDirection >= 0 && isValidNeighbour;
        //        }
        //        if constexpr (yOff > 0) {
        //            int constexpr direction = Neon::index_3d::directionY;
        //            int const cartesianByDirection = getGlobalIndexByDirection<direction>(gidxNgh);
        //            isValidNeighbour = cartesianByDirection < mFullGridSize.v[direction] && isValidNeighbour;
        //        }
        //        if constexpr (yOff < 0) {
        //            int constexpr direction = Neon::index_3d::directionY;
        //            int const cartesianByDirection = getGlobalIndexByDirection<direction>(gidxNgh);
        //            isValidNeighbour = cartesianByDirection >= 0 && isValidNeighbour;
        //        }
        //        if constexpr (zOff > 0) {
        //            int constexpr direction = Neon::index_3d::directionZ;
        //            int const cartesianByDirection = getGlobalIndexByDirection<direction>(gidxNgh);
        //            isValidNeighbour = cartesianByDirection < mFullGridSize.v[direction] && isValidNeighbour;
        //        }
        //        if constexpr (zOff < 0) {
        //            int constexpr direction = Neon::index_3d::directionZ;
        //            int const cartesianByDirection = getGlobalIndexByDirection<direction>(gidxNgh);
        //            isValidNeighbour = cartesianByDirection >= 0 && isValidNeighbour;
        //        }
        //        return isValidNeighbour;
    }


    NEON_CUDA_HOST_DEVICE inline auto
    mem()
        -> T*
    {
        return mMem;
    }

    NEON_CUDA_HOST_DEVICE inline auto
    mem()
        const
        -> const T*
    {
        return mMem;
    }

    NEON_CUDA_HOST_DEVICE inline auto
    mem(const Idx& cell,
        int        cardinalityIdx) -> T*
    {
        int64_t p = getPitch(cell, cardinalityIdx);
        return mMem[p];
    }

    NEON_CUDA_HOST_DEVICE inline auto
    operator()(const Idx& cell,
               int        cardinalityIdx) -> T&
    {
        int64_t p = getPitch(cell, cardinalityIdx);
        return mMem[p];
    }

    NEON_CUDA_HOST_DEVICE inline auto
    operator()(const Idx& cell,
               int        cardinalityIdx) const -> const T&
    {
        int64_t p = getPitch(cell, cardinalityIdx);
        return mMem[p];
    }

    template <typename ComputeType>
    NEON_CUDA_HOST_DEVICE inline auto
    castRead(const Idx& cell,
             int        cardinalityIdx)
        const -> ComputeType
    {
        Type value = this->operator()(cell, cardinalityIdx);
        if constexpr (std::is_same_v<__half, Type>) {

            if constexpr (std::is_same_v<float, ComputeType>) {
                return __half2float(value);
            }
            if constexpr (std::is_same_v<double, ComputeType>) {
                return static_cast<double>(__half2double(value));
            }
        } else {
            return static_cast<ComputeType>(value);
        }
    }

    template <typename ComputeType>
    NEON_CUDA_HOST_DEVICE inline auto
    castWrite(const Idx&         cell,
              int                cardinalityIdx,
              const ComputeType& value)
        -> void
    {
        if constexpr (std::is_same_v<__half, Type>) {
            if constexpr (std::is_same_v<float, ComputeType>) {
                this->operator()(cell, cardinalityIdx) = __float2half(value);
            }
            if constexpr (std::is_same_v<double, ComputeType>) {
                this->operator()(cell, cardinalityIdx) = __double2half(value);
            }
        } else {
            this->operator()(cell, cardinalityIdx) = static_cast<Type>(value);
        }
    }

    NEON_CUDA_HOST_DEVICE inline auto getGlobalIndex(const Idx& local) const -> Neon::index_3d
    {
        //        assert(local.mLocation.x >= 0 &&
        //               local.mLocation.y >= 0 &&
        //               local.mLocation.z >= m_zHaloRadius &&
        //               local.mLocation.x < m_dim.x &&
        //               local.mLocation.y < m_dim.y &&
        //               local.mLocation.z < m_dim.z + m_zHaloRadius);

        Neon::index_3d result = local.mLocation;
        result.z = result.z + mOrigin.z - mZHaloRadius;
        return result;
    }

    template <int direction>
    NEON_CUDA_HOST_DEVICE inline auto getGlobalIndexByDirection(const Idx& local)
        const -> int
    {
        if constexpr (Neon::index_3d::directionZ != direction) {
            return local.mLocation.v[direction];
        } else {
            return local.mLocation.v[Neon::index_3d::directionZ] +
                   mOrigin.v[Neon::index_3d::directionZ] -
                   mZHaloRadius;
        }
    }

    NEON_CUDA_HOST_DEVICE inline auto getDomainSize()
        const -> Neon::index_3d
    {
        return mFullGridSize;
    }

    auto ioToVti(std::string const& fname, std::string const& fieldName)
    {
        auto fnameCommplete = fname + "_" + std::to_string(mPrtID);
        auto haloOrigin = Vec_3d<double>(mOrigin.x, mOrigin.y, mOrigin.z - mZHaloRadius);
        auto haloDim = mDim + Neon::index_3d(0, 0, 2 * mZHaloRadius) + 1;

        IoToVTK<int, int64_t> io(fnameCommplete,
                                 haloDim,
                                 Vec_3d<double>(1, 1, 1),
                                 haloOrigin,
                                 Neon::IoFileType::ASCII);


        io.addField([&](const Neon::index_3d& idx, int i) {
            return operator()(dIndex(idx), i);
        },
                    mCardinality, "Partition", ioToVTKns::VtiDataType_e::voxel);

        io.flushAndClear();
        return;
    }

    auto getDataView()
        const -> Neon::DataView
    {
        return mDataView;
    }

    auto helpGetGlobalToLocalOffets()
        const -> NghIdx*
    {
        return mStencil;
    }

   private:
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


}  // namespace Neon::domain::details::dGrid
