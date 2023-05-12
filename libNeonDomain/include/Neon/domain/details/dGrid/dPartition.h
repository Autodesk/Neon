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
        : m_dataView(dataView),
          m_mem(mem),
          m_dim(dim),
          m_zHaloRadius(zHaloRadius),
          m_zBoundaryRadius(zBoundaryRadius),
          m_pitch(pitch),
          m_prtID(prtID),
          m_origin(origin),
          m_cardinality(cardinality),
          m_fullGridSize(fullGridSize),
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
        return m_prtID;
    }

    inline NEON_CUDA_HOST_DEVICE auto
    cardinality()
        const -> int
    {
        return m_cardinality;
    }

    inline NEON_CUDA_HOST_DEVICE auto
    getPitchData()
        const -> const Pitch&
    {
        return m_pitch;
    }

    inline NEON_CUDA_HOST_DEVICE auto
    getPitch(const Idx& idx,
             int        cardinalityIdx = 0)
        const -> int64_t
    {
        return idx.get().x * int64_t(m_pitch.x) +
               idx.get().y * int64_t(m_pitch.y) +
               idx.get().z * int64_t(m_pitch.z) +
               cardinalityIdx * int64_t(m_pitch.w);
    }

    inline NEON_CUDA_HOST_DEVICE auto
    dim()
        const -> const Neon::index_3d
    {
        return m_dim;
    }

    inline NEON_CUDA_HOST_DEVICE auto
    halo()
        const -> const Neon::index_3d
    {
        return Neon::index_3d(0, 0, m_zHaloRadius);
    }

    inline NEON_CUDA_HOST_DEVICE auto
    origin()
        const -> const Neon::index_3d
    {
        return m_origin;
    }

    NEON_CUDA_HOST_DEVICE inline auto
    getNghData(const Idx& eId,
               NghIdx     nghOffset,
               int        card,
               const T&   alternativeVal)
        const -> NghData
    {
        Idx        cellNgh;
        const bool isValidNeighbour = nghIdx(eId, nghOffset, cellNgh);
        T          val = alternativeVal;
        if (isValidNeighbour) {
            val = operator()(cellNgh, card);
        }
        return NghData(val, isValidNeighbour);
    }

    NEON_CUDA_HOST_DEVICE inline auto
    getNghData(const Idx& eId,
               NghIdx     nghOffset,
               int        card)
        const -> NghData
    {
        Idx        cellNgh;
        const bool isValidNeighbour = nghIdx(eId, nghOffset, cellNgh);
        T          val;
        if (isValidNeighbour) {
            val = operator()(cellNgh, card);
        }
        return NghData(val, isValidNeighbour);
    }

    template <int xOff, int yOff, int zOff, typename LambdaVALID, typename LambdaNOTValid = void*>
    NEON_CUDA_HOST_DEVICE inline auto
    getNghData(const Idx&     eId,
               int            card,
               LambdaVALID    funIfValid,
               LambdaNOTValid funIfNOTValid = nullptr)
        const -> void
    {
        Idx        cellNgh;
        const bool isValidNeighbour = nghIdx<xOff, yOff, zOff>(eId, cellNgh);
        if (isValidNeighbour) {
            T val = operator()(cellNgh, card);
            funIfValid(val);
        }
        if constexpr (!std::same_as<LambdaNOTValid, void*>) {
            if (!isValidNeighbour) {
                funIfNOTValid();
            }
        }
    }

    NEON_CUDA_HOST_DEVICE inline auto
    nghVal(const Idx& eId,
           uint8_t    nghID,
           int        card,
           const T&   alternativeVal)
        const -> NghData
    {
        NghIdx nghOffset = mStencil[nghID];
        return getNghData(eId, nghOffset, card, alternativeVal);
    }
    /**
     * Get the index of the neighbor given the offset
     * @tparam dataView_ta
     * @param[in] eId Index of the current element
     * @param[in] nghOffset Offset of the neighbor of interest from the current element
     * @param[in,out] neighbourIdx Index of the neighbor
     * @return Whether the neighbour is valid
     */
    NEON_CUDA_HOST_DEVICE inline auto
    nghIdx(const Idx&    eId,
           const NghIdx& nghOffset,
           Idx&          neighbourIdx)
        const -> bool
    {
        Idx cellNgh(eId.get().x + nghOffset.x,
                    eId.get().y + nghOffset.y,
                    eId.get().z + nghOffset.z);

        const auto cellNghGlobal = getGlobalIndex(cellNgh);

        bool isValidNeighbour = true;

        if (mPeriodicZ) {
            printf("Error, periodic not implemented yet");
            assert(false);
        }

        isValidNeighbour = (cellNghGlobal.x >= 0) &&
                           (cellNghGlobal.y >= 0) &&
                           (cellNghGlobal.z >= 0);

        //        isValidNeighbour = (cellNgh.get().x < m_dim.x) &&
        //                           (cellNgh.get().y < m_dim.y) &&
        //                           (cellNgh.get().z < m_dim.z + 2 * m_zHaloRadius) && isValidNeighbour;

        isValidNeighbour = (cellNghGlobal.x < m_fullGridSize.x) &&
                           (cellNghGlobal.y < m_fullGridSize.y) &&
                           (cellNghGlobal.z < m_fullGridSize.z) &&
                           isValidNeighbour;

        if (isValidNeighbour) {
            neighbourIdx = cellNgh;
        }
        return isValidNeighbour;
    }

    template <int xOff, int yOff, int zOff>
    NEON_CUDA_HOST_DEVICE inline auto
    nghIdx(const Idx& eId,
           Idx&       cellNgh)
        const -> bool
    {
        cellNgh = Idx(eId.get().x + xOff,
                      eId.get().y + yOff,
                      eId.get().z + zOff);
        Idx cellNgh_global(cellNgh.get() + m_origin);
        // const bool isValidNeighbour = (cellNgh_global >= 0 && cellNgh < (m_dim + m_halo) && cellNgh_global < m_fullGridSize);
        bool isValidNeighbour = true;
        if constexpr (xOff > 0) {
            isValidNeighbour = cellNgh.get().x < (m_dim.x) && isValidNeighbour;
            isValidNeighbour = cellNgh_global.get().x <= m_fullGridSize.x && isValidNeighbour;
        }
        if constexpr (xOff < 0) {
            isValidNeighbour = cellNgh_global.get().x >= 0 && isValidNeighbour;
        }
        if constexpr (yOff > 0) {
            isValidNeighbour = cellNgh.get().y < (m_dim.y) && isValidNeighbour;
            isValidNeighbour = cellNgh_global.get().y <= m_fullGridSize.y && isValidNeighbour;
        }
        if constexpr (yOff < 0) {
            isValidNeighbour = cellNgh_global.get().y >= 0 && isValidNeighbour;
        }
        if constexpr (zOff > 0) {
            isValidNeighbour = cellNgh.get().z < (m_dim.z + m_zHaloRadius * 2) && isValidNeighbour;
            isValidNeighbour = cellNgh_global.get().z <= m_fullGridSize.z && isValidNeighbour;
        }
        if constexpr (zOff < 0) {
            isValidNeighbour = cellNgh_global.get().z >= m_zHaloRadius && isValidNeighbour;
        }
        return isValidNeighbour;
    }


    NEON_CUDA_HOST_DEVICE inline auto
    mem()
        -> T*
    {
        return m_mem;
    }

    NEON_CUDA_HOST_DEVICE inline auto
    mem()
        const
        -> const T*
    {
        return m_mem;
    }

    NEON_CUDA_HOST_DEVICE inline auto
    mem(const Idx& cell,
        int        cardinalityIdx) -> T*
    {
        int64_t p = getPitch(cell, cardinalityIdx);
        return m_mem[p];
    }

    NEON_CUDA_HOST_DEVICE inline auto
    operator()(const Idx& cell,
               int        cardinalityIdx) -> T&
    {
        int64_t p = getPitch(cell, cardinalityIdx);
        return m_mem[p];
    }

    NEON_CUDA_HOST_DEVICE inline auto
    operator()(const Idx& cell,
               int        cardinalityIdx) const -> const T&
    {
        int64_t p = getPitch(cell, cardinalityIdx);
        return m_mem[p];
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

        Neon::index_3d result = local.mLocation + m_origin;
        result.z -= m_zHaloRadius;
        return result;
    }

    NEON_CUDA_HOST_DEVICE inline auto getDomainSize()
        const -> Neon::index_3d
    {
        return m_fullGridSize;
    }

    auto ioToVti(std::string const& fname, std::string const& fieldName)
    {
        auto fnameCommplete = fname + "_" + std::to_string(m_prtID);
        auto haloOrigin = Vec_3d<double>(m_origin.x, m_origin.y, m_origin.z - m_zHaloRadius);
        auto haloDim = m_dim + Neon::index_3d(0, 0, 2 * m_zHaloRadius) + 1;

        IoToVTK<int, int64_t> io(fnameCommplete,
                                 haloDim,
                                 Vec_3d<double>(1, 1, 1),
                                 haloOrigin,
                                 Neon::IoFileType::ASCII);


        io.addField([&](const Neon::index_3d& idx, int i) {
            return operator()(dIndex(idx), i);
        },
                    m_cardinality, "Partition", ioToVTKns::VtiDataType_e::voxel);

        io.flushAndClear();
        return;
    }

   private:
    Neon::DataView m_dataView;
    T*             m_mem;
    Neon::index_3d m_dim;
    int            m_zHaloRadius;
    int            m_zBoundaryRadius;
    Pitch          m_pitch;
    int            m_prtID;
    Neon::index_3d m_origin;
    int            m_cardinality;
    Neon::index_3d m_fullGridSize;
    bool           mPeriodicZ;
    NghIdx*        mStencil;
};


}  // namespace Neon::domain::details::dGrid
