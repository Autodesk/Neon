#pragma once
#include <assert.h>
#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"
#include "Neon/domain/interface/NghInfo.h"
#include "Neon/set/DevSet.h"
#include "Neon/sys/memory/CudaIntrinsics.h"
#include "Neon/sys/memory/mem3d.h"
#include "cuda_fp16.h"
#include "dCell.h"
namespace Neon::domain::internal::dGrid {

/**
 * Local representation for the dField for one device
 * works as a wrapper for the mem3d which represent the allocated memory on a
 * single device.
 **/
class dPartitionIndexSpace;

template <typename T_ta, int cardinality_ta = 0>
struct dPartition
{
   public:
    using PartitionIndexSpace = dPartitionIndexSpace;
    using self_t = dPartition<T_ta, cardinality_ta>;
    using Cell = dCell;
    using nghIdx_t = int8_3d;
    using Type = T_ta;
    using ePitch_t = Neon::size_4d;

   private:
    Neon::DataView m_dataView;
    T_ta*          m_mem;
    Neon::index_3d m_dim;
    int            m_zHaloRadius;
    int            m_zBoundaryRadius;
    ePitch_t       m_pitch;
    int            m_prtID;
    Neon::index_3d m_origin;
    int            m_cardinality;
    Neon::index_3d m_fullGridSize;
    bool           mPeriodicZ;
    nghIdx_t*      mStencil;

   public:
    dPartition() = default;

    ~dPartition() = default;

    explicit dPartition(Neon::DataView dataView,
                        T_ta*          mem,
                        Neon::index_3d dim,
                        int            zHaloRadius,
                        int            zBoundaryRadius,
                        ePitch_t       pitch,
                        int            prtID,
                        Neon::index_3d origin,
                        int            cardinality,
                        Neon::index_3d fullGridSize,
                        nghIdx_t*      stencil = nullptr)
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

    inline NEON_CUDA_HOST_ONLY auto enablePeriodicAlongZ() -> void
    {
        mPeriodicZ = true;
    }

    inline NEON_CUDA_HOST_DEVICE auto prtID() const -> int
    {
        return m_prtID;
    }

    inline NEON_CUDA_HOST_DEVICE auto cardinality() const -> int
    {
        return m_cardinality;
    }

    inline NEON_CUDA_HOST_DEVICE auto ePitch() const -> const ePitch_t&
    {
        return m_pitch;
    }

    inline NEON_CUDA_HOST_DEVICE int64_t elPitch(const Cell& idx,
                                                 int         cardinalityIdx = 0) const
    {
        return idx.get().x * int64_t(m_pitch.x) +
               idx.get().y * int64_t(m_pitch.y) +
               idx.get().z * int64_t(m_pitch.z) +
               cardinalityIdx * int64_t(m_pitch.w);
    }

    inline NEON_CUDA_HOST_DEVICE auto dim() const -> const Neon::index_3d
    {
        return m_dim;
    }

    inline NEON_CUDA_HOST_DEVICE auto halo() const -> const Neon::index_3d
    {
        return Neon::index_3d(0, 0, m_zHaloRadius);
    }

    inline NEON_CUDA_HOST_DEVICE auto origin() const -> const Neon::index_3d
    {
        return m_origin;
    }

    NEON_CUDA_HOST_DEVICE inline auto nghVal(const Cell& eId,
                                             nghIdx_t    nghOffset,
                                             int         card,
                                             const T_ta& alternativeVal) const -> NghInfo<T_ta>
    {
        Cell       cellNgh;
        const bool isValidNeighbour = nghIdx(eId, nghOffset, cellNgh);
        T_ta       val = alternativeVal;
        if (isValidNeighbour) {
            val = operator()(cellNgh, card);
        }
        return NghInfo<T_ta>(val, isValidNeighbour);
    }

    template <int xOff, int yOff, int zOff>
    NEON_CUDA_HOST_DEVICE inline auto nghVal(const Cell& eId,
                                             int         card,
                                             const T_ta& alternativeVal) const -> NghInfo<T_ta>
    {
        Cell       cellNgh;
        const bool isValidNeighbour = nghIdx<xOff, yOff, zOff>(eId, cellNgh);
        T_ta       val = alternativeVal;
        if (isValidNeighbour) {
            val = operator()(cellNgh, card);
        }
        return NghInfo<T_ta>(val, isValidNeighbour);
    }

    NEON_CUDA_HOST_DEVICE inline auto nghVal(const Cell& eId,
                                             uint8_t     nghID,
                                             int         card,
                                             const T_ta& alternativeVal) const -> NghInfo<T_ta>
    {
        nghIdx_t nghOffset = mStencil[nghID];
        return nghVal(eId, nghOffset, card, alternativeVal);
    }
    /**
     * Get the index of the neighbor given the offset
     * @tparam dataView_ta
     * @param[in] eId Index of the current element
     * @param[in] nghOffset Offset of the neighbor of interest from the current element
     * @param[in,out] neighbourIdx Index of the neighbor
     * @return Whether the neighbour is valid
     */
    NEON_CUDA_HOST_DEVICE inline auto nghIdx(const Cell&     eId,
                                             const nghIdx_t& nghOffset,
                                             Cell&           neighbourIdx) const -> bool
    {
        Cell cellNgh(eId.get().x + nghOffset.x,
                     eId.get().y + nghOffset.y,
                     eId.get().z + nghOffset.z);

        Cell cellNgh_global(cellNgh.get() + m_origin);

        bool isValidNeighbour = true;

        isValidNeighbour = (cellNgh_global.get().x >= 0) &&
                           (cellNgh_global.get().y >= 0) &&
                           ((!mPeriodicZ && cellNgh_global.get().z >= m_zHaloRadius) ||
                            (mPeriodicZ && cellNgh_global.get().z >= 0)) &&
                           isValidNeighbour;

        isValidNeighbour = (cellNgh.get().x < m_dim.x) &&
                           (cellNgh.get().y < m_dim.y) &&
                           (cellNgh.get().z < m_dim.z + 2 * m_zHaloRadius) && isValidNeighbour;

        isValidNeighbour = (cellNgh_global.get().x <= m_fullGridSize.x) &&
                           (cellNgh_global.get().y <= m_fullGridSize.y) &&
                           ((!mPeriodicZ && cellNgh_global.get().z <= m_fullGridSize.z) ||
                            (mPeriodicZ && cellNgh_global.get().z <= m_fullGridSize.z + 2 * m_zHaloRadius)) &&
                           isValidNeighbour;

        if (isValidNeighbour) {
            neighbourIdx = cellNgh;
        }
        return isValidNeighbour;
    }

    template <int xOff, int yOff, int zOff>
    NEON_CUDA_HOST_DEVICE inline auto nghIdx(const Cell& eId,
                                             Cell&       cellNgh) const -> bool
    {
        cellNgh = Cell(eId.get().x + xOff,
                       eId.get().y + yOff,
                       eId.get().z + zOff);
        Cell cellNgh_global(cellNgh.get() + m_origin);
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


    NEON_CUDA_HOST_DEVICE inline auto mem() -> T_ta*
    {
        return m_mem;
    }

    NEON_CUDA_HOST_DEVICE inline auto mem() const -> const T_ta*
    {
        return m_mem;
    }

    NEON_CUDA_HOST_DEVICE inline auto cmem() const -> const T_ta*
    {
        return m_mem;
    }


    NEON_CUDA_HOST_DEVICE inline auto operator()(const Cell& cell,
                                                 int         cardinalityIdx) -> T_ta&
    {
        int64_t p = elPitch(cell, cardinalityIdx);
        return m_mem[p];
    }

    NEON_CUDA_HOST_DEVICE inline auto operator()(const Cell& cell,
                                                 int         cardinalityIdx) const -> const T_ta&
    {
        int64_t p = elPitch(cell, cardinalityIdx);
        return m_mem[p];
    }

    template <typename ComputeType>
    NEON_CUDA_HOST_DEVICE inline auto castRead(const Cell& cell,
                                               int         cardinalityIdx) const -> ComputeType
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
    NEON_CUDA_HOST_DEVICE inline auto castWrite(const Cell&        cell,
                                                int                cardinalityIdx,
                                                const ComputeType& value) -> void
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

    NEON_CUDA_HOST_DEVICE inline auto ePrt(const index_3d& cell,
                                           int             cardinalityIdx) -> T_ta*
    {
        int64_t p = elPitch(cell, cardinalityIdx);
        return m_mem + p;
    }

    NEON_CUDA_HOST_DEVICE inline auto mapToGlobal(const Cell& local) const -> Neon::index_3d
    {
        assert(local.mLocation.x >= 0 &&
               local.mLocation.y >= 0 &&
               local.mLocation.z >= -m_zHaloRadius &&
               local.mLocation.x < m_dim.x &&
               local.mLocation.y < m_dim.y &&
               local.mLocation.z < m_dim.z + m_zHaloRadius);

        switch (m_dataView) {
            case Neon::DataView::STANDARD: {
                return local.mLocation + m_origin;
            }
            default: {
            }
        }
#if defined(NEON_PLACE_CUDA_HOST)
        NEON_THROW_UNSUPPORTED_OPTION();
#else
        int* error = nullptr;
        error[0] = 0xBAD;
        return int64_t(size_t(0xffffffffffffffff) - 1);
#endif
    }
};


}  // namespace Neon::domain::internal::dGrid
