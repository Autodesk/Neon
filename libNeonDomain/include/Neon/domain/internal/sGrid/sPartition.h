#pragma once

#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"
#include "Neon/set/DevSet.h"
#include "Neon/set/memory/memSet.h"
#include "Neon/sys/memory/CudaIntrinsics.h"

#include "Neon/domain/internal/sGrid/sCell.h"
#include "sPartitionIndexSpace.h"


namespace Neon::domain::internal::sGrid {


template <typename OuterGridT, typename T, int C = 0>
class sField;

/**
 * Partition abstraction for sGrid
 */
template <typename OuterGridT, typename T, int C = 0>
struct sPartition
{
    /**
     * Internals:
     * Local field memory layout:
     *
     * a. Memory layout for sPartition
     *  |
     *  |   m_mem
     *  |   |
     *  |   |<--------------------><----------------->|
     *  |             INTERNAL             BOUNDARY            GHOST
     *  |
     *  |   SoA or AoS options are managed transparently by mPitch internal variable,
     *  |   which has 2 components, one for the pitch on the element position (eIdx)
     *  |   and the other on the element cardinality.
     *  |--)
     *
     *  b. DataView:
     *  |
     *  |   The data view is handled by the Partition Index Space.
     *  |   The Cell initialized by the Partition Index Space
     *  |   is an offset from top pf the buffer for the partition.
     *  |   In this implementation, mMemory does not change based on the data view.
     *  |--)
     *
     */

   public:
    template <typename O_,
              typename T_,
              int Cardinality>
    friend class sField;


    //-- [PUBLIC TYPES] ----------------------------------------------------------------------------
    using OuterGrid = OuterGridT;
    static constexpr int Cardinality = C;
    using Self = sPartition<OuterGridT, T, Cardinality>;  //<- this type
    using Cell = sCell;                                   //<- type of an index to an element

    using Type = T;        //<- type of the data stored by the field
    using Jump = index_t;  //<- Type of a Jump value
    using Pitch = Neon::index64_2d;
    ;  //<- Type of the pitch representation
    using Count = int;
    using Index = int;

   private:
    //-- [INTERNAL DATA] ----------------------------------------------------------------------------
    T*    mMemory;
    int   mCardinality;
    Pitch mPitch;

    Neon::DataView mDataView;

    //-- [CONNECTIVITY] ----------------------------------------------------------------------------
    typename OuterGrid::Cell::OuterCell const* mTableToOuterCell;

    int mPartitionId;

   public:
    //-- [CONSTRUCTORS] ----------------------------------------------------------------------------

    /**
     * Default constructor
     */
    sPartition() = default;

    /**
     * Default Destructor
     */
    ~sPartition() = default;

    /**
     * Returns the partition ID
     */
    NEON_CUDA_HOST_DEVICE auto
    getPartitionId() const
        -> int;

    /**
     * Non scalar fields stores vectors on each cell.
     * The number of components of the vector also called cardinality
     */
    template <int dummy_ta = Cardinality>
    inline NEON_CUDA_HOST_DEVICE auto
    cardinality() const
        -> std::enable_if_t<dummy_ta == 0, int>;

    /**
     * Non scalar fields stores vectors on each cell.
     * The number of components of the vector also called cardinality
     */
    template <int dummy_ta = Cardinality>
    constexpr inline NEON_CUDA_HOST_DEVICE auto
    cardinality() const
        -> std::enable_if_t<dummy_ta != 0, int>;

    /**
     * Access method for cell metadata
     */
    NEON_CUDA_HOST_DEVICE inline auto
    operator()(Cell const& eId,
               int         cardinalityIdx) const
        -> T;

    /**
     * Access method for cell metadata
     */
    NEON_CUDA_HOST_DEVICE inline auto
    operator()(Cell const& eId,
               int         cardinalityIdx) -> T&;


    template <typename ComputeType>
    NEON_CUDA_HOST_DEVICE inline auto
    castRead(Cell eId, int cardinalityIdx) const
        -> ComputeType;

    template <typename ComputeType>
    NEON_CUDA_HOST_DEVICE inline auto
    castWrite(Cell eId, int cardinalityIdx, const ComputeType& value)
        -> void;

    /**
     * Translate a sGrid cell to a outer grid cell
     */
    NEON_CUDA_HOST_DEVICE inline auto
    mapToOuterGrid(const Cell&) const
        -> typename OuterGrid::Cell::OuterCell const&;

   private:
    /**
     * Helper function to compute the offset associated to a cell
     */
    NEON_CUDA_HOST_DEVICE inline auto
    helpGetJump(Cell const& eId) const
        -> Jump;

    /**
     * Helper function to compute the offset associated to a cell
     */
    NEON_CUDA_HOST_DEVICE inline auto
    helpGetJump(Cell const& eId,
                int         cardinalityIdx) const
        -> Jump;

   private:
    /**
     * Private constructor
     */
    explicit sPartition(const Neon::DataView&                      dataView,
                        int                                        prtId,
                        T*                                         mem,
                        int                                        cardinality,
                        const Pitch&                               ePitch,
                        typename OuterGrid::Cell::OuterCell const* tableToOuterCell);
};
}  // namespace Neon::domain::internal::sGrid
