#pragma once

#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"
#include "Neon/domain/interface/NghInfo.h"
#include "Neon/domain/interface/common.h"
#include "Neon/set/DevSet.h"
#include "Neon/set/memory/memSet.h"
#include "Neon/sys/memory/CudaIntrinsics.h"
#include "eCommon.h"
#include "ePartitionIndexSpace.h"

#include "ePartitionIndexSpace.h"

namespace Neon::domain::internal::eGrid {


/**
 * Local representation for the sparse eGrid.
 *
 * @tparam T_ta: type of data stored by this field
 * @tparam cardinality_ta: cardinality of this filed.
 *                         If cardinality_ta is 0, the actual cardinality is manged dynamically
 */


template <typename T_ta, int cardinality_ta>
class eField;

template <typename T_ta, int cardinality_ta>
class eFieldDevice_t;

template <typename T,
          int cardinality_ta = 1>
struct ePartition
{
    /**
     * Internals:
     * Local field memory layout:
     *
     * a. Fields data:
     *  |
     *  |   m_mem
     *  |   |
     *  |   V                      |-- DW --|-- UP ---|-- DW --|-- UP ---|
     *  |   V                      V        V         V        V         V
     *  |   |<--------------------><-----------------><----------------->|
     *  |             INTERNAL             BOUNDARY            GHOST
     *  |
     *  |   GHOST is allocated only for HALO type fields.
     *  |   SoA or AoS options are managed transparently by m_ePitch,
     *  |   which has 2 components, one for the pitch on the element position (eIdx)
     *  |   and the other on the element cardinality.
     *  |--)
     *
     *  b. DataView:
     *  |
     *  |   The data view option if handled only by the validateThreadIdx method.
     *  |   The eIdx if first computed according to the target runtime (opm or cuda)
     *  |   The value is equivalent to a 1D thread indexing (if no cardinality is used) or 2D
     *  |   The final value of eIdx is than shifted based on the data view parameter.
     *  |   The shift is done by the helper function hApplyDataViewShift
     *  |--)
     *
     *  c. Connectivity table
     *  |
     *  |   Connectivity table has the same layout of a field with cardinality equal to
     *  |   the number of neighbours and an SoA layout. Let's call this field nghField.
     *  |   nghField(e, nghIdx) is the eIdx_t of the neighbour element as in a STANDARD
     *  |   view.
     *  |--)
     */

   public:
    //-- [PUBLIC TYPES] ----------------------------------------------------------------------------
    using Self = ePartition<T, cardinality_ta>;   //<- this type
    using Cell = eCell;                           //<- type of an index to an element
    using OuterCell = typename eCell::OuterCell;  //<- type of an index to an element

    static constexpr int Cardinality = cardinality_ta;

    using nghIdx_t = uint8_t;                                    //<- type of an index to identify a neighbour
    using Type = T;                                              //<- type of the data stored by the field
    using eJump_t = index_t;                                     //<- Type of a jump value
    using ePitch_t = ::Neon::domain::internal::eGrid::ePitch_t;  //<- Type of the pitch representation

    template <typename T_,
              int Cardinality_>
    friend class eFieldDevice_t;

   private:
    //-- [INTERNAL DATA] ----------------------------------------------------------------------------
    T*       m_mem;
    int      m_cardinality;
    ePitch_t m_ePitch;

    Neon::DataView m_dataView;

    //-- [INDEXING] ----------------------------------------------------------------------------
    Cell::Offset m_bdrOff[ComDirection_e::COM_NUM] = {-1, -1};
    Cell::Offset m_ghostOff[ComDirection_e::COM_NUM] = {-1, -1};
    Cell::Offset m_bdrCount[ComDirection_e::COM_NUM] = {-1, -1};
    Cell::Offset m_ghostCount[ComDirection_e::COM_NUM] = {-1, -1};

    //-- [CONNECTIVITY] ----------------------------------------------------------------------------
    Cell::Offset* m_connRaw;
    ePitch_t      m_connPitch;

    //-- [INVERSE MAPPING] ----------------------------------------------------------------------------
    Neon::index_t* m_inverseMapping = {nullptr};
    int            m_prtID;

   public:
    //-- [CONSTRUCTORS] ----------------------------------------------------------------------------

    /**
     * Default constructor
     */
    ePartition() = default;

    /**
     * Default Destructor
     */
    ~ePartition() = default;

    /**
     * Partition id. This is an index inside a devSet
     * @return
     */
    NEON_CUDA_HOST_DEVICE auto
    prtID() const -> int;
    /**
     * Returns cardinality of the field.
     * For example a density field has cardinality one, velocity has cardinality three
     * @return
     */
    template <int dummy_ta = cardinality_ta>
    inline NEON_CUDA_HOST_DEVICE auto
    cardinality() const
        -> std::enable_if_t<dummy_ta == 0, int>;

    /**
     * Returns cardinality of the field.
     * For example a density field has cardinality one, velocity has cardinality three
     * @return
     */
    template <int dummy_ta = cardinality_ta>
    constexpr inline NEON_CUDA_HOST_DEVICE auto
    cardinality() const -> std::enable_if_t<dummy_ta != 0, int>;

    /**
     * Returns the value associated to element eId and cardinality cardinalityIdx
     * @param eId
     * @param cardinalityIdx
     * @return
     */
    NEON_CUDA_HOST_DEVICE inline auto
    operator()(Cell eId, int cardinalityIdx) const
        -> T;

    NEON_CUDA_HOST_DEVICE inline auto
    operator()(Cell eId, int cardinalityIdx)
        -> T&;

    template <typename ComputeType>
    NEON_CUDA_HOST_DEVICE inline auto
    castRead(Cell eId, int cardinalityIdx) const
        -> ComputeType;

    template <typename ComputeType>
    NEON_CUDA_HOST_DEVICE inline auto
    castWrite(Cell eId, int cardinalityIdx, const ComputeType& value)
        -> void;
    /**
     * Retrieve value of a neighbour for a field with multiple cardinalities
     * @tparam dataView_ta
     * @param eId
     * @param nghIdx
     * @param alternativeVal
     * @return
     */
    template <bool enableLDG = true, int shadowCardinality_ta = cardinality_ta>
    NEON_CUDA_HOST_DEVICE inline auto
    nghVal(Cell        eId,
           nghIdx_t    nghIdx,
           int         card,
           const Type& alternativeVal)
        const -> std::enable_if_t<shadowCardinality_ta != 1, NghInfo<Type>>;


    /**
     * Check is the
     * @tparam dataView_ta
     * @param eId
     * @param nghIdx
     * @param neighbourIdx
     * @return
     */
    NEON_CUDA_HOST_DEVICE inline auto
    nghIdx(Cell     eId,
           nghIdx_t nghIdx,
           Cell&    neighbourIdx) const
        -> bool;


    /**
     * Returns the pitch structure used by the grid.
     * @return
     */
    NEON_CUDA_HOST_DEVICE inline auto
    ePitch() const -> const ePitch_t&;

    /**
     * Convert grid local id to globals.
     * @return
     */
    NEON_CUDA_HOST_DEVICE inline auto
    globalLocation(Cell cell) const
        -> Neon::index_3d;

    NEON_CUDA_HOST_DEVICE inline auto
    mem() const
        -> const T*;

   private:
    /**
     * Private constructor only used by the grid.
     *
     * @param bdrOff
     * @param ghostOff
     * @param remoteBdrOff
     */
    explicit ePartition(const Neon::DataView&                                    dataView,
                        int                                                      prtId,
                        T*                                                       mem,
                        int                                                      cardinality,
                        const ePitch_t&                                          ePitch,
                        const std::array<Cell::Offset, ComDirection_e::COM_NUM>& bdrOff,
                        const std::array<Cell::Offset, ComDirection_e::COM_NUM>& ghostOff,
                        const std::array<Cell::Offset, ComDirection_e::COM_NUM>& bdrCount,
                        const std::array<Cell::Offset, ComDirection_e::COM_NUM>& ghostCount,
                        Cell::Offset*                                            connRaw,
                        const ePitch_t&                                          connPitch,
                        index_t*                                                 inverseMapping);

    /**
     * Returns a pointer to element eId with target cardinality cardinalityIdx
     * @tparam dataView_ta
     * @param eId
     * @param cardinalityIdx
     * @return
     */
    NEON_CUDA_HOST_DEVICE auto
    pointer(Cell eId, int cardinalityIdx) const
        -> const Type*;
    /**
     * Computes the jump of a element.
     * Jump is the offset between the head of the raw memory adders
     * and the position of the element defined by eId.
     *
     * Because we handle the data view model when setting the
     * iterator, this function is just an identity.
     *
     * @tparam dataView_ta
     * @param eId
     * @return
     */
    NEON_CUDA_HOST_DEVICE inline auto
    eJump(Cell eId)
        const
        -> eJump_t;

    /**
     * Computes the jump for an element
     *
     * @tparam dataView_ta
     * @param eId
     * @param cardinalityIdx
     * @return
     */
    NEON_CUDA_HOST_DEVICE inline auto
    eJump(Cell eId, int cardinalityIdx) const
        -> eJump_t;

    /**
     * Returns raw pointer of the field
     * @tparam dataView_ta
     * @return
     */
    NEON_CUDA_HOST_DEVICE inline auto
    mem()
        -> T*;
};
}  // namespace Neon::domain::internal::eGrid

#include "Neon/domain/internal/eGrid/ePartition_imp.h"
