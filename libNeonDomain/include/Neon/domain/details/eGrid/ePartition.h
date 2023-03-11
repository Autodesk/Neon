#pragma once
#include <assert.h>
#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"
#include "Neon/domain/details/eGrid/eIndex.h"
#include "Neon/domain/interface/NghData.h"
#include "Neon/set/DevSet.h"
#include "Neon/sys/memory/CudaIntrinsics.h"
#include "Neon/sys/memory/mem3d.h"
#include "cuda_fp16.h"

namespace Neon::domain::details::eGrid {

/**
 * Local representation for the sparse eGrid.
 *
 * @tparam T: type of data stored by this field
 * @tparam C: cardinality of this filed.
 *                         If C is 0, the actual cardinality is manged dynamically
 */


template <typename T, int C>
class eField;

template <typename T,
          int C = 1>
class ePartition
{
   public:
    /**
     * Internals:
     * Local field memory layout:
     *
     * a. Fields data:
     *  |
     *  |   mMem
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
    using Self = ePartition<T, C>;            //<- this type
    using Idx = eIndex;                       //<- index type
    using OuterIdx = typename Idx::OuterIdx;  //<- index type for the subGrid

    static constexpr int Cardinality = C;

    using NghIdx = uint8_t;         //<- type of an index to identify a neighbour
    using Type = T;                 //<- type of the data stored by the field
    using Offset = eIndex::Offset;  //<- Type of a jump value
    using ePitch = eIndex::ePitch;  //<- Type of the pitch representation
    using Count = eIndex::Count;

    template <typename T_,
              int Cardinality_>
    friend class eField;

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
    template <int CardinalitySFINE = C>
    inline NEON_CUDA_HOST_DEVICE auto
    cardinality() const
        -> std::enable_if_t<CardinalitySFINE == 0, int>;

    /**
     * Returns cardinality of the field.
     * For example a density field has cardinality one, velocity has cardinality three
     * @return
     */
    template <int CardinalitySFINE = C>
    constexpr inline NEON_CUDA_HOST_DEVICE auto
    cardinality() const -> std::enable_if_t<CardinalitySFINE != 0, int>;

    /**
     * Returns the value associated to element eId and cardinality cardinalityIdx
     * @param eId
     * @param cardinalityIdx
     * @return
     */
    NEON_CUDA_HOST_DEVICE inline auto
    operator()(Idx eId, int cardinalityIdx) const
        -> T;

    NEON_CUDA_HOST_DEVICE inline auto
    operator()(Idx eId, int cardinalityIdx)
        -> T&;

    template <typename ComputeType>
    NEON_CUDA_HOST_DEVICE inline auto
    castRead(Idx eId, int cardinalityIdx) const
        -> ComputeType;

    template <typename ComputeType>
    NEON_CUDA_HOST_DEVICE inline auto
    castWrite(Idx eId, int cardinalityIdx, const ComputeType& value)
        -> void;
    /**
     * Retrieve value of a neighbour for a field with multiple cardinalities
     * @tparam dataView_ta
     * @param eId
     * @param nghIdx
     * @param alternativeVal
     * @return
     */
    NEON_CUDA_HOST_DEVICE inline auto
    nghVal(Idx         eId,
           NghIdx      nghIdx,
           int         card,
           const Type& alternativeVal)
        const -> NghData<Type>;

    NEON_CUDA_HOST_DEVICE inline auto
    nghVal(eIndex               eId,
           const Neon::int8_3d& nghIdx,
           int                  card,
           const Type&          alternativeVal)
        const -> NghData<Type>;


    /**
     * Check is the
     * @tparam dataView_ta
     * @param eId
     * @param nghIdx
     * @param neighbourIdx
     * @return
     */
    NEON_CUDA_HOST_DEVICE inline auto
    isValidNgh(Idx    eId,
               NghIdx nghIdx,
               Idx&   neighbourIdx) const
        -> bool;


    /**
     * Returns the pitch structure used by the grid.
     * @return
     */
    NEON_CUDA_HOST_DEVICE inline auto
    getPitch() const -> const ePitch&;

    /**
     * Convert grid local id to globals.
     * @return
     */
    NEON_CUDA_HOST_DEVICE inline auto
    getGlobalIndex(Idx Idx) const
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
    explicit ePartition(int             prtId,
                        T*              mem,
                        int32_t         cardinality,
                        int32_t         countAllocated,
                        Offset*         connRaw,
                        Neon::index_3d* toGlobal,
                        int32_t*        stencil3dTo1dOffset,
                        int32_t         stencilRadius);

    /**
     * Returns a pointer to element eId with target cardinality cardinalityIdx
     * @tparam dataView_ta
     * @param eId
     * @param cardinalityIdx
     * @return
     */
    NEON_CUDA_HOST_DEVICE auto
    pointer(Idx eId, int cardinalityIdx) const
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
    getOffset(Idx eId)
        const
        -> Offset;

    /**
     * Computes the jump for an element
     *
     * @tparam dataView_ta
     * @param eId
     * @param cardinalityIdx
     * @return
     */
    NEON_CUDA_HOST_DEVICE inline auto
    getOffset(Idx eId, int cardinalityIdx) const
        -> Offset;

    /**
     * Returns raw pointer of the field
     * @tparam dataView_ta
     * @return
     */
    NEON_CUDA_HOST_DEVICE inline auto
    mem()
        -> T*;

   private:
    //-- [INTERNAL DATA] ----------------------------------------------------------------------------
    T*      mMem;
    int     mCardinality;
    int32_t mCountAllocated;

    //-- [CONNECTIVITY] ----------------------------------------------------------------------------
    Offset* mConnectivity = {nullptr} /** connectivity table */;

    //-- [INVERSE MAPPING] ----------------------------------------------------------------------------
    Neon::int32_3d* mOrigins = {nullptr};
    int             mPrtID;
    int32_t*        mStencil3dTo1dOffset = {nullptr};
    int32_t         mStencilTableYPitch;
};
}  // namespace Neon::domain::details::eGrid

#include "Neon/domain/details/eGrid/ePartition_imp.h"