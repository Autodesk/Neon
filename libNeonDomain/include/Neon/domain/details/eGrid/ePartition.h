#pragma once
#include <assert.h>
#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"
#include "Neon/domain/details/eGrid/eIndex.h"
#include "Neon/domain/interface/NghData.h"
#include "Neon/set/DevSet.h"
#include "Neon/sys/memory/CudaIntrinsics.h"
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

    NEON_CUDA_HOST_DEVICE auto
    getNghIndex(eIndex eId, const int8_3d& ngh3dIdx, eIndex& eIdxNgh)
        const -> bool;

    NEON_CUDA_HOST_DEVICE inline auto
    mem()
        -> T*;

   public:
    //-- [PUBLIC TYPES] ----------------------------------------------------------------------------
    using Self = ePartition<T, C>;            //<- this type
    using Idx = eIndex;                       //<- index type
    using OuterIdx = typename Idx::OuterIdx;  //<- index type for the subGrid

    static constexpr int Cardinality = C;

    using NghIdx = uint8_t;  //<- type of an index to identify a neighbour
    using Ngh3DIdx = Neon::int8_3d;
    using Ngh1DIdx = uint8_t;
    using NghData = Neon::domain::NghData<T>;


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

    //    template <typename ComputeType>
    //    NEON_CUDA_HOST_DEVICE inline auto
    //    castRead(Idx eId, int cardinalityIdx) const
    //        -> ComputeType;
    //
    //    template <typename ComputeType>
    //    NEON_CUDA_HOST_DEVICE inline auto
    //    castWrite(Idx eId, int cardinalityIdx, const ComputeType& value)
    //        -> void;
    /**
     * Retrieve value of a neighbour for a field with multiple cardinalities
     * @tparam dataView_ta
     * @param eId
     * @param nghIdx
     * @param alternativeVal
     * @return
     */
    NEON_CUDA_HOST_DEVICE inline auto
    getNghData(Idx    eId,
               NghIdx nghIdx,
               int    card)
        const -> NghData;

    NEON_CUDA_HOST_DEVICE inline auto
    getNghData(eIndex               eId,
               const Neon::int8_3d& nghIdx,
               int                  card)
        const -> NghData;

    template <int xOff, int yOff, int zOff>
    NEON_CUDA_HOST_DEVICE inline auto
    getNghData(Idx eId,
               int card)
        const -> NghData;

    template <int xOff, int yOff, int zOff>
    NEON_CUDA_HOST_DEVICE inline auto
    getNghData(Idx eId,
               int card,
               T defaultValue)
        const -> NghData;
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
                        ePitch          pitch,
                        int32_t         cardinality,
                        int32_t         countAllocated,
                        Offset*         connRaw,
                        Neon::index_3d* toGlobal,
                        int8_t*         stencil3dTo1dOffset,
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

   protected:
    //-- [INTERNAL DATA] ----------------------------------------------------------------------------
    T*      mMem;
    int     mCardinality;
    int32_t mCountAllocated;
    ePitch  mPitch;

    //-- [CONNECTIVITY] ----------------------------------------------------------------------------
    Offset* mConnectivity = {nullptr} /** connectivity table */;

    //-- [INVERSE MAPPING] ----------------------------------------------------------------------------
    Neon::int32_3d* mOrigins = {nullptr};
    int             mPrtID;
    int8_t*         mStencil3dTo1dOffset = {nullptr};
    int32_t         mStencilTableYPitch;
    int32_t         mStencilRadius;  // Shift to be applied to all 3d offset component to access mStencil3dTo1dOffset table
};
}  // namespace Neon::domain::details::eGrid

#include "Neon/domain/details/eGrid/ePartition_imp.h"