#pragma once

#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"
#include "Neon/set/DevSet.h"
#include "Neon/set/memory/memSet.h"
#include "Neon/sys/memory/CudaIntrinsics.h"

#include "Neon/domain/details/aGrid/aIndex.h"

namespace Neon::domain::details::aGrid {

/**
 * Partition abstraction for the aGrid.
 *
 * Internally aPartition manages a simple 1D array.
 */
template <typename T /**< Type of the element of the partition */,
          int C = 0 /** Cardinality of the field. If zero, the cardinality is determined at runtime */>
class aPartition
{
   public:
    using Type = T;
    using Cell = Neon::domain::details::aGrid::aIndex;

    using count_t = int32_t;
    using index_t = int32_t;
    using eJump_t = typename Cell::Location;
    using pitch_t = Neon::index64_2d;
    using prt_idx = int;


   public:
    aPartition() = default;

    ~aPartition() = default;

    explicit aPartition(const prt_idx& prtId /**< Partition Index */,
                        Type*          mem /**< Raw memory pointer */,
                        const pitch_t& pitch /**< Memory pitch */,
                        const count_t& nElements /**< Number of elements */,
                        const int      cardinality /**< Cardinality of the field */);

    /**
     * Returns the partition index.
     */
    NEON_CUDA_HOST_DEVICE auto prtID() const
        -> const int&;

    /**
     * Return the offset of an element from the raw memory (mem() method)
     */
    NEON_CUDA_HOST_DEVICE inline auto eJump(const Cell& eId)
        const
        -> eJump_t;

    /**
     * Return the offset of an element from the raw memory (mem() method)
     */
    NEON_CUDA_HOST_DEVICE inline auto eJump(const Cell& eId,
                                            const int&  cardinalityIdx) const
        -> eJump_t;

    /**
     * Returns the raw memory pointer used by the data structure
     */
    NEON_CUDA_HOST_DEVICE inline auto mem()
        -> T*;

    /**
     * Returns the raw memory pointer used by the data structure
     */
    NEON_CUDA_HOST_DEVICE inline auto mem() const
        -> const T*;

    /**
     * Returns the metadata associated with cell referred by eIdx.
     * This method should be used only for fields of cardinality 1
     */
    NEON_CUDA_HOST_DEVICE inline auto operator()(Cell eId, int cardinalityIdx) const
        -> const T&;

    /**
     * Returns the metadata associated with cell referred by eIdx.
     * This method should be used only for fields of cardinality 1
     */
    NEON_CUDA_HOST_DEVICE inline auto operator()(Cell eId, int cardinalityIdx)
        -> T&;

    /**
     * Returns the number of cells stored by the partition
     */
    NEON_CUDA_HOST_DEVICE inline auto nElements()
        const
        -> count_t;

    /**
     * Returns pitch information
     */
    NEON_CUDA_HOST_DEVICE inline auto pitch() const
        -> const pitch_t&;

    /**
     * Returns the field cardinality
     */
    NEON_CUDA_HOST_DEVICE inline auto cardinality() const
        -> int;

   private:
    Type*   m_mem /**< Raw memory */;
    pitch_t m_pitch /**< Pitch structure that consider the data layout when computing a cell metadata offset. */;
    index_t m_nElements /**< Number of element for this partition */;
    prt_idx m_prtID /**< Partition index */;
    int     m_cardinality /**< Cardinality of the field.*/;
};
}  // namespace Neon::domain::array

#include "aPartition_imp.h"