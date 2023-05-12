#pragma once

#include "Neon/core/core.h"


namespace Neon::domain::details::bGrid {

// Common forward declarations
template <int8_t dataBlockSizeX, int8_t dataBlockSizeY, int8_t dataBlockSizeZ>
class bGrid;
template <int8_t dataBlockSizeX, int8_t dataBlockSizeY, int8_t dataBlockSizeZ>
class bSpan;
template <typename T, int C, int8_t dataBlockSizeX, int8_t dataBlockSizeY, int8_t dataBlockSizeZ>
class bPartition;

template <int8_t dataBlockSizeX, int8_t dataBlockSizeY, int8_t dataBlockSizeZ>
class bIndex
{
   public:
    template <int8_t, int8_t, int8_t>
    friend class bSpan;
    using OuterIdx = bIndex<dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ>;

    using NghIdx = int8_3d;
    template <typename T, int C, int8_t, int8_t, int8_t>
    friend class bPartition;

    template <typename T, int C, int8_t, int8_t, int8_t>
    friend class bField;

    template <int8_t, int8_t, int8_t>
    friend class bSpan;
    template <int8_t, int8_t, int8_t>
    friend class bGrid;

    using DataBlockCount = uint32_t;
    using DataBlockIdx = uint32_t;
    using InDataBlockIdx = int8_3d;


    bIndex() = default;
    virtual ~bIndex() = default;

    NEON_CUDA_HOST_DEVICE inline explicit bIndex(const DataBlockIdx&            blockIdx,
                                                 const InDataBlockIdx::Integer& x,
                                                 const InDataBlockIdx::Integer& y,
                                                 const InDataBlockIdx::Integer& z);


    // the local index within the block
    InDataBlockIdx mInDataBlockIdx;
    DataBlockIdx   mDataBlockIdx{};
};
}  // namespace Neon::domain::details::bGrid

#include "Neon/domain/details/bGrid/bIndex_imp.h"
