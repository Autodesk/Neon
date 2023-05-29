#pragma once

#include "Neon/core/core.h"


namespace Neon::domain::details::bGrid {

// Common forward declarations
template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
class bGrid;
template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
class bSpan;
template <typename T, int C, uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
class bPartition;

class MicroIndex
{
   public:
    using TrayIdx = uint32_t;
    using InTrayIdx = int8_3d;

    InTrayIdx mInDataBlockIdx;
    TrayIdx   mDataBlockIdx{};
};

template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
class bIndex
{
   public:
    template <uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>
    friend class bSpan;
    using OuterIdx = bIndex<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>;

    using NghIdx = int8_3d;
    template <typename T, int C, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>
    friend class bPartition;

    template <typename T, int C, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>
    friend class bField;

    template <uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>
    friend class bSpan;
    template <uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>
    friend class bGrid;

    using DataBlockCount = uint32_t;
    using DataBlockIdx = uint32_t;
    using InDataBlockIdx = int8_3d;
    using TrayIdx = uint32_t;
    using InTrayIdx = int8_3d;


    bIndex() = default;
    virtual ~bIndex() = default;

    NEON_CUDA_HOST_DEVICE inline explicit bIndex(const DataBlockIdx&            blockIdx,
                                                 const InDataBlockIdx::Integer& x,
                                                 const InDataBlockIdx::Integer& y,
                                                 const InDataBlockIdx::Integer& z);

    NEON_CUDA_HOST_DEVICE inline auto getTrayIdx() -> TrayIdx;
    NEON_CUDA_HOST_DEVICE inline auto getInTrayIdx() -> InTrayIdx;

    NEON_CUDA_HOST_DEVICE inline auto getMicroIndex() -> MicroIndex;
    NEON_CUDA_HOST_DEVICE inline auto init(MicroIndex const &) -> void ;


    // the local index within the block
    InDataBlockIdx mInDataBlockIdx;
    DataBlockIdx   mDataBlockIdx{};
};
}  // namespace Neon::domain::details::bGrid

#include "Neon/domain/details/bGrid/bIndex_imp.h"
