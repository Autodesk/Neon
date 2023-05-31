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
    using InTrayIdx = uint8_3d;

    NEON_CUDA_HOST_DEVICE inline explicit MicroIndex():MicroIndex(0,0,0,0)
    {
    }

    NEON_CUDA_HOST_DEVICE inline explicit MicroIndex(const TrayIdx&            blockIdx,
                                                     const InTrayIdx::Integer& x,
                                                     const InTrayIdx::Integer& y,
                                                     const InTrayIdx::Integer& z)
    {
        mInTrayBlockIdx = blockIdx;
        mInTrayBlockIdx.x = x;
        mInTrayBlockIdx.y = y;
        mInTrayBlockIdx.z = z;
    }

    NEON_CUDA_HOST_DEVICE inline auto getInTrayBlockIdx() const -> InTrayIdx const&
    {
        return mInTrayBlockIdx;
    }

    NEON_CUDA_HOST_DEVICE inline auto getTrayBlockIdx() const -> TrayIdx const&
    {
        return mTrayBlockIdx;
    }

    NEON_CUDA_HOST_DEVICE inline auto setInTrayBlockIdx(InTrayIdx const& inTrayIdx) -> void
    {
        mInTrayBlockIdx = inTrayIdx;
    }

    NEON_CUDA_HOST_DEVICE inline auto setTrayBlockIdx(TrayIdx const& trayIdx) -> void
    {
        mTrayBlockIdx = trayIdx;
    }

    InTrayIdx mInTrayBlockIdx;
    TrayIdx   mTrayBlockIdx{};
};

template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
class bIndex
{
   public:
    template <uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>
    friend class bSpan;
    using OuterIdx = bIndex<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>;

    static constexpr Neon::uint32_3d memBlock3DSize{memBlockSizeX,memBlockSizeY,memBlockSizeZ};

    using NghIdx = int8_3d;
    template <typename T, int C, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>
    friend class bPartition;

    template <typename T, int C, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>
    friend class bField;

    template <uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>
    friend class bSpan;
    template <uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>
    friend class bGrid;


    using TrayIdx = MicroIndex::TrayIdx;
    using InTrayIdx = MicroIndex::InTrayIdx;

    using DataBlockCount = TrayIdx;
    using DataBlockIdx = TrayIdx;
    using InDataBlockIdx = InTrayIdx;

    bIndex() = default;
    virtual ~bIndex() = default;

    NEON_CUDA_HOST_DEVICE inline explicit bIndex(const DataBlockIdx&            blockIdx,
                                                 const InDataBlockIdx::Integer& x,
                                                 const InDataBlockIdx::Integer& y,
                                                 const InDataBlockIdx::Integer& z);

    NEON_CUDA_HOST_DEVICE inline auto getMicroIndex() -> MicroIndex;
    NEON_CUDA_HOST_DEVICE inline auto init(MicroIndex const&) -> void;

    NEON_CUDA_HOST_DEVICE inline auto getInDataBlockIdx() const -> InDataBlockIdx const&;
    NEON_CUDA_HOST_DEVICE inline auto getDataBlockIdx() const -> DataBlockIdx const&;
    NEON_CUDA_HOST_DEVICE inline auto setInDataBlockIdx(InDataBlockIdx const&) -> void;
    NEON_CUDA_HOST_DEVICE inline auto setDataBlockIdx(DataBlockIdx const&) -> void;
    // the local index within the block
    InDataBlockIdx mInDataBlockIdx;
    DataBlockIdx   mDataBlockIdx{};
};

template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
NEON_CUDA_HOST_DEVICE auto bIndex<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::setDataBlockIdx(const bIndex::DataBlockIdx& dataBlockIdx) -> void
{
    mDataBlockIdx = dataBlockIdx;
}

template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
NEON_CUDA_HOST_DEVICE auto bIndex<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::setInDataBlockIdx(const bIndex::InDataBlockIdx& inDataBlockIdx) -> void
{
    mInDataBlockIdx = inDataBlockIdx;
}

template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
NEON_CUDA_HOST_DEVICE auto bIndex<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::getDataBlockIdx() const -> const bIndex::DataBlockIdx&
{
    return mDataBlockIdx;
}
template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
NEON_CUDA_HOST_DEVICE auto bIndex<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::getInDataBlockIdx() const -> const bIndex::InDataBlockIdx&
{
    return mInDataBlockIdx;
}


}  // namespace Neon::domain::details::bGrid

#include "Neon/domain/details/bGrid/bIndex_imp.h"
