#pragma once

#include "Neon/core/core.h"


namespace Neon::domain::details::bGrid {

// Common forward declarations
template <int8_t dataBlockSizeX, int8_t dataBlockSizeY, int8_t dataBlockSizeZ, int8_t userBlockSizeX, int8_t userBlockSizeY, int8_t userBlockSizeZ>
class bGrid;
template <int8_t dataBlockSizeX, int8_t dataBlockSizeY, int8_t dataBlockSizeZ, int8_t userBlockSizeX, int8_t userBlockSizeY, int8_t userBlockSizeZ>
class bSpan;
template <typename T, int C, int8_t dataBlockSizeX, int8_t dataBlockSizeY, int8_t dataBlockSizeZ, int8_t userBlockSizeX, int8_t userBlockSizeY, int8_t userBlockSizeZ>
class bPartition;

template <int8_t dataBlockSizeX, int8_t dataBlockSizeY, int8_t dataBlockSizeZ, int8_t userBlockSizeX, int8_t userBlockSizeY, int8_t userBlockSizeZ>
class Operations
{
    constexpr static uint32_t userBSX_UINT = static_cast<uint32_t>(userBlockSizeX);
    constexpr static uint32_t userBSY_UINT = static_cast<uint32_t>(userBlockSizeY);
    constexpr static uint32_t userBSZ_UINT = static_cast<uint32_t>(userBlockSizeZ);

    constexpr static uint32_t userBSX_POPCOUNT = std::popcount(userBSX_UINT);
    constexpr static uint32_t userBSY_POPCOUNT = std::popcount(userBSY_UINT);
    constexpr static uint32_t userBSZ_POPCOUNT = std::popcount(userBSZ_UINT);

    template <uint32_t userBS>
    static auto moduleOp(uint32_t x) -> uint32_t
    {
        static_assert(userBS == userBSX_UINT || userBS == userBSY_UINT || userBS == userBSZ_UINT);
        if constexpr (userBSX_POPCOUNT == 1 &&
                      userBSY_POPCOUNT == 1 &&
                      userBSZ_POPCOUNT == 1) {
            return x & (userBS);
        } else {
            return x % userBS;
        }
    }

    template <uint32_t userBS>
    static auto divisionOp(uint32_t x) -> uint32_t
    {
        static_assert(userBS == userBSX_UINT || userBS == userBSY_UINT || userBS == userBSZ_UINT);
        if constexpr (userBSX_POPCOUNT == 1 &&
                      userBSY_POPCOUNT == 1 &&
                      userBSZ_POPCOUNT == 1) {
            constexpr auto log2Fun = [](uint32_t i) -> uint32_t { return 8 * sizeof(i) - std::countl_zero(i) - 1; };
            constexpr uint32_t log2= log2Fun(userBS);
            return x >> (log2);
        } else {
            return x / userBS;
        }
    }
};

class MicroIndex
{
   public:
    using TrayIdx = uint32_t;
    using InTrayIdx = int8_3d;

    InTrayIdx mInDataBlockIdx;
    TrayIdx   mDataBlockIdx{};
};

template <int8_t dataBlockSizeX, int8_t dataBlockSizeY, int8_t dataBlockSizeZ, int8_t userBlockSizeX, int8_t userBlockSizeY, int8_t userBlockSizeZ>
class bIndex
{
   public:
    template <int8_t, int8_t, int8_t, int8_t, int8_t, int8_t>
    friend class bSpan;
    using OuterIdx = bIndex<dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>;

    using NghIdx = int8_3d;
    template <typename T, int C, int8_t, int8_t, int8_t, int8_t, int8_t, int8_t>
    friend class bPartition;

    template <typename T, int C, int8_t, int8_t, int8_t, int8_t, int8_t, int8_t>
    friend class bField;

    template <int8_t, int8_t, int8_t, int8_t, int8_t, int8_t>
    friend class bSpan;
    template <int8_t, int8_t, int8_t, int8_t, int8_t, int8_t>
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
