#pragma once

#include "Neon/core/core.h"
#if defined(NEON_WARP_COMPILATION)
namespace std{
// Generic (non-specialized) template for numeric_limits.
template <typename T>
class numeric_limits {
public:
    static constexpr bool is_specialized = false;

    // Default implementations simply return a default-constructed T.
    static constexpr T min() noexcept { return T(); }
    static constexpr T max() noexcept { return T(); }
    static constexpr T epsilon() noexcept { return T(); }
};

// Specialized version for float using literal constants.
template <>
class numeric_limits<float> {
public:
    static constexpr bool is_specialized = true;

    // Smallest positive normalized float (approximately)
    static constexpr float min() noexcept { return 1.17549435e-38F; }
    // Largest finite float value (approximately)
    static constexpr float max() noexcept { return 3.402823466e+38F; }
    // Difference between 1 and the next representable float
    static constexpr float epsilon() noexcept { return 1.19209290e-7F; }
};

// Specialized version for double using literal constants.
template <>
class numeric_limits<double> {
public:
    static constexpr bool is_specialized = true;

    // Smallest positive normalized double (approximately)
    static constexpr double min() noexcept { return 2.2250738585072014e-308; }
    // Largest finite double value (approximately)
    static constexpr double max() noexcept { return 1.7976931348623158e+308; }
    // Difference between 1 and the next representable double
    static constexpr double epsilon() noexcept { return 2.2204460492503131e-16; }
};

// Specialized version for int using literal constants.
template <>
class numeric_limits<int> {
public:
    static constexpr bool is_specialized = true;

    // Minimum and maximum for a 32-bit signed integer.
    static constexpr int min() noexcept { return -2147483648; }
    static constexpr int max() noexcept { return 2147483647; }
    // For integral types, epsilon is not applicable.
    static constexpr int epsilon() noexcept { return 0; }
};

// Specialized version for int64_t using literal constants.
template <>
class numeric_limits<int64_t> {
public:
    static constexpr bool is_specialized = true;

    // Minimum and maximum for a 64-bit signed integer.
    static constexpr int64_t min() noexcept { return -9223372036854775807LL - 1; }
    static constexpr int64_t max() noexcept { return 9223372036854775807LL; }
    // For integral types, epsilon is not applicable.
    static constexpr int64_t epsilon() noexcept { return 0; }
};
}
#endif

namespace Neon::domain::details::bGrid {

// Common forward declarations
template <typename SBlock>
class bGrid;
template <typename SBlock>
class bSpan;
template <typename T, int C, typename SBlock>
class bPartition;

class MicroIndex
{
   public:
    using TrayIdx = int32_t;
    using InTrayIdx = int8_3d;

    NEON_CUDA_HOST_DEVICE inline explicit MicroIndex()
        : MicroIndex(0, 0, 0, 0)
    {
    }

    NEON_CUDA_HOST_DEVICE inline explicit MicroIndex(const TrayIdx&            blockIdx,
                                                     const InTrayIdx::Integer& x,
                                                     const InTrayIdx::Integer& y,
                                                     const InTrayIdx::Integer& z)
    {
        mTrayBlockIdx = blockIdx;
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

template <typename SBlock>
class bIndex
{
   public:
    template <typename SBlock_>
    friend class bSpan;
    using OuterIdx = bIndex<SBlock>;

    using NghIdx = int8_3d;
    template <typename T, int C, typename SBlock_>
    friend class bPartition;

    template <typename T, int C, typename SBlock_>
    friend class bField;

    template <typename SBlock_>
    friend class bSpan;
    template <typename SBlock_>
    friend class bGrid;


    using TrayIdx = MicroIndex::TrayIdx;
    using InTrayIdx = MicroIndex::InTrayIdx;
#if !defined(NEON_WARP_COMPILATION)

    using DataBlockCount = std::make_unsigned_t<TrayIdx>;
    using DataBlockIdx = std::make_unsigned_t<TrayIdx>;
#else
    using DataBlockCount = TrayIdx;
    using DataBlockIdx = TrayIdx;
#endif

    using InDataBlockIdx = InTrayIdx;

    bIndex() = default;
    ~bIndex() = default;

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
    NEON_CUDA_HOST_DEVICE inline auto isActive() const -> bool;
    // the local index within the block
    InDataBlockIdx mInDataBlockIdx;
    DataBlockIdx   mDataBlockIdx{};
};

template <typename SBlock>
NEON_CUDA_HOST_DEVICE auto bIndex<SBlock>::setDataBlockIdx(const bIndex::DataBlockIdx& dataBlockIdx) -> void
{
    mDataBlockIdx = dataBlockIdx;
}

template <typename SBlock>
NEON_CUDA_HOST_DEVICE auto bIndex<SBlock>::setInDataBlockIdx(const bIndex::InDataBlockIdx& inDataBlockIdx) -> void
{
    mInDataBlockIdx = inDataBlockIdx;
}

template <typename SBlock>
NEON_CUDA_HOST_DEVICE auto bIndex<SBlock>::getDataBlockIdx() const -> const bIndex::DataBlockIdx&
{
    return mDataBlockIdx;
}
template <typename SBlock>
NEON_CUDA_HOST_DEVICE auto bIndex<SBlock>::getInDataBlockIdx() const -> const bIndex::InDataBlockIdx&
{
    return mInDataBlockIdx;
}

template <typename SBlock>
NEON_CUDA_HOST_DEVICE auto bIndex<SBlock>::isActive() const -> bool
{
    return mDataBlockIdx != std::numeric_limits<typename bIndex::DataBlockIdx>::max();
}

}  // namespace Neon::domain::details::bGrid

#include "Neon/domain/details/StaticBlock.h"
namespace Neon::domain::details::bGrid {
constexpr int defaultBlockSize = ::Neon::domain::details::StaticBlockSizeDefault;
using BlockDefault = ::Neon::domain::details::StaticBlockDefault;
}

#include "Neon/domain/details/bGrid/bIndex_imp.h"
