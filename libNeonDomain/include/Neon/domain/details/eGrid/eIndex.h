#pragma once

#include "Neon/core/core.h"

namespace Neon::domain::details::eGrid {

// Common forward declarations
class eGrid;
class eSpan;
template <typename T, int C>
class ePartition;

class eIndex
{
   public:
    using OuterIndex = eIndex;

    friend class eSpan;
    friend class eGrid;
    template <typename T,
              int Cardinality>
    friend class ePartition;

    template <typename T,
              int Cardinality>
    friend class eFieldDevice_t;

    // eGrid specific types
    using Offset = int32_t;
    using InternalIdx = int32_t;
    using Count = int32_t;
    using ePitch = Neon::Integer_2d<Offset>;

    NEON_CUDA_HOST_DEVICE inline auto
    manualSet(InternalIdx idx) -> void;

    eIndex() = default;

    NEON_CUDA_HOST_DEVICE inline auto
    helpGet() const -> const InternalIdx&;

   private:
    NEON_CUDA_HOST_DEVICE inline explicit eIndex(const InternalIdx& Idx);

    NEON_CUDA_HOST_DEVICE inline auto
    helpSet() -> InternalIdx&;

    InternalIdx mIdx = 0;
};

enum class ComDirection : uint32_t
{
    UP = 0,
    DW = 1,
    NUM = 2
};

class ComDirectionUtils
{
   public:
    static constexpr auto toInt(ComDirection direction) -> uint32_t
    {
        return static_cast<uint32_t>(direction);
    }
};

}  // namespace Neon::domain::details::eGrid

#include "eIndex_imp.h"
