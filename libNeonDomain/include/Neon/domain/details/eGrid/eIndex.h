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
    using OuterCell = eIndex;

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
    using Idx = int32_t;
    using Count = int32_t;
    using ePitch = Neon::index64_2d;


    eIndex() = default;

   private:
    Idx mIdx = 0;

    NEON_CUDA_HOST_DEVICE inline explicit eIndex(Idx Idx);

    NEON_CUDA_HOST_DEVICE inline auto set() -> Idx&;

    NEON_CUDA_HOST_DEVICE inline auto get() -> const Idx&;
};

enum class ComDirection : uint8_t
{
    UP = 0,
    DW = 1,
    NUM = 2
};

}  // namespace Neon::domain::details::eGrid

#include "eIndex_imp.h"
