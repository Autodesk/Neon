#pragma once

#include "Neon/set/Backend.h"
#include "Neon/set/memory/memory.h"
namespace Neon {
namespace set {

/**
 * API to allocate any type of memory
 */
struct Memory
{
    template <typename T_ta>
    static auto MemSet(const Neon::Backend&                bk,
                       int                                 cardinality,
                       const Neon::set::DataSet<uint64_t>& nElementVec,
                       Neon::DataUse                       dataUse,
                       Neon::MemSetOptions_t               cpuConfig = Neon::MemSetOptions_t(),
                       Neon::MemSetOptions_t               gpuConfig = Neon::MemSetOptions_t()) -> MemSet<T_ta>;

    template <typename T_ta>
    static auto MemSet(const Neon::Backend&  bk,
                       int                   cardinality,
                       const uint64_t&       nElementInEachPartition,
                       Neon::DataUse         dataUse,
                       Neon::MemSetOptions_t cpuConfig = Neon::MemSetOptions_t(),
                       Neon::MemSetOptions_t gpuConfig = Neon::MemSetOptions_t()) -> MemSet<T_ta>;
};

}  // namespace set
}  // namespace Neon

#include "Neon/set/memory/memory.ti.h"
