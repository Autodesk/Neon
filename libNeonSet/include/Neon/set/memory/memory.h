#pragma once

#include "Neon/set/Backend.h"
#include "Neon/set/memory/memSet.h"

namespace Neon::set {

template <typename T_ta>
class MemSet;

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

}  // namespace Neon::set

#include "Neon/set/memory/memory.ti.h"
