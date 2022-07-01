#pragma once

#include <string>

#include "Neon/core/types/Allocator.h"
#include "Neon/core/types/DataUse.h"
#include "Neon/core/types/memOptions.h"

namespace Neon {

/**
 * Type of memory layout for either for scalar or vector types
 */
struct [[deprecated]] MemSetOptions_t
{
   private:
    MemoryLayout      m_memOrder = MemoryLayout::structOfArrays;
    memPadding_e::e   m_memPadding = memPadding_e::OFF;
    memAlignment_e::e m_memAlignment = memAlignment_e::SYSTEM;

    Neon::Allocator m_allocatorByDev[Neon::DeviceTypeUtil::nConfig];

   public:
    MemSetOptions_t();

   public:
    /**
     * Set or get memory order option
     * @return
     */

    auto order() -> MemoryLayout&;

    /**
     * Set or get memory padding option
     * @return
     */
    auto padding() -> memPadding_e::e&;

    /**
     * Set or get memory alignment option
     * @return
     */
    auto alignment() -> memAlignment_e::e&;

    /**
     * Set or get memory allocator option
     * @return
     */
    auto allocator(Neon::DeviceType devType) -> Neon::Allocator&;

    /**
     * Set or get memory order option
     * @return
     */
    auto order() const -> const MemoryLayout&;

    /**
     * Set or get memory padding option
     * @return
     */
    auto padding() const -> const memPadding_e::e&;
    /**
     * Set or get memory alignment option
     * @return
     */
    auto alignment() const -> const memAlignment_e::e&;

    /**
     * Set or get memory allocator option
     * @return
     */
    auto allocator(Neon::DeviceType devType) const -> const Neon::Allocator&;

    /**
     * Return a string with all the information related to the memory configuration
     * @return
     */
    auto toString() const -> std::string;
};
}  // namespace Neon
