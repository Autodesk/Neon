#pragma once

#include <vector>
#include "Neon/sys/devices/cpu/CpuSys.h"
#include "Neon/sys/devices/gpu/GpuSys.h"
#include "Neon/sys/memory/memAlignment.et.h"
#include "Neon//core/types/memSetOptions.h"


namespace Neon {
namespace sys {

struct MemAlignment
{
   private:
    memAlignment_et m_mode{memAlignment_et::system};
    uint32_t        m_exponent{0};

   public:
    ~MemAlignment() = default;
    MemAlignment() = default;

    explicit MemAlignment(memAlignment_et mode_,
                            uint32_t        exponent_);

    explicit MemAlignment(memAlignment_e::e align,
                            uint32_t          exponent =0);


    explicit MemAlignment(memAlignment_et mode_);


    const memAlignment_et& mode() const;

    uint32_t exponent() const;


    uint32_t expAlign(const Neon::sys::GpuMem& gpuMem) const;
    uint32_t expAlign() const;
    uint32_t expAlign(const DeviceType& devType,
                      const DeviceID& devId) const;

    static uint32_t exp2byte(uint32_t exp);

    bool operator==(const MemAlignment& other)
    {
        bool equal = true;
        equal = (m_mode == other.m_mode) ? equal : false;
        equal = (m_exponent == other.m_exponent) ? equal : false;
        return equal;
    }
    bool operator!=(const MemAlignment& other)
    {
        return !(*this == other);
    }
};
}  // namespace sys
}  // namespace Neon
