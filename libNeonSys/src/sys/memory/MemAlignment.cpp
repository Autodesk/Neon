#include "Neon/sys/memory/MemAlignment.h"
#include "Neon/sys/global/CpuSysGlobal.h"
#include "Neon/sys/global/GpuSysGlobal.h"
#include "Neon/sys/memory/GpuMem.h"

#include <vector>


namespace Neon {
namespace sys {

MemAlignment::MemAlignment(memAlignment_et mode,
                               uint32_t        exponent)
    : m_mode(mode), m_exponent(exponent)
{
    // nothing to do
}

MemAlignment::MemAlignment(memAlignment_et mode)
    : m_mode(mode)
{
    // nothing to do
}

MemAlignment::MemAlignment(memAlignment_e::e align,
                               uint32_t          exponent)
{
    m_exponent = exponent;
    switch (align) {
        case memAlignment_e::SYSTEM: {
            m_mode = memAlignment_et::system;
            break;
        }
        case memAlignment_e::L1: {
            m_mode = memAlignment_et::cacheLine;
            break;
        }
        case memAlignment_e::L2: {
            m_mode = memAlignment_et::cacheLine;
            break;
        }
        case memAlignment_e::PAGE: {
            m_mode = memAlignment_et::page;
            break;
        }
    }
}

const memAlignment_et& MemAlignment::mode() const
{
    return m_mode;
}

uint32_t MemAlignment::exponent() const
{
    return m_exponent;
}

uint32_t MemAlignment::expAlign(const Neon::sys::GpuMem& gpuMem) const
{
    switch (m_mode.type()) {
        case memAlignment_et::system: {
            const size_t exponent = 0;
            return exponent;
        }

        case memAlignment_et::cacheLine: {
            size_t L1 = gpuMem.expL1DataCacheLine();
            return static_cast<uint32_t>(L1);
        }

        case memAlignment_et::page: {
            size_t L1 = gpuMem.expL1_TLB_PageSize();
            size_t L2 = gpuMem.expL2_TLB_PageSize();
            size_t max = std::max(L1, L2);
            return static_cast<uint32_t>(max);
        }
        case memAlignment_et::user: {
            size_t exponent = m_exponent;
            return static_cast<uint32_t>(exponent);
        }
        default: {
            Neon::NeonException exp("memAlignment_t");
            exp << "memAlignment_t::expAlign : undefined mode";
            NEON_THROW(exp);
        }
    }
}
uint32_t MemAlignment::expAlign() const
{
    switch (m_mode.type()) {
        case memAlignment_et::system: {
            size_t exponent = 0;
            return static_cast<uint32_t>(exponent);
        }
        case memAlignment_et::cacheLine: {
            NEON_DEV_UNDER_CONSTRUCTION("memAlignment_t - cacheLine");
        }
        case memAlignment_et::page: {
            NEON_DEV_UNDER_CONSTRUCTION("memAlignment_t - page");
        }
        case memAlignment_et::user: {
            size_t exponent = m_exponent;
            return static_cast<uint32_t>(exponent);
        }
        default: {
            NEON_DEV_UNDER_CONSTRUCTION("memAlignment_t - page");
        }
    }
}


uint32_t MemAlignment::expAlign(const DeviceType& devType,
                                  const DeviceID&     devId) const
{

    switch (devType) {
        case Neon::DeviceType::CUDA: {
            Neon::sys::GpuMem& mem = Neon::sys::globalSpace::gpuSysObj().allocator(devId.idx());
            const uint32_t       exp = this->expAlign(mem);
            return exp;
        }
        case Neon::DeviceType::CPU: {
            const uint32_t exp = this->expAlign();
            return exp;
        }
        default: {
            Neon::NeonException exp("Mem_t");
            exp << "Unsupported memory allocation for device " << devType;
            NEON_THROW(exp);
        }
    }
}
uint32_t MemAlignment::exp2byte(uint32_t exp)
{
    uint32_t res = 1;
    res = (res << exp);
    return res;
}

}  // namespace sys
}  // namespace Neon
