#include "Neon/sys/devices/memType.h"

namespace Neon {
namespace sys {


mem_et::mem_et(enum_e type)
    : m_type(type){};

mem_et::enum_e mem_et::type() const
{
    return m_type;
}

const char* mem_et::name(mem_et::enum_e type)
{
    switch (type) {
        case mem_et::enum_e::cpu: {
            return "cpu";
        }
        case mem_et::enum_e::gpu: {
            return "gpu";
        }
        case mem_et::enum_e::device: {
            return "device";
        }
        default:
            return nullptr;
    }
}

const char* mem_et::name() const
{
    return this->name(m_type);
}

}  // namespace sys
}  // namespace Neon
