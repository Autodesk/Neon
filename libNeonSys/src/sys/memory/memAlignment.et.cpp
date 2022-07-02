#include "Neon/sys/memory/memAlignment.et.h"

#include <vector>

namespace Neon {
namespace sys {

static std::vector<std::string> mem3dLayoutNames{std::string("cacheLine"), std::string("page"), std::string("system"), std::string("user")};

memAlignment_et::memAlignment_et(enum_e type)
    : m_type(type) {}

memAlignment_et::enum_e memAlignment_et::type() const
{
    return m_type;
}

const char* memAlignment_et::name(enum_e type)
{
    return mem3dLayoutNames[type].c_str();
}

const char* memAlignment_et::name() const
{
    return memAlignment_et::name(m_type);
}


}  // namespace sys
}  // namespace Neon
