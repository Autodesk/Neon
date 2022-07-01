#include "Neon/skeleton/internal/dependencyTools/Dependency.h"

namespace Neon::skeleton::internal {

Dependency::Dependency(ContainerIdx   t1,
                       Dependencies_e type,
                       DataUId_t      uid,
                       ContainerIdx   t0)
{
    m_t1 = t1;
    m_type = type;
    m_uid = uid;
    m_t0 = t0;
}

bool Dependency::isValid()
{
    return m_type != Dependencies_e::NONE;
}

auto Dependency::toString() -> std::string
{
    return std::to_string(m_t1) +
           " -> (" + Dependencies_et::toString(m_type) +
           " [" + std::to_string(m_uid) +
           "]) -> " + std::to_string(m_t0);
}

auto Dependency::type() -> Dependencies_e
{
    return m_type;
}

Dependency Dependency::getEmpty()
{
    return Dependency();
}
auto Dependency::t0() -> ContainerIdx
{
    return m_t0;
}

auto Dependency::t1() -> ContainerIdx
{
    return m_t1;
}

}  // namespace Neon::skeleton::internal