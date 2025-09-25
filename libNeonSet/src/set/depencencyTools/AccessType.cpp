#include "Neon/set/dependency/AccessType.h"

namespace Neon::set::dataDependency {

auto AccessTypeUtils::toString(AccessType val) -> std::string
{
    switch (val) {
        case AccessType::READ: {
            return "READ";
        }
        case AccessType::WRITE: {
            return "WRITE";
        }
        case AccessType::NONE: {
            return "NONE";
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION();
}

auto AccessTypeUtils::merge(AccessType valA, AccessType valB) -> AccessType
{
    if (valA == AccessType::WRITE)
        return AccessType::WRITE;
    if (valB == AccessType::WRITE)
        return AccessType::WRITE;
    if (valA == AccessType::READ)
        return AccessType::READ;
    if (valB == AccessType::READ)
        return AccessType::READ;
    return AccessType::NONE;
}

auto AccessTypeUtils::fromInt(int val) -> AccessType
{
    if (val == 0)
        return AccessType::NONE;
    if (val == 1)
        return AccessType::READ;
    if (val == 2)
        return AccessType::WRITE;
    return AccessType::NONE;
}
}  // namespace Neon::set::dataDependency
