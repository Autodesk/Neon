#include "Neon/core/types/Access.h"

namespace Neon {

auto AccessUtils::toString(Access mode) -> const char*
{
    switch (mode) {
        case Access::read: {
            return "read";
        }
        case Access::readWrite: {
            return "readWrite";
        }
        default:
            return nullptr;
    }
}
}  // END of namespace Neon

std::ostream& operator<<(std::ostream& os, Neon::Access const& m)
{
    return os << std::string(Neon::AccessUtils::toString(m));
}
