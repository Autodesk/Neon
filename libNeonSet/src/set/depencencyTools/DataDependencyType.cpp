#include "Neon/set/dependency/DataDependencyType.h"

namespace Neon::internal::dataDependency {

auto DataDependencyTypeUtils::toString(DataDependencyType val) -> std::string
{
    switch (val) {
        case DataDependencyType::RAW: {
            return "RAW";
        }
        case DataDependencyType::WAR: {
            return "WAR";
        }
        case DataDependencyType::RAR: {
            return "RAR";
        }
        case DataDependencyType::WAW: {
            return "WAW";
        }
        case DataDependencyType::NONE: {
            return "NONE";
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION();
}

std::ostream& operator<<(std::ostream& os, Neon::internal::dataDependency::DataDependencyType const& m){
   os << DataDependencyTypeUtils::toString(m);
   return os;
}

}