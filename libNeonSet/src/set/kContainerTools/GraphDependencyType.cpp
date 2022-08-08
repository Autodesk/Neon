#include "Neon/set/ContainerTools/graph/GraphDependencyType.h"
#include "Neon/core/core.h"

namespace Neon {


auto GraphDependencyTypeUtil::validOptions() -> std::array<Neon::GraphDependencyType, GraphDependencyTypeUtil::nConfig>
{
    std::array<Neon::GraphDependencyType, GraphDependencyTypeUtil::nConfig> options = {GraphDependencyType::DATA,
                                                                                       GraphDependencyType::SCHEDULING,
                                                                                       GraphDependencyType::USER};
    return options;
}

auto GraphDependencyTypeUtil::toString(GraphDependencyType e) -> std::string
{
    switch (e) {
        case GraphDependencyType::DATA: {
            return "DATA";
        }
        case GraphDependencyType::SCHEDULING: {
            return "SCHEDULING";
        }
        case GraphDependencyType::USER: {
            return "USER";
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("GraphDependencyTypeUtil");
        }
    }
}

auto GraphDependencyTypeUtil::fromInt(int val) -> GraphDependencyType
{
    switch (val) {
        case static_cast<int>(GraphDependencyType::DATA): {
            return GraphDependencyType::DATA;
        }
        case static_cast<int>(GraphDependencyType::SCHEDULING): {
            return GraphDependencyType::SCHEDULING;
        }
        case static_cast<int>(GraphDependencyType::USER): {
            return GraphDependencyType::USER;
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("GraphDependencyTypeUtil");
        }
    }
}

auto GraphDependencyTypeUtil::toInt(GraphDependencyType dataView) -> int
{
    return static_cast<int>(dataView);
}

}  // namespace Neon

std::ostream& operator<<(std::ostream& os, Neon::GraphDependencyType const& m)
{
    return os << std::string(Neon::GraphDependencyTypeUtil::toString(m));
}
