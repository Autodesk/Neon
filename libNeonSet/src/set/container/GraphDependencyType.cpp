#include "Neon/set/ContainerTools/graph/GraphDependencyType.h"
#include "Neon/core/core.h"

namespace Neon {


auto GraphDependencyTypeUtil::validOptions() -> std::array<Neon::GraphDependencyType, GraphDependencyTypeUtil::nConfig>
{
    std::array<Neon::GraphDependencyType, GraphDependencyTypeUtil::nConfig> options = {GraphDependencyType::data,
                                                                                       GraphDependencyType::scheduling,
                                                                                       GraphDependencyType::user};
    return options;
}

auto GraphDependencyTypeUtil::toString(GraphDependencyType e) -> std::string
{
    switch (e) {
        case GraphDependencyType::data: {
            return "data";
        }
        case GraphDependencyType::scheduling: {
            return "scheduling";
        }
        case GraphDependencyType::user: {
            return "user";
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("GraphDependencyTypeUtil");
        }
    }
}

auto GraphDependencyTypeUtil::fromInt(int val) -> GraphDependencyType
{
    switch (val) {
        case static_cast<int>(GraphDependencyType::data): {
            return GraphDependencyType::data;
        }
        case static_cast<int>(GraphDependencyType::scheduling): {
            return GraphDependencyType::scheduling;
        }
        case static_cast<int>(GraphDependencyType::user): {
            return GraphDependencyType::user;
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
