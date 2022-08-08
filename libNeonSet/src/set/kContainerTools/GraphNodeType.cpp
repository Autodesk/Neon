#include "Neon/set/ContainerTools/graph/GraphNodeType.h"
#include "Neon/core/core.h"

namespace Neon {


auto GraphNodeTypeUtil::validOptions() -> std::array<Neon::GraphNodeType, GraphNodeTypeUtil::nConfig>
{
    std::array<Neon::GraphNodeType, GraphNodeTypeUtil::nConfig> options = {GraphNodeType::Compute,
                                                                           GraphNodeType::Halo,
                                                                           GraphNodeType::Sync};
    return options;
}

auto GraphNodeTypeUtil::toString(GraphNodeType e) -> std::string
{
    switch (e) {
        case GraphNodeType::Compute: {
            return "DATA";
        }
        case GraphNodeType::Halo: {
            return "SCHEDULING";
        }
        case GraphNodeType::Sync: {
            return "Sync";
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("GraphNodeTypeUtil");
        }
    }
}

auto GraphNodeTypeUtil::fromInt(int val) -> GraphNodeType
{
    switch (val) {
        case static_cast<int>(GraphNodeType::Compute): {
            return GraphNodeType::Compute;
        }
        case static_cast<int>(GraphNodeType::Halo): {
            return GraphNodeType::Halo;
        }
        case static_cast<int>(GraphNodeType::Sync): {
            return GraphNodeType::Sync;
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("GraphNodeTypeUtil");
        }
    }
}

auto GraphNodeTypeUtil::toInt(GraphNodeType dataView) -> int
{
    return static_cast<int>(dataView);
}

}  // namespace Neon

std::ostream& operator<<(std::ostream& os, Neon::GraphNodeType const& m)
{
    return os << std::string(Neon::GraphNodeTypeUtil::toString(m));
}
