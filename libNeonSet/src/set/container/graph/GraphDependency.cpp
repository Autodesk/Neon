#include "Neon/set/container/graph/GraphDependency.h"

namespace Neon::set::container {

GraphDependency::GraphDependency()
{
}

auto GraphDependency::setType(GraphDependencyType type) -> void
{
    mType = type;
}

auto GraphDependency::getType() const -> GraphDependencyType
{
    return mType;
}
GraphDependency::GraphDependency(GraphDependencyType type)
{
    setType(type);
}

}  // namespace Neon::set::container
