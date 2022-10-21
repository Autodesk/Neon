#include "Neon/set/container/graph/GraphDependency.h"
#include <string>
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

auto GraphDependency::getLabel() -> std::string
{
    return GraphDependencyTypeUtil::toString(getType());
}

}  // namespace Neon::set::container
