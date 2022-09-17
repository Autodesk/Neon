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
    std::stringstream s;
    s << GraphDependencyTypeUtil::toString(getType());
    s << toString([](int ) -> std::pair<std::string, std::string> {
        return {"\\l", ""};
    });
    return s.str();
}

auto GraphDependency::appendInfo(Neon::internal::dataDependency::DataDependencyType dataDependencyType,
                                 Neon::internal::dataDependency::DataUId            dataUId) -> void
{
    mInfo.push_back({dataDependencyType, dataUId});
}

auto GraphDependency::toString(std::function<std::pair<std::string, std::string>(int)> prefixPostfix) -> std::string
{
    std::stringstream s;
    for (int i = 0; i < int(mInfo.size()); ++i) {
        const auto& [pre, post] = prefixPostfix(i);
        s << pre << mInfo[i].dataDependencyType << " (Data Id " << mInfo[i].dataUId << ")" << post;
    }
    return s.str();
}

}  // namespace Neon::set::container
