#include <string>

#include "Neon/set/container/graph/GraphDependency.h"
// #include "Neon/set/dependency/Alias.h"
// #include "Neon/set/dependency/ComputeType.h"

namespace Neon::set::container {

GraphDependency::GraphDependency()
{
    mHasStencilDependency = false;
    mSource = GraphData::notSet;
    mDestination = GraphData::notSet;
}

GraphDependency::GraphDependency(GraphDependencyType type,
                                 GraphData::Uid      source,
                                 GraphData::Uid      destination)
{
    mHasStencilDependency = false;
    setType(type);
    mSource = source;
    mDestination = destination;
}

auto GraphDependency::setType(GraphDependencyType type) -> void
{
    mType = type;
}

auto GraphDependency::getType() const -> GraphDependencyType
{
    return mType;
}


auto GraphDependency::getLabel() -> std::string
{
    std::stringstream s;
    s << GraphDependencyTypeUtil::toString(getType());
    s << toString([](int) -> std::pair<std::string, std::string> {
        return {"\\l", ""};
    });
    return s.str();
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

auto GraphDependency::appendInfo(Neon::internal::dataDependency::DataDependencyType dataDependencyType,
                                 Neon::internal::dataDependency::DataUId            dataUId,
                                 Neon::Compute                                      compute) -> void
{
    mInfo.push_back({dataDependencyType, dataUId, compute});
    if (compute == Neon::Compute::STENCIL) {
        mHasStencilDependency = true;
    }
}

auto GraphDependency::getListStencilData() const -> std::vector<Neon::internal::dataDependency::DataUId>
{
    std::vector<Neon::internal::dataDependency::DataUId> output;
    for (const auto i : mInfo) {
        if (i.compute == Neon::Compute::STENCIL) {
            output.push_back(i.dataUId);
        }
    }
    return output;
}

auto GraphDependency::getSource() const -> GraphData::Uid
{
    return mSource;
}

auto GraphDependency::getDestination() const -> GraphData::Uid
{
    return mDestination;
}

auto GraphDependency::hasStencilDependency() const -> bool
{
    return mHasStencilDependency;
}

}  // namespace Neon::set::container
