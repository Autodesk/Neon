#include <string>

#include "Neon/set/Containter.h"
#include "Neon/set/container/graph/GraphDependency.h"
// #include "Neon/set/dependency/Alias.h"
// #include "Neon/set/dependency/ComputeType.h"

namespace Neon::set::container {

GraphDependency::GraphDependency()
{
    mHasStencilDependency = false;
    mSource = GraphInfo::notSet;
    mDestination = GraphInfo::notSet;
}

GraphDependency::GraphDependency(GraphDependencyType type,
                                 GraphInfo::NodeUid      source,
                                 GraphInfo::NodeUid      destination)
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
                                 Neon::internal::dataDependency::MdObjUid dataUId,
                                 Neon::Compute                                      compute) -> void
{
    mInfo.emplace_back(dataDependencyType, dataUId, compute);
    if (compute == Neon::Compute::STENCIL) {
        NEON_THROW_UNSUPPORTED_OPERATION("The appendStencilInfo method should be used instead.")
    }
}

auto GraphDependency::appendStencilInfo(Neon::internal::dataDependency::DataDependencyType dataDependencyType,
                                        Neon::internal::dataDependency::MdObjUid dataUId,
                                        Neon::set::Container&                              haloUpdate) -> void
{
    auto compute = Neon::Compute::STENCIL;
    mInfo.emplace_back(dataDependencyType, dataUId, compute, haloUpdate);
    mHasStencilDependency = true;
}


auto GraphDependency::getListStencilInfo() const -> std::vector<const Info*>
{
    std::vector<const Info*> output;
    for (const auto& i : mInfo) {
        if (i.compute == Neon::Compute::STENCIL) {
            output.push_back(&i);
        }
    }
    return output;
}

auto GraphDependency::getSource() const -> GraphInfo::NodeUid
{
    return mSource;
}

auto GraphDependency::getDestination() const -> GraphInfo::NodeUid
{
    return mDestination;
}

auto GraphDependency::hasStencilDependency() const -> bool
{
    return mHasStencilDependency;
}

}  // namespace Neon::set::container
