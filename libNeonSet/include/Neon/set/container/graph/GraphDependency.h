#pragma once

#include "GraphDependencyType.h"
#include "Neon/set/dependency/Alias.h"
#include "Neon/set/dependency/ComputeType.h"
#include "Neon/set/dependency/DataDependencyType.h"

namespace Neon::set::container {

struct GraphDependency
{
    std::string getLabel();

   public:
    GraphDependency();

    GraphDependency(GraphDependencyType type);

    auto setType(GraphDependencyType type) -> void;

    auto getType() const -> GraphDependencyType;

    auto appendInfo(Neon::internal::dataDependency::DataDependencyType dataDependencyType,
                    Neon::internal::dataDependency::DataUId            dataUId,
                    Neon::Compute                                      compute) -> void;

    auto toString(std::function<std::pair<std::string, std::string>(int)> prefix) -> std::string;

    auto getListStencilData() const -> std::vector<Neon::internal::dataDependency::DataUId>;

    auto hasStencilDependency() const -> bool;

   private:
    struct Info
    {
        Neon::internal::dataDependency::DataDependencyType dataDependencyType;
        Neon::internal::dataDependency::DataUId            dataUId;
        Neon::Compute                                      compute;
    };

    GraphDependencyType mType;
    std::vector<Info>   mInfo;
    bool                mHasStencilDependency = false;
    // TODO - add information for data and Scheduling dependency
};

}  // namespace Neon::set::container
