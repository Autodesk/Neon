#pragma once

#include "GraphDependencyType.h"
#include "Neon/set/container/graph/GraphInfo.h"
#include "Neon/set/dependency/Alias.h"
#include "Neon/set/dependency/ComputeType.h"
#include "Neon/set/dependency/DataDependencyType.h"
#include "Neon/set/Containter.h"

namespace Neon::set::container {

struct GraphDependency
{

   public:
    struct Info
    {
        Info(Neon::internal::dataDependency::DataDependencyType dataDependencyType,
             Neon::internal::dataDependency::MdObjUid           UId,
             Neon::Compute                                      compute)
            : dataDependencyType(dataDependencyType),
              dataUId(UId),
              compute(compute)
        {
        }

        Info(Neon::internal::dataDependency::DataDependencyType dataDependencyType,
             Neon::internal::dataDependency::MdObjUid           UId,
             Neon::Compute                                      compute,
             Neon::set::Container&                              haloUpdate)
            : dataDependencyType(dataDependencyType),
              dataUId(UId),
              compute(compute),
              haloUpdate(haloUpdate)
        {
        }

        Neon::internal::dataDependency::DataDependencyType dataDependencyType;
        Neon::internal::dataDependency::MdObjUid           dataUId;
        Neon::Compute                                      compute;
        Neon::set::Container                               haloUpdate;
    };

    GraphDependency();

    GraphDependency(GraphDependencyType type,
                    GraphInfo::NodeUid  source,
                    GraphInfo::NodeUid  destination);

    auto setType(GraphDependencyType type) -> void;

    auto getType() const -> GraphDependencyType;

    auto appendInfo(Neon::internal::dataDependency::DataDependencyType dataDependencyType,
                    Neon::internal::dataDependency::MdObjUid           dataUId,
                    Neon::Compute                                      compute) -> void;

    auto appendStencilInfo(Neon::internal::dataDependency::DataDependencyType dataDependencyType,
                           Neon::internal::dataDependency::MdObjUid           dataUId,
                           Neon::set::Container&                              container) -> void;

    auto toString(std::function<std::pair<std::string, std::string>(int)> prefix) -> std::string;

    auto getListStencilInfo() const
        -> std::vector<const Info*>;

    auto hasStencilDependency() const -> bool;

    auto getSource() const -> GraphInfo::NodeUid;

    auto getDestination() const -> GraphInfo::NodeUid;

    auto getLabel() -> std::string;

   private:
    GraphDependencyType mType;
    std::vector<Info>   mInfo;
    bool                mHasStencilDependency = false;
    GraphInfo::NodeUid  mSource;
    GraphInfo::NodeUid  mDestination;
    // TODO - add information for data and Scheduling dependency
};

}  // namespace Neon::set::container
