#pragma once
#include "Neon/core/core.h"

#include "Neon/set/container/ContainerAPI.h"
#include "Neon/set/container/Graph.h"
#include "Neon/set/container/GraphContainer.h"
#include "Neon/set/container/Loader.h"

namespace Neon {
namespace set {
namespace internal {

template <typename MultiXpuDataT>
struct HaloUpdateContainer
    : public GraphContainer
{

   public:
    virtual ~HaloUpdateContainer() override = default;

    HaloUpdateContainer(const Neon::Backend&        bk,
                        const Neon::set::Container& dataTransferContainer,
                        const Neon::set::Container& syncContainer)
    {
        Neon::set::container::Graph graph(bk);

        auto dataTranferNode = graph.addNode(dataTransferContainer);
        auto syncNode = graph.addNode(syncContainer);

        if (dataTransferContainer.getContainerInterface().getTransferMode() ==
            Neon::set::TransferMode::get) {
            graph.addDependency(syncNode, dataTranferNode, GraphDependencyType::data);
        } else {
            graph.addDependency(dataTranferNode, syncNode, GraphDependencyType::data);
        }

        auto           name = std::string("HaloUpdate");
        GraphContainer graphContainer(graph, [&](Neon::set::Loader& loader) {
            // Nothing to load
        });

        this->GraphContainer = graphContainer;

        setContainerOperationType(ContainerOperationType::communication);
        setDataViewSupport(DataViewSupport::off);
    }

   private:
};

}  // namespace internal
}  // namespace set
}  // namespace Neon
