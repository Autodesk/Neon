#pragma once

#include "Neon/set/ContainerTools/GraphNodeOrganization.h"
#include "Neon/set/ContainerTools/GraphNodeScheduling.h"

#include "Neon/set/Containter.h"

namespace Neon::set::container {

struct GraphNode
{
   public:
    GraphNode();
    GraphNode(const Container&           container,
              GraphNodeOrganization::Uid uid);

    /**
     * Factory method to generate a begin node
     */
    static auto getBeginNode() -> GraphNode;

    /**
     * Factory method to generate a end node
     */
    static auto getEndNode() -> GraphNode;

    /**
     * Execute the scheduling operation associated to the node
     */
    auto execute() -> void;

    auto getOrganization() -> GraphNodeOrganization&;
    auto getOrganization() const -> const GraphNodeOrganization&;

    auto getScheduling() -> GraphNodeScheduling&;
    auto getScheduling() const -> const GraphNodeScheduling&;

    auto getContianer() -> Container&;
    auto getContianer() const -> const Container&;

   private:
    Container             mContainer /**< Any Neon container */;
    GraphNodeScheduling   mGraphNodeScheduling /**< Scheduling information for the node */;
    GraphNodeOrganization mGraphNodeOrganization /**< Information to organize the node w.r.t. the rest of the graph */;
};

}  // namespace Neon::set::container
