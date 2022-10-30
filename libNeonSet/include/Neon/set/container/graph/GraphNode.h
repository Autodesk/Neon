#pragma once

#include "GraphData.h"
#include "GraphNodeScheduling.h"
#include "Neon/set/Containter.h"

namespace Neon::set::container {

struct GraphNode
{

   public:
    GraphNode();

    GraphNode(const Container& container,
              GraphData::Uid   uid);

    /**
     * Factory method to generate a begin node
     */
    static auto newBeginNode()
        -> GraphNode;

    /**
     * Factory method to generate a end node
     */
    static auto newEndNode()
        -> GraphNode;

    /**
     * Executes the scheduling operation associated to the node
     */
    auto execute()
        -> void;

    /**
     * Returns a reference to graph information related to this node.
     * */
    auto getGraphData()
        -> GraphData&;

    /**
     * Returns a reference to some graph information related to this node.
     * */
    auto getGraphData()
        const -> const GraphData&;

    /**
     * Returns the scheduling information to run this node
     */
    auto getScheduling()
        -> GraphNodeScheduling&;

    /**
     * Returns the scheduling information to run this node
     */
    auto getScheduling()
        const -> const GraphNodeScheduling&;

    /**
     * Returns a reference to the container stored by this node.
     */
    auto getContainer()
        -> Container&;

    /**
     * Returns a reference to the container stored by this node.
     */
    auto getContainer()
        const -> const Container&;

    /**
     * Returns the operation type of the container associated to this node
     */
    auto getContainerOperationType()
        const -> Neon::set::ContainerOperationType;

    auto toString()
        const -> std::string;

    auto getLabel(bool debug)
        const -> std::string;

    auto getLabelProperty()
        const -> std::string;

   private:
    auto helpGetDotProperties()
        const -> std::string;

    auto helpGetDotName()
        const -> std::string;

    auto helpGetDotInfo()
        const -> std::string;

    Container            mContainer /**< Any Neon container */;
    GraphNodeScheduling  mGraphNodeScheduling /**< Scheduling information for the node */;
    GraphData            mGraphNodeOrganization /**< Information to organize the node w.r.t. the rest of the graph */;
    ContainerPatternType getContainerpatternType() const;
};

}  // namespace Neon::set::container
