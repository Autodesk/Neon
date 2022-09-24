#pragma once

#include "GraphInfo.h"
#include "GraphNodeScheduling.h"
#include "Neon/set/Containter.h"

namespace Neon::set::container {

struct GraphNode
{
    auto getLabel(bool debug) -> std::string;

    auto getLabelProperty() -> std::string;

   public:
    GraphNode();

    GraphNode(const Container& container,
              GraphInfo::NodeUid   uid);

    /**
     * Factory method to generate a begin node
     */
    static auto newBeginNode() -> GraphNode;

    /**
     * Factory method to generate a end node
     */
    static auto newEndNode() -> GraphNode;

    /**
     * Executes the scheduling operation associated to the node
     */
    auto execute() -> void;

    /**
     * Returns a reference to graph information related to this node.
     * */
    auto getGraphData() -> GraphInfo&;

    /**
     * Returns a reference to some graph information related to this node.
     * */
    auto getGraphData() const -> const GraphInfo&;

    /**
     * Returns the scheduling information to run this node
     */
    auto getScheduling() -> GraphNodeScheduling&;

    /**
     * Returns the scheduling information to run this node
     */
    auto getScheduling() const -> const GraphNodeScheduling&;

    /**
     * Returns a reference to the container stored by this node.
     */
    auto getContainer() -> Container&;

    /**
     * Returns a reference to the container stored by this node.
     */
    auto getContainer() const -> const Container&;

    /**
     * Returns the operation type of the container associated to this node
     */
    auto getContainerOperationType() const -> Neon::set::ContainerOperationType;

    auto toString() -> std::string;


   private:
    auto helpGetDotProperties() -> std::string;
    auto helpGetDotName() -> std::string;
    auto helpGetDotInfo() -> std::string;

    Container            mContainer /**< Any Neon container */;
    GraphNodeScheduling  mGraphNodeScheduling /**< Scheduling information for the node */;
    GraphInfo            mGraphNodeOrganization /**< Information to organize the node w.r.t. the rest of the graph */;
    ContainerPatternType getContainerpatternType() const;
};

}  // namespace Neon::set::container
