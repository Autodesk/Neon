#pragma once

#include "Neon/core/core.h"
#include "Neon/core/types/digraph.h"

#include "Neon/set/ContainerTools/GraphDependency.h"
#include "Neon/set/ContainerTools/GraphNode.h"

namespace Neon::set::container {

struct Graph
{
    using Uid = GraphNodeOrganization::Uid;
    using Index = GraphNodeOrganization::Index;

   public:
    Graph();

    /**
     * Get a reference to the begin node
     */
    auto getBeginNode() const -> const GraphNode&;

    /**
     * Get a reference to the end node of the graph
     */
    auto getEndNode() const -> const GraphNode&;

    /**
     * add a node between the begin and end nodes
     */
    auto addNode(Container& container) -> GraphNode&;

    /**
     * Remove Node
     */
    auto removeNode(GraphNode& gn) -> GraphNode;

    /**
     * Add node between two other nodes
     */
    auto addNodeInBetween(const GraphNode& nodeA,
                          Container&       containerB,
                          const GraphNode& nodeC) -> GraphNode&;

    auto addNodeInBetween(const GraphNode& nodeA,
                          Container&       containerB,
                          const GraphNode& nodeC,
                          GraphDependency& ab,
                          GraphDependency& bc) -> GraphNode&;
    /**
     * Add a dependency between two nodes of the graph
     */
    auto addDependency(const GraphNode&    source,
                       const GraphNode&    destination,
                       GraphDependencyType graphDependencyType = GraphDependencyType::USER) -> GraphDependency&;

    /**
     * Clone a node and return a reference to the new clone.
     * The cloning process connects the clone the the same nodes of the original
     *
     * @param graphNode
     * @return
     */
    auto clone(const GraphNode& graphNode) -> GraphNode&;

    /**
     * Returns all proceeding graph nodes.
     * The begin node is excluded
     */
    auto getProceedingGraphNodes() -> std::vector<GraphNode*>;

    /**
     * Returns all subsequent graph nodes.
     * The end node is excluded
     */
    auto getSubsequentGraphNodes() -> std::vector<GraphNode*>;

    /**
     * Execute the scheduling operation associated to the node
     */
    auto execute() -> void;

    auto computeScheduling() -> void;

   private:
    using RawGraph = DiGraph<GraphNode, GraphDependency>;
    Uid      mUidCounter;
    RawGraph mRawGraph;
};

}  // namespace Neon::set::container
