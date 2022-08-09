#pragma once

#include "Neon/core/core.h"
#include "Neon/core/types/digraph.h"

#include "Neon/set/ContainerTools/graph/GraphDependency.h"
#include "Neon/set/ContainerTools/graph/GraphNode.h"

namespace Neon::set::container {

struct Graph
{
    using Uid = GraphData::Uid;
    using Index = GraphData::Index;

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
     * Adds a node between the begin and end nodes
     */
    auto addNode(const Container& container,
                 GraphNodeType    graphNodeType) -> GraphNode&;

    /**
     * Remove Node
     */
    auto removeNode(GraphNode& gn) -> GraphNode;

    /**
     * Adds a dependency between two nodes of the graph
     */
    auto addNodeInBetween(const GraphNode&          nodeA,
                          Container&                containerB,
                          const GraphNode&          nodeC,
                          const GraphDependencyType ab = GraphDependencyType::user,
                          const GraphDependencyType bc = GraphDependencyType::user) -> GraphNode&;

    /**
     * Adds a dependency between two node of the graph
     */
    auto addDependency(const GraphNode&    nodeA,
                       const GraphNode&    nodeB,
                       GraphDependencyType type) -> GraphDependency&;

    /**
     * Returns the dependency type between two nodes.
     */
    auto getDependencyType(const GraphNode& nodeA,
                           const GraphNode& nodeB) -> GraphDependencyType;

    /**
     * Clone a node and return a reference to the new clone.
     * The cloning process connects the clone the the same nodes of the original
     */
    auto cloneNode(const GraphNode& graphNode)
        -> GraphNode&;

    /**
     * Returns all proceeding graph nodes.
     * The begin node is excluded
     */
    auto getProceedingGraphNodes(const GraphNode& graphNode)
        -> std::vector<GraphNode*>;

    /**
     * Returns all subsequent graph nodes.
     * The end node is excluded
     */
    auto getSubsequentGraphNodes(const GraphNode& graphNode) -> std::vector<GraphNode*>;

    /**
     * Execute the scheduling operation associated to the node
     */
    auto execute() -> void;

    auto computeScheduling() -> void;

    auto ioToDot(const std::string& fame) -> void;


   private:
    /**
     * Invalidate all scheduling information that were computed
     */
    auto helpInvalidateScheduling() -> void;

    /**
     * Remove redundant dependencies
     */
    auto helpRemoteRedundantDependencies() -> void;

    /*
     * Compute BFS
     */
    auto computeBFS(const std::vector<GraphDependencyType>& depednenciesToBeConsidered) -> void;

    /**
     * Returns the out-neighbour of a target node
     */
    auto helpGetOutNeighbors(GraphData::Uid,
                             bool                                    fileterOutEnd = true,
                             const std::vector<GraphDependencyType>& dependencyTypes = {GraphDependencyType::user,
                                                                                        GraphDependencyType::data})
        -> std::set<GraphData::Uid>;

    /**
     * Returns the in-neighbour of a target node
     */
    auto helpGetInNeighbors(GraphData::Uid                          nodeUid,
                            bool                                    fileterOutBegin = true,
                            const std::vector<GraphDependencyType>& dependencyTypes = {GraphDependencyType::user,
                                                                                       GraphDependencyType::data})
        -> std::set<GraphData::Uid>;

    /**
     * Returns the out-edges of a target node
     */
    auto helpGetOutEdges(GraphData::Uid,
                         bool                                    filterOutEnd = true,
                         const std::vector<GraphDependencyType>& dependencyTypes = {GraphDependencyType::user,
                                                                                    GraphDependencyType::data})
        -> std::set<std::pair<GraphData::Uid, GraphData::Uid>>;

    /**
     * Returns the in-edges of a target node
     */
    auto helpGetInEdges(GraphData::Uid                          nodeUid,
                        bool                                    filterOutBegin = true,
                        const std::vector<GraphDependencyType>& dependencyTypes = {GraphDependencyType::user,
                                                                                   GraphDependencyType::data})
        -> std::set<std::pair<GraphData::Uid, GraphData::Uid>>;

    /**
     * Returns nodes Ids for a BFS visit
     */
    auto helpGetBFS(bool                                    filterOutBegin = false,
                    const std::vector<GraphDependencyType>& dependencyTypes = {GraphDependencyType::user,
                                                                               GraphDependencyType::data})
        -> std::vector<std::vector<GraphData::Uid>>;

    using RawGraph = DiGraph<GraphNode, GraphDependency>;
    Uid      mUidCounter;
    RawGraph mRawGraph;
    bool     mSchedulingStatusIsValid;
};

}  // namespace Neon::set::container
