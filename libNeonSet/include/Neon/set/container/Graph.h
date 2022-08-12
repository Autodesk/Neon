#pragma once

#include "Neon/core/core.h"
#include "Neon/core/types/digraph.h"

#include "Neon/set/container/graph/GraphDependency.h"
#include "Neon/set/container/graph/GraphNode.h"

namespace Neon::set::container {
struct Bfs;

struct Graph
{
    using Uid = GraphData::Uid;
    using Index = GraphData::Index;
    friend struct Bfs;

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
    auto addNode(const Container& container) -> GraphNode&;

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

    auto ioToDot(const std::string& fame) -> void;


   protected:
    /**
     * Invalidate all scheduling information that were computed
     */
    auto helpInvalidateScheduling() -> void;

    /**
     * Remove redundant dependencies
     */
    auto helpRemoteRedundantDependencies() -> void;

    /**
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
    auto helpGetBFS(bool                                    filterOutBeginEnd = false,
                    const std::vector<GraphDependencyType>& dependencyTypes = {GraphDependencyType::user,
                                                                               GraphDependencyType::data})
        -> Bfs;

    /**
     * Extract a graph node from its id
     */
    auto helpGetGraphNode(GraphData::Uid)
        -> GraphNode&;

    /**
     * Extract a graph node from its id
     */
    auto helpGetGraphNode(GraphData::Uid) const
        -> const GraphNode&;

    /**
     *
     * Computes two elements:
     * - order of execution
     * - mapping between streams and graph nodes
     */
    auto helpComputeScheduling(bool filterOutAnchors = true) -> void;

    /**
     * Resetting node's data related to scheduling
     */
    auto helpComputeScheduling_00_resetData(Bfs& bfs) -> void;
    /**
     * Maps node to streams
     */
    auto helpComputeScheduling_01_mappingStreams(Bfs& bfs) -> void;

    /**
     * Define events to be waited and fired from each node
     */
    auto helpComputeScheduling_02_events(Bfs& bfs) -> void;

    using RawGraph = DiGraph<GraphNode, GraphDependency>;

    Uid      mUidCounter;
    RawGraph mRawGraph;
    bool     mSchedulingStatusIsValid;
    int      mMaxNumberStreams;
};

}  // namespace Neon::set::container
