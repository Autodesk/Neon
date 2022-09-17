#pragma once

#include "Neon/core/core.h"
#include "Neon/core/types/digraph.h"

#include "Neon/set/container/graph/Bfs.h"
#include "Neon/set/container/graph/GraphDependency.h"
#include "Neon/set/container/graph/GraphNode.h"

namespace Neon::set::container {

/**
 * Abstraction for a graph of containers.
 * Each graph node represents a Neon container,
 * directed edges of the graph are dependencies between the containers.
 * Dependencies my be data driven or user provided.
 *
 *
 */
struct Graph
{
    using Uid = GraphData::Uid;
    using Index = GraphData::Index;
    friend struct Bfs;

   public:
    Graph();
    explicit Graph(const Backend& bk);

    /**
     * Get a reference to the begin node
     */
    auto getBeginNode() const
        -> const Neon::set::container::GraphNode&;

    /**
     * Get a reference to the end node of the graph
     */
    auto getEndNode() const
        -> const GraphNode&;

    /**
     * Adds a node between the begin and end nodes
     */
    auto addNode(const Container& container)
        -> GraphNode&;

    /**
     * Adds a dependency between two nodes of the graph
     */
    auto addNodeInBetween(const GraphNode&    nodeA,
                          Container           containerB,
                          const GraphNode&    nodeC,
                          GraphDependencyType ab = GraphDependencyType::user,
                          GraphDependencyType bc = GraphDependencyType::user)
        -> GraphNode&;

    /**
     * Remove node by maintaining the dependencies between
     * proceeding and following nodes
     */
    auto removeNode(GraphNode& gn)
        -> Neon::set::Container;

    /**
     * Remove a node and its dependencies
     */
    auto removeNodeAndItsDependencies(GraphNode& gn)
        -> Neon::set::Container;

    /**
     * Adds a dependency between two node of the graph
     */
    auto addDependency(const GraphNode&    nodeA,
                       const GraphNode&    nodeB,
                       GraphDependencyType type)
        -> GraphDependency&;

    auto appendDataDependency(const GraphNode&                                   nodeA,
                              const GraphNode&                                   nodeB,
                              Neon::internal::dataDependency::DataDependencyType dataDependencyType,
                              Neon::internal::dataDependency::DataUId            dataUId) -> GraphDependency&;
    /**
     * Returns the dependency type between two nodes.
     */
    auto getDependencyType(const GraphNode& nodeA,
                           const GraphNode& nodeB)
        -> GraphDependencyType;

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
    auto getProceedingGraphNodes(const GraphNode&                        graphNode,
                                 const std::vector<GraphDependencyType>& dependencyTypes = {GraphDependencyType::user,
                                                                                            GraphDependencyType::data})
        -> std::vector<GraphNode*>;

    /**
     * Returns all subsequent graph nodes.
     * The end node is excluded
     */
    auto getSubsequentGraphNodes(const GraphNode&                        graphNode,
                                 const std::vector<GraphDependencyType>& dependencyTypes = {GraphDependencyType::user,
                                                                                            GraphDependencyType::data}) -> std::vector<GraphNode*>;
    /**
     * Set the stream to run the graph.
     * We provide a preset function so that some initialization
     * could be only once, before executing the run method.
     */
    auto runtimePreSet(int anchorStream)
        -> void;

    /**
     * Execute the graph on all devices
     */
    auto run(int            streamIdx = 0,
             Neon::DataView dataView = Neon::DataView::STANDARD)
        -> void;

    /**
     * Run the graph on a target device.
     */
    auto run(Neon::SetIdx   setIdx,
             int            streamIdx = 0,
             Neon::DataView dataView = Neon::DataView::STANDARD)
        -> void;

    auto ioToDot(const std::string& fname,
                 const std::string& graphName,
                 bool               debug) -> void;

    /**
     * Returns a reference for the used backend.
     */
    auto getBackend() const -> const Neon::Backend&;

    /**
     * Recursively iterate through the graph nodes
     * to expand any graph type of node.
     */
    auto expandSubGraphs()
        -> void;

    auto getNumberOfNodes()
        -> int;

    /**
     * it removes redundant dependencies
     */
    auto removeRedundantDependencies() -> void;

    /**
     * Extract a graph node from its id
     */
    auto getGraphNode(GraphData::Uid)
        -> GraphNode&;

    /**
     * Extract a graph node from its id
     */
    auto getGraphNode(GraphData::Uid) const
        -> const GraphNode&;

   protected:
    /**
     * Invalidate all scheduling information that were computed
     */
    auto helpInvalidateScheduling() -> void;

    /**
     * Helper - it checks the initialization of the backend.
     */
    auto helpCheckBackendStatus() -> void;


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
     *
     * Computes two elements:
     * - order of execution
     * - mapping between streams and graph nodes
     */
    auto helpComputeScheduling(bool filterOutAnchors, int anchorStream)
        -> void;

    /**
     * Helper - it executes the graph on all devices
     */
    auto helpExecute(int anchorStream)
        -> void;

    /**
     * Helper - it executes the graph on a target device
     */
    auto helpExecute(Neon::SetIdx setIdx, int anchorStream)
        -> void;
    /**
     * Helper - It resets node scheduling data
     */
    auto helpComputeScheduling_00_resetData()
        -> void;

    /**
     * Helper - it generates a BFS visit structure for the graph
     */
    auto helpComputeScheduling_01_generatingBFS(bool filterOutAnchors)
        -> Bfs;

    /**
     * Helper - it maps node to streams.
     * Returns the max stream Id used by the scheduling
     */
    auto helpComputeScheduling_02_mappingStreams(Bfs& bfs, bool filterOutAnchors, int anchorStream)
        -> int;

    /**
     * Helper - it defines events to be waited and fired from each node
     * Returns the max event Id used by the scheduling.
     */
    auto helpComputeScheduling_03_events(Bfs& bfs)
        -> int;

    /**
     * Helper - it Books the required resources from the backend.
     */
    auto helpComputeScheduling_04_ensureResources(int maxStreamId, int maxEventId)
        -> void;


    using RawGraph = DiGraph<GraphNode, GraphDependency>;

    Uid      mUidCounter /**< internal counter to create node uids */;
    RawGraph mRawGraph /**< raw graph structure */;
    bool     mSchedulingStatusIsValid;
    int      mMaxNumberStreams /**< Max number of streams used in the scheduling phase */;
    Bfs      mBfs /**< Structure for the BFS visit. */;

    Backend mBackend /**< Backend used to create streams and events */;
    bool    mBackendIsSet = false;

    bool mFilterOutAnchorsPreSet = false;
    int  mAnchorStreamPreSet = 0;
};

}  // namespace Neon::set::container

#include "Neon/set/container/graph/Bfs_imp.h"
