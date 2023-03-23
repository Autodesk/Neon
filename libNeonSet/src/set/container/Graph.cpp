#include "Neon/set/container/Graph.h"
#include <algorithm>
#include "Neon/set/Containter.h"
#include "Neon/set/container/graph/Bfs.h"
#ifdef NEON_USE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif
namespace Neon::set::container {

Graph::Graph()
{
    auto begin = GraphNode::newBeginNode();
    auto end = GraphNode::newEndNode();
    mUidCounter = GraphData::firstInternal;

    mRawGraph.addVertex(begin.getGraphData().getUid(), begin);
    mRawGraph.addVertex(end.getGraphData().getUid(), end);

    helpInvalidateScheduling();
}

auto Graph::
    getBeginNode()
        const -> const GraphNode&
{
    return mRawGraph.getVertexProperty(GraphData::beginUid);
}

auto Graph::
    getEndNode()
        const -> const GraphNode&
{
    return mRawGraph.getVertexProperty(GraphData::endUid);
}

auto Graph::
    addNodeInBetween(const GraphNode&          nodeA,
                     Container                 containerB,
                     const GraphNode&          nodeC,
                     const GraphDependencyType ab,
                     const GraphDependencyType bc)
        -> GraphNode&
{
    helpCheckBackendStatus();
    helpInvalidateScheduling();

    auto& nodeB = addNode(containerB);
    addDependency(nodeA, nodeB, ab);
    addDependency(nodeB, nodeC, bc);
    return nodeB;
}

auto Graph::
    addNode(const Container& container)
        -> GraphNode&
{
    helpCheckBackendStatus();
    helpInvalidateScheduling();

    auto const& node = GraphNode(container, mUidCounter);
    mRawGraph.addVertex(node.getGraphData().getUid(), node);
    mUidCounter++;

    addDependency(getBeginNode(), node, GraphDependencyType::user);
    addDependency(node, getEndNode(), GraphDependencyType::user);

    return mRawGraph.getVertexProperty(node.getGraphData().getUid());
}

auto Graph::
    addDependency(const GraphNode&    nodeA,
                  const GraphNode&    nodeB,
                  GraphDependencyType type)
        -> GraphDependency&
{
    helpCheckBackendStatus();
    if (nodeA.getGraphData().getUid() == nodeB.getGraphData().getUid()) {
        NEON_THROW_UNSUPPORTED_OPERATION("");
    }
    helpInvalidateScheduling();

    auto aUid = nodeA.getGraphData().getUid();
    auto bUid = nodeB.getGraphData().getUid();

    GraphDependency ab(type, {aUid, bUid});

    mRawGraph.addEdge(nodeA.getGraphData().getUid(),
                      nodeB.getGraphData().getUid(),
                      ab);

    if (GraphDependencyType::scheduling != type) {
        if (mRawGraph.hasEdge(aUid, GraphData::endUid) &&
            GraphData::endUid != bUid) {
            mRawGraph.removeEdge(aUid, GraphData::endUid);
        }

        if (mRawGraph.hasEdge(GraphData::beginUid, bUid) &&
            GraphData::beginUid != aUid) {
            mRawGraph.removeEdge(GraphData::beginUid, bUid);
        }
    }

    return mRawGraph.getEdgeProperty({nodeA.getGraphData().getUid(),
                                      nodeB.getGraphData().getUid()});
}

auto Graph::
    addDependency(const GraphNode&                        nodeA,
                  const GraphNode&                        nodeB,
                  const Neon::set::dataDependency::Token& token)
        -> GraphDependency&
{
    helpCheckBackendStatus();
    if (nodeA.getGraphData().getUid() == nodeB.getGraphData().getUid()) {
        NEON_THROW_UNSUPPORTED_OPERATION("");
    }
    helpInvalidateScheduling();

    if (mRawGraph.hasEdge(nodeA.getGraphData().getUid(),
                          nodeB.getGraphData().getUid())) {
        GraphDependency& graphDependency = mRawGraph.getEdgeProperty({nodeA.getGraphData().getUid(),
                                                                      nodeB.getGraphData().getUid()});
        graphDependency.addToken(token);
    } else {
        GraphDependency ab(token, {nodeA.getGraphData().getUid(),
                                   nodeB.getGraphData().getUid()});

        mRawGraph.addEdge(nodeA.getGraphData().getUid(),
                          nodeB.getGraphData().getUid(),
                          ab);
    }


    return mRawGraph.getEdgeProperty({nodeA.getGraphData().getUid(),
                                      nodeB.getGraphData().getUid()});
}

auto Graph::
    removeDependency(const GraphDependency& graphDependency)
        -> void
{
    helpCheckBackendStatus();
    helpInvalidateScheduling();

    const auto& nodeA = graphDependency.getSourceNode(*this);
    const auto& nodeB = graphDependency.getDestinationNode(*this);

    const auto nodeAuid = nodeA.getGraphData().getUid();
    const auto nodeBuid = nodeB.getGraphData().getUid();
    ;

    mRawGraph.removeEdge(nodeAuid,
                         nodeBuid);

    if (mRawGraph.outEdgesCount(nodeAuid) == 0) {
        this->addDependency(nodeA, getEndNode(), GraphDependencyType::data);
    }

    if (mRawGraph.inEdgesCount(nodeBuid) == 0) {
        this->addDependency(getBeginNode(), nodeB, GraphDependencyType::data);
    }
}


auto Graph::
    removeNode(GraphNode& gn)
        -> Container
{
    helpInvalidateScheduling();

    auto uidB = gn.getGraphData().getUid();

    if (uidB == getBeginNode().getGraphData().getUid() ||
        uidB == getEndNode().getGraphData().getUid()) {
        NeonException ex("");
        ex << "Begin or end nodes can not be removed";
        NEON_THROW(ex);
    }

    // a. get all in and our edges
    // b. connect each in node all the out nodes

    auto inNodes = mRawGraph.inNeighbors(uidB);
    auto outNodes = mRawGraph.outNeighbors(uidB);

    for (auto inNodeId : inNodes) {
        for (auto outNodeId : outNodes) {
            auto& inNode = mRawGraph.getVertexProperty(inNodeId);
            auto& outNode = mRawGraph.getVertexProperty(outNodeId);
            auto& inNodeDepWithTarget = mRawGraph.getEdgeProperty({inNodeId, uidB});
            this->addDependency(inNode, outNode, inNodeDepWithTarget.getType());
        }
    }

    for (auto inNodeId : inNodes) {
        mRawGraph.removeEdge({inNodeId, uidB});
    }
    for (auto outNodeId : outNodes) {
        mRawGraph.removeEdge({uidB, outNodeId});
    }

    mRawGraph.removeVertex(uidB);
    gn.getGraphData().setUid(GraphData::notSet);

    return gn.getContainer();
}

auto Graph::
    removeNodeAndItsDependencies(GraphNode& gn)
        -> Container
{
    auto uidB = gn.getGraphData().getUid();

    if (uidB == getBeginNode().getGraphData().getUid() ||
        uidB == getEndNode().getGraphData().getUid()) {
        NeonException ex("");
        ex << "Begin or end nodes can not be removed";
        NEON_THROW(ex);
    }

    helpInvalidateScheduling();

    auto inNodes = mRawGraph.inNeighbors(uidB);
    auto outNodes = mRawGraph.outNeighbors(uidB);

    for (auto inNodeId : inNodes) {
        auto& inNode = mRawGraph.getVertexProperty(inNodeId);
        if (inNodeId != getBeginNode().getGraphData().getUid()) {
            this->addDependency(getBeginNode(), inNode, GraphDependencyType::data);
        }
    }

    for (auto outNodeId : outNodes) {
        auto& outNode = mRawGraph.getVertexProperty(outNodeId);
        if (outNodeId != getEndNode().getGraphData().getUid()) {
            this->addDependency(outNode, getBeginNode(), GraphDependencyType::data);
        }
    }

    for (auto inNodeId : inNodes) {
        mRawGraph.removeEdge({inNodeId, uidB});
    }
    for (auto outNodeId : outNodes) {
        mRawGraph.removeEdge({uidB, outNodeId});
    }

    mRawGraph.removeVertex(uidB);
    gn.getGraphData().setUid(GraphData::notSet);

    return gn.getContainer();
}

auto Graph::
    getProceedingGraphNodes(const GraphNode&                        graphNode,
                            const std::vector<GraphDependencyType>& dependencyTypes)
        -> std::vector<GraphNode*>
{
    std::vector<GraphNode*> nodes;

    auto nodeUID = graphNode.getGraphData().getUid();
    auto nodeUidSet = mRawGraph.inNeighbors(nodeUID);
    for (auto&& proceedingUid : nodeUidSet) {
        const auto& connection = mRawGraph.getEdgeProperty({proceedingUid, nodeUID});
        bool        toBeReturned = false;

        for (auto& t : dependencyTypes) {
            if (t == connection.getType()) {
                toBeReturned = true;
                break;
            }
        }

        if (!toBeReturned) {
            continue;
        }

        auto& proceedingNode = mRawGraph.getVertexProperty(proceedingUid);
        nodes.push_back(&proceedingNode);
    }

    return nodes;
}

auto Graph::
    getSubsequentGraphNodes(const GraphNode&                        graphNode,
                            const std::vector<GraphDependencyType>& dependencyTypes)
        -> std::vector<GraphNode*>
{
    std::vector<GraphNode*> nodes;

    auto nodeUID = graphNode.getGraphData().getUid();
    auto nodeUidSet = mRawGraph.outNeighbors(nodeUID);
    for (auto&& subsequentUID : nodeUidSet) {
        const auto& connection = mRawGraph.getEdgeProperty({nodeUID, subsequentUID});
        bool        toBeReturned = false;

        for (auto& t : dependencyTypes) {
            if (t == connection.getType()) {
                toBeReturned = true;
                break;
            }
        }

        if (!toBeReturned) {
            continue;
        }
        auto& subsequentNode = mRawGraph.getVertexProperty(subsequentUID);
        nodes.push_back(&subsequentNode);
    }

    return nodes;
}

auto Graph::
    cloneNode(const GraphNode& graphNode)
        -> GraphNode&
{
    helpInvalidateScheduling();

    auto proceedings = getProceedingGraphNodes(graphNode);
    auto subsequents = getSubsequentGraphNodes(graphNode);

    auto& newNode = addNode(graphNode.getContainer());

    for (const auto& proPtr : proceedings) {
        auto dependencyType = getDependencyType(*proPtr, graphNode);
        addDependency(*proPtr, newNode, dependencyType);
    }

    for (const auto& post : subsequents) {
        auto dependencyType = getDependencyType(graphNode, *post);
        addDependency(newNode, *post, dependencyType);
    }
    return newNode;
}

auto Graph::
    getDependencyType(const GraphNode& nodeA,
                      const GraphNode& nodeB)
        -> GraphDependencyType
{
    auto uidA = nodeA.getGraphData().getUid();
    auto uidB = nodeB.getGraphData().getUid();

    auto edgePropetry = mRawGraph.getEdgeProperty({uidA, uidB});
    auto dependencyType = edgePropetry.getType();

    return dependencyType;
}

auto Graph::
    getDependency(const GraphNode& nodeA,
                  const GraphNode& nodeB)
        const -> const GraphDependency&
{
    auto uidA = nodeA.getGraphData().getUid();
    auto uidB = nodeB.getGraphData().getUid();

    const auto& dependency = mRawGraph.getEdgeProperty({uidA, uidB});

    return dependency;
}

auto Graph::
    getMutableDependency(const GraphNode& nodeA,
                         const GraphNode& nodeB)
        -> GraphDependency&
{
    auto uidA = nodeA.getGraphData().getUid();
    auto uidB = nodeB.getGraphData().getUid();

    auto& dependency = mRawGraph.getEdgeProperty({uidA, uidB});

    return dependency;
}

auto Graph::
    helpInvalidateScheduling()
        -> void
{
    mSchedulingStatusIsValid = false;
}

auto Graph::
    removeRedundantDependencies()
        -> void
{
    // Vectors of edges to be removed
    std::vector<std::pair<size_t, size_t>> edgesToBeRemoved;
    mRawGraph.forEachVertex([&](size_t diGraphNodeId) {
        // In this body we are looping over all nodes
        // For each node do:

        // Check node's children
        auto        visitingNode = this->helpGetGraphNode(diGraphNodeId);
        const auto& childrenPtr = this->getSubsequentGraphNodes(visitingNode, {GraphDependencyType::user,
                                                                               GraphDependencyType::data});
        if (childrenPtr.size() <= 1) {
            // If no more than one, move to the next node
            // Nothing to do for the visiting node as there are no redundant paths
            // Let's move to another
            return;
        }
        // Start checking for redundant paths
        for (const auto& targetChildPtr : childrenPtr) {
            if (this->getProceedingGraphNodes(*targetChildPtr, {GraphDependencyType::user,
                                                                GraphDependencyType::data})
                    .size() <= 1) {
                //                helpGetInEdges(targetChild, false).size() <= 1) {
                // This targetChild can only be reached by one father
                // No redundant path here
                continue;
            }
            bool foundRedundant = false;
            // Checking all siblings' paths.
            // We are looking for the targetChild in the path of any of its siblings.
            // A BFS visit is used for the process.
            for (const auto& targetSiblingPtr : childrenPtr) {
                if (targetSiblingPtr->getGraphData().getUid() ==
                    targetChildPtr->getGraphData().getUid()) {
                    continue;
                }
                // first BFS frontier are the targetSibling's child
                auto frontierOfUid = [&] {
                    auto             vec = this->getSubsequentGraphNodes(*targetSiblingPtr);
                    std::set<size_t> output;
                    for (auto v : vec) {
                        output.insert(v->getGraphData().getUid());
                    }
                    return output;
                }();
                while (!frontierOfUid.empty()) {
                    auto nextFrontier = std::set<size_t>();

                    for (const auto& nodeUidInFrontier : frontierOfUid) {
                        if (nodeUidInFrontier == targetChildPtr->getGraphData().getUid()) {
                            // We have found a redundant path
                            foundRedundant = true;
                            break;
                        } else {
                            // Let's continue by updating the frontier for the next iteration
                            for (auto nodeFrontierForNextIteration : this->getSubsequentGraphNodes(this->helpGetGraphNode(nodeUidInFrontier))) {
                                nextFrontier.insert(nodeFrontierForNextIteration->getGraphData().getUid());
                            }
                        }
                    }

                    if (foundRedundant) {
                        // A longer and altenrative path has been found.
                        break;
                    }
                    frontierOfUid = nextFrontier;
                }
                if (foundRedundant)
                    break;
            }
            if (foundRedundant) {
                edgesToBeRemoved.push_back({visitingNode.getGraphData().getUid(),
                                            targetChildPtr->getGraphData().getUid()});
            }
        }
    });
    for (const auto& toBeRemoved : edgesToBeRemoved) {
        mRawGraph.removeEdge(toBeRemoved);
    }
}

auto Graph::
    helpGetOutNeighbors(GraphData::Uid                          nodeUid,
                        bool                                    filteredOut,
                        const std::vector<GraphDependencyType>& dependencyTypes)
        -> std::set<GraphData::Uid>
{
    std::set<GraphData::Uid> outNgh;
    mRawGraph.forEachOutEdge(
        nodeUid,
        [&](std::pair<GraphData::Uid, GraphData::Uid> edge) {
            auto& edgeProp = mRawGraph.getEdgeProperty(edge);
            for (auto& depType : dependencyTypes) {
                if (depType == edgeProp.getType()) {
                    bool isAnchor = mRawGraph.getVertexProperty(edge.second).getContainerOperationType() == Neon::set::ContainerOperationType::anchor;
                    if (isAnchor && filteredOut) {
                        break;
                    }
                    outNgh.insert(edge.second);
                }
            }
        });
    return outNgh;
}

auto Graph::
    helpGetInNeighbors(GraphData::Uid                          nodeUid,
                       bool                                    filterOutBegin,
                       const std::vector<GraphDependencyType>& dependencyTypes)
        -> std::set<GraphData::Uid>
{
    std::set<GraphData::Uid> inNgh;
    mRawGraph.forEachInEdge(
        nodeUid,
        [&](std::pair<GraphData::Uid, GraphData::Uid> edge) {
            auto& edgeProp = mRawGraph.getEdgeProperty(edge);
            for (auto& depType : dependencyTypes) {
                if (depType == edgeProp.getType()) {
                    if (mRawGraph.getVertexProperty(edge.second).getContainerOperationType() == Neon::set::ContainerOperationType::anchor && filterOutBegin) {
                        break;
                    }
                    inNgh.insert(edge.first);
                }
            }
        });
    return inNgh;
}

auto Graph::
    helpGetOutEdges(GraphData::Uid                          nodeUid,
                    bool                                    filterOutEnd,
                    const std::vector<GraphDependencyType>& dependencyTypes)
        -> std::set<std::pair<GraphData::Uid, GraphData::Uid>>
{
    std::set<std::pair<GraphData::Uid, GraphData::Uid>> outEdges;
    mRawGraph.forEachOutEdge(
        nodeUid,
        [&](std::pair<GraphData::Uid, GraphData::Uid> edge) {
            auto& edgeProp = mRawGraph.getEdgeProperty(edge);
            for (auto& depType : dependencyTypes) {
                if (depType == edgeProp.getType()) {
                    if (mRawGraph.getVertexProperty(edge.second).getContainerOperationType() == Neon::set::ContainerOperationType::anchor && filterOutEnd) {
                        break;
                    }
                    outEdges.insert(edge);
                }
            }
        });
    return outEdges;
}

auto Graph::
    helpGetInEdges(GraphData::Uid                          nodeUid,
                   bool                                    filterOutBegin,
                   const std::vector<GraphDependencyType>& dependencyTypes)
        -> std::set<std::pair<GraphData::Uid, GraphData::Uid>>
{
    std::set<std::pair<GraphData::Uid, GraphData::Uid>> inEdges;
    mRawGraph.forEachInEdge(
        nodeUid,
        [&](std::pair<GraphData::Uid, GraphData::Uid> edge) {
            auto& edgeProp = mRawGraph.getEdgeProperty(edge);
            for (auto& depType : dependencyTypes) {
                if (depType == edgeProp.getType()) {
                    if (mRawGraph.getVertexProperty(edge.second).getContainerOperationType() == Neon::set::ContainerOperationType::anchor && filterOutBegin) {
                        break;
                    }
                    inEdges.insert(edge);
                }
            }
        });
    return inEdges;
}

auto Graph::
    helpGetBFS(bool                                    filterOutBeginEnd,
               const std::vector<GraphDependencyType>& dependencyTypes)
        -> Bfs
{
    using Frontier = std::unordered_map<GraphData::Uid, size_t>;

    Bfs bfs;

    Frontier a;
    Frontier b;
    Frontier tmp;

    Frontier* currentFrontier = &a;
    Frontier* nextFrontier = &b;

    [[maybe_unused]] auto printFrontier = [](const Frontier* frontier,
                                             std::string     prefix = "") {
        std::stringstream s;
        s << "[";
        s << prefix << " ";
        for (auto& el : *frontier) {
            auto [nodeId, counter] = el;
            s << "(" << nodeId << "," << counter << ")";
        }
        s << "]";
        std::cout << s.str() << std::endl;
    };

    if (!filterOutBeginEnd) {
        currentFrontier->insert({getBeginNode().getGraphData().getUid(), 0});
    } else {
        auto beginNode = getBeginNode().getGraphData().getUid();
        auto outNgh = helpGetOutNeighbors(beginNode, filterOutBeginEnd, dependencyTypes);
        for (const auto& ngh : outNgh) {
            size_t stillUnsatisfiedDependencies = helpGetInEdges(ngh).size() - 1;
            currentFrontier->insert({ngh, stillUnsatisfiedDependencies});
        }
    }

    while (currentFrontier->size() != 0) {
        auto [newLevel, newLevelIdx] = bfs.getNewLevel();
        nextFrontier->clear();
        tmp.clear();

        {  // all nodes with a counter higher than 1 are moved directly to the next frontier
            for (auto& currentFrontierNode : *currentFrontier) {
                if (currentFrontierNode.second == 0) {
                    tmp.insert(currentFrontierNode);
                } else {
                    nextFrontier->insert(currentFrontierNode);
                }
            }
            *currentFrontier = tmp;
        }

        // printFrontier(currentFrontier, "currentFrontier");
        // printFrontier(nextFrontier, "nextFrontier");
        for (auto& currentFrontierNode : *currentFrontier) {
            // At this any currentFrontierNode has .second ==0;
            // printFrontier(currentFrontier, "currentFrontier");
            // std::cout << "selecting " << currentFrontierNode.first << std::endl;

            // All incoming dependencies have been resolved.
            // The node is ready to be added to the current BFS level
            newLevel.push_back(currentFrontierNode.first);

            // Adding the node's children to the next frontier
            auto currentFrontierNodeChildren = helpGetOutNeighbors(currentFrontierNode.first,
                                                                   filterOutBeginEnd, dependencyTypes);
            for (const auto& child : currentFrontierNodeChildren) {
                auto it = std::find_if(nextFrontier->begin(), nextFrontier->end(),
                                       [&child](auto& entry) {
                                           return (entry.first == child);
                                       });
                if (it != nextFrontier->end()) {
                    it->second--;
                } else {
                    auto numChildDependencies = helpGetInEdges(child, filterOutBeginEnd, dependencyTypes).size();
                    nextFrontier->insert({child, numChildDependencies - 1});
                }
            }
            // printFrontier(nextFrontier, "nextFrontier");
        }
        std::swap(currentFrontier, nextFrontier);
    }

    return bfs;
}

auto Graph::
    helpComputeScheduling(bool filterOutAnchors, int anchorStream)
        -> void
{
    helpComputeScheduling_00_resetData();
    mBfs = helpComputeScheduling_01_generatingBFS(filterOutAnchors);
    int maxStreamId = helpComputeScheduling_02_mappingStreams(mBfs, filterOutAnchors, anchorStream);
    int maxEventId = helpComputeScheduling_03_events(mBfs);
    helpComputeScheduling_04_ensureResources(maxStreamId, maxEventId);
    helpComputeScheduling_05_executionOrder(filterOutAnchors, mBfs);
}

auto Graph::
    helpComputeScheduling_01_generatingBFS(bool filterOutAnchors)
        -> Bfs
{
    return helpGetBFS(filterOutAnchors, {GraphDependencyType::data,
                                         GraphDependencyType::user});
}

auto Graph::helpGetGraphNode(GraphData::Uid uid) -> GraphNode&
{
    return mRawGraph.getVertexProperty(uid);
}

auto Graph::helpGetGraphNode(GraphData::Uid uid) const -> const GraphNode&
{
    return mRawGraph.getVertexProperty(uid);
}

auto Graph::helpComputeScheduling_00_resetData() -> void
{
    mRawGraph.forEachVertex([&](const GraphData::Uid& graphNodeId) {
        auto& targetNode = mRawGraph.getVertexProperty(graphNodeId);
        targetNode.getScheduling().reset();
    });
}

auto Graph::
    helpComputeScheduling_02_mappingStreams(Bfs& bfs,
                                            bool filterOutAnchors,
                                            int  anchorStream)
        -> int
{
    // used stream -> true
    // available -> false
    constexpr bool usedStream = true;
    constexpr bool availableStream = false;

    const auto bookFirstAvailableStream = [&](std::vector<bool>& streamsStatus) {
        for (int i = 0; i < int(streamsStatus.size()); i++) {
            if (availableStream == streamsStatus.at(i)) {
                streamsStatus[i] = usedStream;
                return i;
            }
        }
        NEON_THROW_UNSUPPORTED_OPERATION("");
    };

    const auto bookStream = [&](std::vector<bool>& streamsStatus,
                               int                streamId)
        -> bool {
        if (streamsStatus.at(streamId) == availableStream) {
            streamsStatus[streamId] = usedStream;
            return true;
        }
        return false;
    };

    const auto resetBooking = [&](std::vector<bool>& streamsStatus)
        -> void {
        for (auto&& entry : streamsStatus) {
            entry = availableStream;
        }
    };

    mSchedulingStatusIsValid = true;
    mMaxNumberStreams = bfs.getMaxLevelWidth();


    std::vector<bool> streamsStatus(mMaxNumberStreams,
                                    availableStream);


    // The stream mapping process is computer per level.
    // On each level we have two steps
    // a. map nodes with the streams of one of the proceeding nodes when possible
    // b. map the remaining nodes by mapping the first available stream
    bfs.forEachLevel([&](const Bfs::Level& level,
                         int               levelIdx) {
        std::vector<GraphNode*> delayedBooking;

        if (!filterOutAnchors && levelIdx == 0 && anchorStream > -1) {
            // If anchors are represented than
            // the first level contains only the begin node
            bool checkA = getBeginNode().getGraphData().getUid() == level[0];
            bool checkB = level.size() == 1;
            if (!checkA || !checkB) {
                Neon::NeonException ex("");
                ex << "Inconsistent status of the container graph.";
                NEON_THROW(ex);
            }
            this->helpGetGraphNode(level[0]).getScheduling().setStream(anchorStream);
            return;
        }
        // Step a.
        for (auto& nodeUid : level) {
            auto& node = this->helpGetGraphNode(nodeUid);

            const int helpAssociatedStream = [&] {
                if (levelIdx == 0) {
                    return bookFirstAvailableStream(streamsStatus);
                }
                const auto preNodePtrs = this->getProceedingGraphNodes(node);
                for (auto& preNode : preNodePtrs) {
                    auto preNodeStream = preNode->getScheduling().getStream();
                    if (bookStream(streamsStatus, preNodeStream)) {
                        return preNodeStream;
                    }
                }
                return -1;
            }();

            if (helpAssociatedStream == -1) {
                // track nodes that need to go through step 2
                delayedBooking.push_back(&node);
            } else {
                node.getScheduling().setStream(helpAssociatedStream);
            }
        }

        // Step b.
        for (auto nodePrt : delayedBooking) {
            auto associatedStream = bookFirstAvailableStream(streamsStatus);
            nodePrt->getScheduling().setStream(associatedStream);
        }

        resetBooking(streamsStatus);
    });

    if (!filterOutAnchors && anchorStream > -1) {
        auto  endNodeId = getEndNode().getGraphData().getUid();
        auto& endNode = mRawGraph.getVertexProperty(endNodeId);
        endNode.getScheduling().setStream(anchorStream);
    }

    return mMaxNumberStreams - 1;
}

auto Graph::
    helpComputeScheduling_03_events(Bfs& bfs)
        -> int
{
    int eventCount = 0;

    bfs.forEachNodeByLevel(*this, [&](GraphNode& targetNode, int /*levelId*/) {
        int targetStreamId = targetNode.getScheduling().getStream();

        auto preNodeIds = this->getProceedingGraphNodes(targetNode);
        if (preNodeIds.empty()) {
            // this is the first node, no need to have dependencies
            return;
        }
        // Processing PreNodes on different streams than the node
        std::vector<GraphNode*> preNodesOnDifferentStreams;
        for (auto* preNode : preNodeIds) {
            auto preNodeStream = preNode->getScheduling().getStream();
            if (preNodeStream == targetStreamId) {
                continue;
            }
            // For pre-nodes on different streams we do the following
            // a. check if the pre-node has an event associated, if not create one
            // b. add the pre-node event to the list of the target node

            // a.
            int preNodeEvent = preNode->getScheduling().getEvent();
            if (preNodeEvent == -1) {
                preNodeEvent = eventCount;
                preNode->getScheduling().setEvent(preNodeEvent);
                eventCount++;
            }
            // b.
            targetNode.getScheduling().getDependentEvents().push_back(preNodeEvent);
        }
    });

    return eventCount - 1;
}

auto Graph::
    helpComputeScheduling_04_ensureResources(int maxStreamId,
                                             int maxEventId)
        -> void
{
    auto bk = getBackend();
    bk.setAvailableStreamSet(maxStreamId + 1);
    bk.setAvailableUserEvents(maxEventId + 1);
}

auto Graph::
    ioToDot(const std::string& fname,
            const std::string& graphName,
            bool               debug)
        -> void
{
    this->removeRedundantDependencies();

    auto vertexLabel = [&](size_t v) -> std::string {
        auto& node = mRawGraph.getVertexProperty(v);
        return node.getLabel(debug);
    };

    auto edgeLabel = [&](const std::pair<size_t, size_t>& edge)
        -> std::string {
        auto& edgeMeta = mRawGraph.getEdgeProperty(edge);
        return edgeMeta.getLabel();
    };

    auto edgeLabelProperty = [&](const std::pair<size_t, size_t>& edge)
        -> std::string {
        auto& edgeMeta = mRawGraph.getEdgeProperty(edge);

        if (edgeMeta.getType() == GraphDependencyType::scheduling) {
            // return "style=dashed, color=\"#2A9D8F\"";
            return "style=dashed, color=\"#F4A261\", penwidth=3";
        }
        return " penwidth=3";
    };

    auto vertexLabelProperty = [&](const size_t& v) {
        auto& node = mRawGraph.getVertexProperty(v);
        return node.getLabelProperty();
    };
    mRawGraph.exportDotFile(fname + ".dot", graphName, vertexLabel, edgeLabel,
                            vertexLabelProperty, edgeLabelProperty);
}

auto Graph::helpComputeScheduling_05_executionOrder(bool filterOutAnchors, Bfs& bfs) -> void
{
    bfs = helpGetBFS(filterOutAnchors, {GraphDependencyType::data,
                                        GraphDependencyType::user,
                                        GraphDependencyType::scheduling});

    int levels = bfs.getNumberOfLevels();
    int execOrder = 0;
    for (int i = 0; i < levels; i++) {
        bfs.forEachNodeAtLevel(i, *this, [&](Neon::set::container::GraphNode& graphNode) {
            graphNode.getScheduling().setExecutionOerder(execOrder);
            execOrder++;
        });
    }
}

Graph::
    Graph(const Backend& bk)
{
    mBackend = bk;
    mBackendIsSet = true;

    auto begin = GraphNode::newBeginNode();
    auto end = GraphNode::newEndNode();
    mUidCounter = GraphData::firstInternal;

    mRawGraph.addVertex(begin.getGraphData().getUid(), begin);
    mRawGraph.addVertex(end.getGraphData().getUid(), end);

    helpInvalidateScheduling();
}

auto Graph::
    getBackend()
        const -> const Neon::Backend&
{
    if (mBackendIsSet) {
        return mBackend;
    }
    NeonException ex("Graph");
    ex << "A backend was not set.";
    NEON_THROW(ex);
}

auto Graph::
    run(Neon::SetIdx   setIdx,
        int            streamIdx,
        Neon::DataView dataView)
        -> void
{
    if (dataView != Neon::DataView::STANDARD) {
        NEON_THROW_UNSUPPORTED_OPERATION("");
    }

    this->runtimePreSet(streamIdx);
    this->helpExecute(setIdx, streamIdx);
}

auto Graph::
    run(int            streamIdx,
        Neon::DataView dataView)
        -> void
{
    if (dataView != Neon::DataView::STANDARD) {
        NEON_THROW_UNSUPPORTED_OPERATION("");
    }

    this->runtimePreSet(streamIdx);
    this->helpExecuteWithOmpAtNodeLevel(streamIdx);
}

auto Graph::
    helpExecute(Neon::SetIdx setIdx,
                int          anchorStream)
        -> void
{
    if (anchorStream > -1 && (anchorStream != mAnchorStreamPreSet || mFilterOutAnchorsPreSet == true)) {
        Neon::NeonException ex("");
        ex << "Execution parameters are inconsistent with the preset ones.";
        NEON_THROW(ex);
    }

    int levels = mBfs.getNumberOfLevels();
    for (int i = 0; i < levels; i++) {
        mBfs.forEachNodeAtLevel(i, *this, [&](Neon::set::container::GraphNode& graphNode) {
            auto&       scheduling = graphNode.getScheduling();
            auto&       container = graphNode.getContainer();
            const auto& waitingEvents = scheduling.getDependentEvents();
            const auto& signalEvents = scheduling.getEvent();
            const auto& stream = scheduling.getStream();

            for (auto toBeWaited : waitingEvents) {
                mBackend.waitEventOnStream(setIdx, toBeWaited, stream);
            }
            container.run(setIdx, stream, scheduling.getDataView());

            if (signalEvents >= 0) {
                mBackend.pushEventOnStream(setIdx, signalEvents, stream);
            }
        });
    }
}

auto Graph::
    helpExecuteWithOmpAtNodeLevel(int anchorStream)
        -> void
{
    if (anchorStream > -1 && (anchorStream != mAnchorStreamPreSet || mFilterOutAnchorsPreSet == true)) {
        Neon::NeonException ex("");
        ex << "Execution parameters are inconsistent with the preset ones.";
        NEON_THROW(ex);
    }

#ifdef NEON_USE_NVTX
    nvtxRangePush("Graph Iteration");
#endif
    int levels = mBfs.getNumberOfLevels();
    for (int i = 0; i < levels; i++) {
        mBfs.forEachNodeAtLevel(i, *this, [&](Neon::set::container::GraphNode& graphNode) {
            auto&       scheduling = graphNode.getScheduling();
            auto&       container = graphNode.getContainer();
            const auto& waitingEvents = scheduling.getDependentEvents();
            const auto& signalEvents = scheduling.getEvent();
            const auto& stream = scheduling.getStream();

            for (auto toBeWaited : waitingEvents) {
                mBackend.waitEventOnStream(toBeWaited, stream);
            }
#ifdef NEON_USE_NVTX
            nvtxRangePush((std::string("Container Run") + container.getName()).c_str());
#endif
            container.run(stream, scheduling.getDataView());
#ifdef NEON_USE_NVTX
            nvtxRangePop();
#endif
            if (signalEvents >= 0) {
                mBackend.pushEventOnStream(signalEvents, stream);
            }
        });
    }
#ifdef NEON_USE_NVTX
    nvtxRangePop();
#endif

    // #ifdef NEON_USE_NVTX
    //     nvtxRangePush("Graph Iteration");
    // #endif
    //
    //     const int setCardinality = mBackend.devSet().setCardinality();
    //
    //     int levels = mBfs.getNumberOfLevels();
    // #pragma omp parallel num_threads(setCardinality)
    //     {
    //         const int threadRank = omp_get_thread_num();
    //
    //         for (int i = 0; i < levels; i++) {
    //             mBfs.forEachNodeAtLevel(i, *this, [&](Neon::set::container::GraphNode& graphNode) {
    //                 auto&       scheduling = graphNode.getScheduling();
    //                 auto&       container = graphNode.getContainer();
    //                 const auto& waitingEvents = scheduling.getDependentEvents();
    //                 const auto& signalEvents = scheduling.getEvent();
    //                 const auto& stream = scheduling.getStream();
    //
    //                 for (auto toBeWaited : waitingEvents) {
    //                     mBackend.waitEventOnStream(threadRank, toBeWaited, stream);
    //                 }
    // #ifdef NEON_USE_NVTX
    //                 nvtxRangePush((container.getName()+Neon::DataViewUtil::toString(scheduling.getDataView())).c_str());
    // #endif
    //                 container.run(threadRank, stream, scheduling.getDataView());
    // #ifdef NEON_USE_NVTX
    //                 nvtxRangePop();
    // #endif
    //                 if (signalEvents >= 0) {
    //                     mBackend.pushEventOnStream(threadRank, signalEvents, stream);
    //                 }
    //             });
    //         }
    //     }
    // #ifdef NEON_USE_NVTX
    //     nvtxRangePop();
    // #endif
}

auto Graph::runtimePreSet(int anchorStream) -> void
{
    helpComputeScheduling(false, anchorStream);
    mAnchorStreamPreSet = anchorStream;
    mFilterOutAnchorsPreSet = false;
    for (int i = 0; i < mBfs.getNumberOfLevels(); i++) {
        mBfs.forEachNodeAtLevel(i, *this, [&](Neon::set::container::GraphNode& graphNode) {
            graphNode.getContainer().getContainerInterface().configureWithScheduling(graphNode);
        });
    }
}

auto Graph::expandSubGraphs() -> void
{
    bool atLeastOneWasFound = false;

    auto searchForSubGraph = [&]() -> std::tuple<size_t, bool> {
        bool   found = false;
        size_t target = 0;
        mRawGraph.forEachVertex([&](size_t id) {
            auto& node = mRawGraph.getVertexProperty(id);
            if (node.getContainer().getContainerExecutionType() == ContainerExecutionType::graph) {
                target = id;
                found = true;
                atLeastOneWasFound = true;
            }
        });
        return {target, found};
    };
    int i = -1;
    while (true) {
        i++;
        auto [newTargetId, validTarget] = searchForSubGraph();
        if (!validTarget) {
            if (atLeastOneWasFound) {
                helpInvalidateScheduling();
                this->removeRedundantDependencies();
                // this->ioToDot(std::to_string(i) + "_t_05", "kllkj", true);
            }
            return;
        }
        // this->ioToDot(std::to_string(i) + "_t_00", "kllkj", true);

        auto&       newTarget = mRawGraph.getVertexProperty(newTargetId);
        const auto& subGraph = newTarget.getContainer().getContainerInterface().getGraph();

        std::unordered_map<size_t, size_t> fromOldToNew;

        // Cloning the subGraph nodes into the graph
        subGraph.mRawGraph.forEachVertex([&](size_t id) {
            const auto& oldNode = subGraph.mRawGraph.getVertexProperty(id);
            const auto& newNode = this->addNode(oldNode.getContainer());
            fromOldToNew.insert(std::pair<size_t, size_t>(oldNode.getGraphData().getUid(),
                                                          newNode.getGraphData().getUid()));
        });
        /// this->ioToDot(std::to_string(i) + "_t_01", "kllkj", true);

        // Cloning the subGraph edges into the graph
        subGraph.mRawGraph.forEachEdge([&](std::pair<size_t, size_t> edge) {
            const auto& oldDep = subGraph.mRawGraph.getEdgeProperty(edge);
            size_t      newA = fromOldToNew.at(edge.first);
            size_t      newB = fromOldToNew.at(edge.second);

            const auto& nodeA = mRawGraph.getVertexProperty(newA);
            const auto& nodeB = mRawGraph.getVertexProperty(newB);

            this->addDependency(nodeA, nodeB, oldDep.getType());
        });
        // this->ioToDot(std::to_string(i) + "_t_02", "kllkj", true);

        {  // Removing the cloned begin and end from the subGraph
            auto oldBeginId = subGraph.getBeginNode().getGraphData().getUid();
            auto oldEndId = subGraph.getEndNode().getGraphData().getUid();

            auto toBeRemovedBeginId = fromOldToNew.at(oldBeginId);
            auto toBeRemovedEndId = fromOldToNew.at(oldEndId);

            auto& toBeRemovedBeginNode = mRawGraph.getVertexProperty(toBeRemovedBeginId);
            auto& toBeRemovedEndNode = mRawGraph.getVertexProperty(toBeRemovedEndId);

            this->removeNode(toBeRemovedBeginNode);
            this->removeNode(toBeRemovedEndNode);
        }
        // this->ioToDot(std::to_string(i) + "_t_03", "kllkj", true);

        {  // Removing subGraph and depedencies

            this->removeNodeAndItsDependencies(newTarget);
        }
        // this->ioToDot(std::to_string(i) + "_t_04", "kllkj", true);
    }
}
auto Graph::
    helpCheckBackendStatus()
        -> void
{
    if (!mBackendIsSet) {
        NeonException exception("Container Graph");
        exception << "A backend was not registered with the Graph";
        NEON_THROW(exception);
    }
}

auto Graph::
    getNumberOfNodes()
        -> int
{
    // We remove 2 because of the head and tail holders.
    return int(mRawGraph.numVertices() - 2);
}

auto Graph::
    forEachDependency(const std::function<void(const GraphDependency&)>& fun)
        const -> void
{
    mRawGraph.forEachEdge([&](const auto& edge) {
        const auto& dep = this->mRawGraph.getEdgeProperty(edge);
        fun(dep);
    });
}

auto Graph::
    forEachNode(const std::function<void(const GraphNode&)>& fun)
        const -> void
{
    mRawGraph.forEachVertex([&](size_t nodeId) {
        auto node = this->mRawGraph.getVertexProperty(nodeId);
        fun(node);
    });
}

auto Graph::
    expandAndMerge(const GraphNode& A,
                   const Container& container,
                   const GraphNode& B,
                   bool             propagateSchedulingHints)
        -> int
{
    if (container.getContainerExecutionType() != ContainerExecutionType::graph) {
        NEON_THROW_UNSUPPORTED_OPERATION("Container should be of type Graph");
    }

    Graph toMergeGraph = container.getContainerInterface().getGraph();
    toMergeGraph.expandSubGraphs();
    int numNodes = toMergeGraph.getNumberOfNodes();

    auto schedulingDepOfB = getProceedingGraphNodes(B, {GraphDependencyType::scheduling});

    std::unordered_map<size_t, size_t> fromInputToLocal;

    auto isInternalNode = [&](const GraphNode& inputNode) {
        auto inputUid = inputNode.getGraphData().getUid();
        if (GraphData::beginUid != inputUid &&
            GraphData::endUid != inputUid) {
            return true;
        }
        return false;
    };

    auto isInternalDependency = [&](const GraphDependency& inputDependency) {
        const auto& source = inputDependency.getSourceNode(toMergeGraph);
        const auto& dest = inputDependency.getDestinationNode(toMergeGraph);

        if (isInternalNode(source) &&
            isInternalNode(dest)) {
            return true;
        }
        return false;
    };

    // adding nodes to the graph
    toMergeGraph.forEachNode([&](const GraphNode& inputNode) {
        // discarding begin and end nodes
        if (isInternalNode(inputNode)) {
            auto& container = inputNode.getContainer();
            auto& newGraphNode = this->addNode(container);

            fromInputToLocal[inputNode.getGraphData().getUid()] =
                newGraphNode.getGraphData().getUid();
        }
    });

    // adding dependency to the graph
    toMergeGraph.forEachDependency([&](const GraphDependency& inputDependency) {
        // discarding begin and end nodes
        if (isInternalDependency(inputDependency)) {
            const auto& sourceInput = inputDependency.getSourceNode(toMergeGraph);
            const auto& destInput = inputDependency.getDestinationNode(toMergeGraph);

            size_t sourceUid = fromInputToLocal[sourceInput.getGraphData().getUid()];
            size_t destUid = fromInputToLocal[destInput.getGraphData().getUid()];

            const GraphNode& source = this->helpGetGraphNode(sourceUid);
            const GraphNode& dest = this->helpGetGraphNode(destUid);

            this->addDependency(source, dest, inputDependency.getType());
        }
    });


    {  // Connecting node A with all the children of the input graph begin node
        // And propagating the scheduling dependencies of B too
        const auto&             inputBeginNode = toMergeGraph.getBeginNode();
        std::vector<GraphNode*> subsequentGraphNodes = toMergeGraph.getSubsequentGraphNodes(inputBeginNode);

        for (auto inputGraphNodePtr : subsequentGraphNodes) {
            const auto&  source = A;
            const size_t destUid = fromInputToLocal[inputGraphNodePtr->getGraphData().getUid()];
            auto         dest = this->helpGetGraphNode(destUid);
            const auto&  inputDependency = toMergeGraph.getDependency(inputBeginNode, *inputGraphNodePtr);
            this->addDependency(source, dest, inputDependency.getType());

            if (propagateSchedulingHints) {
                for (auto& schedulingAnteNode : schedulingDepOfB) {
                    this->addDependency(*schedulingAnteNode, dest, GraphDependencyType::scheduling);
                }
            }
        }
    }


    {  // Connecting node B with all the parents of the input graph end node
        const auto&             inputEndNode = toMergeGraph.getEndNode();
        std::vector<GraphNode*> proceedingGraphNodes = toMergeGraph.getProceedingGraphNodes(inputEndNode);
        for (auto inputGraphNodePtr : proceedingGraphNodes) {
            const size_t sourceUid = fromInputToLocal[inputGraphNodePtr->getGraphData().getUid()];

            const auto& source = this->helpGetGraphNode(sourceUid);
            const auto& dest = B;

            const auto& inputDependency = toMergeGraph.getDependency(*inputGraphNodePtr, inputEndNode);
            this->addDependency(source, dest, inputDependency.getType());
        }
    }

    return numNodes;
}


}  // namespace Neon::set::container
