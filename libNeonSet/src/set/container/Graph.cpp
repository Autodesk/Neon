#include "Neon/set/container/Graph.h"
#include <algorithm>
#include "Neon/set/container/graph/Bfs.h"

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

auto Graph::getBeginNode() const -> const GraphNode&
{
    return mRawGraph.getVertexProperty(GraphData::beginUid);
}

auto Graph::getEndNode() const -> const GraphNode&
{
    return mRawGraph.getVertexProperty(GraphData::endUid);
}

auto Graph::addNodeInBetween(const GraphNode&          nodeA,
                             Container                 containerB,
                             const GraphNode&          nodeC,
                             const GraphDependencyType ab,
                             const GraphDependencyType bc) -> GraphNode&
{
    helpCheckBackendStatus();
    helpInvalidateScheduling();

    auto& nodeB = addNode(containerB);
    addDependency(nodeA, nodeB, ab);
    addDependency(nodeB, nodeC, bc);
    return nodeB;
}

auto Graph::addNode(const Container& container) -> GraphNode&
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

auto Graph::addDependency(const GraphNode&    nodeA,
                          const GraphNode&    nodeB,
                          GraphDependencyType type) -> GraphDependency&
{
    helpCheckBackendStatus();
    if (nodeA.getGraphData().getUid() == nodeB.getGraphData().getUid()) {
        NEON_THROW_UNSUPPORTED_OPERATION("");
    }
    helpInvalidateScheduling();

    auto nodeAuid = nodeA.getGraphData().getUid();
    auto nodeBuid = nodeB.getGraphData().getUid();

    GraphDependency ab(type, nodeAuid, nodeBuid);

    mRawGraph.addEdge(nodeAuid, nodeBuid, ab);

    return mRawGraph.getEdgeProperty({nodeAuid,
                                      nodeBuid});
}

auto Graph::appendDataDependency(const GraphNode&                                   nodeA,
                                 const GraphNode&                                   nodeB,
                                 Neon::internal::dataDependency::DataDependencyType dataDependencyType,
                                 size_t                                             dataUId,
                                 Compute                                            compute) -> GraphDependency&
{
    helpCheckBackendStatus();

    if (nodeA.getGraphData().getUid() == nodeB.getGraphData().getUid()) {
        NEON_THROW_UNSUPPORTED_OPERATION("");
    }
    helpInvalidateScheduling();


    bool hasEdge = mRawGraph.hasEdge(nodeA.getGraphData().getUid(),
                                     nodeB.getGraphData().getUid());

    if (!hasEdge) {
        GraphDependencyType type = GraphDependencyType::data;
        addDependency(nodeA, nodeB, type);
    }

    auto& output = mRawGraph.getEdgeProperty({nodeA.getGraphData().getUid(),
                                              nodeB.getGraphData().getUid()});

    output.appendInfo(dataDependencyType, dataUId, compute);

    return output;
}

auto Graph::removeNode(GraphNode& gn) -> Container
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

auto Graph::removeNodeAndItsDependencies(GraphNode& gn)
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

auto Graph::getProceedingGraphNodes(const GraphNode&                        graphNode,
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

auto Graph::getSubsequentGraphNodes(const GraphNode&                        graphNode,
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

auto Graph::cloneNode(const GraphNode& graphNode)
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

auto Graph::getDependencyType(const GraphNode& nodeA,
                              const GraphNode& nodeB)
    -> GraphDependencyType
{
    auto dependency = getDependency(nodeA, nodeB);
    auto dependencyType = dependency.getType();
    return dependencyType;
}

auto Graph::getDependency(const GraphNode& nodeA,
                          const GraphNode& nodeB) -> const GraphDependency&
{
    auto uidA = nodeA.getGraphData().getUid();
    auto uidB = nodeB.getGraphData().getUid();

    const auto& dependency = mRawGraph.getEdgeProperty({uidA, uidB});
    return dependency;
}

auto Graph::helpInvalidateScheduling()
    -> void
{
    mSchedulingStatusIsValid = false;
}

auto Graph::removeRedundantDependencies()
    -> void
{
    // Vectors of edges to be removed
    std::vector<std::pair<size_t, size_t>> edgesToBeRemoved;
    mRawGraph.forEachVertex([&](size_t diGraphNodeId) {
        // In this body we are looping over all nodes
        // For each node do:

        // Check node's childrenUid
        auto        visitingNodeUid = mRawGraph.getVertexProperty(diGraphNodeId).getGraphData().getUid();
        const auto& childrenUid = helpGetOutNeighbors(visitingNodeUid, false);
        if (childrenUid.size() <= 1) {
            // If no more than one, move to the next node
            // Nothing to do for the visiting node as there are no redundant paths
            // Let's move to another
            return;
        }
        // Start checking for redundant paths
        for (const auto& targetChildUid : childrenUid) {
            if (helpGetInEdges(targetChildUid, false).size() <= 1) {
                // This targetChildUid can only be reached by one father
                // No redundant path here
                continue;
            }
            bool foundRedundant = false;
            // Checking all siblings' paths.
            // We are looking for the targetChildUid in the path of any of its siblings.
            // A BFS visit is used for the process.
            for (const auto& targetSiblingUid : childrenUid) {
                if (targetSiblingUid == targetChildUid) {
                    continue;
                }
                // first BFS frontier are the targetSiblingUid's child
                auto frontier = mRawGraph.outNeighbors(targetSiblingUid);
                while (frontier.size() != 0) {
                    auto nextFrontier = std::set<size_t>();

                    for (const auto& nodeInFrontier : frontier) {
                        if (nodeInFrontier == targetChildUid) {
                            // We have found a redundant path
                            foundRedundant = true;
                            break;
                        } else {
                            // Let's continue by updating the frontier for the next iteration
                            for (auto nodeFrontierForNextIteration : helpGetOutNeighbors(nodeInFrontier)) {
                                nextFrontier.insert(nodeFrontierForNextIteration);
                            }
                        }
                    }

                    if (foundRedundant) {
                        // A longer and altenrative path has been found.
                        break;
                    }
                    frontier = nextFrontier;
                }
                if (foundRedundant)
                    break;
            }
            if (foundRedundant) {
                // Flag as redundant only if the edge is not of type stencil
                const bool isStencilEdge = std::invoke([&] {
                    const auto& visitingNode = mRawGraph.getVertexProperty(visitingNodeUid);
                    const auto& targetChild = mRawGraph.getVertexProperty(targetChildUid);
                    const auto& dependency = this->getDependency(visitingNode, targetChild);
                    const bool  output = dependency.hasStencilDependency();
                    return output;
                });

                if (!isStencilEdge) {
                    edgesToBeRemoved.push_back({visitingNodeUid, targetChildUid});
                }
            }
        }
    });
    for (const auto& toBeRemoved : edgesToBeRemoved) {
        mRawGraph.removeEdge(toBeRemoved);
    }
}

auto Graph::helpGetOutNeighbors(GraphData::Uid                          nodeUid,
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

auto Graph::helpGetInNeighbors(GraphData::Uid                          nodeUid,
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

auto Graph::helpGetOutEdges(GraphData::Uid                          nodeUid,
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

auto Graph::helpGetInEdges(GraphData::Uid                          nodeUid,
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

auto Graph::helpGetBFS(bool                                    filterOutBeginEnd,
                       const std::vector<GraphDependencyType>& dependencyTypes)
    -> Bfs
{
    using Frontier = std::unordered_map<GraphData::Uid, size_t>;

    Bfs bfs;

    Frontier a;
    Frontier b;

    Frontier* currentFrontier = &a;
    Frontier* nextFrontier = &b;


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

        for (auto& currentFrontierNode : *currentFrontier) {
            if (currentFrontierNode.second == 0) {
                // All incoming dependencies have been resolved.
                // The node is ready to be added to the current BFS level
                newLevel.push_back(currentFrontierNode.first);

                // Adding the node's children to the next frontier
                auto currentFrontierNodeChildren = helpGetOutNeighbors(currentFrontierNode.first, filterOutBeginEnd);
                for (const auto& child : currentFrontierNodeChildren) {
                    auto it = std::find_if(nextFrontier->begin(), nextFrontier->end(),
                                           [&child](auto& entry) {
                                               return (entry.first == child);
                                           });
                    if (it != nextFrontier->end()) {
                        it->second--;
                    } else {
                        auto numChildDependencies = helpGetInEdges(child, filterOutBeginEnd).size();
                        nextFrontier->insert({child, numChildDependencies - 1});
                    }
                }
            } else {
                nextFrontier->insert(currentFrontierNode);
            }
        }
        std::swap(currentFrontier, nextFrontier);
    }

    return bfs;
}

auto Graph::helpComputeScheduling(bool filterOutAnchors, int anchorStream)
    -> void
{
    helpComputeScheduling_00_resetData();
    mBfs = helpComputeScheduling_01_generatingBFS(filterOutAnchors);
    int maxStreamId = helpComputeScheduling_02_mappingStreams(mBfs, filterOutAnchors, anchorStream);
    int maxEventId = helpComputeScheduling_03_events(mBfs);
    helpComputeScheduling_04_ensureResources(maxStreamId, maxEventId);
}

auto Graph::helpComputeScheduling_01_generatingBFS(bool filterOutAnchors)
    -> Bfs
{
    return helpGetBFS(filterOutAnchors, {GraphDependencyType::data,
                                         GraphDependencyType::user});
}

auto Graph::getGraphNode(GraphData::Uid uid) -> GraphNode&
{
    return mRawGraph.getVertexProperty(uid);
}

auto Graph::getGraphNode(GraphData::Uid uid) const -> const GraphNode&
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

auto Graph::helpComputeScheduling_02_mappingStreams(Bfs& bfs,
                                                    bool filterOutAnchors,
                                                    int  anchorStream)
    -> int
{
    mSchedulingStatusIsValid = true;
    mMaxNumberStreams = bfs.getMaxLevelWidth();

    // used stream -> true
    // available -> false
    constexpr bool usedStream = true;
    constexpr bool availableStream = false;

    std::vector<bool> streamsStatus(mMaxNumberStreams,
                                    availableStream);

    auto bookFirstAvailableStream = [&]() {
        for (int i = 0; i < int(streamsStatus.size()); i++) {
            if (availableStream == streamsStatus[i]) {
                streamsStatus[i] = usedStream;
                return i;
            }
        }
        NEON_THROW_UNSUPPORTED_OPERATION("");
    };

    auto isStreamAvailable = [&](int streamId)
        -> bool {
        return streamsStatus[streamId] == availableStream;
    };

    auto bookStream = [&](int streamId)
        -> bool {
        if (isStreamAvailable(streamId)) {
            streamsStatus[streamId] = usedStream;
            return true;
        }
        return false;
    };

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
            mRawGraph.getVertexProperty(level[0]).getScheduling().setStream(anchorStream);
            return;
        }
        // Step a.
        for (auto& nodeUid : level) {
            auto& node = mRawGraph.getVertexProperty(nodeUid);

            int associatedStream = [&]() -> int {
                if (levelIdx == 0) {
                    return bookFirstAvailableStream();
                }
                auto& preNodes = mRawGraph.inNeighbors(node.getGraphData().getUid());
                for (auto& preNodeIdx : preNodes) {
                    auto preNodeStream = mRawGraph.getVertexProperty(preNodeIdx).getScheduling().getStream();
                    if (bookStream(preNodeStream)) {
                        return preNodeStream;
                    }
                }
                return -1;
            }();

            if (associatedStream == -1) {
                // track nodes that need to go through step 2
                delayedBooking.push_back(&node);
            } else {
                node.getScheduling().setStream(associatedStream);
            }
        }

        // Step b.
        for (auto nodePrt : delayedBooking) {
            auto associatedStream = bookFirstAvailableStream();
            (*nodePrt).getScheduling().setStream(associatedStream);
        }

        streamsStatus = std::vector<bool>(mMaxNumberStreams, false);
    });

    if (!filterOutAnchors && anchorStream > -1) {
        auto  endNodeId = getEndNode().getGraphData().getUid();
        auto& endNode = mRawGraph.getVertexProperty(endNodeId);
        endNode.getScheduling().setStream(anchorStream);
    }

    this->ioToDot("tge", "tge", true);
    return mMaxNumberStreams - 1;
}

auto Graph::helpComputeScheduling_03_events(Bfs& bfs)
    -> int
{
    int eventCount = 0;

    bfs.forEachNodeByLevel(*this, [&](GraphNode& targetNode, int /*levelId*/) {
        int  targetStreamId = targetNode.getScheduling().getStream();
        auto preNodeIds = mRawGraph.inNeighbors(targetNode.getGraphData().getUid());
        if (preNodeIds.size() == 0) {
            // this is the first node, no need to have dependencies
            return;
        }
        // Processing PreNodes on different streams than the node
        std::vector<GraphNode*> preNodesOnDifferentStreams;
        for (auto nodeId : preNodeIds) {
            auto& preNode = mRawGraph.getVertexProperty(nodeId);
            auto  preNodeStream = preNode.getScheduling().getStream();
            if (preNodeStream == targetStreamId) {
                continue;
            }
            // For pre-nodes on different streams we do the following
            // a. check if the pre-node has an event associated, if not create one
            // b. add the pre-node event to the list of the target node

            // a.
            int preNodeEvent = preNode.getScheduling().getEvent();
            if (preNodeEvent == -1) {
                preNodeEvent = eventCount;
                preNode.getScheduling().setEvent(preNodeEvent);
                eventCount++;
            }
            // b.
            targetNode.getScheduling().getDependentEvents().push_back(preNodeEvent);
        }
    });

    return eventCount - 1;
}

auto Graph::helpComputeScheduling_04_ensureResources(int maxStreamId,
                                                     int maxEventId)
    -> void
{
    auto bk = getBackend();
    bk.setAvailableStreamSet(maxStreamId + 1);
    bk.setAvailableUserEvents(maxEventId + 1);
}

auto Graph::ioToDot(const std::string& fname,
                    const std::string& graphName,
                    bool               debug) -> void
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
            return "style=dashed, color=\"#F4A261\", penwidth=7";
        }
        return "color=\"#d9d9d9\", penwidth=7";
    };

    auto vertexLabelProperty = [&](const size_t& v) {
        auto& node = mRawGraph.getVertexProperty(v);
        return node.getLabelProperty();
    };
    mRawGraph.exportDotFile(fname + ".dot", graphName, vertexLabel, edgeLabel,
                            vertexLabelProperty, edgeLabelProperty);
}

Graph::Graph(const Backend& bk)
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

auto Graph::getBackend() const
    -> const Neon::Backend&
{
    if (mBackendIsSet) {
        return mBackend;
    }
    NeonException ex("Graph");
    ex << "A backend was not set.";
    NEON_THROW(ex);
}

auto Graph::run(Neon::SetIdx   setIdx,
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

auto Graph::run(int            streamIdx,
                Neon::DataView dataView)
    -> void
{
    if (dataView != Neon::DataView::STANDARD) {
        NEON_THROW_UNSUPPORTED_OPERATION("");
    }

    this->runtimePreSet(streamIdx);
    this->helpExecute(streamIdx);
}

auto Graph::helpExecute(Neon::SetIdx setIdx,
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

auto Graph::helpExecute(int anchorStream)
    -> void
{
    if (anchorStream > -1 && (anchorStream != mAnchorStreamPreSet || mFilterOutAnchorsPreSet == true)) {
        Neon::NeonException ex("");
        ex << "Execution parameters are inconsistent with the preset ones.";
        NEON_THROW(ex);
    }

    int ndevs = getBackend().devSet().setCardinality();
    {
#pragma omp parallel for num_threads(ndevs)
        for (int setIdx = 0; setIdx < ndevs; setIdx++) {
            helpExecute(setIdx, anchorStream);
        }
    }
}

auto Graph::runtimePreSet(int anchorStream) -> void
{
    helpComputeScheduling(false, anchorStream);
    mAnchorStreamPreSet = anchorStream;
    mFilterOutAnchorsPreSet = false;
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
auto Graph::helpCheckBackendStatus()
    -> void
{
    if (!mBackendIsSet) {
        NeonException exception("Container Graph");
        exception << "A backend was not registered with the Graph";
        NEON_THROW(exception);
    }
}
auto Graph::getNumberOfNodes() -> int
{
    // We remove 2 because of the head and tail holders.
    return int(mRawGraph.numVertices() - 2);
}


}  // namespace Neon::set::container
