#include "Neon/set/ContainerTools/Graph.h"

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
                             Container&                containerB,
                             const GraphNode&          nodeC,
                             const GraphDependencyType ab,
                             const GraphDependencyType bc) -> GraphNode&
{
    helpInvalidateScheduling();

    auto& nodeB = addNodeInBetween(nodeA, containerB, nodeC);

    auto& abEdge = mRawGraph.getEdgeProperty({nodeA.getGraphData().getUid(),
                                              nodeB.getGraphData().getUid()});
    abEdge.setType(ab);

    auto& bcEdge = mRawGraph.getEdgeProperty({nodeB.getGraphData().getUid(),
                                              nodeC.getGraphData().getUid()});

    bcEdge.setType(bc);


    return nodeB;
}

auto Graph::addNode(const Container& container,
                    GraphNodeType    graphNodeType) -> GraphNode&
{
    helpInvalidateScheduling();

    auto const& node = GraphNode(container, mUidCounter, graphNodeType);
    mRawGraph.addVertex(node.getGraphData().getUid(), node);
    mUidCounter++;
    return mRawGraph.getVertexProperty(node.getGraphData().getUid());
}

auto Graph::addDependency(const GraphNode&    nodeA,
                          const GraphNode&    nodeB,
                          GraphDependencyType type) -> GraphDependency&
{
    helpInvalidateScheduling();

    GraphDependency ab(type);

    mRawGraph.addEdge(nodeA.getGraphData().getUid(),
                      nodeB.getGraphData().getUid(),
                      ab);

    return mRawGraph.getEdgeProperty({nodeA.getGraphData().getUid(),
                                      nodeB.getGraphData().getUid()});
}

auto Graph::removeNode(GraphNode& gn) -> GraphNode
{
    helpInvalidateScheduling();

    auto uidB = gn.getGraphData().getUid();

    std::vector<GraphData::Uid> a_toBeConnectedToEnd;
    mRawGraph.forEachInEdge(uidB, [&](const RawGraph::Edge& edge) -> void {
        auto uidA = edge.first;
        int  outEdgesFromA = mRawGraph.outEdgesCount(uidA);
        if (outEdgesFromA == 1) {
            a_toBeConnectedToEnd.push_back(uidA);
        }
    });

    std::vector<GraphData::Uid> c_toBeConnectedToBegin;
    mRawGraph.forEachOutEdge(uidB, [&](const RawGraph::Edge& edge) -> void {
        auto uidC = edge.second;
        int  inEdgesIntoC = mRawGraph.outEdgesCount(uidC);
        if (inEdgesIntoC == 1) {
            a_toBeConnectedToEnd.push_back(uidC);
        }
    });

    for (auto&& uidA : a_toBeConnectedToEnd) {
        auto& nodeA = mRawGraph.getVertexProperty(uidA);
        addDependency(nodeA, this->getEndNode());
    }

    for (auto&& uidC : c_toBeConnectedToBegin) {
        auto& nodeC = mRawGraph.getVertexProperty(uidC);
        addDependency(this->getBeginNode(), nodeC);
    }

    GraphNode removed = mRawGraph.getVertexProperty(gn.getGraphData().getUid());
    removed.getGraphData().setUid(GraphData::notSet);

    return removed;
}

auto Graph::getProceedingGraphNodes(const GraphNode& graphNode)
    -> std::vector<GraphNode*>
{
    std::vector<GraphNode*> nodes;

    auto nodeUID = graphNode.getGraphData().getUid();
    auto nodeUidSet = mRawGraph.inNeighbors(nodeUID);
    for (auto&& proceedingUid : nodeUidSet) {
        auto& proceedingNode = mRawGraph.getVertexProperty(proceedingUid);
        nodes.push_back(&proceedingNode);
    }

    return nodes;
}

auto Graph::getSubsequentGraphNodes(const GraphNode& graphNode) -> std::vector<GraphNode*>
{
    std::vector<GraphNode*> nodes;

    auto nodeUID = graphNode.getGraphData().getUid();
    auto nodeUidSet = mRawGraph.outNeighbors(nodeUID);
    for (auto&& subsequentUID : nodeUidSet) {
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

    auto& newNode = addNode(graphNode.getContianer());

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
    auto uidA = nodeA.getGraphData().getUid();
    auto uidB = nodeB.getGraphData().getUid();

    auto edgePropetry = mRawGraph.getEdgeProperty({uidA, uidB});
    auto dependencyType = edgePropetry.getType();

    return dependencyType;
}

auto Graph::helpInvalidateScheduling() -> void
{
    mSchedulingStatusIsValid = false;
}

auto Graph::helpRemoteRedundantDependencies() -> void
{
    // Vectors of edges to be removed
    std::vector<std::pair<size_t, size_t>> edgesToBeRemoved;
    mRawGraph.forEachVertex([&](size_t visitingNode) {
        // In this body we are looping over all nodes
        // For each node do:

        // Check node's children
        const auto& children = helpGetOutNeighbors(visitingNode);
        if (children.size() <= 1) {
            // If no more than one, move to the next node
            // Nothing to do for the visiting node as there are no redundant paths
            // Let's move to another
            return;
        }
        // Start checking for redundant paths
        for (const auto& targetChild : children) {
            if (helpGetInEdges(targetChild).size() <= 1) {
                // This targetChild can only be reached by one father
                // No redundant path here
                continue;
            }
            bool foundRedundant = false;
            // Checking all siblings' paths.
            // We are looking for the targetChild in the path of any of its siblings.
            // A BFS visit is used for the process.
            for (const auto& targetSibling : children) {
                if (targetSibling == targetChild) {
                    continue;
                }
                // first BFS frontier are the targetSibling's child
                auto frontier = mRawGraph.outNeighbors(targetSibling);
                while (frontier.size() != 0) {
                    auto nextFrontier = std::set<size_t>();

                    for (const auto& nodeInFrontier : frontier) {
                        if (nodeInFrontier == targetChild) {
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
                edgesToBeRemoved.push_back({visitingNode, targetChild});
            }
        }
    });
    for (const auto& toBeRemoved : edgesToBeRemoved) {
        mRawGraph.removeEdge(toBeRemoved);
    }
}

auto Graph::helpGetOutNeighbors(GraphData::Uid                          nodeUid,
                                bool                                    filteredOut,
                                const std::vector<GraphDependencyType>& dependencyTypes) -> std::set<GraphData::Uid>
{
    std::set<GraphData::Uid> outNgh;
    mRawGraph.forEachOutEdge(
        nodeUid,
        [&](std::pair<GraphData::Uid, GraphData::Uid> edge) {
            auto& edgeProp = mRawGraph.getEdgeProperty(edge);
            for (auto& depType : dependencyTypes) {
                if (depType == edgeProp.getType()) {
                    if (mRawGraph.getVertexProperty(edge.second).getContainerOperationType() == Neon::set::ContainerOperationType::anchor) {
                        break;
                    }
                    outNgh.insert(edge.second);
                }
            }
        });
    return outNgh;
}

auto Graph::helpGetInNeighbors(GraphData::Uid nodeUid, bool filterOutBegin, const std::vector<GraphDependencyType>& dependencyTypes) -> std::set<GraphData::Uid>
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
                            const std::vector<GraphDependencyType>& dependencyTypes) -> std::set<std::pair<GraphData::Uid, GraphData::Uid>>
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
                           const std::vector<GraphDependencyType>& dependencyTypes) -> std::set<std::pair<GraphData::Uid, GraphData::Uid>>
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

auto Graph::helpGetBFS(bool filterOutBegin, const std::vector<GraphDependencyType>& dependencyTypes) -> std::vector<std::vector<GraphData::Uid>>
{
    std::vector<std::vector<GraphData::Uid>> bfs;
    bfs.push_back(std::vector<GraphData::Uid>());
    int currentLevel = 0;
}


}  // namespace Neon::set::container
