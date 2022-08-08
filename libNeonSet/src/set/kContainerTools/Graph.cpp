#include "Neon/set/ContainerTools/Graph.h"

namespace Neon::set::container {

Graph::Graph()
{
    auto begin = GraphNode::newBeginNode();
    auto end = GraphNode::newEndNode();
    mUidCounter = GraphData::firstInternal;

    mRawGraph.addVertex(begin.getGraphData().getUid(), begin);
    mRawGraph.addVertex(end.getGraphData().getUid(), end);
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
    auto& nodeB = addNodeInBetween(nodeA, containerB, nodeC);

    auto& abEdge = mRawGraph.getEdgeProperty({nodeA.getGraphData().getUid(),
                                              nodeB.getGraphData().getUid()});
    abEdge.setType(ab);

    auto& bcEdge = mRawGraph.getEdgeProperty({nodeB.getGraphData().getUid(),
                                              nodeC.getGraphData().getUid()});

    bcEdge.setType(bc);


    return nodeB;
}

auto Graph::addNode(const Container& container) -> GraphNode&
{
    auto const& node = GraphNode(container, mUidCounter);
    mRawGraph.addVertex(node.getGraphData().getUid(), node);
    mUidCounter++;
    return mRawGraph.getVertexProperty(node.getGraphData().getUid());
}

auto Graph::addDependency(const GraphNode&    nodeA,
                          const GraphNode&    nodeB,
                          GraphDependencyType type) -> GraphDependency&
{
    GraphDependency ab(type);

    mRawGraph.addEdge(nodeA.getGraphData().getUid(),
                      nodeB.getGraphData().getUid(),
                      ab);

    return mRawGraph.getEdgeProperty({nodeA.getGraphData().getUid(),
                                      nodeB.getGraphData().getUid()});
}

auto Graph::removeNode(GraphNode& gn) -> GraphNode
{
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

}  // namespace Neon::set::container
