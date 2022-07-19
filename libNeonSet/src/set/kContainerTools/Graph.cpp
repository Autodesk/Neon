#include "Neon/set/ContainerTools/Graph.h"

namespace Neon::set::container {

Graph::Graph()
{
    auto begin = GraphNode::getBeginNode();
    auto end = GraphNode::getEndNode();
    mUidCounter = GraphNodeOrganization::firstInternal;

    mRawGraph.addVertex(begin.getOrganization().getUid(), begin);
    mRawGraph.addVertex(end.getOrganization().getUid(), end);
}

auto Graph::getBeginNode() const -> const GraphNode&
{
    return mRawGraph.getVertexProperty(GraphNodeOrganization::beginUid);
}

auto Graph::getEndNode() const -> const GraphNode&
{
    return mRawGraph.getVertexProperty(GraphNodeOrganization::endUid);
}

auto Graph::addNodeInBetween(const GraphNode& nodeA,
                             Container&       containerB,
                             const GraphNode& nodeC) -> GraphNode&
{
    auto& nodeB = addNode(containerB);

    GraphDependency ab(GraphDependencyType::USER);
    GraphDependency bc(GraphDependencyType::USER);

    mRawGraph.addEdge(nodeA.getOrganization().getUid(),
                      nodeB.getOrganization().getUid(),
                      ab);

    mRawGraph.addEdge(nodeB.getOrganization().getUid(),
                      nodeC.getOrganization().getUid(),
                      bc);

    return nodeB;
}

auto Graph::addNodeInBetween(const GraphNode& nodeA,
                             Container&       containerB,
                             const GraphNode& nodeC,
                             GraphDependency& ab,
                             GraphDependency& bc) -> GraphNode&
{
    auto& nodeB = addNodeInBetween(nodeA, containerB, nodeC);

    ab = mRawGraph.getEdgeProperty({nodeA.getOrganization().getUid(),
                                    nodeB.getOrganization().getUid()});


    bc = mRawGraph.getEdgeProperty({nodeB.getOrganization().getUid(),
                                    nodeC.getOrganization().getUid()});

    return nodeB;
}

auto Graph::addNode(Container& container) -> GraphNode&
{
    auto const& node = GraphNode(container, mUidCounter);
    mRawGraph.addVertex(node.getOrganization().getUid(), node);
    mUidCounter++;
    return mRawGraph.getVertexProperty(node.getOrganization().getUid());
}

auto Graph::addDependency(const GraphNode&    nodeA,
                          const GraphNode&    nodeB,
                          GraphDependencyType graphDependencyType) -> GraphDependency&
{
    GraphDependency ab(graphDependencyType);

    mRawGraph.addEdge(nodeA.getOrganization().getUid(),
                      nodeB.getOrganization().getUid(),
                      ab);

    return mRawGraph.getEdgeProperty({nodeA.getOrganization().getUid(),
                                      nodeB.getOrganization().getUid()});
}
auto Graph::removeNode(GraphNode& gn) -> GraphNode
{
    auto uidB = gn.getOrganization().getUid();

    std::vector<GraphNodeOrganization::Uid> a_toBeConnectedToEnd;
    mRawGraph.forEachInEdge(uidB, [&](const RawGraph::Edge& edge) -> void {
        auto uidA = edge.first;
        int  outEdgesFromA = mRawGraph.outEdgesCount(uidA);
        if (outEdgesFromA == 1) {
            a_toBeConnectedToEnd.push_back(uidA);
        }
    });

    std::vector<GraphNodeOrganization::Uid> c_toBeConnectedToBegin;
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

    GraphNode removed = mRawGraph.getVertexProperty(gn.getOrganization().getUid());
    removed.getOrganization().setUid(GraphNodeOrganization::notSet);

    return removed;
}

}  // namespace Neon::set::container
