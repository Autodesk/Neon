#include "Neon/set/ContainerTools/GraphNode.h"

namespace Neon::set::container {

GraphNode::GraphNode()
{
}

auto GraphNode::getBeginNode() -> GraphNode
{
    GraphNode node;
    node.mGraphNodeOrganization.setUid(GraphNodeOrganization::beginUid);
    return node;
}

auto GraphNode::getEndNode() -> GraphNode
{
    GraphNode node;
    node.mGraphNodeOrganization.setUid(GraphNodeOrganization::endUid);
    return node;
}

auto GraphNode::getOrganization() -> GraphNodeOrganization&
{
    return mGraphNodeOrganization;
}

auto GraphNode::getOrganization() const -> const GraphNodeOrganization&
{
    return mGraphNodeOrganization;
}

auto GraphNode::getScheduling() -> GraphNodeScheduling&
{
    return mGraphNodeScheduling;
}

auto GraphNode::getScheduling() const -> const GraphNodeScheduling&
{
    return mGraphNodeScheduling;
}

auto GraphNode::getContianer() -> Container&
{
    return mContainer;
}

auto GraphNode::getContianer() const -> const Container&
{
    return mContainer;
}

GraphNode::GraphNode(const Container& container, GraphNodeOrganization::Uid uid)
{
    mContainer = container;
    mGraphNodeOrganization.setUid(uid);
}

}  // namespace Neon::set::container
