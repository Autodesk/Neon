#include "Neon/set/ContainerTools/graph/GraphNode.h"

namespace Neon::set::container {

GraphNode::GraphNode()
{
}

auto GraphNode::getBeginNode() -> GraphNode
{
    GraphNode node;
    node.mGraphNodeOrganization.setUid(GraphData::beginUid);
    return node;
}

auto GraphNode::getEndNode() -> GraphNode
{
    GraphNode node;
    node.mGraphNodeOrganization.setUid(GraphData::endUid);
    return node;
}

auto GraphNode::getGraphData() -> GraphData&
{
    return mGraphNodeOrganization;
}

auto GraphNode::getGraphData() const -> const GraphData&
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

GraphNode::GraphNode(const Container& container, GraphData::Uid uid)
{
    mContainer = container;
    mGraphNodeOrganization.setUid(uid);
}
auto GraphNode::toString() -> std::string
{
    return std::string();
}

auto GraphNode::helpGetDotProperties() -> std::string
{
    if (mGraphNodeType == GraphNodeType::Anchor) {
        return R"(shape=octagon, style="rounded,filled", fillcolor="#fb8072", color="#b11605")";
    }
    if (mGraphNodeType == GraphNodeType::Compute) {
        return R"(shape=octagon, style="rounded,filled", fillcolor="#fb8072", color="#b11605")";
    }
    if (mGraphNodeType == GraphNodeType::Halo) {
        return R"(shape=octagon, style="rounded,filled", fillcolor="#fb8072", color="#b11605")";
    }
    if (mGraphNodeType == GraphNodeType::Sync) {
        return R"(shape=octagon, style="rounded,filled", fillcolor="#fb8072", color="#b11605")";
    }
}
auto GraphNode::helpGetDotName() -> std::string
{
    return getContianer().getName();
}
auto GraphNode::helpGetDotInfo() -> std::string
{
    if (mGraphNodeType == GraphNodeType::Anchor) {
        return std::string();
    }
    if (mGraphNodeType == GraphNodeType::Compute) {
        std::stringstream s;
        s << "Uid = " << getContianer().getUid();
        s << "DataView = " << Neon::DataViewUtil::toString(getScheduling().getDataView());
        return s.str();
    }
    if (mGraphNodeType == GraphNodeType::Halo) {
        return R"(shape=octagon, style="rounded,filled", fillcolor="#fb8072", color="#b11605")";
    }
    if (mGraphNodeType == GraphNodeType::Sync) {
        return R"(shape=octagon, style="rounded,filled", fillcolor="#fb8072", color="#b11605")";
    }
}

}  // namespace Neon::set::container
