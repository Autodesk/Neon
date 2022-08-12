#include "Neon/set/ContainerTools/graph/GraphNode.h"

namespace Neon::set::container {

GraphNode::GraphNode()
{
}

auto GraphNode::newBeginNode() -> GraphNode
{
    GraphNode node;
    node.mGraphNodeOrganization.setUid(GraphData::beginUid);
    return node;
}

auto GraphNode::newEndNode() -> GraphNode
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

auto GraphNode::getContainer() -> Container&
{
    return mContainer;
}

auto GraphNode::getContainer() const -> const Container&
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
    if (getContainerOperationType() == Neon::set::ContainerOperationType::anchor) {
        return R"(shape=octagon, style="rounded,filled", fillcolor="#fb8072", color="#b11605")";
    }
    if (getContainerOperationType() == Neon::set::ContainerOperationType::compute) {
        return R"(shape=octagon, style="rounded,filled", fillcolor="#fb8072", color="#b11605")";
    }
    if (getContainerOperationType() == Neon::set::ContainerOperationType::halo) {
        return R"(shape=octagon, style="rounded,filled", fillcolor="#fb8072", color="#b11605")";
    }
    if (getContainerOperationType() == Neon::set::ContainerOperationType::sync) {
        return R"(shape=octagon, style="rounded,filled", fillcolor="#fb8072", color="#b11605")";
    }
    NEON_DEV_UNDER_CONSTRUCTION("");
}
auto GraphNode::helpGetDotName() -> std::string
{
    return getContainer().getName();
}
auto GraphNode::helpGetDotInfo() -> std::string
{
    if (getContainerOperationType() == Neon::set::ContainerOperationType::anchor) {
        return std::string();
    }
    if (getContainerOperationType() == Neon::set::ContainerOperationType::compute) {
        std::stringstream s;
        s << "Uid = " << getContainer().getUid();
        s << "DataView = " << Neon::DataViewUtil::toString(getScheduling().getDataView());
        return s.str();
    }
    if (getContainerOperationType() == Neon::set::ContainerOperationType::halo) {
        return R"(shape=octagon, style="rounded,filled", fillcolor="#fb8072", color="#b11605")";
    }
    if (getContainerOperationType() == Neon::set::ContainerOperationType::sync) {
        return R"(shape=octagon, style="rounded,filled", fillcolor="#fb8072", color="#b11605")";
    }
    NEON_DEV_UNDER_CONSTRUCTION("");

}
auto GraphNode::getContainerOperationType() const -> Neon::set::ContainerOperationType
{
    return getContainer().getContainerInterface().getContainerOperationType();
}

}  // namespace Neon::set::container
