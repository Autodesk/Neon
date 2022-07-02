#include "Neon/skeleton/internal/MultiGpuGraph.h"
#include <list>

namespace Neon::skeleton::internal {


// auto UserGraph_t::h_newHaloNode()
//{
//     auto node = Node_t::factory(NodeType_e::HELPER, -1);
//     m_graph().addVertex(node.nodeId(), node);
//     return node.nodeId();
// }
//
// auto UserGraph_t::h_newNeonNode() -> size_t
//{
//     auto node = Node_t::factory(NodeType_e::HELPER, -1);
//     m_graph().addVertex(node.nodeId(), node);
//     return node.nodeId();
// }
void MultiGpuGraph::init(Neon::Backend&                           bk,
                         const std::vector<Neon::set::Container>& operations,
                         std::string                              name,
                         Options                                  options)
{
    parse(bk.devSet().setCardinality(), std::forward<const std::vector<Neon::set::Container>&&>(operations));
    m_storage->m_userAppGraph = m_graph();
    h_removeRedundantDependencies();
    h_io2Dot("t0_" + name + ".dot", "i");
    optimizations(options);
    h_io2Dot("t1_" + name + ".dot", "i");
    addSyncAndMemoryTransfers(options);
    h_removeRedundantDependencies();
    checkCoherency();
    h_io2Dot("t2_" + name + ".dot", "i");
    resetLinearContinuousIndexing();
}

void MultiGpuGraph::parse(int setCardinalty, const std::vector<Neon::set::Container>&& operations)
{
    m_setCardinality() = setCardinalty;

    std::vector<size_t>                   containerId2gNodeId;
    std::vector<std::tuple<size_t, Edge>> specialClosureNodes;
    int                                   i = 0;
    for (auto&& k : operations) {
        h_parseContainer(k,
                         containerId2gNodeId,
                         specialClosureNodes);
        i++;
    }
    h_closure(specialClosureNodes);
}

void MultiGpuGraph::h_parseContainer(const Neon::set::Container&            inContainer,
                                     std::vector<size_t>&                   containerId2gNodeId,
                                     std::vector<std::tuple<size_t, Edge>>& specialClosureNodes)
{
    // Register and retrieve the id for the new container
    // add a node
    ContainerIdx containerIdx = h_trackNewContainer(inContainer, containerId2gNodeId);

    // Parsing all the data toke used by the kernel container
    std::vector<DataToken_t> tokens = h_parse(containerIdx);
    int                      detectedDependencies = 0;
    for (auto& token : tokens) {
        // TODO@Max(if the kernelData is used for stencil store it for future halo update)
        if (token.compute() == Neon::Compute::REDUCE) {
            const NodeId nodeId = containerId2gNodeId[containerIdx];
            auto&        nodeMeta = m_graph().getVertexProperty(nodeId);
            if (nodeMeta.isStencil()) {
                NEON_THROW_UNSUPPORTED_OPTION("The skeleton does not suport container with both reduce and stencil operations.");
            }
            nodeMeta.setCompute(Neon::Compute::REDUCE);
        }
        // update the dependency state machine with the new toke.
        // newDependencies are the detected dependencies
        auto newDependencies = m_dataRecords().updateStatus(containerIdx, token.access(), token.uid());
        if (newDependencies.size() == 0 &&
            token.compute() == Neon::Compute::STENCIL) {

            const size_t stencilNode = containerId2gNodeId[containerIdx];
            auto&        nodeMeta = m_graph().getVertexProperty(stencilNode);
            nodeMeta.setCompute(Neon::Compute::STENCIL);

            auto newEdge = Edge::factory(token, Dependencies_e::RAW);
            specialClosureNodes.push_back({stencilNode, newEdge});

            detectedDependencies++;
        }
        for (auto& dep : newDependencies) {
            detectedDependencies++;

            const size_t nodeid_t0 = containerId2gNodeId[dep.t0()];
            const size_t nodeid_t1 = containerId2gNodeId[dep.t1()];

            if (!m_graph().hasEdge(nodeid_t0, nodeid_t1)) {
                auto newEdge = Edge::factory(token, dep.type());
                m_graph().addEdge(nodeid_t0, nodeid_t1, newEdge);
            } else {
                auto& edgeVal = m_graph().getEdgeProperty({nodeid_t0, nodeid_t1});
                edgeVal.append(token, dep.type());
            }
            if (token.compute() == Neon::Compute::STENCIL) {
                auto& node = m_graph().getVertexProperty(nodeid_t1);
                node.setCompute(Neon::Compute::STENCIL);
            }
        }
    }
}

auto MultiGpuGraph::getContainer(ContainerIdx id)
    -> Neon::set::Container&
{
    return m_kContainers()[id];
}

auto MultiGpuGraph::getContainer(ContainerIdx id) const
    -> const Neon::set::Container&
{
    return m_kContainers()[id];
}


/**
 * Helper function to close the graph with a begin and end emtpy nodes
 * @param specialClosureNodes
 */
auto MultiGpuGraph::h_closure(const std::vector<std::tuple<size_t, Edge>>& specialClosureNodes)
    -> void
{
    // using NodeType_e = internal::userGraph::NodeType_e;

    // Adding the kernel container into the graph
    auto node = MetaNode::factory(MetaNodeType_e::HELPER, "Begin", -1);
    m_graph().addVertex(node.nodeId(), node);
    m_setRootNodeId(node.nodeId());

    // Adding the kernel container into the graph
    node = MetaNode::factory(MetaNodeType_e::HELPER, "End", -1);
    m_graph().addVertex(node.nodeId(), node);
    m_setFinalNodeId(node.nodeId());

    m_graph().forEachVertex([&](size_t nodeId) {
        if (m_graph().inEdges(nodeId).size() == 0) {
            if (nodeId != m_rootNodeId() && nodeId != m_finalNodeId()) {
                m_graph().addEdge(m_rootNodeId(), nodeId);
            }
        }
        if (m_graph().outEdges(nodeId).size() == 0) {
            if (nodeId != m_rootNodeId() && nodeId != m_finalNodeId()) {
                m_graph().addEdge(nodeId, m_finalNodeId());
            }
        }
    });

    for (auto& specialClosureNode : specialClosureNodes) {
        size_t node_t0 = m_rootNodeId();
        size_t node_t1 = std::get<size_t>(specialClosureNode);
        auto&  edgeMetaData = std::get<Edge>(specialClosureNode);
        if (m_graph().hasEdge({node_t0, node_t1})) {
            Edge& edge = m_graph().getEdgeProperty({node_t0, node_t1});
            edge = edgeMetaData;
        }
    }
    this->h_io2Dot("DB.dot", "");

    // h_io2Dot("test.dot", "g");
}


auto MultiGpuGraph::h_parse(ContainerIdx kernelContainerIdx)
    -> std::vector<DataToken_t>
{
    auto& container = getContainer(kernelContainerIdx);
    auto& kcInterface = container.getContainerInterface();
    auto& tokens = kcInterface.parse();
    return tokens;
}

auto MultiGpuGraph::h_io2Dot([[maybe_unused]] const std::string& fname,
                             [[maybe_unused]] const std::string& graphName) -> void
{
    // io2Dot(fname, graphName);
    return;
}

auto MultiGpuGraph::io2Dot(const std::string& fname, const std::string& graphName) -> void
{
    // http://www.graphviz.org/doc/info/shapes.html
    auto clone = m_graph();
    m_schedulingGraph().forEachEdge([&](const DiGraphScheduling::Edge& edge) {
        auto edgeMeta = Edge::factorySchedulingEdge();
        clone.addEdge(edge.first, edge.second, edgeMeta);
    });

    auto vertexLabel = [&](size_t v) -> std::string {
        if (v == m_finalNodeId()) {
            return std::string("END");  //+ "DEBUG" + std::to_string(m_graph().getVertexProperty(v).nodeId());
        }
        if (v == m_rootNodeId()) {
            return std::string("BEGIN");  // + "DEBUG" + std::to_string(m_graph().getVertexProperty(v).nodeId());
        }
        return clone.getVertexProperty(v).toString();
    };

    auto edgeLabel = [&](const std::pair<size_t, size_t>& edge)
        -> std::string {
        const auto& metaEdge = clone.getEdgeProperty(edge);
        if (metaEdge.m_isSchedulingEdge) {
            return "";
        }
        if (metaEdge.nDependencies() > 0) {
            return metaEdge.toString();
        }
        return "";
    };

    auto edgeLabelProperty = [&](const std::pair<size_t, size_t>& edge)
        -> std::string {
        const auto& metaEdge = clone.getEdgeProperty(edge);
        if (metaEdge.m_isSchedulingEdge) {
            // return "style=dashed, color=\"#2A9D8F\"";
            return "style=dashed, color=\"#F4A261\", penwidth=7";
        }
        return "color=\"#d9d9d9\", penwidth=7";
    };

    auto vertexLabelProperty = [&](const size_t& v) {
        if (v == m_finalNodeId() || (v == m_rootNodeId())) {
            return R"(shape=doublecircle, style=filled, fillcolor="#d9d9d9", color="#6c6c6c")";
        }
        const auto& metaNode = clone.getVertexProperty(v);
        if (metaNode.isHu()) {
            return R"(shape=octagon, style="rounded,filled", fillcolor="#fb8072", color="#b11605")";
        }
        if (metaNode.isSync()) {
            return R"(shape=octagon, style="rounded,filled", fillcolor="#fb8072", color="#b11605")";
        }

        if (metaNode.isMap()) {
            return R"(style=filled, fillcolor="#b3de69", color="#5f861d")";
        }
        if (metaNode.isReduce()) {
            return R"(style=filled, fillcolor="#80b1d3", color="#2b5c7d")";
        }
        if (metaNode.isStencil()) {
            return R"(style=filled, fillcolor="#bebada", color="#4e4683")";
        }
        return "";
    };

    clone.exportDotFile(fname, graphName, vertexLabel, edgeLabel, vertexLabelProperty, edgeLabelProperty);
}

auto MultiGpuGraph::io2DotOriginalApp(const std::string& fname, const std::string& graphName) -> void
{
    // http://www.graphviz.org/doc/info/shapes.html
    auto& clone = m_storage->m_userAppGraph;

    auto vertexLabel = [&](size_t v) -> std::string {
        if (v == m_finalNodeId()) {
            return std::string("END");  // + "DEBUG" + std::to_string(m_graph().getVertexProperty(v).nodeId());
        }
        if (v == m_rootNodeId()) {
            return std::string("BEGIN");  // + "DEBUG" + std::to_string(m_graph().getVertexProperty(v).nodeId());
        }
        return clone.getVertexProperty(v).toString();
    };

    auto edgeLabel = [&](const std::pair<size_t, size_t>& edge)
        -> std::string {
        const auto& metaEdge = clone.getEdgeProperty(edge);
        if (metaEdge.m_isSchedulingEdge) {
            return "";
        }
        if (metaEdge.nDependencies() > 0) {
            return metaEdge.toString();
        }
        return "";
    };

    auto edgeLabelProperty = [&](const std::pair<size_t, size_t>& edge)
        -> std::string {
        const auto& metaEdge = clone.getEdgeProperty(edge);
        if (metaEdge.m_isSchedulingEdge) {
            // return "style=dashed, color=\"#2A9D8F\"";
            return "style=dashed, color=\"#F4A261\", penwidth=7";
        }
        return "color=\"#d9d9d9\", penwidth=7";
    };

    auto vertexLabelProperty = [&](const size_t& v) {
        if (v == m_finalNodeId() || (v == m_rootNodeId())) {
            return R"(shape=doublecircle, style=filled, fillcolor="#d9d9d9", color="#6c6c6c")";
        }
        const auto& metaNode = clone.getVertexProperty(v);
        if (metaNode.isHu()) {
            return R"(shape=octagon, style="rounded,filled", fillcolor="#fb8072", color="#b11605")";
        }
        if (metaNode.isSync()) {
            return R"(shape=octagon, style="rounded,filled", fillcolor="#fb8072", color="#b11605")";
        }

        if (metaNode.isMap()) {
            return R"(style=filled, fillcolor="#b3de69", color="#5f861d")";
        }
        if (metaNode.isReduce()) {
            return R"(style=filled, fillcolor="#80b1d3", color="#2b5c7d")";
        }
        if (metaNode.isStencil()) {
            return R"(style=filled, fillcolor="#bebada", color="#4e4683")";
        }
        return "";
    };

    clone.exportDotFile(fname, graphName, vertexLabel, edgeLabel, vertexLabelProperty, edgeLabelProperty);
}

auto MultiGpuGraph::h_trackNewContainer(const Neon::set::Container& container,
                                        std::vector<size_t>&        containerId2gNodeId) -> ContainerIdx
{
    ContainerIdx kernelContainerIdx = m_kContainers().size();
    m_kContainers().push_back(container);

    // Adding the kernel container into the graph
    auto node = MetaNode::factory(MetaNodeType_e::CONTAINER, container.getName(), kernelContainerIdx);
    m_graph().addVertex(node.nodeId(), node);

    containerId2gNodeId.push_back(node.nodeId());

    return kernelContainerIdx;
}

auto MultiGpuGraph::h_removeRedundantDependencies() -> void
{
    std::vector<std::pair<size_t, size_t>> edgesToBeRemoved;
    m_graph().forEachVertex([&](size_t node) {
        // Looping over all nodes
        const auto& children = m_graph().outNeighbors(node);
        // Checking sons. If no more than one, move to the next node
        if (children.size() <= 1) {
            return;
        }
        for (const auto& child : children) {
            if (m_graph().inEdges(child).size() <= 1) {
                // This child can only be reach by once father
                // No redundant path here
                continue;
            }
            bool foundRedundant = false;
            for (const auto& sibling : children) {
                if (sibling == child) {
                    continue;
                }
                auto frontiar = m_graph().outNeighbors(sibling);
                while (frontiar.size() != 0) {
                    auto nextFrontiar = std::set<size_t>();

                    for (const auto& eInFrintier : frontiar) {
                        if (eInFrintier == child) {
                            foundRedundant = true;
                            break;
                        } else {
                            for (auto nextE : m_graph().outNeighbors(eInFrintier)) {
                                nextFrontiar.insert(nextE);
                            }
                        }
                    }  // end FOF
                    if (foundRedundant) {
                        break;
                    }
                    frontiar = nextFrontiar;
                }
                if (foundRedundant)
                    break;
            }
            if (foundRedundant) {
                edgesToBeRemoved.push_back({node, child});
            }
        }
    });
    for (const auto& toBeRemoved : edgesToBeRemoved) {
        m_graph().removeEdge(toBeRemoved);
    }
}

/**
 * Creates 2 nodes for
 * @param uid
 * @param edge
 * @param infoIdx
 * @param transferModeE
 * @return
 */
auto MultiGpuGraph::h_add_hUpdateSyncNodes(size_t                      t0Node,
                                           size_t                      t1Node,
                                           const internal::Edge::Info& info,
                                           Neon::set::TransferMode     transferModeE)
    -> void
{
    auto getNewEdge = [&]() {
        const bool haloUpdate = true;
        return internal::Edge::factory(info.token, info.dependency, haloUpdate);
    };

    auto hupNode = MetaNode::haloUpdateFactory(transferModeE,
                                               info.token.uid(),
                                               info.token.getHaloUpdate(),
                                               info.token.getHaloUpdatePerDevice());

    m_graph().addVertex(hupNode.nodeId(), hupNode);

    auto syncNode = MetaNode::syncLeftRightFactory(info.token.uid());
    m_graph().addVertex(syncNode.nodeId(), syncNode);

    size_t syncNodet0;
    size_t syncNodet1;

    if (transferModeE == Neon::set::TransferMode::put) {
        syncNodet0 = hupNode.nodeId();
        syncNodet1 = syncNode.nodeId();
    } else {
        syncNodet0 = syncNode.nodeId();
        syncNodet1 = hupNode.nodeId();
    }

    {  // linking the two halo and sync nodes
        auto hpSycInternlaEdge = getNewEdge();
        m_graph().addEdge(syncNodet0, syncNodet1, hpSycInternlaEdge);
    }

    {
        {
            auto fromT0 = getNewEdge();
            m_graph().addEdge(t0Node, syncNodet0, fromT0);
        }
        {
            auto toT1 = getNewEdge();
            m_graph().addEdge(syncNodet1, t1Node, toT1);
        }
    }

    const auto& inNgh = m_schedulingGraph().inNeighbors(t1Node);
    if (!inNgh.empty()) {
        m_schedulingGraph().addVertex(syncNodet0);
    }
    for (const auto& toBeFirst : inNgh) {
        m_schedulingGraph().addEdge(toBeFirst, syncNodet0);
    }

    for (const auto& toBeFirst : inNgh) {
        m_schedulingGraph().removeEdge(toBeFirst, t1Node);
    }
    return;
}

auto MultiGpuGraph::optimizations(const Neon::skeleton::Options& options) -> void
{
    switch (options.occ()) {
        case Neon::skeleton::Occ::none:
            return;
        case Neon::skeleton::Occ::standard:
            return optimizeStandardOCC(options);
        case Neon::skeleton::Occ::extended:
            return optimizeExtendedOCC(options);
        case Neon::skeleton::Occ::twoWayExtended:
            return optimizeTwoWayExtendedOCC(options);
    }
}

auto MultiGpuGraph::optimizeStandardOCC(const Neon::skeleton::Options&) -> void
{
    /**
     * Objective:
     * a. detect all stencil nodes
     * b. each stencil node is slipped into internal and boundary
     * c. all the dependency must be cloned too
     */
    using Compute_e = Neon::Compute;

    // Detects all stencil nodes
    std::vector<size_t> stencilNodes;
    m_graph().forEachVertex([&](size_t nodeId) {
        const auto& node = m_graph().getVertexProperty(nodeId);
        if (node.getCompute() == Compute_e::STENCIL) {
            if (node.getDataView() != Neon::DataView::INTERNAL) {
                stencilNodes.push_back(nodeId);
            }
        }
    });
    // Cloning each stencil node
    for (auto&& stencilNode : stencilNodes) {
        // Data used by the helper functions
        const auto inEdges = m_graph().inEdges(stencilNode);
        const auto outEdges = m_graph().outEdges(stencilNode);
        // HELPER //////////////////////////////////////////////////////////////////////////
        auto h_cloneStencilNode = [&](const Neon::DataView& dataView) -> size_t {
            const auto& toBeCloned = m_graph().getVertexProperty(stencilNode);
            auto        clone = toBeCloned.clone();
            clone.setDataView(dataView);
            m_graph().addVertex(clone.nodeId(), clone);
            return clone.nodeId();
        };
        // HELPER //////////////////////////////////////////////////////////////////////////
        auto h_cloneIncomingConnections = [&](size_t clone) -> void {
            std::for_each(outEdges.begin(), outEdges.end(),
                          [&](const DiGraph::Edge& edge) {
                              auto clonedEdge = m_graph().getEdgeProperty(edge).clone();
                              m_graph().addEdge(clone, edge.second, clonedEdge);
                          });
        };
        // HELPER //////////////////////////////////////////////////////////////////////////
        auto h_cloneOutgoingConnections = [&](size_t clone) -> void {
            std::for_each(inEdges.begin(), inEdges.end(),
                          [&](const DiGraph::Edge& edge) {
                              auto clonedEdge = m_graph().getEdgeProperty(edge).clone();
                              m_graph().addEdge(edge.first, clone, clonedEdge);
                          });
        };
        // HELPER //////////////////////////////////////////////////////////////////////////
        auto cloningProcess = [&](const Neon::DataView& dataView) -> NodeId {
            NodeId cloneId = h_cloneStencilNode(dataView);
            h_cloneIncomingConnections(cloneId);
            h_cloneOutgoingConnections(cloneId);
            return cloneId;
        };
        ////////////////////////////////////////////////////////////////////////////
        NodeId internalNode = cloningProcess(Neon::DataView::INTERNAL);
        NodeId boundaryNode = cloningProcess(Neon::DataView::BOUNDARY);
        for (auto& node : {internalNode, boundaryNode}) {
            m_schedulingGraph().hasVertex(node);
            m_schedulingGraph().addVertex(node);
        }

        if (!m_schedulingGraph().hasEdge({boundaryNode, internalNode})) {
            m_schedulingGraph().addEdge(internalNode, boundaryNode);
        }
    }

    for (auto&& stencilNode : stencilNodes) {
        m_graph().removeVertex(stencilNode);
    }
}

auto MultiGpuGraph::optimizeExtendedOCC(const Neon::skeleton::Options&) -> void
{

    // Detects all stencil nodes
    std::vector<size_t> potentialStencilNodes;
    m_graph().forEachVertex([&](size_t nodeId) {
        const auto& node = m_graph().getVertexProperty(nodeId);
        if (node.getCompute() == Neon::Compute::STENCIL) {
            if (node.getDataView() == Neon::DataView::STANDARD) {
                potentialStencilNodes.push_back(nodeId);
            }
        }
    });
    std::vector<size_t> targetStencilNodes;

    for (auto& potentialStencilNode : potentialStencilNodes) {
        bool attachedToRoot = false;
        bool proceedByMap = false;
        m_graph().forEachInEdge(potentialStencilNode, [&](const std::pair<size_t, size_t>& edge) {
            if (edge.first == m_rootNodeId()) {
                attachedToRoot = true;
            }
            if (!m_graph().getVertexProperty(edge.first).isStencil()) {
                proceedByMap = true;
            }
        });
        if (!attachedToRoot && proceedByMap) {
            targetStencilNodes.push_back(potentialStencilNode);
        }
    }
    // HELPER //////////////////////////////////////////////////////////////////////////
    auto h_cloneKernelNodeNode = [&](size_t node, const Neon::DataView& dataView) -> size_t {
        const auto& toBeCloned = m_graph().getVertexProperty(node);
        auto        clone = toBeCloned.clone();
        clone.setDataView(dataView);
        m_graph().addVertex(clone.nodeId(), clone);
        return clone.nodeId();
    };

    // HELPER //////////////////////////////////////////////////////////////////////////
    auto h_cloneOutgoingConnections = [&](std::set<std::pair<size_t, size_t>> outEdges, size_t clone) -> void {
        std::for_each(outEdges.begin(), outEdges.end(),
                      [&](const DiGraph::Edge& edge) {
                          auto clonedEdge = m_graph().getEdgeProperty(edge).clone();
                          m_graph().addEdge(clone, edge.second, clonedEdge);
                      });
    };

    // HELPER //////////////////////////////////////////////////////////////////////////
    auto h_cloneIncomingConnections = [&](std::set<std::pair<size_t, size_t>> inEdges, size_t clone) -> void {
        std::for_each(inEdges.begin(), inEdges.end(),
                      [&](const DiGraph::Edge& edge) {
                          auto clonedEdge = m_graph().getEdgeProperty(edge).clone();
                          m_graph().addEdge(edge.first, clone, clonedEdge);
                      });
    };

    // HELPER //////////////////////////////////////////////////////////////////////////
    auto h_cloningProcess = [&](size_t st1, std::vector<size_t> mt0s) -> void {
        /**
         * st1: stencil node
         * mt0s: set of map nodes at the previous level
         *
         * st1 stencil t1
         * mto are map at t0
         */
        this->h_io2Dot("A.dot", "A");
        // Cloning the stencil node
        size_t st1_internal = h_cloneKernelNodeNode(st1, Neon::DataView::INTERNAL);
        size_t st1_boundary = h_cloneKernelNodeNode(st1, Neon::DataView::BOUNDARY);
        m_schedulingGraph().addVertex(st1_internal);
        m_schedulingGraph().addVertex(st1_boundary);
        m_schedulingGraph().addEdge(st1_internal, st1_boundary);

        this->h_io2Dot("B.dot", "B");

        {
            auto inE = m_graph().inEdges(st1);
            h_cloneIncomingConnections(inE, st1_internal);
            h_cloneIncomingConnections(inE, st1_boundary);
        }
        {
            auto outE = m_graph().outEdges(st1);
            h_cloneOutgoingConnections(outE, st1_internal);
            h_cloneOutgoingConnections(outE, st1_boundary);
        }
        this->h_io2Dot("C.dot", "B");

        for (auto& mt0 : mt0s) {
            // cloning map node into boundary and internal
            size_t mapInternal = h_cloneKernelNodeNode(mt0, Neon::DataView::INTERNAL);
            size_t mapBoundary = h_cloneKernelNodeNode(mt0, Neon::DataView::BOUNDARY);
            m_schedulingGraph().addVertex(mapInternal);
            m_schedulingGraph().addVertex(mapBoundary);
            m_schedulingGraph().addEdge(mapBoundary, mapInternal);

            {
                auto inE = m_graph().inEdges(mt0);
                h_cloneIncomingConnections(inE, mapInternal);
                h_cloneIncomingConnections(inE, mapBoundary);
            }
            this->h_io2Dot("D.dot", "B");

            {
                auto outE = m_graph().outEdges(mt0);
                h_cloneOutgoingConnections(outE, mapInternal);
                h_cloneOutgoingConnections(outE, mapBoundary);
            }

            {  // the stencil arc from map_internal and stencil{internal/boundary} mast be set to map
                for (auto& st1Target : {st1_internal, st1_boundary}) {
                    auto& edge = m_graph().getEdgeProperty({mapInternal, st1Target});
                    auto& infos = edge.infoMutable();
                    for (auto& info : infos) {
                        info.flag_discardStencilDep = true;
                    }
                }
            }

            this->h_io2Dot("E.dot", "B");
            //            //m_graph().removeEdge({internal, st1_boundary});
            //            //m_graph().removeEdge({boundary, st1_internal});
            this->h_io2Dot("F.dot", "B");
        }
        m_graph().removeVertex(st1);
        this->h_io2Dot("G.dot", "B");
        for (auto& mt0 : mt0s) {
            m_graph().removeVertex(mt0);
        }
        this->h_io2Dot("H.dot", "B");
    };

    ////////////////////////////////////////////////////////////////////////////
    auto node2Level = h_getBFSIndexes();
    for (auto&& stencilNode : targetStencilNodes) {
        int                 stencilNodeLevel = node2Level[stencilNode];
        std::vector<size_t> toClone;
        auto                inNodes = m_graph().inNeighbors(stencilNode);
        for (const auto& inNode : inNodes) {
            if (node2Level[inNode] == stencilNodeLevel - 1) {
                toClone.push_back(inNode);
            }
        }
        h_cloningProcess(stencilNode, toClone);
    }
}

auto MultiGpuGraph::optimizeTwoWayExtendedOCC(const Neon::skeleton::Options&) -> void
{

    // Detects all stencil nodes
    std::vector<size_t> potentialStencilNodes;
    m_graph().forEachVertex([&](size_t nodeId) {
        const auto& node = m_graph().getVertexProperty(nodeId);
        if (node.getCompute() == Neon::Compute::STENCIL) {
            if (node.getDataView() == Neon::DataView::STANDARD) {
                potentialStencilNodes.push_back(nodeId);
            }
        }
    });
    std::vector<size_t> targetStencilNodes;

    for (auto& potentialStencilNode : potentialStencilNodes) {
        bool attachedToRoot = false;
        bool attachedToFinal = false;
        bool proceedByMap = false;
        bool followedByMapOrDot = false;

        m_graph().forEachInEdge(potentialStencilNode, [&](const std::pair<size_t, size_t>& edge) {
            if (edge.first == m_rootNodeId()) {
                attachedToRoot = true;
            }
            // only MAP front kernels can be splitted
            if (m_graph().getVertexProperty(edge.first).isMap()) {
                proceedByMap = true;
            }
        });
        m_graph().forEachOutEdge(potentialStencilNode, [&](const std::pair<size_t, size_t>& edge) {
            if (edge.second == m_finalNodeId()) {
                attachedToFinal = true;
            }

            // only MAP or DOT (-> not STENCIL) back kernels can be splitted
            if (!m_graph().getVertexProperty(edge.second).isStencil()) {
                followedByMapOrDot = true;
            }
        });
        if (!attachedToRoot && !attachedToFinal && proceedByMap && followedByMapOrDot) {
            targetStencilNodes.push_back(potentialStencilNode);
        }
    }
    // HELPER //////////////////////////////////////////////////////////////////////////
    auto h_cloneKernelNodeNode = [&](size_t node, const Neon::DataView& dataView) -> size_t {
        const auto& toBeCloned = m_graph().getVertexProperty(node);
        auto        clone = toBeCloned.clone();
        clone.setDataView(dataView);
        m_graph().addVertex(clone.nodeId(), clone);
        return clone.nodeId();
    };

    // HELPER //////////////////////////////////////////////////////////////////////////
    auto h_cloneOutgoingConnections = [&](std::set<std::pair<size_t, size_t>> outEdges, size_t clone) -> void {
        std::for_each(outEdges.begin(), outEdges.end(),
                      [&](const DiGraph::Edge& edge) {
                          auto clonedEdge = m_graph().getEdgeProperty(edge).clone();
                          m_graph().addEdge(clone, edge.second, clonedEdge);
                      });
    };

    // HELPER //////////////////////////////////////////////////////////////////////////
    auto h_cloneIncomingConnections = [&](std::set<std::pair<size_t, size_t>> inEdges, size_t clone) -> void {
        std::for_each(inEdges.begin(), inEdges.end(),
                      [&](const DiGraph::Edge& edge) {
                          auto clonedEdge = m_graph().getEdgeProperty(edge).clone();
                          m_graph().addEdge(edge.first, clone, clonedEdge);
                      });
    };

    // HELPER //////////////////////////////////////////////////////////////////////////
    auto h_cloningProcess = [&](const std::set<size_t>& frontNodes,
                                size_t                  StencilNode,
                                const std::set<size_t>& backNodes) -> void {
        std::set<size_t> backNodesInternal;
        std::set<size_t> backNodesBoundary;
        std::set<size_t> frontMapInternal;

        /**
         * st1: stencil node
         * mt0s: set of map nodes at the previous level
         *
         * st1 stencil t1
         * mto are map at t0
         */
        this->h_io2Dot("A.dot", "A");

        // Cloning from the back toward the front
        for (auto& mapBack : backNodes) {
            // cloning map node into boundary and internal
            size_t mapInternal = h_cloneKernelNodeNode(mapBack, Neon::DataView::INTERNAL);
            size_t mapBoundary = h_cloneKernelNodeNode(mapBack, Neon::DataView::BOUNDARY);

            backNodesInternal.insert(mapInternal);
            backNodesBoundary.insert(mapBoundary);

            m_schedulingGraph().addVertex(mapInternal);
            m_schedulingGraph().addVertex(mapBoundary);

            const auto& metaNode = m_graph().getVertexProperty(mapBack);
            if (metaNode.isReduce()) {
                m_graph().addEdge(mapInternal, mapBoundary);
            } else {
                m_schedulingGraph().addEdge(mapInternal, mapBoundary);
            }


            {
                auto inE = m_graph().inEdges(mapBack);
                h_cloneIncomingConnections(inE, mapInternal);
                h_cloneIncomingConnections(inE, mapBoundary);
            }
            this->h_io2Dot("D.cloning.back.dot", "B");

            {
                auto outE = m_graph().outEdges(mapBack);
                h_cloneOutgoingConnections(outE, mapInternal);
                h_cloneOutgoingConnections(outE, mapBoundary);
            }

            this->h_io2Dot("E.cloning.back.dot", "B");
        }

        for (auto& mapBack : backNodes) {
            m_graph().removeVertex(mapBack);
        }


        for (auto& mapFront : frontNodes) {
            // cloning map node into boundary and internal
            size_t mapBoundary = h_cloneKernelNodeNode(mapFront, Neon::DataView::BOUNDARY);
            size_t mapInternal = h_cloneKernelNodeNode(mapFront, Neon::DataView::INTERNAL);

            frontMapInternal.insert(mapInternal);

            m_schedulingGraph().addVertex(mapInternal);
            m_schedulingGraph().addVertex(mapBoundary);
            m_schedulingGraph().addEdge(mapBoundary, mapInternal);

            {
                auto inE = m_graph().inEdges(mapFront);
                h_cloneIncomingConnections(inE, mapInternal);
                h_cloneIncomingConnections(inE, mapBoundary);
            }
            this->h_io2Dot("D.cloning.front.dot", "B");

            {
                auto outE = m_graph().outEdges(mapFront);
                h_cloneOutgoingConnections(outE, mapInternal);
                h_cloneOutgoingConnections(outE, mapBoundary);
            }

            this->h_io2Dot("E.cloning.front.dot", "B");
        }
        for (auto& mapFront : frontNodes) {
            m_graph().removeVertex(mapFront);
        }

        // Cloning the stencil node
        size_t st1_internal = h_cloneKernelNodeNode(StencilNode, Neon::DataView::INTERNAL);
        size_t st1_boundary = h_cloneKernelNodeNode(StencilNode, Neon::DataView::BOUNDARY);
        m_schedulingGraph().addVertex(st1_internal);
        m_schedulingGraph().addVertex(st1_boundary);
        m_schedulingGraph().addEdge(st1_internal, st1_boundary);

        this->h_io2Dot("B.dot", "B");

        {
            auto inE = m_graph().inEdges(StencilNode);
            h_cloneIncomingConnections(inE, st1_internal);
            h_cloneIncomingConnections(inE, st1_boundary);
        }
        {
            auto outE = m_graph().outEdges(StencilNode);
            h_cloneOutgoingConnections(outE, st1_internal);
            h_cloneOutgoingConnections(outE, st1_boundary);
        }

        for (auto& backNodeIternal : backNodesInternal) {
            m_schedulingGraph().addVertex(backNodeIternal);
            m_schedulingGraph().addVertex(st1_boundary);
            m_schedulingGraph().addEdge(backNodeIternal, st1_boundary);
        }
        this->h_io2Dot("C.dot", "B");

        m_graph().removeVertex(StencilNode);


        for (const auto& node : backNodesBoundary) {
            m_graph().removeEdge(st1_internal, node);
        }
        for (const auto& node : backNodesInternal) {
            m_graph().removeEdge(st1_boundary, node);
        }
        {  // the stencil arc from map_internal and stencil{internal/boundary} mast be set to map
            for (auto& st1Target : {st1_internal, st1_boundary}) {
                for (auto& mapInternal : frontMapInternal) {
                    auto& edge = m_graph().getEdgeProperty({mapInternal, st1Target});
                    auto& infos = edge.infoMutable();
                    for (auto& info : infos) {
                        info.flag_discardStencilDep = true;
                    }
                }
            }
        }


        this->h_io2Dot("HHH.dot", "B");
    };

    ////////////////////////////////////////////////////////////////////////////
    auto node2Level = h_getBFSIndexes();
    for (auto&& stencilNode : targetStencilNodes) {
        int              stencilNodeLevel = node2Level[stencilNode];
        std::set<size_t> frontToClone;
        std::set<size_t> backToClone;

        auto inNodes = m_graph().inNeighbors(stencilNode);
        for (const auto& inNode : inNodes) {
            if (node2Level[inNode] == stencilNodeLevel - 1) {
                frontToClone.insert(inNode);
            }
        }

        auto outNodes = m_graph().outNeighbors(stencilNode);
        for (const auto& outNode : outNodes) {
            if (outNode != m_finalNodeId()) {
                backToClone.insert(outNode);
            }
        }
        h_cloningProcess(frontToClone, stencilNode, backToClone);
    }
}

/**
 * Breadth First Traversal
 * @param f
 * @return
 */
auto MultiGpuGraph::h_BFT(std::function<void(int                     level,
                                             const std::set<size_t>& nodes)> f)
    -> void
{
    std::set<size_t>              visited;
    std::vector<std::set<size_t>> levels;
    levels.push_back(std::set<size_t>());

    size_t level = 0;
    levels[level].insert(m_rootNodeId());

    auto h_recordNewNode = [&](size_t node, size_t nodeLevel) -> bool {
        if (!visited.insert(node).second) {
            return false;
        }
        levels.at(nodeLevel).insert(node);
        return true;
    };

    while (level < levels.size() && levels[level].size() > 0) {
        levels.push_back(std::set<size_t>());
        for (const auto& n : levels[level]) {
            auto nghSet = m_graph().neighbors(n);
            for (const auto& ngh : nghSet) {
                h_recordNewNode(ngh, level + 1);
            }
        }
        level++;
    }

    // Reset level for visiting
    level = 0;
    while (level < levels.size() && levels[level].size() > 0) {
        f(int(level), levels[level]);
        level++;
    }
    return;
}

auto MultiGpuGraph::h_getBFSIndexes() -> std::unordered_map<size_t, int>
{
    std::unordered_map<size_t, int> map;

    h_BFT([&](int                     level,
              const std::set<size_t>& nodes) {
        for (const auto& n : nodes) {
            map[n] = level;
        }
    });

    return map;
}

auto MultiGpuGraph::getDiGraph() -> DiGraph&
{
    return m_graph();
}
auto MultiGpuGraph::getSchedulingDiGraph() -> DiGraphScheduling&
{
    return m_schedulingGraph();
}
//    auto fuseMaps(const Neon::skeleton::Options_t& options) -> void
//    {
//        // BFS visit
//        std::vector<int> frontiar;
//        m_graph().forEachOutEdge(m_rootNodeId, [&](const std::pair<size_t, size_t>& edge) {
//            frontiar.push_back(edge.second);
//        });
//
//        size_t frontiarParserIdx = 0;
//
//        /**
//         * Return -1 if there are no condition for a fusion
//         * Otherwise, it returns the node to fuse with
//         */
//        auto h_conditionForFusion = [&](const size_t n0) -> int64_t {
//            if (m_graph().outEdgesCount(n0) != 1) {
//                return -1;
//            }
//            const size_t n1 = m_graph().outEdges(n0).begin()->second;
//            if (m_graph().inEdgesCount(n1) != 1) {
//                return -1;
//            }
//            if(m_graph().getVertexProperty(n1).isStencil()){
//                return -1;
//            }
//            return n1;
//        };
//
//        auto h_fuse= [&](const size_t n0, const size_t n1){
//            //1. create a fused kernel container
//        };
//
//        while (frontiarParserIdx < frontiar.size()) {
//            size_t nodeId = frontiar[frontiarParserIdx];
//            if (m_graph().outEdges(nodeId).size() == 1 &&) {
//                // a. Fuse
//
//                // b. update nodeId to the new node
//            }
//            // c. continue with BFS
//        }
//    }

auto MultiGpuGraph::addSyncAndMemoryTransfers(const Neon::skeleton::Options& options) -> void
{
    if (m_setCardinality() == 1) {
        return;
    }

    // Detects all stencil nodes
    std::vector<size_t> stencilNodes;
    m_graph().forEachVertex([&](size_t nodeId) {
        const auto& node = m_graph().getVertexProperty(nodeId);
        if (node.getCompute() == Neon::Compute::STENCIL) {
            if (node.getDataView() != Neon::DataView::INTERNAL) {
                stencilNodes.push_back(nodeId);
            }
        }
    });
    for (auto&& stencilNode : stencilNodes) {

        const auto             inEdges = m_graph().inEdges(stencilNode);
        int                    detectedStencilEdges = 0;
        [[maybe_unused]] auto& nodeInfo = m_graph().getVertexProperty(stencilNode);


        // m_graph().forEachInEdge(stencilNode,
        // We are using a copy of the inEdges,
        // so we can modify the graph without any issue
        std::for_each(inEdges.begin(), inEdges.end(),
                      [&](const DiGraph::Edge& edge) {
                          bool  visited = false;
                          auto& edgeP = m_graph().getEdgeProperty(edge);
                          auto& targetProperty = edgeP.infoMutable();

                          // remove_if only  move elements at the end of the vector
                          auto end = std::remove_if(targetProperty.begin(),
                                                    targetProperty.end(),
                                                    [&](Edge::Info& info) -> bool {
                                                        if (info.token.compute() == Neon::Compute::STENCIL && !info.flag_discardStencilDep) {
                                                            visited = true;
                                                            detectedStencilEdges++;
                                                            h_add_hUpdateSyncNodes(edge.first,
                                                                                   edge.second,
                                                                                   info,
                                                                                   options.transferMode());

                                                            return true;
                                                        } else {
                                                            visited = true;
                                                            return false;
                                                        }
                                                    });

                          targetProperty.erase(end, targetProperty.end());
                          if (detectedStencilEdges > 0 && m_graph().getEdgeProperty(edge).nDependencies() == 0) {
                              m_graph().removeEdge(edge);
                          }
                      });

        if (detectedStencilEdges != 1) {
            NEON_THROW_UNSUPPORTED_OPTION("Only one stencil in field is supported for now");
        }
    }
}

auto MultiGpuGraph::checkCoherency() -> void
{
    // Detects all stencil nodes
    std::vector<size_t> stencilNodes;
    m_graph().forEachVertex([&](size_t nodeId) {
        const auto& node = m_graph().getVertexProperty(nodeId);
        if (node.getCompute() == Neon::Compute::STENCIL) {
            if (m_setCardinality() == 1) {
                m_graph().getVertexProperty(nodeId).setAsCoherent();
                return;
            }

            if (node.getDataView() == Neon::DataView::INTERNAL) {
                m_graph().getVertexProperty(nodeId).setAsCoherent();
            } else {
                stencilNodes.push_back(nodeId);
            }
        }
    });
    for (auto&& stencilNode : stencilNodes) {
        bool isMemoryCoherent = true;
        m_graph().forEachInEdge(stencilNode, [&](const DiGraph::Edge& edge) -> void {
            const auto& e = m_graph().getEdgeProperty(edge);
            for (const auto& i : e.info()) {
                if (i.isStencil() && !i.isHu()) {
                    isMemoryCoherent = false;
                }
            }
        });
        if (isMemoryCoherent) {
            m_graph().getVertexProperty(stencilNode).setAsCoherent();
        }
    }
}

auto MultiGpuGraph::resetLinearContinuousIndexing() -> void
{
    m_storage->m_resetLinearContinuousIndexingCounter = 0;
    m_graph().forEachVertex([&](size_t nodeId) {
        auto& node = m_graph().getVertexProperty(nodeId);
        node.setLinearContinuousIndex(m_storage->m_resetLinearContinuousIndexingCounter);
        m_storage->m_resetLinearContinuousIndexingCounter++;
    });
}

auto MultiGpuGraph::getLinearContinuousIndexingCounter() -> size_t
{
    return m_storage->m_resetLinearContinuousIndexingCounter;
}

/**
 * Access methods to members
 * @return
 */
auto MultiGpuGraph::rootNodeId() const -> const size_t&
{
    return m_rootNodeId();
}

/**
 * Access methods to members
 * @return
 */
auto MultiGpuGraph::finalNodeId() const -> const size_t&
{
    return m_finalNodeId();
}
MultiGpuGraph::MultiGpuGraph()
{
    m_storage = std::make_shared<Storage>();
}
}  // namespace Neon::skeleton::internal