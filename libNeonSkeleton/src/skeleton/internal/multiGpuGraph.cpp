#include <list>
#include "Neon/skeleton/internal/MultiXpuGraph.h"

namespace Neon::skeleton::internal {

void MultiXpuGraph::init(Neon::Backend&                           bk,
                         const std::vector<Neon::set::Container>& operations,
                         std::string                              /*name*/,
                         Options                                  options)
{
    mGraph() = Neon::set::container::Graph(bk);
    parse(bk.devSet().setCardinality(),
          std::forward<const std::vector<Neon::set::Container>&&>(operations));
    mGraph().removeRedundantDependencies();

    // h_ioToDot("t0_" + name + ".dot", "i");
    optimizations(options);
    // h_ioToDot("t1_" + name + ".dot", "i");
    addCommunications(options);
    mGraph().removeRedundantDependencies();

    checkCoherency();
    // h_ioToDot("t2_" + name + ".dot", "i");
    this->computeScheduling();
    // h_ioToDot("final" + name + ".dot", "i");
}

void MultiXpuGraph::parse(int                                       setCardinalty,
                          const std::vector<Neon::set::Container>&& operations)
{
    m_setCardinality() = setCardinalty;

    for (auto&& k : operations) {
        helpParseNewContainer(k);
    }
}

void MultiXpuGraph::
    helpParseNewContainer(const Neon::set::Container& inContainer)
{

    // Register and retrieve the id for the new container
    // add a node
    Neon::set::container::GraphInfo::NodeUid graphNodeUid;
    graphNodeUid = helpAddNewContainerToGraph(inContainer);

    // Parsing all the data toke used by the kernel container
    std::vector<Neon::set::dataDependency::Token> tokens;
    tokens = helpParseContainer(mGraph().helpGetGraphNode(graphNodeUid).getContainer());

    // Tokens are based on the multi-GPU data loaded by Containers
    for (auto& token : tokens) {
        // update the dependency state machine with the new token.
        // newDependencies are the detected dependencies

        auto newDependencies = m_dataRecords().updateStatus(graphNodeUid,
                                                            token.access(),
                                                            token.uid());

        for (auto& dep : newDependencies) {
            const auto& n0 = mGraph().helpGetGraphNode(dep.t0());
            const auto& n1 = mGraph().helpGetGraphNode(dep.t1());
            mGraph().addDependency(n0, n1, token);
        }
    }
}

auto MultiXpuGraph::
    helpParseContainer(Neon::set::Container& container)
        -> std::vector<Neon::set::dataDependency::Token>
{
    auto& kcInterface = container.getContainerInterface();
    auto& tokens = kcInterface.parse();
    return tokens;
}

auto MultiXpuGraph::h_ioToDot([[maybe_unused]] const std::string& fname,
                             [[maybe_unused]] const std::string& graphName) -> void
{
    ioToDot(fname, graphName, true);
}

auto MultiXpuGraph::
    ioToDot(const std::string& fname,
           const std::string& graphName,
           bool               debug) -> void
{
    mGraph().ioToDot(fname, graphName, debug);
}

auto MultiXpuGraph::
    io2DotOriginalApp(const std::string& fname,
                      const std::string& graphName,
                      bool               debug) -> void
{
    mGraph().ioToDot(fname, graphName, debug);
}

auto MultiXpuGraph::helpAddNewContainerToGraph(const Neon::set::Container& container) -> Neon::set::container::GraphInfo::NodeUid
{
    const auto&                              graphNode = mGraph().addNode(container);
    Neon::set::container::GraphInfo::NodeUid uid;
    uid = graphNode.getGraphData().getUid();
    return uid;
}


auto MultiXpuGraph::optimizations(const Neon::skeleton::Options& options) -> void
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

auto MultiXpuGraph::optimizeStandardOCC(const Neon::skeleton::Options&) -> void
{
    std::vector<Neon::set::container::GraphData::Uid> stencilNodeUidList;
    mGraph().forEachNode([&](const Neon::set::container::GraphNode& graphNode) {
        const auto type =
            graphNode.getContainer().getContainerInterface().getContainerPatternType();
        if (Neon::set::ContainerPatternType::stencil == type) {
            if (Neon::DataView::STANDARD == graphNode.getScheduling().getDataView()) {
                Neon::set::container::GraphData::Uid nodeUid = graphNode.getGraphData().getUid();
                stencilNodeUidList.push_back(nodeUid);
            }
        }
    });

    for (auto stencilNodeUid : stencilNodeUidList) {
        auto& boundary_sten = mGraph().helpGetGraphNode(stencilNodeUid);
        auto& internal_sten = mGraph().cloneNode(boundary_sten);

        internal_sten.getScheduling().setDataView(Neon::DataView::INTERNAL);
        boundary_sten.getScheduling().setDataView(Neon::DataView::BOUNDARY);

        mGraph().addDependency(internal_sten, boundary_sten, Neon::GraphDependencyType::scheduling);
    }
}

auto MultiXpuGraph::optimizeExtendedOCC(const Neon::skeleton::Options&) -> void
{
    std::vector<Neon::set::container::GraphData::Uid> stencilNodeUidList;
    mGraph().forEachNode([&](const Neon::set::container::GraphNode& graphNode) {
        const auto type =
            graphNode.getContainer().getContainerInterface().getContainerPatternType();
        if (Neon::set::ContainerPatternType::stencil == type) {
            if (Neon::DataView::STANDARD == graphNode.getScheduling().getDataView()) {
                Neon::set::container::GraphData::Uid nodeUid = graphNode.getGraphData().getUid();
                stencilNodeUidList.push_back(nodeUid);
            }
        }
    });

    auto proceedingNodesAreMapOnly = [&](const Neon::set::container::GraphNode& node)
        -> std::vector<Neon::set::container::GraphNode*> {
        auto proceeding = this->mGraph().getProceedingGraphNodes(node);
        for (auto preNodePtr : proceeding) {
            if (preNodePtr->getContainer().getContainerInterface().getContainerPatternType() !=
                Neon::set::ContainerPatternType::map) {
                return {};
            }
            if (preNodePtr->getScheduling().getDataView() != Neon::DataView::STANDARD) {
                return {};
            }
        }
        return proceeding;
    };

    for (auto stencilNodeUid : stencilNodeUidList) {
        auto proceedingNodes = proceedingNodesAreMapOnly(mGraph().helpGetGraphNode(stencilNodeUid));

        if (proceedingNodes.empty()) {
            continue;
        }

        for (auto mapNodePtr : proceedingNodes) {
            auto& boundary_map = *mapNodePtr;
            auto& internal_map = mGraph().cloneNode(boundary_map);

            internal_map.getScheduling().setDataView(Neon::DataView::INTERNAL);
            boundary_map.getScheduling().setDataView(Neon::DataView::BOUNDARY);

            mGraph().addDependency(boundary_map, internal_map, Neon::GraphDependencyType::scheduling);
        }

        auto& boundary_sten = mGraph().helpGetGraphNode(stencilNodeUid);
        auto& internal_sten = mGraph().cloneNode(boundary_sten);

        internal_sten.getScheduling().setDataView(Neon::DataView::INTERNAL);
        boundary_sten.getScheduling().setDataView(Neon::DataView::BOUNDARY);

        mGraph().addDependency(internal_sten, boundary_sten, Neon::GraphDependencyType::scheduling);
    }
}

auto MultiXpuGraph::optimizeTwoWayExtendedOCC(const Neon::skeleton::Options&) -> void
{
    std::vector<Neon::set::container::GraphData::Uid> stencilNodeUidList;
    mGraph().forEachNode([&](const Neon::set::container::GraphNode& graphNode) {
        const auto type =
            graphNode.getContainer().getContainerInterface().getContainerPatternType();
        if (Neon::set::ContainerPatternType::stencil == type) {
            if (Neon::DataView::STANDARD == graphNode.getScheduling().getDataView()) {
                Neon::set::container::GraphData::Uid nodeUid = graphNode.getGraphData().getUid();
                stencilNodeUidList.push_back(nodeUid);
            }
        }
    });

    auto proceedingNodesAreMapOnly = [&](const Neon::set::container::GraphNode& node)
        -> std::vector<Neon::set::container::GraphNode*> {
        auto proceeding = this->mGraph().getProceedingGraphNodes(node);
        for (auto preNodePtr : proceeding) {
            if (preNodePtr->getContainer().getContainerInterface().getContainerPatternType() !=
                Neon::set::ContainerPatternType::map) {
                return {};
            }
            if (preNodePtr->getScheduling().getDataView() != Neon::DataView::STANDARD) {
                return {};
            }
        }
        return proceeding;
    };
    auto subsequentNodesAreMapOnly = [&](const Neon::set::container::GraphNode& node)
        -> std::vector<Neon::set::container::GraphNode*> {
        auto subsequent = this->mGraph().getSubsequentGraphNodes(node);
        for (auto postNodePtr : subsequent) {
            if (postNodePtr->getContainer().getContainerInterface().getContainerPatternType() !=
                    Neon::set::ContainerPatternType::map &&
                postNodePtr->getContainer().getContainerInterface().getContainerPatternType() !=
                    Neon::set::ContainerPatternType::reduction) {
                return {};
            }
            if (postNodePtr->getScheduling().getDataView() != Neon::DataView::STANDARD) {
                return {};
            }
        }
        return subsequent;
    };

    for (auto stencilNodeUid : stencilNodeUidList) {
        auto proceedingNodes = proceedingNodesAreMapOnly(mGraph().helpGetGraphNode(stencilNodeUid));
        auto subsequentNodes = subsequentNodesAreMapOnly(mGraph().helpGetGraphNode(stencilNodeUid));

        if (proceedingNodes.empty() || subsequentNodes.empty()) {
            continue;
        }

        for (auto mapNodePtr : proceedingNodes) {
            auto& boundary_map = *mapNodePtr;
            auto& internal_map = mGraph().cloneNode(boundary_map);

            internal_map.getScheduling().setDataView(Neon::DataView::INTERNAL);
            boundary_map.getScheduling().setDataView(Neon::DataView::BOUNDARY);

            mGraph().addDependency(boundary_map, internal_map, Neon::GraphDependencyType::scheduling);
        }

        auto& boundary_stencil = mGraph().helpGetGraphNode(stencilNodeUid);
        auto& internal_stencil = mGraph().cloneNode(boundary_stencil);

        internal_stencil.getScheduling().setDataView(Neon::DataView::INTERNAL);
        boundary_stencil.getScheduling().setDataView(Neon::DataView::BOUNDARY);

        mGraph().addDependency(internal_stencil, boundary_stencil, Neon::GraphDependencyType::scheduling);

        for (auto mapOrReduceNodePtr : subsequentNodes) {
            if(mapOrReduceNodePtr->getContainer().getContainerInterface().getContainerPatternType()
                == Neon::set::ContainerPatternType::map) {
                auto& boundary_map = *mapOrReduceNodePtr;
                auto& internal_map = mGraph().cloneNode(boundary_map);

                internal_map.getScheduling().setDataView(Neon::DataView::INTERNAL);
                boundary_map.getScheduling().setDataView(Neon::DataView::BOUNDARY);

                mGraph().addDependency(internal_map, boundary_stencil, Neon::GraphDependencyType::scheduling);

                mGraph().removeDependency(mGraph().getDependency(internal_stencil, boundary_map));
                mGraph().removeDependency(mGraph().getDependency(boundary_stencil, internal_map));
            }
            if(mapOrReduceNodePtr->getContainer().getContainerInterface().getContainerPatternType()
                == Neon::set::ContainerPatternType::reduction) {
                auto& boundary_red = *mapOrReduceNodePtr;
                auto& internal_red = mGraph().cloneNode(boundary_red);

                internal_red.getScheduling().setDataView(Neon::DataView::INTERNAL);
                boundary_red.getScheduling().setDataView(Neon::DataView::BOUNDARY);

                mGraph().addDependency(internal_red, boundary_stencil, Neon::GraphDependencyType::scheduling);

                mGraph().removeDependency(mGraph().getDependency(internal_stencil, boundary_red));
                mGraph().removeDependency(mGraph().getDependency(boundary_stencil, internal_red));

                mGraph().addDependency(internal_red, boundary_red, Neon::GraphDependencyType::data);
            }
        }
    }
}


auto MultiXpuGraph::addCommunications(const Neon::skeleton::Options& skeletonOptions) -> void
{
    if (m_setCardinality() == 1) {
        return;
    }

    // List all dependencies related to stencil operations
    std::vector<const Neon::set::container::GraphDependency*> stencilTypeDependencies;
    mGraph().forEachDependency([&](const Neon::set::container::GraphDependency& dep) {
        if (dep.getType() != Neon::GraphDependencyType::scheduling) {
            if (dep.hasStencilDependency()) {
                auto dw = dep.getDestinationNode(mGraph()).getScheduling().getDataView();
                if (Neon::DataView::INTERNAL != dw) {
                    stencilTypeDependencies.push_back(&dep);
                }
            }
        }
    });

    std::vector<const Neon::set::container::GraphDependency*> toBeRemoved;


    for (auto depPtr : stencilTypeDependencies) {
        const auto& dep = *depPtr;
        const auto  rawEdge = dep.getRawEdge();
        auto        nodeA = mGraph().helpGetGraphNode(rawEdge.first);
        auto        nodeB = mGraph().helpGetGraphNode(rawEdge.second);

        const Neon::set::container::GraphDependency::Tokens& tokens = dep.getTokens();
        int                                                  numNewNodes = 0;

        for (const auto& token : tokens) {
            if (token.compute() == Neon::Compute::STENCIL) {
                auto container = token.getDataTransferContainer(skeletonOptions.transferMode());
                numNewNodes += mGraph().expandAndMerge(nodeA, container, nodeB, true);
            }
        }

        auto schedulingDepOfB = this->mGraph().getProceedingGraphNodes(nodeB, {GraphDependencyType::scheduling});
        for (auto nodePtr : schedulingDepOfB) {
            const auto& dependency = mGraph().getDependency(*nodePtr, nodeB);
            toBeRemoved.push_back(&dependency);
        }
    }

    for (auto depPtr : toBeRemoved) {
        mGraph().removeDependency(*depPtr);
    }
}

auto MultiXpuGraph::checkCoherency() -> void
{
    //    // Detects all stencil nodes
    //    std::vector<size_t> stencilNodes;
    //    m_graph().forEachVertex([&](size_t nodeId) {
    //        const auto& node = m_graph().getVertexProperty(nodeId);
    //        if (node.getCompute() == Neon::Compute::STENCIL) {
    //            if (m_setCardinality() == 1) {
    //                m_graph().getVertexProperty(nodeId).setAsCoherent();
    //                return;
    //            }
    //
    //            if (node.getDataView() == Neon::DataView::INTERNAL) {
    //                m_graph().getVertexProperty(nodeId).setAsCoherent();
    //            } else {
    //                stencilNodes.push_back(nodeId);
    //            }
    //        }
    //    });
    //    for (auto&& stencilNode : stencilNodes) {
    //        bool isMemoryCoherent = true;
    //        m_graph().forEachInEdge(stencilNode, [&](const DiGraph::Edge& edge) -> void {
    //            const auto& e = m_graph().getEdgeProperty(edge);
    //            for (const auto& i : e.info()) {
    //                if (i.isStencil() && !i.isHu()) {
    //                    isMemoryCoherent = false;
    //                }
    //            }
    //        });
    //        if (isMemoryCoherent) {
    //            m_graph().getVertexProperty(stencilNode).setAsCoherent();
    //        }
    //    }
}

MultiXpuGraph::
    MultiXpuGraph()
{
    m_storage = std::make_shared<Storage>();
}

auto MultiXpuGraph::
    computeScheduling() -> void
{
    mGraph().runtimePreSet(Neon::Backend::mainStreamIdx);
}

auto MultiXpuGraph::
    execute()
        -> void
{
    this->mGraph().helpExecute(Neon::Backend::mainStreamIdx);
}
}  // namespace Neon::skeleton::internal