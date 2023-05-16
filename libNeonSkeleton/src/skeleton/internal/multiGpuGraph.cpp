#include <list>
#include "Neon/skeleton/internal/MultiXpuGraph.h"

namespace Neon::skeleton::internal {

void MultiXpuGraph::init(Neon::Backend&                           bk,
                         const std::vector<Neon::set::Container>& operations,
                         std::string /*name*/,
                         Options options)
{
    getGraph() = Neon::set::container::Graph(bk);
    parse(bk.devSet().setCardinality(),
          std::forward<const std::vector<Neon::set::Container>&&>(operations));
    getGraph().removeRedundantDependencies();

    // Stencil dependencies with the beginNode are not detected by the data dependency analysis.
    // We fix them manually after all redundant dependencies are cleaned.
    fixingDependenciesWithBeginNode();

    // h_ioToDot("t0_" + name + ".dot", "i");
    optimizations(options);
    // h_ioToDot("t1_" + name + ".dot", "i");
    communications(options);
    getGraph().removeRedundantDependencies();


    // h_ioToDot("t2_" + name + ".dot", "i");
    this->computeScheduling();
    // h_ioToDot("final" + name + ".dot", "i");
}

void MultiXpuGraph::parse(int                                       setCardinalty,
                          const std::vector<Neon::set::Container>&& operations)
{
    getSetCardinality() = setCardinalty;

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
    tokens = helpParseContainer(getGraph().helpGetGraphNode(graphNodeUid).getContainer());

    // Tokens are based on the multi-GPU data loaded by Containers
    for (auto& token : tokens) {
        // update the dependency state machine with the new token.
        // newDependencies are the detected dependencies

        auto newDependencies = getDataRecords().updateStatus(graphNodeUid,
                                                             token.access(),
                                                             token.uid());

        for (auto& dep : newDependencies) {
            const auto& n0 = getGraph().helpGetGraphNode(dep.t0());
            const auto& n1 = getGraph().helpGetGraphNode(dep.t1());
            getGraph().addDependency(n0, n1, token);
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

auto MultiXpuGraph::
    ioToDot(const std::string& fname,
            const std::string& graphName,
            bool               debug) -> void
{
    getGraph().ioToDot(fname, graphName, debug);
}

auto MultiXpuGraph::
    io2DotOriginalApp(const std::string& fname,
                      const std::string& graphName,
                      bool               debug) -> void
{
    getGraph().ioToDot(fname, graphName, debug);
}

auto MultiXpuGraph::helpAddNewContainerToGraph(const Neon::set::Container& container) -> Neon::set::container::GraphInfo::NodeUid
{
    const auto&                              graphNode = getGraph().addNode(container);
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
    getGraph().forEachNode([&](const Neon::set::container::GraphNode& graphNode) {
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
        auto& boundary_sten = getGraph().helpGetGraphNode(stencilNodeUid);
        auto& internal_sten = getGraph().cloneNode(boundary_sten);

        internal_sten.getScheduling().setDataView(Neon::DataView::INTERNAL);
        boundary_sten.getScheduling().setDataView(Neon::DataView::BOUNDARY);

        getGraph().addDependency(internal_sten, boundary_sten, Neon::GraphDependencyType::scheduling);
    }
}

auto MultiXpuGraph::optimizeExtendedOCC(const Neon::skeleton::Options&) -> void
{
    std::vector<Neon::set::container::GraphData::Uid> stencilNodeUidList;
    getGraph().forEachNode([&](const Neon::set::container::GraphNode& graphNode) {
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
        auto proceeding = this->getGraph().getProceedingGraphNodes(node);
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
        auto proceedingNodes = proceedingNodesAreMapOnly(getGraph().helpGetGraphNode(stencilNodeUid));

        if (proceedingNodes.empty()) {
            continue;
        }

        for (auto mapNodePtr : proceedingNodes) {
            auto& boundary_map = *mapNodePtr;
            auto& internal_map = getGraph().cloneNode(boundary_map);

            internal_map.getScheduling().setDataView(Neon::DataView::INTERNAL);
            boundary_map.getScheduling().setDataView(Neon::DataView::BOUNDARY);

            getGraph().addDependency(boundary_map, internal_map, Neon::GraphDependencyType::scheduling);
        }

        auto& boundary_sten = getGraph().helpGetGraphNode(stencilNodeUid);
        auto& internal_sten = getGraph().cloneNode(boundary_sten);

        internal_sten.getScheduling().setDataView(Neon::DataView::INTERNAL);
        boundary_sten.getScheduling().setDataView(Neon::DataView::BOUNDARY);

        getGraph().addDependency(internal_sten, boundary_sten, Neon::GraphDependencyType::scheduling);
    }
}

auto MultiXpuGraph::optimizeTwoWayExtendedOCC(const Neon::skeleton::Options&) -> void
{
    std::vector<Neon::set::container::GraphData::Uid> stencilNodeUidList;
    getGraph().forEachNode([&](const Neon::set::container::GraphNode& graphNode) {
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
        auto proceeding = this->getGraph().getProceedingGraphNodes(node);
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
        auto subsequent = this->getGraph().getSubsequentGraphNodes(node);
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
        auto proceedingNodes = proceedingNodesAreMapOnly(getGraph().helpGetGraphNode(stencilNodeUid));
        auto subsequentNodes = subsequentNodesAreMapOnly(getGraph().helpGetGraphNode(stencilNodeUid));

        if (proceedingNodes.empty() || subsequentNodes.empty()) {
            continue;
        }

        for (auto mapNodePtr : proceedingNodes) {
            auto& boundary_map = *mapNodePtr;
            auto& internal_map = getGraph().cloneNode(boundary_map);

            internal_map.getScheduling().setDataView(Neon::DataView::INTERNAL);
            boundary_map.getScheduling().setDataView(Neon::DataView::BOUNDARY);

            getGraph().addDependency(boundary_map, internal_map, Neon::GraphDependencyType::scheduling);
        }

        auto& boundary_stencil = getGraph().helpGetGraphNode(stencilNodeUid);
        auto& internal_stencil = getGraph().cloneNode(boundary_stencil);

        internal_stencil.getScheduling().setDataView(Neon::DataView::INTERNAL);
        boundary_stencil.getScheduling().setDataView(Neon::DataView::BOUNDARY);

        getGraph().addDependency(internal_stencil, boundary_stencil, Neon::GraphDependencyType::scheduling);

        for (auto mapOrReduceNodePtr : subsequentNodes) {
            if (mapOrReduceNodePtr->getContainer().getContainerInterface().getContainerPatternType() == Neon::set::ContainerPatternType::map) {
                auto& boundary_map = *mapOrReduceNodePtr;
                auto& internal_map = getGraph().cloneNode(boundary_map);

                internal_map.getScheduling().setDataView(Neon::DataView::INTERNAL);
                boundary_map.getScheduling().setDataView(Neon::DataView::BOUNDARY);

                getGraph().addDependency(internal_map, boundary_stencil, Neon::GraphDependencyType::scheduling);

                getGraph().removeDependency(getGraph().getDependency(internal_stencil, boundary_map));
                getGraph().removeDependency(getGraph().getDependency(boundary_stencil, internal_map));
            }
            if (mapOrReduceNodePtr->getContainer().getContainerInterface().getContainerPatternType() == Neon::set::ContainerPatternType::reduction) {
                auto& boundary_red = *mapOrReduceNodePtr;
                auto& internal_red = getGraph().cloneNode(boundary_red);

                internal_red.getScheduling().setDataView(Neon::DataView::INTERNAL);
                boundary_red.getScheduling().setDataView(Neon::DataView::BOUNDARY);

                getGraph().addDependency(internal_red, boundary_stencil, Neon::GraphDependencyType::scheduling);

                getGraph().removeDependency(getGraph().getDependency(internal_stencil, boundary_red));
                getGraph().removeDependency(getGraph().getDependency(boundary_stencil, internal_red));

                getGraph().addDependency(internal_red, boundary_red, Neon::GraphDependencyType::data);
            }
        }
    }
}


auto MultiXpuGraph::communications(const Neon::skeleton::Options& skeletonOptions) -> void
{
    if (getSetCardinality() == 1) {
        return;
    }

    // List all dependencies related to stencil operations
    std::vector<const Neon::set::container::GraphDependency*> stencilTypeDependencies;
    getGraph().forEachDependency([&](const Neon::set::container::GraphDependency& dep) {
        if (dep.getType() != Neon::GraphDependencyType::scheduling) {
            if (dep.hasStencilDependency()) {
                auto dw = dep.getDestinationNode(getGraph()).getScheduling().getDataView();
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
        auto        nodeA = getGraph().helpGetGraphNode(rawEdge.first);
        auto        nodeB = getGraph().helpGetGraphNode(rawEdge.second);

        const Neon::set::container::GraphDependency::Tokens& tokens = dep.getTokens();
        int                                                  numNewNodes = 0;

        for (const auto& token : tokens) {
            if (token.compute() == Neon::Pattern::STENCIL) {
                auto container = token.getDataTransferContainer(skeletonOptions.transferMode());
                numNewNodes += getGraph().expandAndMerge(nodeA, container, nodeB, true);
            }
        }

        auto schedulingDepOfB = this->getGraph().getProceedingGraphNodes(nodeB, {GraphDependencyType::scheduling});
        for (auto nodePtr : schedulingDepOfB) {
            const auto& dependency = getGraph().getDependency(*nodePtr, nodeB);
            toBeRemoved.push_back(&dependency);
        }
    }

    for (auto depPtr : toBeRemoved) {
        getGraph().removeDependency(*depPtr);
    }
}

auto MultiXpuGraph::fixingDependenciesWithBeginNode() -> void
{
    auto& beginNode = getGraph().getBeginNode();
    auto  nodesAfterBeginPtrVec = getGraph().getSubsequentGraphNodes(beginNode);

    for (auto nodePtr : nodesAfterBeginPtrVec) {
        const auto& tokens = nodePtr->getContainer().getContainerInterface().getTokens();
        for (const auto& token : tokens) {
            const auto computeType = token.compute();
            if (Neon::Pattern::STENCIL == computeType) {
                // 1. remove current dependency between begin and target node
                auto& dep = getGraph().getMutableDependency(beginNode, *nodePtr);
                dep.addToken(token);
            }
        }
    }
}

MultiXpuGraph::
    MultiXpuGraph()
{
    mStorage = std::make_shared<Storage>();
}

auto MultiXpuGraph::
    computeScheduling() -> void
{
    getGraph().runtimePreSet(Neon::Backend::mainStreamIdx);
}

auto MultiXpuGraph::
    execute(const Neon::skeleton::Options& options)
        -> void
{
    if (options.executor() == Neon::skeleton::Executor::ompAtNodeLevel) {
        this->getGraph().helpExecuteWithOmpAtNodeLevel(Neon::Backend::mainStreamIdx);
    } else {
        NEON_DEV_UNDER_CONSTRUCTION("");
    };
}
}  // namespace Neon::skeleton::internal