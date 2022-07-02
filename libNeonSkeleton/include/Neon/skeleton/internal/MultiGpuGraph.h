#pragma once
#include <list>
#include "Neon/core/types/digraph.h"
#include "Neon/set//Containter.h"
#include "Neon/set/Backend.h"
#include "Neon/skeleton/internal/dependencyTools/UserDataManager.h"
#include "Neon/skeleton/internal/multiGpuGraph/Edge.h"
#include "Neon/skeleton/internal/multiGpuGraph/MetaNode.h"
#include "Neon/skeleton/Options.h"
namespace Neon::skeleton::internal {
/**
 * Graph storing dependency between user kernels 
 */
struct MultiGpuGraph
{
   public:
    using DiGraph = Neon::DiGraph<MetaNode,
                                  Edge>;

    using DiGraphScheduling = Neon::DiGraph<Empty,
                                              Empty>;

   public:
    /**
     * Default empty constructor
     */
    MultiGpuGraph();

    void init(Neon::Backend&                    bk,
              const std::vector<Neon::set::Container>& operations,
              std::string                              name,
              Options                                  options);

    /**
     * Function to retrieve the container index in the user container list
     * @param id
     * @return
     */
    auto getContainer(ContainerIdx id) -> Neon::set::Container&;
    auto getContainer(ContainerIdx id) const -> const Neon::set::Container&;

    /**
     * Return a reference to the data dependency graph
     * @return
     */
    auto getDiGraph() -> DiGraph&;

    /**
     * Return a reference to the scheduling dependency graph
     * @return
     */
    auto getSchedulingDiGraph() -> DiGraphScheduling&;

    /**
     * Export the graph of the user application
     * @param fname
     * @param graphName
     */
    auto io2DotOriginalApp(const std::string& fname, const std::string& graphName) -> void;

    /**
     * Export both the data dependency graph and the scheduling graph
     * @param fname
     * @param graphName
     */
    auto io2Dot(const std::string& fname, const std::string& graphName) -> void;

    /**
     * Return the ID for the start node
     * @return
     */
    auto rootNodeId() const -> const size_t&;

    /**
     * Return the ID for the end node
     * @return
     */
    auto finalNodeId() const -> const size_t&;

    /**
     * Return the counter used to map all nodes to a 1D indexing
     * @return
     */
    auto getLinearContinuousIndexingCounter() -> size_t;

   private:
    struct Storage
    {
        std::vector<Neon::set::Container>                       m_kContainers;
        UserDataManager                                         m_dataRecords;

        DiGraph           m_userAppGraph;
        DiGraph           m_graph;
        DiGraphScheduling m_schedulingGraph;

        size_t m_rootNodeId;
        size_t m_finalNodeId;
        int    m_setCardinality = 0;
        size_t m_resetLinearContinuousIndexingCounter = 0;
    };
    std::shared_ptr<Storage> m_storage;

    /**
     * Access methods to members
     * @return
     */
    inline auto m_kContainers() -> std::vector<Neon::set::Container>&
    {
        return m_storage->m_kContainers;
    }

    inline auto m_kContainers() const -> const std::vector<Neon::set::Container>&
    {
        return m_storage->m_kContainers;
    }

    /**
     * Access methods to members
     * @return
     */
    inline auto m_dataRecords() -> UserDataManager&
    {
        return m_storage->m_dataRecords;
    }

    /**
     * Access methods to members
     * @return
     */
    inline auto m_graph() -> DiGraph&
    {
        return m_storage->m_graph;
    }

    /**
     * Access methods to members
     * @return
     */
    inline auto m_schedulingGraph() -> DiGraphScheduling&
    {
        return m_storage->m_schedulingGraph;
    }

    /**
     * Access methods to members
     * @return
     */
    inline auto m_rootNodeId() const -> const size_t&
    {
        return m_storage->m_rootNodeId;
    }

    inline auto m_setRootNodeId(const size_t& rootNodeId) -> void
    {
        m_storage->m_rootNodeId = rootNodeId;
    }


    /**
     * Access methods to members
     * @return
     */
    inline auto m_finalNodeId() const -> const size_t&
    {
        return m_storage->m_finalNodeId;
    }

    /**
     * set final node
     * @return
     */
    inline auto m_setFinalNodeId(const size_t& finalNode) -> void
    {
        m_storage->m_finalNodeId = finalNode;
    }

    /**
     * Access methods to members
     * @return
     */
    inline auto m_setCardinality() -> int&
    {
        return m_storage->m_setCardinality;
    }


    enum heloUpdate
    {
        PUT,
        GET
    };

    /**
     * Parser method
     * @param setCardinalty
     * @param operations
     */
    void parse(int setCardinalty, const std::vector<Neon::set::Container>&& operations);

    /**
     *
     * @param options
     */
    auto optimizations(const Neon::skeleton::Options& options)
        -> void;

    /**
     * Helper function to add a new kernel to the graph
     * The dependencies are extracted from the kernel container
     * @param container
     */
    auto h_parseContainer(const Neon::set::Container&              inContainer,
                          std::vector<size_t>&                     containerId2gNodeId,
                          std::vector<std::tuple<size_t, Edge>>& specialClosureNodes)
        -> void;

    /**
     * helper function to close the graph with a strart and end node
     * @param specialClosureNodes
     */
    auto h_closure(const std::vector<std::tuple<size_t, Edge>>& specialClosureNodes)
        -> void;

    /**
     * Parsing of a container of the task list
     * @param kernelContainerIdx
     * @return
     */
    auto h_parse(ContainerIdx kernelContainerIdx)
        -> std::vector<DataToken_t>;

    /**
     * helper function to export a dot file
     * @param fname
     * @param graphName
     */
    auto h_io2Dot([[maybe_unused]] const std::string& fname,
                  [[maybe_unused]] const std::string& graphName) -> void;


   private:
    auto h_trackNewContainer(const Neon::set::Container& container,
                             std::vector<size_t>&        containerId2gNodeId) -> ContainerIdx;

    auto h_removeRedundantDependencies() -> void;

    /**
     * Creates 2 nodes for
     * @param uid
     * @param edge
     * @param infoIdx
     * @param transferModeE
     * @return
     */
    auto h_add_hUpdateSyncNodes(size_t                          t0Node,
                                size_t                          t1Node,
                                const internal::Edge::Info& info,
                                Neon::set::TransferMode transferModeE = Neon::set::TransferMode::put)
        -> void;

    auto optimizeStandardOCC(const Neon::skeleton::Options&) -> void;

    auto optimizeExtendedOCC(const Neon::skeleton::Options&) -> void;

    auto optimizeTwoWayExtendedOCC(const Neon::skeleton::Options&) -> void;

    /**
     * Breadth First Traversal
     * @param f
     * @return
     */
    auto
    h_BFT(std::function<void(int                     level,
                             const std::set<size_t>& nodes)> f) -> void;

    auto h_getBFSIndexes() -> std::unordered_map<size_t, int>;

    //    auto fuseMaps(const Neon::skeleton::Options_t& options) -> void
    //    {
    //        // BFS visit
    //        std::vector<int> frontiar;
    //        m_graph.forEachOutEdge(m_rootNodeId, [&](const std::pair<size_t, size_t>& edge) {
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
    //            if (m_graph.outEdgesCount(n0) != 1) {
    //                return -1;
    //            }
    //            const size_t n1 = m_graph.outEdges(n0).begin()->second;
    //            if (m_graph.inEdgesCount(n1) != 1) {
    //                return -1;
    //            }
    //            if(m_graph.getVertexProperty(n1).isStencil()){
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
    //            if (m_graph.outEdges(nodeId).size() == 1 &&) {
    //                // a. Fuse
    //
    //                // b. update nodeId to the new node
    //            }
    //            // c. continue with BFS
    //        }
    //    }

    auto addSyncAndMemoryTransfers(const Neon::skeleton::Options& options) -> void;

    auto checkCoherency() -> void;


    auto resetLinearContinuousIndexing() -> void;
};
}  // namespace Neon::skeleton::internal
