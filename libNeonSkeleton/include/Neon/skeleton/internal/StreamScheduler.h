#pragma once
#include <unordered_set>
#include "Neon/skeleton/internal/MultiGpuGraph.h"

namespace Neon::skeleton::internal {

class StreamScheduler
{
   public:
    using DiGraph = MultiGpuGraph::DiGraph;

    using Level = std::vector<MetaNodeExtendedIdx>;
    struct MetaNodeExtended
    {
        struct SchedulingInfo
        {
            SchedulingInfo();
            SchedulingInfo(size_t nodeId, size_t extendedNodeIdx, ContainerIdx cIdx);
            NodeId                       nodeId;
            MetaNodeExtendedIdx          metaNodeExtendedIdx;
            ContainerIdx                 containerIdx;
            StreamIdx                    streamIdx = {-1};
            std::unordered_set<EventIdx> waitEventIdxList;
            EventIdx                     setEventIdx = {-1};
            EventIdx                     barrierLR = {-1};
            bool                         isHaloUpdateNode = false;
            int                          schedulingOrder{-1};
        };

        int            blockingDependencies{-1};
        SchedulingInfo schedulingInfo;
    };


    auto io2Dot(const std::string& nodeId, const std::string& graphName) -> void;
    auto io2DotOrder(const std::string& nodeId, const std::string& graphName) -> void;

   private:
    struct Storage
    {
       public:
        std::vector<Level>  m_levels;
        std::vector<int>    m_streamFlags;
        MultiGpuGraph&      m_graph;
        bool                m_initDone = {false};
        Neon::Backend       m_bk;
        std::vector<NodeId> m_executionOrder;
        std::vector<NodeId> m_linearization;

        Storage(Neon::Backend& bk, MultiGpuGraph& graph)
            : m_graph(graph), m_bk(bk)
        {
        }
        std::vector<MetaNodeExtended> m_metaNodeExtendedList;
        bool                          useFullBarrierOnAllStreamsAtTheEnd = true;
        bool                          canEndNodeLastBarrierBeOptimizedOut = false;
    };

    std::shared_ptr<Storage> m_storage;

   public:
    auto init(Neon::Backend& bk, MultiGpuGraph& multiGpuGraph) -> void;

    /**
     * Returns number of levels
     * @return
     */
    auto nLevels() -> int;

    /**
     *
     * @return
     */
    auto addNewLevel() -> LevelIdx;

    auto getLevel(int levelIdx) -> Level&;

    auto resetLevels() -> void;

    /**
     *
     */
    auto getStreamFlags() -> std::vector<int>&;

    /**
     * Counting the number of streams required
     * @return
     */
    auto initStreamFlags() -> int;

    /**
     *
     * @return
     */
    auto resetStreamFlags() -> void;

    /**
     *
     * @param containerIdx
     * @return
     */
    auto registerNodeToLastLevel(size_t nodeId) -> void;

    /**
     * Creating the level for the scheduling
     */
    auto initCreateLevels() -> void;
    /**
     * Schedule a stream for each node
     */
    auto initScheduleStreams() -> void;

    /**
     * Schedule required events for each node
     */
    auto initScheduleEvents() -> void;

    /**
     * Create a linearizex indexing
     */
    auto initLinearisationList() -> void;

    /**
     * Remove redundant synchronizations
     */
    auto initCleanRedundantSync() -> void;

    /**
     * Remove redundant events
     */
    auto initCleanRedundantEvents() -> void;

    /**
     * Define the execution order for each task
     */
    auto initExecutionOrder() -> void;

    /**
     * Execute the graph
     */
    auto run(const Neon::skeleton::Options& options) -> void;

   private:
    auto h_getMetaNodeExtended(NodeId id) -> MetaNodeExtended&;
    auto h_getMetaNodeExtended(MetaNode& id) -> MetaNodeExtended&;
    auto h_getMetaNode(NodeId id) -> MetaNode&;

    auto helpRunOmpAtGraphLevel() -> void;
    auto helpRunOmpAtNodeLevel() -> void;

    template <typename Graph>
    auto h_dissolveDataDependencies(Graph& graph, NodeId nodeId) -> void
    {
        // helper function to update the dependency count of child nodes
        auto outNeighbours = graph.outNeighbors(nodeId);
        for (const auto& ngh : outNeighbours) {
            auto& metaNodeExtended = h_getMetaNodeExtended(ngh);
            // std::cout << "Node " << nodeId << " dissolve " << ngh << " from " << metaNodeExtended.blockingDependencies << " to " << metaNodeExtended.blockingDependencies - 1 << std::endl;
            metaNodeExtended.blockingDependencies--;
        }
        //        for (auto huNode : h_getMetaNodeExtended(nodeId).childHaloNodes) {
        //            h_dissolveDataDependencies(huNode);
        //        }
    };

    auto helpRun(NodeId nodeId, StreamIdx streamIdx) -> void;
    auto helpRun(Neon::SetIdx setIdx, NodeId nodeId, StreamIdx streamIdx) -> void;
    auto helpNvtxName(NodeId nodeId) -> std::string;
    auto helpWaitForEventCompletion(StreamIdx streamIdx, EventIdx eventIdx) -> void;
    auto helpWaitForEventCompletion(Neon::SetIdx setIdx, StreamIdx streamIdx, EventIdx eventIdx) -> void;
    auto helpEnqueueEvent(StreamIdx streamIdx, EventIdx eventIdx) -> void;
    auto helpEnqueueEvent(Neon::SetIdx setIdx, StreamIdx streamIdx, EventIdx eventIdx) -> void;
};

}  // namespace Neon::skeleton::internal
