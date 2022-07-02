#include "Neon/skeleton/internal/StreamScheduler.h"
#include <unordered_set>
#include "Neon/set/syncrhonizations/event_LR_barrier.h"

namespace Neon::skeleton::internal {

#ifdef NEON_USE_NVTX
#include <nvToolsExt.h>
#endif

StreamScheduler::MetaNodeExtended::SchedulingInfo::SchedulingInfo(size_t       nodeId_,
                                                                  size_t       extendedNodeIdx_,
                                                                  ContainerIdx cIdx)
{
    nodeId = nodeId_;
    metaNodeExtendedIdx = extendedNodeIdx_;
    containerIdx = cIdx;
}

StreamScheduler::MetaNodeExtended::SchedulingInfo::SchedulingInfo()
{
    nodeId = 0;
    metaNodeExtendedIdx = 0;
    containerIdx = 0;
    streamIdx = -1;
    setEventIdx = -1;
}

auto StreamScheduler::init(Neon::Backend& bk, MultiGpuGraph& multiGpuGraph) -> void
{
    m_storage = std::make_shared<Storage>(bk, multiGpuGraph);
    m_storage->m_metaNodeExtendedList = std::vector<MetaNodeExtended>(
        multiGpuGraph.getLinearContinuousIndexingCounter());

    initCleanRedundantSync();
    initCreateLevels();
    initScheduleStreams();
    initScheduleEvents();
    initCleanRedundantEvents();
    initLinearisationList();
    initExecutionOrder();
}

auto StreamScheduler::initCleanRedundantSync() -> void
{
    if (m_storage->useFullBarrierOnAllStreamsAtTheEnd) {
        {  // Removing events from root nodes as we have a full barrier at the end
            NodeId           root = m_storage->m_graph.rootNodeId();
            auto&            diGraph = m_storage->m_graph.getDiGraph();
            std::set<size_t> rootChildList = diGraph.outNeighbors(root);

            for (auto& rootChild : rootChildList) {
                const auto& rootChildMeta = h_getMetaNode(rootChild);
                if (rootChildMeta.isSync()) {
                    auto& schedulingrGraph = m_storage->m_graph.getSchedulingDiGraph();
                    if (diGraph.outNeighbors(rootChild).size() != 1) {
                        continue;
                    }
                    NodeId      huNode = *diGraph.outNeighbors(rootChild).begin();
                    const auto& huMeta = diGraph.getVertexProperty(huNode);
                    if (!huMeta.isHu()) {
                        continue;
                    }
                    // we are in a sutuation of a SYNC follow by a HALO Up.
                    if (schedulingrGraph.hasVertex(rootChild)) {
                        if (!schedulingrGraph.hasVertex(huNode)) {
                            schedulingrGraph.addVertex(huNode);
                        }
                        const auto& inSchedulingEdges = schedulingrGraph.inEdges(rootChild);
                        for (auto& inSchedulingEdge : inSchedulingEdges) {
                            schedulingrGraph.addEdge(inSchedulingEdge.first, huNode);
                        }
                        schedulingrGraph.removeVertex(rootChild);
                    }
                    for (auto& neighbour : diGraph.neighbors(rootChild)) {
                        diGraph.addEdge(root, neighbour);
                    }
                    diGraph.removeVertex(rootChild);
                }
            }
        }
    }
}

auto StreamScheduler::initCleanRedundantEvents() -> void
{
    if (m_storage->useFullBarrierOnAllStreamsAtTheEnd) {
        {  // Removing events from root nodes as we have a full barrier at the end
            NodeId root = m_storage->m_graph.rootNodeId();
            auto&  metaNodeExtendedRoot = h_getMetaNodeExtended(root);
            metaNodeExtendedRoot.schedulingInfo.setEventIdx = -1;

            auto&            diGraph = m_storage->m_graph.getDiGraph();
            std::set<size_t> rootChildList = diGraph.outNeighbors(root);

            for (auto& rootChild : rootChildList) {
                auto& metaNodeExtended = h_getMetaNodeExtended(rootChild);
                metaNodeExtended.schedulingInfo.waitEventIdxList = std::unordered_set<EventIdx>();
            }
        }
    }

    // Checking if we can remove the last sync all
    NodeId           endNode = m_storage->m_graph.finalNodeId();
    auto&            diGraph = m_storage->m_graph.getDiGraph();
    std::set<size_t> endNodeChildList = diGraph.inNeighbors(endNode);
    if (endNodeChildList.size() == 1) {
        bool allReduceNodes = true;
        for (auto& endNodeChild : endNodeChildList) {
            const auto& metaNode = m_storage->m_graph.getDiGraph().getVertexProperty(endNodeChild);
            if (!metaNode.isReduce()) {
                allReduceNodes = false;
            } else {
                auto& metaNodeExtendedChild = h_getMetaNodeExtended(endNodeChild);
                metaNodeExtendedChild.schedulingInfo.setEventIdx = -1;
            }
        }
        if (allReduceNodes) {
            m_storage->canEndNodeLastBarrierBeOptimizedOut = true;
        }
    }
}


/**
 * Returns number of levels
 * @return
 */
auto StreamScheduler::nLevels() -> int
{
    return static_cast<int>(m_storage->m_levels.size());
}

/**
 *
 * @return
 */
auto StreamScheduler::addNewLevel() -> LevelIdx
{
    m_storage->m_levels.emplace_back();
    return LevelIdx(m_storage->m_levels.size() - 1);
}

auto StreamScheduler::getLevel(int levelIdx) -> Level&
{
    return m_storage->m_levels.at(levelIdx);
}

auto StreamScheduler::resetLevels() -> void
{
    m_storage->m_levels.clear();
}

/**
 *
 */
auto StreamScheduler::getStreamFlags() -> std::vector<int>&
{
    return m_storage->m_streamFlags;
}

/**
 *
 * @return
 */
auto StreamScheduler::resetStreamFlags() -> void
{
    for (auto& flag : m_storage->m_streamFlags) {
        flag = 0;
    }
}

auto StreamScheduler::h_getMetaNodeExtended(size_t id) -> MetaNodeExtended&
{
    auto& meta = m_storage->m_graph.getDiGraph().getVertexProperty(id);
    return m_storage->m_metaNodeExtendedList.at(meta.getLinearContinuousIndex());
};

auto StreamScheduler::h_getMetaNode(NodeId id) -> MetaNode&
{
    auto& meta = m_storage->m_graph.getDiGraph().getVertexProperty(id);
    return meta;
};

auto StreamScheduler::h_getMetaNodeExtended(MetaNode& id) -> MetaNodeExtended&
{
    return m_storage->m_metaNodeExtendedList.at(id.getLinearContinuousIndex());
}

/**
 *
 * @param nodeId
 */
auto StreamScheduler::registerNodeToLastLevel(NodeId nodeId) -> void
{
    const size_t levelTarget = m_storage->m_levels.size() - 1;
    this->m_storage->m_levels.at(levelTarget).push_back(nodeId);
}

auto StreamScheduler::initCreateLevels() -> void
{
    auto& diGraph = m_storage->m_graph.getDiGraph();

    // Set the dependency counter of each node
    diGraph.forEachVertex([&](NodeId nodeId) {
        auto&     metaNodeExtended = h_getMetaNodeExtended(nodeId);
        const int nDependencies = int(diGraph.inEdgesCount(nodeId));
        metaNodeExtended.blockingDependencies = nDependencies;
    });

    std::unordered_set<NodeId> frontier;
    const NodeId               beginning = m_storage->m_graph.rootNodeId();
    frontier.insert(beginning);
    // h_dissolveDataDependencies(beginning);

    // Walking the graph following a BFS with constraints
    int level = -1;
    while (!frontier.empty()) {
        addNewLevel();
        level++;
        // printf("Level %d \n", level);
        std::unordered_set<NodeId> nextFrontier;
        std::unordered_set<NodeId> nodeToApplyDissolveTo;
        for (auto& frontierNodeId : frontier) {
            {  // 2. If the node has still some dependencies that are not satisfy,
                // don't process the node now ad add it to the next frontier
                // printf("frontierNodeId %ld \n", frontierNodeId);
                auto& frontierMetaNode = h_getMetaNode(frontierNodeId);
                auto  frontierMetaNodeExtended = h_getMetaNodeExtended(frontierMetaNode);
                if (frontierMetaNodeExtended.blockingDependencies != 0) {
                    // There are some data dependencies still active before processing this node
                    // Moving the node to the next frontier.
                    nextFrontier.insert(frontierNodeId);
                    // std:: cout<< "frontierMetaNode " << frontierNodeId <<" postpone" <<std::endl;
                    continue;
                }
            }
            {  // 3. Populating the next frontier
                std::set<size_t> childList = diGraph.outNeighbors(frontierNodeId);
                for (auto& child : childList) {
                    nextFrontier.insert(child);
                }
            }
            {  // 4. Upgrade information in the metaNodeExtended
                auto& metaNode = h_getMetaNode(frontierNodeId);
                auto& metaNodeExtended = h_getMetaNodeExtended(frontierNodeId);
                metaNodeExtended.schedulingInfo = MetaNodeExtended::SchedulingInfo(frontierNodeId,
                                                                                   metaNode.getLinearContinuousIndex(),
                                                                                   metaNode.getContainerId());
            }
            {  // 5. Register the target frontier node to the last level
                registerNodeToLastLevel(frontierNodeId);
                nodeToApplyDissolveTo.insert(frontierNodeId);
            }
        }

        // Update all the dependencies
        for (auto& frontierNode : nodeToApplyDissolveTo) {
            h_dissolveDataDependencies(diGraph, frontierNode);
        }
        // h_print_depcountALL();
        frontier = nextFrontier;
    }
}

auto StreamScheduler::initStreamFlags() -> int
{
    int maxNstream = 0;
    for (int i = 0; i < nLevels(); i++) {
        Level& level = getLevel(i);
        maxNstream = int(level.size()) > maxNstream ? int(level.size()) : maxNstream;
    }
    m_storage->m_streamFlags = std::vector<int>(maxNstream, 0);
    m_storage->m_bk.setAvailableStreamSet(maxNstream);
    return maxNstream;
}

auto StreamScheduler::initScheduleStreams() -> void
{
    auto& diGraph = m_storage->m_graph.getDiGraph();

    NodeId startNodeId = m_storage->m_graph.rootNodeId();
    NodeId endNodeId = m_storage->m_graph.finalNodeId();

    auto h_setStream = [this](NodeId nodeId, int streamId) {
        auto& metaNode = h_getMetaNodeExtended(nodeId);
        metaNode.schedulingInfo.streamIdx = streamId;
    };

    initStreamFlags();
    auto& streamFlags = getStreamFlags();

    for (int i = 0; i < nLevels(); i++) {
        resetStreamFlags();
        Level&              level = getLevel(i);
        std::vector<NodeId> leftToMatch;
        //
        for (auto it = level.begin(), end = level.end(); it != end; ++it) {
            NodeId nodeId = *it;

            if (nodeId == endNodeId || nodeId == startNodeId) {
                h_setStream(nodeId, 0);
                continue;
            }

            if (h_getMetaNodeExtended(nodeId).schedulingInfo.isHaloUpdateNode) {
                // Halo nodes are associated to the same stream as the parent node
                continue;
            }
            auto parents = diGraph.inNeighbors(nodeId);
            bool matchDone = false;
            // Try to match the stream id with one of the parent nodes
            for (auto& parent : parents) {
                const auto& parentMetaNodeExtended = h_getMetaNodeExtended(parent);
                int         parentStreamId = parentMetaNodeExtended.schedulingInfo.streamIdx;

                // checking if at this level the stream of the parent is still available
                const bool available = (streamFlags[parentStreamId] == 0);
                if (available) {
                    streamFlags[parentStreamId] = 1;
                    h_setStream(nodeId, parentStreamId);
                    matchDone = true;
                    break;
                }
            }
            if (matchDone)
                continue;
            // nodes left to match
            // no halo node is added to this vector
            leftToMatch.push_back(nodeId);
        }
        for (auto& toMatch : leftToMatch) {
            bool matchFound = false;
            for (int streamIdx = 0; streamIdx < static_cast<int>(streamFlags.size()); streamIdx++) {
                const bool available = streamFlags[streamIdx] == 0;
                if (available) {
                    streamFlags[streamIdx] = 1;
                    h_setStream(toMatch, streamIdx);
                    matchFound = true;
                    break;
                }
            }
            if (!matchFound) {
                NEON_THROW_UNSUPPORTED_OPTION("");
            }
        }
    }
}

auto StreamScheduler::initScheduleEvents() -> void
{
    int eventCounter = 0;

    auto h_waitEvent = [this, &eventCounter](NodeId nodeId) -> EventIdx {
        auto& metaNode = h_getMetaNodeExtended(nodeId);
        if (metaNode.schedulingInfo.setEventIdx == -1) {
            metaNode.schedulingInfo.setEventIdx = eventCounter;
            eventCounter++;
        }
        return metaNode.schedulingInfo.setEventIdx;
    };

    auto h_barrierLREvent = [&eventCounter](MetaNodeExtended& metaNodeExtended) -> void {
        if (metaNodeExtended.schedulingInfo.barrierLR == -1) {
            metaNodeExtended.schedulingInfo.barrierLR = eventCounter;
            eventCounter++;
        }
    };

    auto h_setEvent = [this](NodeId nodeId, EventIdx eventIdx) -> void {
        auto& metaNode = h_getMetaNodeExtended(nodeId);
        metaNode.schedulingInfo.waitEventIdxList.insert(eventIdx);
    };

    auto h_addEventsIfNeeded = [this, &h_waitEvent, &h_setEvent, &h_barrierLREvent](NodeId nodeId) -> void {
        const auto&           diGraph = m_storage->m_graph.getDiGraph();
        auto&                 metaNodeExtended = h_getMetaNodeExtended(nodeId);
        const Neon::StreamIdx nodeStreamIdx = metaNodeExtended.schedulingInfo.streamIdx;
        auto&                 parentNodeIdSet = diGraph.inNeighbors(nodeId);

        for (auto& parentNodeId : parentNodeIdSet) {
            auto                  parentMetaNode = h_getMetaNodeExtended(parentNodeId);
            const Neon::StreamIdx parentStreamIdx = parentMetaNode.schedulingInfo.streamIdx;
            if (parentStreamIdx != nodeStreamIdx) {
                EventIdx eventIdx = h_waitEvent(parentNodeId);
                h_setEvent(nodeId, eventIdx);
            }
        }
        {  // If the hode is a LR barrier, then we need to book an extra event
            auto nodeMeta = h_getMetaNode(nodeId);
            if (nodeMeta.nodeType() == MetaNodeType_te::SYNC_LEFT_RIGHT) {
                h_barrierLREvent(metaNodeExtended);
            }
            if (nodeMeta.nodeType() == MetaNodeType_te::BARRIER) {
                NEON_THROW_UNSUPPORTED_OPTION("");
            }
        }
    };

    for (int i = 0; i < nLevels(); i++) {
        Level& level = getLevel(i);
        for (auto it = level.begin(), end = level.end(); it != end; ++it) {
            NodeId nodeId = *it;
            h_addEventsIfNeeded(nodeId);
        }
    }

    m_storage->m_bk.setAvailableUserEvents(eventCounter);
}

auto StreamScheduler::initLinearisationList() -> void
{

    std::vector<NodeId>& linear = m_storage->m_linearization;
    for (int i = 0; i < nLevels(); i++) {
        Level& level = getLevel(i);
        for (auto it = level.begin(), end = level.end(); it != end; ++it) {
            NodeId            nodeId = *it;
            MetaNodeExtended& metaNodeExtended = h_getMetaNodeExtended(nodeId);
            int               schedulingOrder = int(linear.size());
            metaNodeExtended.schedulingInfo.schedulingOrder = schedulingOrder;
            linear.push_back(nodeId);
        }
    }
}

#if 0
    auto StreamScheduler::initExecutionOrder() -> void
    {
        // Extract a linear indexing of all the nodes
        const std::vector<NodeId>& linear = m_storage->m_linearization;
        // Aliasing m_executionOrder with variable order
        std::vector<NodeId>& order = m_storage->m_executionOrder;

        // A bool map to track nodes that were already processed
        std::vector<bool> alreadyProceed(linear.size(), false);

        // Aliasing: the scheduling graph from m_graph
        const auto& schedulingGraph = m_storage->m_graph.getSchedulingDiGraph();

        // Lambda to extract all nodes that come before nodeId
        // in the SCHEDULING graph
        auto findSchedulingExecutionDep = [schedulingGraph](NodeId nodeId) -> std::set<NodeId> {
            if (schedulingGraph.hasVertex(nodeId)) {
                const auto& inNhg = schedulingGraph.inNeighbors(nodeId);
                return inNhg;
            }
            return std::set<NodeId>();
        };

        // Looping over all the nodes based on a linearized index.
        // There is no specific order in the indexing
        for (size_t i = 0; i < linear.size(); i++) {
            NodeId   nodeId = linear[i];
            MetaNode metaNode = h_getMetaNode(nodeId);
            if (alreadyProceed[metaNode.getLinearContinuousIndex()]) {
                continue;
            }
            const auto& SchedulingDeps = findSchedulingExecutionDep(nodeId);
            for (const auto& schedulingDepNode : SchedulingDeps) {
                const auto& depMeta = h_getMetaNode(schedulingDepNode);
                if (!alreadyProceed[depMeta.getLinearContinuousIndex()]) {
                    order.push_back(schedulingDepNode);
                    alreadyProceed[depMeta.getLinearContinuousIndex()] = true;
                }
            }
            order.push_back(nodeId);
            alreadyProceed[metaNode.getLinearContinuousIndex()] = true;
        }

        if (linear.size() != order.size()) {
            NEON_THROW_UNSUPPORTED_OPTION("");
        }

        for (size_t i = 0; i < order.size(); i++) {
            NodeId            nodeId = order[i];
            MetaNodeExtended& metaNodeExtended = h_getMetaNodeExtended(nodeId);
            metaNodeExtended.schedulingInfo.schedulingOrder = static_cast<int>(i);
        }
    }
#else


auto StreamScheduler::initExecutionOrder() -> void
{
    // Extract a linear indexing of all the nodes
    // const std::vector<NodeId>& linear = m_storage->m_linearization;
    // Aliasing m_executionOrder with variable order
    std::vector<NodeId>& order = m_storage->m_executionOrder;


    auto dataAndSchedulingGraph = m_storage->m_graph.getDiGraph();
    {
        const auto& schedulingGraph = m_storage->m_graph.getSchedulingDiGraph();
        for (auto& edge : schedulingGraph.edges()) {
            if (!dataAndSchedulingGraph.hasEdge(edge)) {
                dataAndSchedulingGraph.addEdge(edge.first, edge.second);
            }
        }
    }

    // Set the dependency counter of each node
    dataAndSchedulingGraph.forEachVertex([&](NodeId nodeId) {
        auto&     metaNodeExtended = h_getMetaNodeExtended(nodeId);
        const int nDependencies = int(dataAndSchedulingGraph.inEdgesCount(nodeId));
        metaNodeExtended.blockingDependencies = nDependencies;
    });

    std::unordered_set<NodeId> frontier;
    const NodeId               beginning = m_storage->m_graph.rootNodeId();
    frontier.insert(beginning);

    // Reseting Levels;
    resetLevels();

    // Walking the graph following a BFS with constraints
    int level = -1;
    while (!frontier.empty()) {
        addNewLevel();
        level++;
        // printf("Level %d \n", level);
        std::unordered_set<NodeId> nextFrontier;
        std::unordered_set<NodeId> nodeToApplyDissolveTo;
        for (auto& frontierNodeId : frontier) {
            {  // 2. If the node has still some dependencies that are not satisfy,
                // don't process the node now ad add it to the next frontier
                // printf("frontierNodeId %ld \n", frontierNodeId);
                auto& frontierMetaNode = h_getMetaNode(frontierNodeId);
                auto  frontierMetaNodeExtended = h_getMetaNodeExtended(frontierMetaNode);
                if (frontierMetaNodeExtended.blockingDependencies != 0) {
                    // There are some data dependencies still active before processing this node
                    // Moving the node to the next frontier.
                    nextFrontier.insert(frontierNodeId);
                    // std:: cout<< "frontierMetaNode " << frontierNodeId <<" postpone" <<std::endl;
                    continue;
                }
            }
            {  // 3. Populating the next frontier
                std::set<size_t> childList = dataAndSchedulingGraph.outNeighbors(frontierNodeId);
                for (auto& child : childList) {
                    nextFrontier.insert(child);
                }
            }
            {  // 5. Register the target frontier node to the last level
                registerNodeToLastLevel(frontierNodeId);
                nodeToApplyDissolveTo.insert(frontierNodeId);
            }
        }

        // Update all the dependencies
        for (auto& frontierNode : nodeToApplyDissolveTo) {
            h_dissolveDataDependencies(dataAndSchedulingGraph, frontierNode);
        }
        // h_print_depcountALL();
        frontier = nextFrontier;
    }
    for (int i = 0; i < nLevels(); i++) {
        Level& lev = getLevel(i);
        for (auto it = lev.begin(), end = lev.end(); it != end; ++it) {
            NodeId nodeId = *it;
            order.push_back(nodeId);
        }
    }
}

#endif


auto StreamScheduler::helpRunOmpAtNodeLevel() -> void
{
    for (auto& nodeId : m_storage->m_executionOrder) {
#ifdef NEON_USE_NVTX
        const auto nvtxName = helpNvtxName(nodeId);
        nvtxRangePush(nvtxName.c_str());
#endif
        MetaNodeExtended      metaNodeExtended = h_getMetaNodeExtended(nodeId);
        const Neon::StreamIdx streamIdx = metaNodeExtended.schedulingInfo.streamIdx;
        const EventIdx        eventIdx = metaNodeExtended.schedulingInfo.setEventIdx;
        for (const auto& eventToBeWaited : metaNodeExtended.schedulingInfo.waitEventIdxList) {
            if (nodeId == m_storage->m_graph.finalNodeId()) {
                continue;
            }
            helpWaitForEventCompletion(streamIdx, eventToBeWaited);
        }

        helpRun(nodeId, streamIdx);
        if (eventIdx != -1 && nodeId != m_storage->m_graph.finalNodeId()) {
            helpEnqueueEvent(streamIdx, eventIdx);
        }
        if (nodeId == m_storage->m_graph.finalNodeId()) {
            if (!m_storage->canEndNodeLastBarrierBeOptimizedOut) {
                m_storage->m_bk.syncAll();
            }
        }
#ifdef NEON_USE_NVTX
        nvtxRangePop();
#endif
    }
}

auto StreamScheduler::helpRunOmpAtGraphLevel() -> void
{
    const int nNodes = int(m_storage->m_executionOrder.size());

    for (int setIdx = 0; setIdx < m_storage->m_bk.devSet().setCardinality(); setIdx++) {
        for (int a = 0; a < nNodes; a++) {
            auto& nodeId = m_storage->m_executionOrder[a];

#ifdef NEON_USE_NVTX
            const auto nvtxName = helpNvtxName(nodeId);
            nvtxRangePush(nvtxName.c_str());
#endif
            MetaNodeExtended      metaNodeExtended = h_getMetaNodeExtended(nodeId);
            const Neon::StreamIdx streamIdx = metaNodeExtended.schedulingInfo.streamIdx;
            const EventIdx        eventIdx = metaNodeExtended.schedulingInfo.setEventIdx;
            for (const auto& eventToBeWaited : metaNodeExtended.schedulingInfo.waitEventIdxList) {
                if (nodeId == m_storage->m_graph.finalNodeId()) {
                    continue;
                }
                helpWaitForEventCompletion(setIdx, streamIdx, eventToBeWaited);
            }

            helpRun(setIdx, nodeId, streamIdx);
            if (eventIdx != -1 && nodeId != m_storage->m_graph.finalNodeId()) {
                helpEnqueueEvent(setIdx, streamIdx, eventIdx);
            }
            if (nodeId == m_storage->m_graph.finalNodeId()) {
                if (!m_storage->canEndNodeLastBarrierBeOptimizedOut) {
                    m_storage->m_bk.syncAll();
                }
            }
#ifdef NEON_USE_NVTX
            nvtxRangePop();
#endif
        }
    }
}
auto StreamScheduler::run(const Neon::skeleton::Options& options) -> void
{
    if (Neon::skeleton::Executor::ompAtNodeLevel == options.executor()) {
#ifdef NEON_USE_NVTX
        nvtxRangePush("Skeleton Iteration - ompAtNodeLevel");
#endif
        helpRunOmpAtNodeLevel();
#ifdef NEON_USE_NVTX
        nvtxRangePop();
#endif
        return;
    }

    if (Neon::skeleton::Executor::ompAtGraphLevel == options.executor()) {
#ifdef NEON_USE_NVTX
        nvtxRangePush("Skeleton Iteration - ompAtGraphLevel");
#endif
        helpRunOmpAtGraphLevel();
#ifdef NEON_USE_NVTX
        nvtxRangePop();
#endif
        return;
    }
    NEON_THROW_UNSUPPORTED_OPTION("No supported Executor option.");
}

auto StreamScheduler::helpRun(NodeId nodeId, Neon::StreamIdx streamIdx) -> void
{
    MetaNode& metaNode = h_getMetaNode(nodeId);


    switch (metaNode.nodeType()) {
        case MetaNodeType_te::CONTAINER: {

            ContainerIdx containerIdx = metaNode.getContainerId();
            auto&        container = m_storage->m_graph.getContainer(containerIdx);
            container.run(streamIdx, metaNode.getDataView());
            return;
        }
        case MetaNodeType_te::SYNC_LEFT_RIGHT: {
            // Neon::set::Event_LR_barrier::sync(m_storage->m_bk, streamIdx, h_getMetaNodeExtended(nodeId).schedulingInfo.barrierLR);
            m_storage->m_bk.sync(streamIdx);
            return;
        }
        case MetaNodeType_te::HALO_UPDATE: {
            const bool           startWithBarrier = false;
            Neon::set::HuOptions huOptions(metaNode.transferMode(), startWithBarrier, streamIdx);
            metaNode.hu(huOptions);
            return;
        }
        case MetaNodeType_te::HELPER: {
            return;
        }
        case MetaNodeType_te::BARRIER:
        case MetaNodeType_te::UNDEFINED: {
            NEON_THROW_UNSUPPORTED_OPTION("");
        }
    }
}

auto StreamScheduler::helpRun(Neon::SetIdx setIdx, NodeId nodeId, Neon::StreamIdx streamIdx) -> void
{
    MetaNode& metaNode = h_getMetaNode(nodeId);


    switch (metaNode.nodeType()) {
        case MetaNodeType_te::CONTAINER: {

            ContainerIdx containerIdx = metaNode.getContainerId();
            auto&        container = m_storage->m_graph.getContainer(containerIdx);
            container.run(setIdx, streamIdx, metaNode.getDataView());
            return;
        }
        case MetaNodeType_te::SYNC_LEFT_RIGHT: {
            // Neon::set::Event_LR_barrier::sync(m_storage->m_bk, streamIdx, h_getMetaNodeExtended(nodeId).schedulingInfo.barrierLR);
            m_storage->m_bk.sync(setIdx, streamIdx);
            return;
        }
        case MetaNodeType_te::HALO_UPDATE: {
            const bool           startWithBarrier = false;
            Neon::set::HuOptions huOptions(metaNode.transferMode(), startWithBarrier, streamIdx);
            metaNode.hu(setIdx, huOptions);
            return;
        }
        case MetaNodeType_te::HELPER: {
            return;
        }
        case MetaNodeType_te::BARRIER:
        case MetaNodeType_te::UNDEFINED: {
            NEON_THROW_UNSUPPORTED_OPTION("");
        }
    }
}

auto StreamScheduler::helpNvtxName(NodeId nodeId) -> std::string
{
    MetaNode& metaNode = h_getMetaNode(nodeId);
    switch (metaNode.nodeType()) {
        case MetaNodeType_te::CONTAINER: {

            ContainerIdx containerIdx = metaNode.getContainerId();
            auto&        container = m_storage->m_graph.getContainer(containerIdx);
            std::string  name = container.getName() + "_" + Neon::DataViewUtil::toString(metaNode.getDataView());
            return name;
        }
        case MetaNodeType_te::SYNC_LEFT_RIGHT: {
            // Neon::set::Event_LR_barrier::sync(m_storage->m_bk, streamIdx, h_getMetaNodeExtended(nodeId).schedulingInfo.barrierLR);
            return "LeftRightSync";
        }
        case MetaNodeType_te::HALO_UPDATE: {
            return "Halo Update";
        }
        case MetaNodeType_te::HELPER: {
            return "HELPER";
        }
        case MetaNodeType_te::BARRIER:
        case MetaNodeType_te::UNDEFINED:
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("");
        }
    }
}


auto StreamScheduler::helpWaitForEventCompletion(StreamIdx streamIdx, Neon::EventIdx eventIdx) -> void
{
    auto& bk = m_storage->m_bk;
    bk.waitEventOnStream(eventIdx, streamIdx);
}

auto StreamScheduler::helpEnqueueEvent(StreamIdx streamIdx, Neon::EventIdx eventIdx) -> void
{
    auto& bk = m_storage->m_bk;
    bk.pushEventOnStream(eventIdx, streamIdx);
}

auto StreamScheduler::helpWaitForEventCompletion(Neon::SetIdx setIdx, StreamIdx streamIdx, Neon::EventIdx eventIdx) -> void
{
    auto& bk = m_storage->m_bk;
    bk.waitEventOnStream(setIdx, eventIdx, streamIdx);
}

auto StreamScheduler::helpEnqueueEvent(Neon::SetIdx setIdx, StreamIdx streamIdx, Neon::EventIdx eventIdx) -> void
{
    auto& bk = m_storage->m_bk;
    bk.pushEventOnStream(setIdx, eventIdx, streamIdx);
}

#if 0
    auto StreamScheduler::io2Dot(const std::string& fname, const std::string& graphName) -> void
    {
        auto clone = m_storage->m_graph.getDiGraph();
        m_storage->m_graph.getSchedulingDiGraph().forEachEdge([&](const MultiGpuGraph::DiGraphScheduling::Edge& edge) {
            auto edgeMeta = Edge_t::factorySchedulingEdge();
            clone.addEdge(edge.first, edge.second, edgeMeta);
        });

        auto& mGpuGraph = m_storage->m_graph;

        // HELPER //////////////////////////////////////////////////////////////////////////
        auto vertexLabel = [&](NodeId nodeId) -> std::string {
            auto& metaNodeExtended = h_getMetaNodeExtended(nodeId);

            auto h_intro = [&]() -> std::string {
                return std::string("Scheduling info: \\l- Scheduling ID ") + std::to_string(metaNodeExtended.schedulingInfo.schedulingOrder) + "\\l";
            };

            auto h_eventWaitString = [&]() -> std::string {
                std::string ret;
                if (metaNodeExtended.schedulingInfo.waitEventIdxList.size() == 0) {
                    ret += "- No waiting\\l";
                } else {
                    ret += "- Waiting events:";
                    for (auto& e : metaNodeExtended.schedulingInfo.waitEventIdxList) {
                        ret += " " + std::to_string(e);
                    }
                    ret += "\\l";
                }
                return ret;
            };

            auto h_streamString = [&]() -> std::string {
                return "- Stream " + std::to_string(metaNodeExtended.schedulingInfo.streamIdx) + "\\l";
            };

            [[maybe_unused]] auto h_barrierString = [&]() -> std::string {
                if (metaNodeExtended.schedulingInfo.barrierLR != -1) {
                    return "- LR Event " + std::to_string(metaNodeExtended.schedulingInfo.barrierLR) + "\\l";
                    ;
                }
                return "";
            };

            auto h_eventRegisterString = [&]() -> std::string {
                if (metaNodeExtended.schedulingInfo.setEventIdx != -1) {
                    return "- Registering Event " + std::to_string(metaNodeExtended.schedulingInfo.setEventIdx) + "\\l";
                    ;
                }
                return "";
            };

            if (nodeId == mGpuGraph.finalNodeId()) {
                return std::string("END (") + std::to_string(nodeId) + ")\n\n" + h_intro() + h_eventWaitString() + h_streamString() + "\\l";
            }
            if (nodeId == mGpuGraph.rootNodeId()) {
                return std::string("START(") + std::to_string(nodeId) + ")\n\n" + h_intro() + h_eventWaitString() + h_streamString() + h_eventRegisterString() + "\\l";
            }


            return clone.getVertexProperty(nodeId).toString() + h_intro() + h_eventWaitString() + h_streamString() + h_eventRegisterString() + "\\l";
        };

        // HELPER //////////////////////////////////////////////////////////////////////////
        auto edgeLabel = [&](const std::pair<size_t, size_t>& edge)
            -> std::string {
            const auto& metaEdge = clone.getEdgeProperty(edge);
            if (metaEdge.m_isSchedulingEdge) {
                return "";
            }
            //if (property.nDependencies() > 0) {
            auto& metaNodeExtendedStart = h_getMetaNodeExtended(edge.first);
            auto& metaNodeExtendedEnd = h_getMetaNodeExtended(edge.second);
            if (metaNodeExtendedStart.schedulingInfo.streamIdx != metaNodeExtendedEnd.schedulingInfo.streamIdx) {
                // bool found = false;
                for (auto& e : metaNodeExtendedEnd.schedulingInfo.waitEventIdxList) {
                    if (e == metaNodeExtendedStart.schedulingInfo.setEventIdx) {
                        // found = true;
                    }
                }
                // if (!found) {
                //      NEON_THROW_UNSUPPORTED_OPTION("");
                // }
                if (metaEdge.nDependencies() > 0) {
                    return metaEdge.toString() +
                           "Synchronizations:\\l- Stream " +
                           std::to_string(metaNodeExtendedStart.schedulingInfo.streamIdx) +
                           "\\l- Event " + std::to_string(metaNodeExtendedStart.schedulingInfo.setEventIdx) + "\\l";
                } else {
                    return "Synchronization:\\l- Stream " +
                           std::to_string(metaNodeExtendedStart.schedulingInfo.streamIdx) +
                           "\\l- Event " + std::to_string(metaNodeExtendedStart.schedulingInfo.setEventIdx) + "\\l";
                }
            }
            //return property.toString() + "\\l";
            //        }
            return "";
        };

        // HELPER //////////////////////////////////////////////////////////////////////////
        auto edgeLabelProperty = [&](const std::pair<size_t, size_t>& edge)
            -> std::string {
          const auto& metaEdge = clone.getEdgeProperty(edge);
          if (metaEdge.m_isSchedulingEdge) {
              // return "style=dashed, color=\"#2A9D8F\"";
              return "style=dashed, color=\"#F4A261\", penwidth=7";
          }
          return "color=\"#d9d9d9\", penwidth=7";
        };
        // HELPER //////////////////////////////////////////////////////////////////////////
        auto vertexLabelProperty = [&](const size_t& v) {
          if (v == mGpuGraph.finalNodeId() || (v == mGpuGraph.rootNodeId())) {
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

        ////////////////////////////////////////////////////////////////////////////
        clone.exportDotFile(fname, graphName, vertexLabel, edgeLabel,
                            vertexLabelProperty, edgeLabelProperty);
    }
#else

auto StreamScheduler::io2Dot(const std::string& fname, const std::string& graphName) -> void
{
    auto clone = m_storage->m_graph.getDiGraph();
    m_storage->m_graph.getSchedulingDiGraph().forEachEdge([&](const MultiGpuGraph::DiGraphScheduling::Edge& edge) {
        auto edgeMeta = Edge::factorySchedulingEdge();
        clone.addEdge(edge.first, edge.second, edgeMeta);
    });

    auto& mGpuGraph = m_storage->m_graph;

    auto vertexLabel = [&](NodeId nodeId) -> std::string {
        auto& metaNodeExtended = h_getMetaNodeExtended(nodeId);

        auto h_intro = [&]() -> std::string {
            return std::string("Scheduling info: \\l- Taks idx ") +
                   std::to_string(metaNodeExtended.schedulingInfo.schedulingOrder) + "\\l";
        };
        // HELPER //////////////////////////////////////////////////////////////////////////
        auto h_eventWaitString = [&]() -> std::string {
            std::string ret;
            if (metaNodeExtended.schedulingInfo.waitEventIdxList.size() == 0) {
                ret += "- Waiting events: []\\l";
            } else {
                ret += "- Waiting events: [";
                for (auto& e : metaNodeExtended.schedulingInfo.waitEventIdxList) {
                    ret += " " + std::to_string(e);
                }
                ret += "]\\l";
            }
            return ret;
        };
        // HELPER //////////////////////////////////////////////////////////////////////////
        auto h_streamString = [&]() -> std::string {
            return "- Stream idx: " + std::to_string(metaNodeExtended.schedulingInfo.streamIdx) + "\\l";
        };

        [[maybe_unused]] auto h_barrierString = [&]() -> std::string {
            if (metaNodeExtended.schedulingInfo.barrierLR != -1) {
                return "- LR Event " + std::to_string(metaNodeExtended.schedulingInfo.barrierLR) + "\\l";
                ;
            }
            return "";
        };
        // HELPER //////////////////////////////////////////////////////////////////////////
        auto h_eventRegisterString = [&]() -> std::string {
            if (metaNodeExtended.schedulingInfo.setEventIdx != -1) {
                return "- Completion Event idx: " + std::to_string(metaNodeExtended.schedulingInfo.setEventIdx) +
                       "\\l";
            } else {
                return "- Completion Event idx: None";
            }
        };

        if (nodeId == mGpuGraph.finalNodeId()) {
            return std::string("END \n") + h_intro() + h_streamString() + h_eventWaitString() +
                   h_eventRegisterString() + "\\l";
        }
        if (nodeId == mGpuGraph.rootNodeId()) {
            return std::string("START \n") + h_intro() + h_streamString() + h_eventWaitString() +
                   h_eventRegisterString() + "\\l";
        }
        return clone.getVertexProperty(nodeId).toString() + h_intro() + h_streamString() + h_eventWaitString() +
               h_eventRegisterString() + "\\l";
    };
    // HELPER //////////////////////////////////////////////////////////////////////////
    auto edgeLabel = [&](const std::pair<size_t, size_t>&)
        -> std::string {
        return "";
    };

    // HELPER //////////////////////////////////////////////////////////////////////////
    auto edgeLabelProperty = [&](const std::pair<size_t, size_t>& edge)
        -> std::string {
        const auto& metaEdge = clone.getEdgeProperty(edge);
        if (metaEdge.m_isSchedulingEdge) {
            // return "style=dashed, color=\"#2A9D8F\"";
            return "style=dashed, color=\"#F4A261\", penwidth=7";
        }
        return "color=\"#d9d9d9\", penwidth=7";
    };
    // HELPER //////////////////////////////////////////////////////////////////////////
    auto vertexLabelProperty = [&](const size_t& v) {
        if (v == mGpuGraph.finalNodeId() || (v == mGpuGraph.rootNodeId())) {
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

    ////////////////////////////////////////////////////////////////////////////
    clone.exportDotFile(fname, graphName, vertexLabel, edgeLabel,
                        vertexLabelProperty, edgeLabelProperty);
}

#endif

auto StreamScheduler::io2DotOrder(const std::string& fname, const std::string& graphName) -> void
{
    MultiGpuGraph::DiGraphScheduling orderGraph;
    for (int i = 0; i < int(m_storage->m_executionOrder.size()); i++) {
        orderGraph.addVertex(m_storage->m_executionOrder[i]);
        if (i > 0) {
            orderGraph.addEdge(m_storage->m_executionOrder[i - 1], m_storage->m_executionOrder[i]);
        }
    }


    auto& mGpuGraph = m_storage->m_graph;

    auto vertexLabel = [&](NodeId nodeId) -> std::string {
        auto& metaNodeExtended = h_getMetaNodeExtended(nodeId);

        auto h_intro = [&]() -> std::string {
            return std::string("Scheduling info: \\l- Taks idx ") +
                   std::to_string(metaNodeExtended.schedulingInfo.schedulingOrder) + "\\l";
        };
        // HELPER //////////////////////////////////////////////////////////////////////////
        auto h_eventWaitString = [&]() -> std::string {
            std::string ret;
            if (metaNodeExtended.schedulingInfo.waitEventIdxList.size() == 0) {
                ret += "- Waiting events: []\\l";
            } else {
                ret += "- Waiting events: [";
                for (auto& e : metaNodeExtended.schedulingInfo.waitEventIdxList) {
                    ret += " " + std::to_string(e);
                }
                ret += "]\\l";
            }
            return ret;
        };
        // HELPER //////////////////////////////////////////////////////////////////////////
        auto h_streamString = [&]() -> std::string {
            return "- Stream idx: " + std::to_string(metaNodeExtended.schedulingInfo.streamIdx) + "\\l";
        };

        [[maybe_unused]] auto h_barrierString = [&]() -> std::string {
            if (metaNodeExtended.schedulingInfo.barrierLR != -1) {
                return "- LR Event " + std::to_string(metaNodeExtended.schedulingInfo.barrierLR) + "\\l";
                ;
            }
            return "";
        };
        // HELPER //////////////////////////////////////////////////////////////////////////
        auto h_eventRegisterString = [&]() -> std::string {
            if (metaNodeExtended.schedulingInfo.setEventIdx != -1) {
                return "- Completion Event idx: " + std::to_string(metaNodeExtended.schedulingInfo.setEventIdx) +
                       "\\l";
            } else {
                return "- Completion Event idx: None";
            }
        };

        if (nodeId == mGpuGraph.finalNodeId()) {
            return std::string("Task List \n END");
        }
        if (nodeId == mGpuGraph.rootNodeId()) {
            return std::string("Task List \n BEGIN");
        }
        return mGpuGraph.getDiGraph().getVertexProperty(nodeId).toString() + h_intro() + h_streamString() +
               h_eventWaitString() + h_eventRegisterString() + "\\l";
    };
    // HELPER //////////////////////////////////////////////////////////////////////////
    auto edgeLabel = [&](const std::pair<size_t, size_t>&)
        -> std::string {
        return "";
    };

    // HELPER //////////////////////////////////////////////////////////////////////////
    auto edgeLabelProperty = [&]([[maybe_unused]] const std::pair<size_t, size_t>& edge)
        -> std::string {
        return "color=\"#d9d9d9\", penwidth=7";
    };
    // HELPER //////////////////////////////////////////////////////////////////////////
    auto vertexLabelProperty = [&](const size_t& v) {
        if (v == mGpuGraph.finalNodeId() || (v == mGpuGraph.rootNodeId())) {
            return R"(shape=box, style="rounded,filled", fillcolor="#d9d9d9", color="#6c6c6c")";
        }
        const auto& metaNode = mGpuGraph.getDiGraph().getVertexProperty(v);
        if (metaNode.isHu()) {
            return R"(shape=box, style="rounded,filled", fillcolor="#fb8072", color="#b11605")";
        }
        if (metaNode.isSync()) {
            return R"(shape=box, style="rounded,filled", fillcolor="#fb8072", color="#b11605")";
        }

        if (metaNode.isMap()) {
            return R"(shape=box, style="rounded,filled", fillcolor="#b3de69", color="#5f861d")";
        }
        if (metaNode.isReduce()) {
            return R"(shape=box, style="rounded,filled", fillcolor="#80b1d3", color="#2b5c7d")";
        }
        if (metaNode.isStencil()) {
            return R"(shape=box, style="rounded,filled", fillcolor="#bebada", color="#4e4683")";
        }
        return "";
    };

    ////////////////////////////////////////////////////////////////////////////
    orderGraph.exportDotFile(fname, graphName, vertexLabel, edgeLabel,
                             vertexLabelProperty, edgeLabelProperty);
}


}  // namespace Neon::skeleton::internal
