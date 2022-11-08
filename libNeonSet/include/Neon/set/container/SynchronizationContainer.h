#pragma once
#include "Neon/core/core.h"

#include "Neon/set/container/ContainerAPI.h"
#include "Neon/set/container/DeviceContainer.h"
#include "Neon/set/container/Graph.h"
#include "Neon/set/container/Loader.h"
#include "Neon/set/container/graph/GraphNode.h"
#include "Neon/set/container/types/SynchronizationContainerType.h"

#include <omp.h>

namespace Neon::set::internal {

template <typename MxpuDataT>
struct SynchronizationContainer
    : ContainerAPI
{
    ~SynchronizationContainer() override = default;

    explicit SynchronizationContainer(const MxpuDataT&             multiXpuData,
                                      SynchronizationContainerType syncType)
        : mMultiXpuData(multiXpuData),
          mSynchronizationType(syncType)
    {
        setName("SynchronizationContainer");

        setContainerExecutionType(ContainerExecutionType::deviceManaged);
        setContainerOperationType(ContainerOperationType::synchronization);
        setDataViewSupport(DataViewSupport::off);

        if (syncType != SynchronizationContainerType::hostOmpBarrier) {
            NEON_THROW_UNSUPPORTED_OPERATION("");
        }
    }

    auto run(int            streamIdx,
             Neon::DataView /*dataView*/) -> void override
    {
        const Neon::Backend& bk = mMultiXpuData.getBackend();
        bk.sync(streamIdx);
    }

    auto
    run(Neon::SetIdx   setIdx,
        int            streamIdx,
        Neon::DataView /*dataView*/)
        -> void override
    {
        const auto& bk = mMultiXpuData.getBackend();
        if (mEvents.empty()) {
            bk.sync(setIdx, streamIdx);
        } else {
            for (auto eventId : mEvents) {
                bk.syncEvent(setIdx, eventId);
            }
        }

#pragma omp barrier
//#pragma omp critical
//        {
//            const int threadRank = omp_get_thread_num();
//            NEON_TRACE("TRACE SynchronizationContainer rank {} setIdx {} stream {} ", threadRank, setIdx.idx(), streamIdx);
//        };
    }

    auto configureWithScheduling(Neon::set::container::GraphNode& /*graphNode*/)
        -> void override
    {
//        mEvents = graphNode.getScheduling().getDependentEvents();
//        graphNode.getScheduling().getDependentEvents().clear();
    }

   private:
    MxpuDataT                    mMultiXpuData;
    SynchronizationContainerType mSynchronizationType;
    std::vector<int>             mEvents;

};  // namespace Neon::set::internal

}  // namespace Neon::set::internal
