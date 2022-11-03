#pragma once
#include "Neon/core/core.h"

#include "Neon/set/container/ContainerAPI.h"
#include "Neon/set/container/DeviceContainer.h"
#include "Neon/set/container/Graph.h"
#include "Neon/set/container/Loader.h"
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
             Neon::DataView dataView) -> void override
    {
        const Neon::Backend& bk = mMultiXpuData.getBackend();
        const int            setCardinality = bk.devSet().setCardinality();

#pragma omp parallel num_threads(setCardinality)
        {
            const int threadRank = omp_get_thread_num();
            run(Neon::SetIdx(threadRank), streamIdx, dataView);
        }
    }

    auto
    run(Neon::SetIdx   setIdx,
        int            streamIdx,
        Neon::DataView dataView) -> void override
    {
        const auto& bk = mMultiXpuData.getBackend();
        bk.sync(setIdx, streamIdx);
#pragma omp barrier
    }

   private:
    MxpuDataT                    mMultiXpuData;
    SynchronizationContainerType mSynchronizationType;

};  // namespace Neon::set::internal

}  // namespace Neon::set::internal
