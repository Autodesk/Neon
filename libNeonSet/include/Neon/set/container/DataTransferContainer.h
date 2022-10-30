#pragma once
#include "Neon/core/core.h"

#include "Neon/set/container/ContainerAPI.h"
#include "Neon/set/container/Loader.h"

namespace Neon::set::internal {

template <typename MxpuDataT>
struct DataTransferContainer
    : ContainerAPI
{
    virtual ~DataTransferContainer() override = default;

    DataTransferContainer(const MxpuDataT&        multiXpuData,
                          Neon::set::TransferMode     transferMode,
                          Neon::set::TransferSemantic transferSemantic)
        : mMultiXpuData(multiXpuData),
          mTransferMode(transferMode),
          mTransferSemantic(transferSemantic)
    {
        setName("DataTransferContainer");

        setContainerExecutionType(ContainerExecutionType::deviceManaged);
        setContainerOperationType(ContainerOperationType::communication);
        setDataViewSupport(DataViewSupport::off);

        mDataTransferFun = [&](Neon::SetIdx setIdx,
                       int          streamIdx) {
            Neon::set::HuOptions options(this->mTransferMode,
                                         false,
                                         streamIdx,
                                         mTransferSemantic);
            this->mMultiXpuData.haloUpdate(setIdx, options);
        };
    }

    auto run(int            streamIdx,
             Neon::DataView dataView) -> void override
    {
        const Neon::Backend& bk = mMultiXpuData.getBackend();
        const int            setCardinality = bk.devSet().setCardinality();

#pragma omp parallel for num_threads(setCardinality)
        for (int i = 0; i < setCardinality; ++i) {
            run(Neon::SetIdx(i), streamIdx, dataView);
        }
    }

    auto run(Neon::SetIdx   setIdx,
             int            streamIdx,
             Neon::DataView dataView) -> void override
    {
        if (ContainerExecutionType::deviceManaged == this->getContainerType()) {
            mDataTransferFun(setIdx, streamIdx);
            return;
        }
        NEON_THROW_UNSUPPORTED_OPTION("");
    }

   private:
    std::function<void(Neon::SetIdx setIdx,
                       int          streamIdx)>
                                mDataTransferFun;
    MxpuDataT               mMultiXpuData;
    Neon::set::TransferMode     mTransferMode;
    Neon::set::TransferSemantic mTransferSemantic;
};

}  // namespace Neon::set::internal
