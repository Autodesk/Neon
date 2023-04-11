#pragma once
#include "Neon/core/core.h"

#include "Neon/set/MemoryTransfer.h"
#include "Neon/set/container/ContainerAPI.h"
#include "Neon/set/container/Loader.h"

namespace Neon::set::internal {

template <typename MxpuDataT>
struct DataTransferContainer
    : ContainerAPI
{
    virtual ~DataTransferContainer() override = default;

    DataTransferContainer(const MxpuDataT&                                                  multiXpuData,
                          Neon::set::TransferMode                                           transferMode,
                          Neon::set::StencilSemantic                                        transferSemantic,
                          Neon::set::DataSet<std::vector<Neon::set::MemoryTransfer>> const& memoryTransfers,
                          Neon::Execution                                                   execution)
        : mMultiXpuData(multiXpuData),
          mTransferMode(transferMode),
          mTransferSemantic(transferSemantic),
          mMemoryTransfers(memoryTransfers)
    {
        setName("DataTransferContainer");

        if (Execution::host == execution) {
            NEON_DEV_UNDER_CONSTRUCTION("");
        }

        setContainerExecutionType(ContainerExecutionType::deviceManaged);
        setContainerOperationType(ContainerOperationType::communication);

        setDataViewSupport(DataViewSupport::off);
    }

    auto run(int            streamIdx,
             Neon::DataView /*dataView*/) -> void override
    {
        const Neon::Backend& bk = mMultiXpuData.getBackend();

        bk.forEachDeviceSeq([&](SetIdx setIdx) {
            //            std::cout <<"Sending Section ("<<setIdx<<") " <<std::endl;
            for (auto& memoryTransfer : mMemoryTransfers[setIdx]) {
                bk.template deviceToDeviceTransfer<char>(streamIdx,
                                                         memoryTransfer.size,
                                                         mTransferMode,
                                                         memoryTransfer.dst.setIdx, (char*)memoryTransfer.dst.mem,
                                                         memoryTransfer.src.setIdx, (char*)memoryTransfer.src.mem);
                // std::cout <<"Sending ("<<setIdx<<") " << memoryTransfer.toString()<<std::endl;
                // std::cout<< " val " << ((int64_t*)memoryTransfer.src.mem)[0]<<std::endl;
            }
        });
    }

    auto run(Neon::SetIdx setIdx,
             int          streamIdx,
             Neon::DataView /*dataView*/) -> void override
    {
        if (ContainerExecutionType::deviceManaged == this->getContainerExecutionType()) {
            const Neon::Backend& bk = mMultiXpuData.getBackend();
            for (auto& memoryTransfer : mMemoryTransfers[setIdx]) {
                bk.template deviceToDeviceTransfer<char>(streamIdx,
                                                         memoryTransfer.size,
                                                         mTransferMode,
                                                         memoryTransfer.dst.setIdx, (char*)memoryTransfer.dst.mem,
                                                         memoryTransfer.src.setIdx, (char*)memoryTransfer.src.mem);
            }
        }
        NEON_THROW_UNSUPPORTED_OPTION("");
    }

   private:
    MxpuDataT                                                  mMultiXpuData;
    Neon::set::TransferMode                                    mTransferMode;
    Neon::set::StencilSemantic                                 mTransferSemantic;
    Neon::set::DataSet<std::vector<Neon::set::MemoryTransfer>> mMemoryTransfers;
};

}  // namespace Neon::set::internal
