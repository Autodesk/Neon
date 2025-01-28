#pragma once
#include "Neon/core/core.h"

#include "Neon/set/MemoryTransfer.h"
#include "Neon/set/container/ContainerAPI.h"
#include "Neon/set/container/Loader.h"

namespace Neon::set::internal {

template <typename Grid>
struct NcclTransferContainer
    : ContainerAPI
{
    virtual ~NcclTransferContainer() override = default;

    NcclTransferContainer(const Grid&                      multiXpuData,
                          Neon::set::StencilSemantic       transferSemantic,
                          std::vector<Neon::set::NcclPtoP> NcclSession,
                          Neon::Execution                  execution)
        : mGrid(multiXpuData),
          mTransferSemantic(transferSemantic)
    {
        setName("NcclTransferContainer");

        if (Execution::host == execution) {
            NEON_DEV_UNDER_CONSTRUCTION("");
        }

        setContainerExecutionType(ContainerExecutionType::deviceManaged);
        setContainerOperationType(ContainerOperationType::communication);

        setDataViewSupport(DataViewSupport::off);
        ncclSession = NcclSession;
    }

    auto run(int streamIdx,
             Neon::DataView /*dataView*/) -> void override
    {
        const Neon::Backend& bk = mGrid.getBackend();
        ncclGroupStart();
        for (auto& session : ncclSession) {
            session.execute(bk, streamIdx);
        }
        ncclGroupEnd();
    }

    auto run(Neon::SetIdx /*setIdx*/,
             int /*streamIdx*/,
             Neon::DataView /*dataView*/) -> void override
    {
        NEON_THROW_UNSUPPORTED_OPTION("");
    }

   private:
    Grid                             mGrid;
    Neon::set::StencilSemantic       mTransferSemantic;
    std::vector<Neon::set::NcclPtoP> ncclSession;
};

}  // namespace Neon::set::internal
