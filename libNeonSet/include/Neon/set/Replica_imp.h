#pragma once

#include <memory>
#include "Neon/set/DevSet.h"
#include "Neon/set/Replica.h"

namespace Neon::set {

template <typename Obj>
Replica<Obj>::Replica(Neon::Backend&      bk,
                      Neon::DataUse       dataUse,
                      Neon::MemoryOptions memoryOptions)
{
    memoryOptions = bk.devSet().sanitizeMemoryOption(memoryOptions);

    auto& storage = this->getStorage();
    storage.memoryOptions = memoryOptions;
    storage.dataUse = dataUse;
    storage.bk = bk;
    auto nEntryPerGPU = bk.devSet().template newDataSet<size_t>(1);
    storage.obj = bk.devSet().template newMemSet<Obj>(Neon::DataUse::HOST_DEVICE,
                                                      1,
                                                      Neon::MemoryOptions(),
                                                      nEntryPerGPU);


    const int nDev = bk.devSet().setCardinality();
    for (auto target : {Neon::Execution::host, Neon::Execution::device}) {
        int targetIdx = Neon::ExecutionUtils::toInt(target);
        storage.partitionByView[targetIdx] = bk.devSet().template newDataSet<Self::Partition>();
        for (int setIdx = 0; setIdx < nDev; setIdx++) {
            auto addr = storage.obj.rawMem(target, setIdx);
            storage.partitionByView[targetIdx][setIdx].objPrt = addr;
        }
        if (target == Neon::Execution::device && (bk.devType() == Neon::DeviceType::CPU || bk.devType() == Neon::DeviceType::OMP)) {
            int hostIdx = Neon::ExecutionUtils::toInt(Neon::Execution::host);
            int deviceIdx = Neon::ExecutionUtils::toInt(Neon::Execution::device);
            for (int setIdx = 0; setIdx < nDev; setIdx++) {
                storage.partitionByView[deviceIdx][setIdx].objPrt = storage.partitionByView[hostIdx][setIdx].objPrt;
            }
        }
    }
}

template <typename Obj>
auto Replica<Obj>::updateHostData(int streamId)
    -> void
{
    auto&                storage = this->getStorage();
    const Neon::Backend& bk = storage.bk;
    if (storage.dataUse == Neon::DataUse::HOST_DEVICE) {
        if (storage.memoryOptions.getDeviceType() == Neon::DeviceType::CPU) {
            return;
        }
        if (storage.memoryOptions.getDeviceType() == Neon::DeviceType::CUDA) {
            storage.obj.updateHostData(bk, streamId);
            return;
        }
        NEON_THROW_UNSUPPORTED_OPTION("");
    }
}

template <typename Obj>
auto Replica<Obj>::updateDeviceData(int streamId)
    -> void
{
    auto&                storage = this->getStorage();
    const Neon::Backend& bk = storage.bk;
    if (storage.dataUse == Neon::DataUse::HOST_DEVICE) {
        if (storage.memoryOptions.getDeviceType() == Neon::DeviceType::CPU) {
            return;
        }
        if (storage.memoryOptions.getDeviceType() == Neon::DeviceType::CUDA) {
            storage.obj.updateDeviceData(bk, streamId);
            return;
        }
        NEON_THROW_UNSUPPORTED_OPTION("");
    }
}

/**
 * Return a partition based on a set of parameters: execution type, target device, dataView
 */
template <typename Obj>
auto Replica<Obj>::getPartition(Neon::Execution execution,
                                Neon::SetIdx    setIdx,
                                const Neon::DataView&) const
    -> const Self::Partition&
{
    auto& storage = this->getStorage();
    return storage.partitionByView[Neon::ExecutionUtils::toInt(execution)][setIdx];
}

template <typename Obj>
auto Replica<Obj>::getPartition(Neon::Execution execution,
                                Neon::SetIdx    setIdx,
                                const Neon::DataView&)
    -> Self::Partition&
{
    auto& storage = this->getStorage();
    return storage.partitionByView[Neon::ExecutionUtils::toInt(execution)][setIdx];
}

template <typename Obj>
auto Replica<Obj>::getPartition(Neon::DeviceType      execution,
                                Neon::SetIdx          setIdx,
                                const Neon::DataView& dw) const
    -> const Self::Partition&
{
    auto&          storage = this->getStorage();
    const Backend& bk = storage.bk;
    if (execution == DeviceType::CUDA) {
        return getPartition(Neon::Execution::device, setIdx, dw);
    }
    if (execution == DeviceType::OMP || execution == DeviceType::CPU) {
        if (bk.devSet().type() == DeviceType::CUDA)
            return getPartition(Neon::Execution::host, setIdx, dw);
        else
            return getPartition(Neon::Execution::device, setIdx, dw);
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

template <typename Obj>
auto Replica<Obj>::getPartition(Neon::DeviceType      execution,
                                Neon::SetIdx          setIdx,
                                const Neon::DataView& dw)
    -> Self::Partition&
{
    auto&          storage = this->getStorage();
    const Backend& bk = storage.bk;
    if (execution == DeviceType::CUDA) {
        return getPartition(Neon::Execution::device, setIdx, dw);
    }
    if (execution == DeviceType::OMP || execution == DeviceType::CPU) {
        if (bk.devSet().type() == DeviceType::CUDA)
            return getPartition(Neon::Execution::host, setIdx, dw);
        else
            return getPartition(Neon::Execution::device, setIdx, dw);
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

template <typename Obj>
auto Replica<Obj>::operator()(Neon::SetIdx setIdx) -> Obj&
{
    auto&     storage = this->getStorage();
    const int targetElement = 0;
    const int targetCardinality = 0;

    auto& obj = storage.obj.eRef(setIdx.idx(), targetElement,
                                 targetCardinality);
    return obj;
}
template <typename Obj>
auto Replica<Obj>::operator()(Neon::SetIdx setIdx) const -> const Obj&
{
    auto&     storage = this->getStorage();
    const int targetDevice = 0;
    const int targetElement = 0;
    const int targetCardinality = 0;

    auto& obj = storage.obj.eRef(setIdx.idx(), targetElement,
                                 targetCardinality);
    return obj;
}

template <typename Obj>
template <Neon::Execution execution,
          typename LoadingLambda>
auto Replica<Obj>::newContainer(const std::string& name,
                                LoadingLambda      lambda) const -> Neon::set::Container
{
    const Neon::index_3d defaultBlockSize(32, 1, 1);
    Neon::set::Container container = Neon::set::Container::factory<execution>(name,
                                                                              Neon::set::internal::ContainerAPI::DataViewSupport::off,
                                                                              *this,
                                                                              lambda,
                                                                              defaultBlockSize,
                                                                              [](const Neon::index_3d&) { return 0; });
    return container;
}
template <typename Obj>
auto Replica<Obj>::getBackend() -> Neon::Backend&
{
    return this->getStorage().bk;
}
template <typename Obj>
auto Replica<Obj>::getLaunchParameters(Neon::DataView,
                                       const index_3d& blockDim,
                                       const size_t&   shareMem) const -> Neon::set::LaunchParameters
{

    if (blockDim.y != 1 || blockDim.z != 1) {
        NeonException exc("Replica");
        exc << "CUDA block size should be 1D\n";
        NEON_THROW(exc);
    }

    auto      newLaunchParameters = this->getBackend().devSet().newLaunchParameters();
    const int nDevices = this->getBackend().devSet().setCardinality();

    for (int i = 0; i < nDevices; i++) {

        auto           gridMode = Neon::sys::GpuLaunchInfo::mode_e::domainGridMode;
        Neon::index_3d gridDim(1, 1, 1);
        newLaunchParameters[i].set(gridMode, gridDim, blockDim, shareMem);
    }
    return newLaunchParameters;
}
template <typename Obj>
auto Replica<Obj>::getSpan(Neon::Execution, Neon::SetIdx, const DataView&) const -> const Replica::Span&
{
    return this->getStorage().indexSpace;
}
template <typename Obj>
auto Replica<Obj>::getSpan(Neon::Execution, Neon::SetIdx, const DataView&) -> Replica::Span&
{
    return this->getStorage().indexSpace;
}
template <typename Obj>
auto Replica<Obj>::getBackend() const -> const Neon::Backend&
{
    return this->getStorage().bk;
}


}  // namespace Neon::set
