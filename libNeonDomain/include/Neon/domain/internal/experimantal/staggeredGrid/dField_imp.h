#pragma once
#include "dField.h"

namespace Neon::domain::internal::dGrid {

template <typename T, int C>
dField<T, C>::dField(const std::string&                        fieldUserName,
                     Neon::DataUse                             dataUse,
                     const Neon::MemoryOptions&                memoryOptions,
                     const Grid&                               grid,
                     const Neon::set::DataSet<Neon::index_3d>& dims,
                     int                                       zHaloDim,
                     Neon::domain::haloStatus_et::e            haloStatus,
                     int                                       cardinality)
    : Neon::domain::interface::FieldBaseTemplate<T, C, Grid, Partition, int>(&grid,
                                                                             fieldUserName,
                                                                             "dField",
                                                                             cardinality,
                                                                             T(0),
                                                                             dataUse,
                                                                             memoryOptions,
                                                                             haloStatus)
{
    mDataUse = dataUse;
    mMemoryOptions = memoryOptions;

    // only works if dims in x and y direction for all partitions match
    for (int i = 0; i < dims.size() - 1; ++i) {
        for (int j = i + 1; j < dims.size(); ++j) {
            if (dims[i].x != dims[j].x || dims[i].y != dims[j].y) {
                NeonException exc("dField_t");
                exc << "New dField only works on partitioning along z axis.";
                NEON_THROW(exc);
            }
        }
    }


    m_gpu = dFieldDev<T, C>(grid,
                            dims,
                            zHaloDim,
                            haloStatus,
                            memoryOptions.getComputeType(),
                            Neon::memLayout_et::convert(memoryOptions.getOrder()),
                            memoryOptions.getComputeAllocator(dataUse),
                            cardinality);

    m_cpu = dFieldDev<T, C>(grid,
                            dims,
                            zHaloDim,
                            haloStatus,
                            memoryOptions.getIOType(),
                            Neon::memLayout_et::convert(memoryOptions.getOrder()),
                            memoryOptions.getIOAllocator(dataUse),
                            cardinality);
}

template <typename T, int C>
auto dField<T, C>::deviceField(const Neon::Backend& backendConfig) -> FieldDev&
{
    switch (backendConfig.devType()) {
        case Neon::DeviceType::OMP:
        case Neon::DeviceType::CPU: {
            return m_cpu;
        }
        case Neon::DeviceType::CUDA: {
            return m_gpu;
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPERATION("");
        }
    }
}

template <typename T, int C>
template <Neon::run_et::et runMode_ta>
auto dField<T, C>::update(const Neon::set::StreamSet& streamSet,
                          const Neon::DeviceType&     devEt) -> void
{
    if (self().getGrid().getDevSet().type() == Neon::DeviceType::CPU) {
        return;
    }

    if (m_cpu.devType() == Neon::DeviceType::NONE ||
        m_gpu.devType() == Neon::DeviceType::NONE ||
        m_cpu.m_data->memAlloc == Neon::Allocator::NULL_MEM ||
        m_gpu.m_data->memAlloc == Neon::Allocator::NULL_MEM) {
        NeonException exp("dField_t");
        exp << "CPU/GPU field is not initialized properly.";
        NEON_THROW(exp);
    }

    switch (devEt) {
        case Neon::DeviceType::CPU: {
            m_cpu.m_data->memory.template updateFrom<runMode_ta>(streamSet,
                                                                 m_gpu.m_data->memory);
            break;
        }
        case Neon::DeviceType::CUDA: {
            m_gpu.m_data->memory.template updateFrom<runMode_ta>(streamSet,
                                                                 m_cpu.m_data->memory);
            break;
        }
        default: {
            NeonException exp("dField_t");
            NEON_THROW(exp);
        }
    }
}

template <typename T, int C>
auto dField<T, C>::updateCompute(const Neon::set::StreamSet& streamSet)
    -> void
{
    if (m_cpu.devType() == Neon::DeviceType::NONE && m_gpu.devType() == Neon::DeviceType::NONE) {
        NeonException exp("dField");
        exp << "Invalid operation on a not fully initialized mirror.";
        NEON_THROW(exp);
    }

    if (mDataUse == Neon::DataUse::IO_COMPUTE) {
        if (mMemoryOptions.getComputeType() == Neon::DeviceType::CPU) {
            return;
        }
        if (mMemoryOptions.getComputeType() == Neon::DeviceType::CUDA) {
            this->update(streamSet, Neon::DeviceType::CUDA);
            return;
        }
        NEON_THROW_UNSUPPORTED_OPTION("");
    }
}

template <typename T, int C>
auto dField<T, C>::updateCompute(int streamSetId)
    -> void
{
    const Neon::Backend& backendConfig = self().getGrid().getBackend();
    if (m_cpu.devType() == Neon::DeviceType::NONE && m_gpu.devType() == Neon::DeviceType::NONE) {
        NeonException exp("dField");
        exp << "Invalid operation on a not fully initialized mirror.";
        NEON_THROW(exp);
    }

    if (mDataUse == Neon::DataUse::IO_COMPUTE) {
        if (mMemoryOptions.getComputeType() == Neon::DeviceType::CPU) {
            return;
        }
        if (mMemoryOptions.getComputeType() == Neon::DeviceType::CUDA) {
            this->update(backendConfig.streamSet(streamSetId), Neon::DeviceType::CUDA);
            return;
        }
        NEON_THROW_UNSUPPORTED_OPTION("");
    }
}

template <typename T, int C>
auto dField<T, C>::updateIO(int streamSetId)
    -> void
{
    const Neon::Backend& backendConfig = self().getGrid().getBackend();
    if (m_cpu.devType() == Neon::DeviceType::NONE && m_gpu.devType() == Neon::DeviceType::NONE) {
        NeonException exp("dField");
        exp << "Invalid operation on a not fully initialized mirror.";
        NEON_THROW(exp);
    }
    if (mDataUse == Neon::DataUse::IO_COMPUTE) {
        if (mMemoryOptions.getComputeType() == Neon::DeviceType::CPU) {
            return;
        }
        if (mMemoryOptions.getComputeType() == Neon::DeviceType::CUDA) {
            this->update(backendConfig.streamSet(streamSetId), Neon::DeviceType::CPU);
            return;
        }
        NEON_THROW_UNSUPPORTED_OPTION("");
    }
}

template <typename T, int C>
auto dField<T, C>::updateIO(const Neon::set::StreamSet& streamSet)
    -> void
{
    if (m_cpu.devType() == Neon::DeviceType::NONE && m_gpu.devType() == Neon::DeviceType::NONE) {
        NeonException exp("dField");
        exp << "Invalid operation on a not fully initialized mirror.";
        NEON_THROW(exp);
    }
    if (mDataUse == Neon::DataUse::IO_COMPUTE) {
        if (mMemoryOptions.getComputeType() == Neon::DeviceType::CPU) {
            return;
        }
        if (mMemoryOptions.getComputeType() == Neon::DeviceType::CUDA) {
            this->update(streamSet, Neon::DeviceType::CPU);
            return;
        }
    }
}

template <typename T, int C>
auto dField<T, C>::field(const Neon::DeviceType& devType) const -> const FieldDev&
{
    switch (devType) {
        case Neon::DeviceType::CPU:
        case Neon::DeviceType::OMP: {
            return m_cpu;
        }
        case Neon::DeviceType::CUDA: {
            return m_gpu;
        }
        default: {
            NeonException exp("dField_t");
            NEON_THROW(exp);
        }
    }
}

template <typename T, int C>
auto dField<T, C>::field(Neon::Backend& backendConfig) const -> const FieldDev&
{
    switch (backendConfig.devType()) {
        case Neon::DeviceType::CPU:
        case Neon::DeviceType::OMP: {
            return m_cpu;
        }
        case Neon::DeviceType::CUDA: {
            return m_gpu;
        }
        default: {
            NeonException exp("dField_t");
            NEON_THROW(exp);
        }
    }
}

template <typename T, int C>
auto dField<T, C>::field(const Neon::DeviceType& devType) -> FieldDev&
{
    switch (devType) {
        case Neon::DeviceType::CPU:
        case Neon::DeviceType::OMP: {
            return m_cpu;
        }
        case Neon::DeviceType::CUDA: {
            return m_gpu;
        }
        default: {
            NeonException exp("dField_t");
            NEON_THROW(exp);
        }
    }
}

template <typename T, int C>
auto dField<T, C>::field(Neon::Backend& backendConfig) -> FieldDev&
{
    switch (backendConfig.devType()) {
        case Neon::DeviceType::CPU:
        case Neon::DeviceType::OMP: {
            return m_cpu;
        }
        case Neon::DeviceType::CUDA: {
            return m_gpu;
        }
        default: {
            NeonException exp("dField_t");
            NEON_THROW(exp);
        }
    }
}

template <typename T, int C>
auto dField<T, C>::cpu()
    -> FieldDev&
{
    return m_cpu;
}

template <typename T, int C>
auto dField<T, C>::gpu()
    -> FieldDev&
{
    return m_gpu;
}

template <typename T, int C>
auto dField<T, C>::getLaunchInfo(const Neon::DataView dataView) const
    -> Neon::set::LaunchParameters
{
    return m_gpu.getLaunchInfo(dataView);
}

template <typename T, int C>
auto dField<T, C>::ccpu() const
    -> const FieldDev&
{
    return m_cpu;
}

template <typename T, int C>
auto dField<T, C>::cgpu() const
    -> const FieldDev&
{
    return m_gpu;
}

template <typename T, int C>
auto dField<T, C>::getPartition(const Neon::DeviceType& devType,
                                const Neon::SetIdx&     idx,
                                const Neon::DataView&   dataView) const
    -> const Partition&
{
    switch (devType) {
        case Neon::DeviceType::CPU:
        case Neon::DeviceType::OMP: {
            return m_cpu.getPartition(devType, idx, dataView);
        }
        case Neon::DeviceType::CUDA: {
            return m_gpu.getPartition(devType, idx, dataView);
        }
        default: {
            NeonException exp("dField_t");
            NEON_THROW(exp);
        }
    }
}

template <typename T, int C>
auto dField<T, C>::getPartition(const Neon::DeviceType& devType,
                                const Neon::SetIdx&     idx,
                                const Neon::DataView&   dataView)
    -> Partition&
{
    switch (devType) {
        case Neon::DeviceType::CPU:
        case Neon::DeviceType::OMP: {
            return m_cpu.getPartition(devType, idx, dataView);
        }
        case Neon::DeviceType::CUDA: {
            return m_gpu.getPartition(devType, idx, dataView);
        }
        default: {
            NeonException exp("dField_t");
            NEON_THROW(exp);
        }
    }
}

template <typename T, int C>
auto dField<T, C>::getPartition([[maybe_unused]] Neon::Execution,
                                [[maybe_unused]] Neon::SetIdx,
                                [[maybe_unused]] const Neon::DataView& dataView)
    const
    -> const Partition&
{
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename T, int C>
auto dField<T, C>::getPartition([[maybe_unused]] Neon::Execution,
                                [[maybe_unused]] Neon::SetIdx          idx,
                                [[maybe_unused]] const Neon::DataView& dataView)
    -> Partition&
{
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename T, int C>
auto dField<T, C>::operator()(const Neon::index_3d& idx,
                              const int&            cardinality) const
    -> Type
{
    if (m_cpu.devType() == Neon::DeviceType::NONE) {
        NeonException exc("dField_t");
        NEON_THROW(exc);
    }
    return m_cpu.eRef(idx, cardinality);
}

template <typename T, int C>
auto dField<T, C>::getReference(const Neon::index_3d& idx,
                                const int&            cardinality)
    -> Type&
{
    if (m_cpu.devType() == Neon::DeviceType::NONE) {
        NeonException exc("dField_t");
        NEON_THROW(exc);
    }
    return m_cpu.eRef(idx, cardinality);
}


template <typename T, int C>
template <Neon::set::TransferMode transferMode_ta>
auto dField<T, C>::haloUpdate(const Neon::Backend& bk,
                              bool                 startWithBarrier,
                              int                  streamSetIdx)
    -> void
{
    auto fieldDev = this->field(bk.devType());
    fieldDev.template haloUpdate<transferMode_ta>(bk, startWithBarrier, streamSetIdx);
}

template <typename T, int C>
template <Neon::set::TransferMode transferMode_ta>
auto dField<T, C>::haloUpdate(const Neon::Backend& bk,
                              int                  cardinality,
                              bool                 startWithBarrier,
                              int                  streamSetIdx)
    -> void
{
    auto fieldDev = field(bk.devType());
    fieldDev.template haloUpdate<transferMode_ta>(bk, cardinality, startWithBarrier, streamSetIdx);
}

template <typename T, int C>
auto dField<T, C>::haloUpdate(Neon::set::HuOptions& opt) const
    -> void
{
    auto& bk = self().getBackend();
    auto  fieldDev = field(bk.devType());
    switch (opt.transferMode()) {
        case Neon::set::TransferMode::put:
            fieldDev.template haloUpdate<Neon::set::TransferMode::put>(bk, -1, opt.startWithBarrier(), opt.streamSetIdx());
            break;
        case Neon::set::TransferMode::get:
            fieldDev.template haloUpdate<Neon::set::TransferMode::get>(bk, -1, opt.startWithBarrier(), opt.streamSetIdx());
            break;
        default:
            NEON_THROW_UNSUPPORTED_OPTION();
            break;
    }
}

template <typename T, int C>
auto dField<T, C>::haloUpdate(Neon::SetIdx          setIdx,
                              Neon::set::HuOptions& opt) const
    -> void
{
    auto& bk = self().getBackend();
    auto  fieldDev = field(bk.devType());
    switch (opt.transferMode()) {
        case Neon::set::TransferMode::put:
            fieldDev.template haloUpdate<Neon::set::TransferMode::put>(setIdx, bk, -1, opt.startWithBarrier(), opt.streamSetIdx());
            break;
        case Neon::set::TransferMode::get:
            fieldDev.template haloUpdate<Neon::set::TransferMode::get>(setIdx, bk, -1, opt.startWithBarrier(), opt.streamSetIdx());
            break;
        default:
            NEON_THROW_UNSUPPORTED_OPTION();
            break;
    }
}

template <typename T, int C>
auto dField<T, C>::haloUpdate(Neon::set::HuOptions& opt)
    -> void
{
    auto& bk = self().getBackend();
    auto  fieldDev = field(bk.devType());
    switch (opt.transferMode()) {
        case Neon::set::TransferMode::put:
            fieldDev.template haloUpdate<Neon::set::TransferMode::put>(bk, -1, opt.startWithBarrier(), opt.streamSetIdx());
            break;
        case Neon::set::TransferMode::get:
            fieldDev.template haloUpdate<Neon::set::TransferMode::get>(bk, -1, opt.startWithBarrier(), opt.streamSetIdx());
            break;
        default:
            NEON_THROW_UNSUPPORTED_OPTION();
            break;
    }
}

template <typename T, int C>
auto dField<T, C>::haloUpdate(Neon::SetIdx          setIdx,
                              Neon::set::HuOptions& opt)
    -> void
{
    auto& bk = self().getBackend();
    auto  fieldDev = field(bk.devType());
    switch (opt.transferMode()) {
        case Neon::set::TransferMode::put:
            fieldDev.template haloUpdate<Neon::set::TransferMode::put>(setIdx, bk, -1, opt.startWithBarrier(), opt.streamSetIdx());
            break;
        case Neon::set::TransferMode::get:
            fieldDev.template haloUpdate<Neon::set::TransferMode::get>(setIdx, bk, -1, opt.startWithBarrier(), opt.streamSetIdx());
            break;
        default:
            NEON_THROW_UNSUPPORTED_OPTION();
            break;
    }
}

template <typename T, int C>
auto dField<T, C>::dot(Neon::set::patterns::BlasSet<T>& blasSet,
                       const dField<T>&                 input,
                       Neon::set::MemDevSet<T>&         output,
                       const Neon::DataView&            dataView) -> T
{
    if (self().getBackend().devType() == Neon::DeviceType::CUDA) {
        T val = m_gpu.dot(blasSet, input.field(Neon::DeviceType::CUDA), output, dataView);
        return val;
    } else {
        T val = m_cpu.dot(blasSet, input.field(Neon::DeviceType::CPU), output, dataView);
        return val;
    }
}

template <typename T, int C>
auto dField<T, C>::dotCUB(Neon::set::patterns::BlasSet<T>& blasSet,
                          const dField<T>&                 input,
                          Neon::set::MemDevSet<T>&         output,
                          const Neon::DataView&            dataView) -> void
{
    if (self().getBackend().devType() != Neon::DeviceType::CUDA) {
        NeonException exc("dField_t");
        exc << "dotCUB only works for CUDA backend";
        NEON_THROW(exc);
    }
    m_gpu.dotCUB(blasSet, input.field(Neon::DeviceType::CUDA), output, dataView);
}

template <typename T, int C>
auto dField<T, C>::norm2(Neon::set::patterns::BlasSet<T>& blasSet,
                         Neon::set::MemDevSet<T>&         output,
                         const Neon::DataView&            dataView) -> T
{
    if (self().getBackend().devType() == Neon::DeviceType::CUDA) {
        return m_gpu.norm2(blasSet, output, dataView);
    } else {
        return m_cpu.norm2(blasSet, output, dataView);
    }
}

template <typename T, int C>
auto dField<T, C>::norm2CUB(Neon::set::patterns::BlasSet<T>& blasSet,
                            Neon::set::MemDevSet<T>&         output,
                            const Neon::DataView&            dataView) -> void
{
    if (self().getBackend().devType() != Neon::DeviceType::CUDA) {
        NeonException exc("dField_t");
        exc << "norm2CUB only works for CUDA backend";
        NEON_THROW(exc);
    }
    m_gpu.norm2CUB(blasSet, output, dataView);
}

template <typename T, int C>
auto dField<T, C>::self() -> dField::Self&
{
    return *this;
}

template <typename T, int C>
auto dField<T, C>::self() const -> const dField::Self&
{
    return *this;
}
template <typename T, int C>
auto dField<T, C>::swap(dField::Field& A, dField::Field& B) -> void
{
    Neon::domain::interface::FieldBaseTemplate<T, C, Grid, Partition, int>::swapUIDBeforeFullSwap(A, B);
    std::swap(A, B);
}

}  // namespace Neon::domain::internal::dGrid
