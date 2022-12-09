#pragma once

#include "eField.h"

namespace Neon::domain::internal::eGrid {

template <typename T, int C>
eField<T, C>::eField(const std::string&             fieldUserName,
                     int                            cardinality,
                     T                              outsideVal,
                     Neon::DataUse                  dataUse,
                     Neon::MemoryOptions            memoryOptions,
                     Neon::domain::haloStatus_et::e haloStatus,
                     const Neon::set::DataConfig&   dataConfig,
                     FieldDev&                      CPU,
                     FieldDev&                      GPU)
    : Neon::domain::interface::FieldBaseTemplate<T, C, typename Self::Grid, typename Self::Partition, int>(&CPU.grid(),
                                                                                                           fieldUserName,
                                                                                                           "eField",
                                                                                                           cardinality,
                                                                                                           outsideVal,
                                                                                                           dataUse,
                                                                                                           memoryOptions,
                                                                                                           haloStatus)
{

    m_dataConfig = dataConfig;

    if (GPU.devType() != Neon::DeviceType::CUDA) {
        NeonException exp("eFieldMirror_t");
        exp << "Incompatible second input parameter for initialization.";
        NEON_THROW(exp);
    }
    if (CPU.devType() != Neon::DeviceType::CPU) {
        NeonException exp("eFieldMirror_t");
        exp << "Incompatible first input parameter for initialization.";
        NEON_THROW(exp);
    }

    self().helpLink(CPU);
    self().helpLink(GPU);
}

template <typename T, int C>
auto eField<T, C>::operator()(const Neon::index_3d& idx,
                              const int&            cardinality) const
    -> Type
{
    if (mCpu.devType() == Neon::DeviceType::NONE) {
        NeonException exc("eField");
        exc << "eRef operation is can not be run on a GPU field.";
        NEON_THROW(exc);
    }
    if (!helpIsActive(idx)) {
        return this->getOutsideValue();
    }
    return mCpu.eRef(idx, cardinality);
}

template <typename T, int C>
auto eField<T, C>::getReference(const Neon::index_3d& idx,
                                const int&            cardinality)
    -> Type&
{
    if (mCpu.devType() == Neon::DeviceType::NONE) {
        NeonException exc("eField");
        exc << "eRef operation is can not be run on a GPU field.";
        NEON_THROW(exc);
    }
    if (!helpIsActive(idx)) {
        return this->getOutsideValue();
    }
    return mCpu.eRef(idx, cardinality);
}

template <typename T, int C>
auto eField<T, C>::self() -> Self&
{
    return *this;
}

template <typename T, int C>
auto eField<T, C>::self() const -> const Self&
{
    return *this;
}

template <typename T, int C>
auto eField<T, C>::cSelf() const -> const Self&
{
    return *this;
}

template <typename T, int C>
auto eField<T, C>::genericSelf() -> GenericSelf<T>&
{
    return *(GenericSelf<T>*)this;
}

template <typename T, int C>
template <int Card>
auto eField<T, C>::specificSelf() -> SpecificSelf<T, Card>&
{
    if (C != self().getcardinality()) {
        NeonException exp("eField_t");
        exp << "Incompatible getcardinality for specificSelf casting.";
        NEON_THROW(exp);
    }
    return &(SpecificSelf<T, C>*)this;
}

template <typename T, int C>
template <int Card>
auto eField<T, C>::specificSelf() const -> const SpecificSelf<T, Card>&
{
    if (C != self().getcardinality()) {
        NeonException exp("eField_t");
        exp << "Incompatible getcardinality for specificSelf casting.";
        NEON_THROW(exp);
    }
    const SpecificSelf<T, C>& ret = *(SpecificSelf<T, C>*)this;
    return ret;
}

template <typename T, int C>
auto eField<T, C>::cardinality() const -> int
{
    return mCpu.cardinality();
}

template <typename T, int C>
auto eField<T, C>::helpLink(FieldDev& field)
    -> void
{
    FieldDev* target = nullptr;
    bool*     targetFlag = nullptr;

    switch (field.devType()) {
        case Neon::DeviceType::CPU: {
            target = &mCpu;
            targetFlag = &mCpuLink;
            break;
        }
        case Neon::DeviceType::CUDA: {
            target = &mGpu;
            targetFlag = &mGpuLink;
            break;
        }
        default: {
            NeonException exp("eField_t");
            exp << "Incompatible input parameter for initialization.";
            NEON_THROW(exp);
        }
    }

    if ((*target).devType() != Neon::DeviceType::NONE) {
        NeonException exp("eField_t");
        exp << "Reinitializing a mirror component is not an allowed operation. Please do a reset first";
        NEON_THROW(exp);
    }

    (*target) = field;
    (*targetFlag) = true;
}

template <typename T, int C>
auto eField<T, C>::helpIsActive(const Neon::index_3d& idx) const
    -> bool
{
    if (mCpu.devType() == Neon::DeviceType::NONE) {
        NeonException exc("eField");
        exc << "helpUpdate cannot be called on a GPU field.";
        NEON_THROW(exc);
    }
    return mCpu.eActive(idx);
}

template <typename T, int C>
auto eField<T, C>::helpUpdate(const Neon::set::StreamSet& streamSet,
                              const Neon::DeviceType&     devEt)
    -> void
{
    if (m_dataConfig.backendConfig().devType() == Neon::DeviceType::CPU) {
        return;
    }

    if (mCpu.devType() == Neon::DeviceType::NONE || mGpu.devType() == Neon::DeviceType::NONE) {
        NeonException exp("eField_t");
        exp << "Invalid operation on a not fully initialized mirror.";
        NEON_THROW(exp);
    }
    Neon::set::MemDevSet<T>* src = nullptr;
    Neon::set::MemDevSet<T>* dst = nullptr;
    switch (devEt) {
        case Neon::DeviceType::CPU: {
            dst = &(mCpu.m_data->memoryStorage);
            src = &(mGpu.m_data->memoryStorage);
            break;
        }
        case Neon::DeviceType::CUDA: {
            src = &(mCpu.m_data->memoryStorage);
            dst = &(mGpu.m_data->memoryStorage);
            break;
        }
        default: {
            NeonException exp("eField_t");
            exp << "Incompatible second template parameter for transfer.";
            NEON_THROW(exp);
        }
    }
    dst->template updateFrom<Neon::run_et::et::async>(streamSet, *src);
}

template <typename T, int C>
auto eField<T, C>::updateCompute(const Neon::set::StreamSet& streamSet)
    -> void
{
    if (mCpu.devType() == Neon::DeviceType::NONE && mGpu.devType() == Neon::DeviceType::NONE) {
        NeonException exp("eField_t");
        exp << "Invalid operation on a not fully initialized mirror.";
        NEON_THROW(exp);
    }

    if (m_dataConfig.dataUse() == Neon::DataUse::IO_COMPUTE) {
        if (m_dataConfig.backendConfig().devType() == Neon::DeviceType::CUDA) {
            self().update(streamSet, Neon::DeviceType::CUDA);
            return;
        }
    }
}

template <typename T, int C>
auto eField<T, C>::updateIO(int streamSetId)
    -> void
{
    const Neon::Backend& backendConfig = self().getGrid().getBackend();
    if (mCpu.devType() == Neon::DeviceType::NONE && mGpu.devType() == Neon::DeviceType::NONE) {
        NeonException exp("eField_t");
        exp << "Invalid operation on a not fully initialized mirror.";
        NEON_THROW(exp);
    }
    if (m_dataConfig.dataUse() == Neon::DataUse::IO_COMPUTE) {
        if (m_dataConfig.backendConfig().devType() == Neon::DeviceType::CUDA) {
            self().helpUpdate(backendConfig.streamSet(streamSetId), Neon::DeviceType::CPU);
            return;
        }
    }
}

template <typename T, int C>
auto eField<T, C>::updateCompute(int streamSetId)
    -> void
{
    const Neon::Backend& backendConfig = self().getGrid().getBackend();

    if (mCpu.devType() == Neon::DeviceType::NONE && mGpu.devType() == Neon::DeviceType::NONE) {
        NeonException exp("eField_t");
        exp << "Invalid operation on a not fully initialized mirror.";
        NEON_THROW(exp);
    }

    if (m_dataConfig.dataUse() == Neon::DataUse::IO_COMPUTE) {
        if (m_dataConfig.backendConfig().devType() == Neon::DeviceType::CUDA) {
            self().helpUpdate(backendConfig.streamSet(streamSetId), Neon::DeviceType::CUDA);
            return;
        }
    }
}

template <typename T, int C>
auto eField<T, C>::updateIO(const Neon::set::StreamSet& streamSet)
    -> void
{
    if (mCpu.devType() == Neon::DeviceType::NONE && mGpu.devType() == Neon::DeviceType::NONE) {
        NeonException exp("eField_t");
        exp << "Invalid operation on a not fully initialized mirror.";
        NEON_THROW(exp);
    }
    if (m_dataConfig.dataUse() == Neon::DataUse::IO_COMPUTE) {
        if (m_dataConfig.backendConfig().devType() == Neon::DeviceType::CUDA) {
            self().update(streamSet, Neon::DeviceType::CPU);
            return;
        }
    }
}

template <typename T, int C>
auto eField<T, C>::getPartition(const Neon::DeviceType& devType,
                                Neon::SetIdx            idx,
                                const Neon::DataView&   dataView)
    const
    -> const Partition&
{
    switch (devType) {
        case Neon::DeviceType::CPU:
        case Neon::DeviceType::OMP: {
            return mCpu.getPartition(devType, idx, dataView);
        }
        case Neon::DeviceType::CUDA: {
            return mGpu.getPartition(devType, idx, dataView);
        }
        default: {
            NeonException exp("eField_t");
            exp << "Incompatible device parameter.";
            NEON_THROW(exp);
        }
    }
}

template <typename T, int C>
auto eField<T, C>::getPartition(const Neon::DeviceType& devType,
                                Neon::SetIdx            idx,
                                const Neon::DataView&   dataView)
    -> Partition&
{
    switch (devType) {
        case Neon::DeviceType::CPU:
        case Neon::DeviceType::OMP: {
            return mCpu.getPartition(devType, idx, dataView);
        }
        case Neon::DeviceType::CUDA: {
            return mGpu.getPartition(devType, idx, dataView);
        }
        default: {
            NeonException exp("eField_t");
            exp << "Incompatible device parameter.";
            NEON_THROW(exp);
        }
    }
}

template <typename T, int C>
auto eField<T, C>::getPartition([[maybe_unused]] Neon::Execution,
                                [[maybe_unused]] Neon::SetIdx,
                                [[maybe_unused]] const Neon::DataView& dataView)
    const
    -> const Partition&
{
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename T, int C>
auto eField<T, C>::getPartition([[maybe_unused]] Neon::Execution,
                                [[maybe_unused]] Neon::SetIdx          idx,
                                [[maybe_unused]] const Neon::DataView& dataView)
    -> Partition&
{
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename T, int C>
auto eField<T, C>::haloUpdate(Neon::set::HuOptions& opt) const
    -> void
{
    auto field = [this](const Neon::DeviceType& devType) {
        switch (devType) {
            case Neon::DeviceType::CPU:
            case Neon::DeviceType::OMP: {
                return mCpu;
            }
            case Neon::DeviceType::CUDA: {
                return mGpu;
            }
            default: {
                NeonException exp("eField_t");
                exp << "Incompatible device parameter.";
                NEON_THROW(exp);
            }
        }
    };
    auto& bk = self().getBackend();
    auto  fieldDev = field(bk.devType());
    fieldDev.haloUpdate__(bk, opt);
}

template <typename T, int C>
auto eField<T, C>::haloUpdate(Neon::SetIdx          setIdx,
                              Neon::set::HuOptions& opt)
    -> void
{
    auto field = [this](const Neon::DeviceType& devType) {
        switch (devType) {
            case Neon::DeviceType::CPU:
            case Neon::DeviceType::OMP: {
                return mCpu;
            }
            case Neon::DeviceType::CUDA: {
                return mGpu;
            }
            default: {
                NeonException exp("eField_t");
                exp << "Incompatible device parameter.";
                NEON_THROW(exp);
            }
        }
    };
    auto& bk = self().getBackend();
    auto  fieldDev = field(bk.devType());
    fieldDev.haloUpdate__(bk, setIdx, opt);
}

template <typename T, int C>
auto eField<T, C>::haloUpdate(Neon::set::HuOptions& opt)
    -> void
{
    auto field = [this](const Neon::DeviceType& devType) {
        switch (devType) {
            case Neon::DeviceType::CPU:
            case Neon::DeviceType::OMP: {
                return mCpu;
            }
            case Neon::DeviceType::CUDA: {
                return mGpu;
            }
            default: {
                NeonException exp("eField_t");
                exp << "Incompatible device parameter.";
                NEON_THROW(exp);
            }
        }
    };
    auto& bk = self().getBackend();
    auto  fieldDev = field(bk.devType());
    fieldDev.haloUpdate__(bk, opt);
}


template <typename T, int C>
auto eField<T, C>::
    haloUpdateContainer(Neon::set::TransferMode    transferMode,
                        Neon::set::StencilSemantic stencilSemantic)
        const -> Neon::set::Container
{
    Neon::set::Container dataTransferContainer =
        Neon::set::Container::factoryDataTransfer(*this,
                                                  transferMode,
                                                  stencilSemantic);

    Neon::set::Container SyncContainer =
        Neon::set::Container::factorySynchronization(*this,
                                                     Neon::set::SynchronizationContainerType::hostOmpBarrier);

    Neon::set::container::Graph graph(this->getBackend());
    const auto&                 dataTransferNode = graph.addNode(dataTransferContainer);
    const auto&                 syncNode = graph.addNode(SyncContainer);

    switch (transferMode) {
        case Neon::set::TransferMode::put:
            graph.addDependency(dataTransferNode, syncNode, Neon::GraphDependencyType::data);
            break;
        case Neon::set::TransferMode::get:
            graph.addDependency(syncNode, dataTransferNode, Neon::GraphDependencyType::data);
            break;
        default:
            NEON_THROW_UNSUPPORTED_OPTION();
            break;
    }

    graph.removeRedundantDependencies();

    Neon::set::Container output =
        Neon::set::Container::factoryGraph("dGrid-Halo-Update",
                                           graph,
                                           [](Neon::SetIdx, Neon::set::Loader&) {});
    return output;
}

template <typename T, int C>
auto eField<T, C>::swap(Field& A, Field& B) -> void
{
    Neon::domain::interface::FieldBaseTemplate<T, C, typename Self::Grid, typename Self::Partition, int>::swapUIDBeforeFullSwap(A, B);
    std::swap(A, B);
}


}  // namespace Neon::domain::internal::eGrid