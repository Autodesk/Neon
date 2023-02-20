#pragma once
#include "dField.h"

namespace Neon::domain::internal::exp::dGrid {

template <typename T, int C>
dField<T, C>::dField()
{
    mData = std::make_shared<Self::Data>();
}

template <typename T, int C>
dField<T, C>::dField(const std::string&                        fieldUserName,
                     Neon::DataUse                             dataUse,
                     const Neon::MemoryOptions&                memoryOptions,
                     const Grid&                               grid,
                     const Neon::set::DataSet<Neon::index_3d>& dims,
                     int                                       zHaloDim,
                     Neon::domain::haloStatus_et::e            haloStatus,
                     int                                       cardinality,
                     const Neon::set::MemSet_t<Neon::int8_3d>& stencilIdTo3dOffset)
    : Neon::domain::interface::FieldBaseTemplate<T, C, Grid, Partition, int>(&grid,
                                                                             fieldUserName,
                                                                             "dField",
                                                                             cardinality,
                                                                             T(0),
                                                                             dataUse,
                                                                             memoryOptions,
                                                                             haloStatus) {

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

    mData = std::make_shared<Self::Data>();
    mData->dataUse = dataUse;
    mData->memoryOptions = memoryOptions;
    mData->cardinality = cardinality;
    mData->memoryOptions = memoryOptions;
    mData->grid = std::make_shared<Grid>(grid);
    mData->haloStatus = (mData->grid->getDevSet().setCardinality() == 1) ? haloStatus_et::e::OFF : haloStatus;
    const int haloRadius = mData->haloStatus == Neon::domain::haloStatus_et::ON ? mData->zHaloDim : 0;
    mData->zHaloDim = zHaloDim;

    Neon::set::DataSet<index_3d> origins = this->getGrid().getBackend().template newDataSet<index_3d>({0, 0, 0});
    {  // Computing origins
        origins.forEachSeq(
            [&](Neon::SetIdx setIdx, Neon::index_3d& val) {
                if (setIdx == 0) {
                    val.z = 0;
                    return;
                }
                const Neon::SetIdx proceedingIdx = setIdx - 1;
                val.z = origins[proceedingIdx].z + dims[proceedingIdx].z;
            });
    }

    {  // Computing Pitch
        mData.pitch.forEachSeq(
            [&](Neon::SetIdx setIdx, Neon::size_4d& pitch) {
                switch (mData->memoryOptions.getOrder()) {
                    case MemoryLayout::structOfArrays: {
                        pitch.x = 1;
                        pitch.y = pitch.x * dims[setIdx.idx()].x;
                        pitch.z = pitch.y * dims[setIdx.idx()].y;
                        pitch.w = pitch.z * (dims[setIdx.idx()].z + 2 * haloRadius);
                        break;
                    }
                    case MemoryLayout::arrayOfStructs: {
                        pitch.x = mData->cardinality;
                        pitch.y = pitch.x * dims[setIdx.idx()].x;
                        pitch.z = pitch.y * dims[setIdx.idx()].y;
                        pitch.w = 1;
                        break;
                    }
                }
            });
    }

    {  // Setting up partitions
        Neon::domain::aGrid& aGrid = mData.grid.getMemoryGrid();
        mData.memoryField = aGrid.newField(fieldUserName + "-storage", cardinality, T(), dataUse, memoryOptions);
        const int setCardinality = mData.grid.devSet.setCardinality();
        mData.partitionTable.forEachConfiguration(
            [&](Neon::Execution           execution,
                Neon::SetIdx              setIdx,
                Neon::DataView            dw,
                typename Self::Partition& partition, typename Data::PartitionUserData& userData) {
                auto memoryFieldPartition = mData->memoryField.getPartition(execution, setIdx, dw);

                partition = dPartition<T, C>(dw,
                                             memoryFieldPartition.mem(),
                                             dims[setIdx],
                                             haloRadius,
                                             mData->zHaloDim,
                                             mData->pitch[setIdx],
                                             setIdx.idx(),
                                             origins[setIdx],
                                             mData->cardinality,
                                             mData->grid->getDimension(),
                                             stencilIdTo3dOffset.rawMem(execution, setIdx));
            });
    }

}


template <typename T, int C>
auto dField<T, C>::updateCompute(int streamSetId)
    -> void
{
mData.memoryField.updateCompute(streamSetId);
}

template <typename T, int C>
auto dField<T, C>::updateIO(int streamSetId)
    -> void
{
mData.memoryField.updateIO(streamSetId);

}

template <typename T, int C>
auto dField<T, C>::getLaunchInfo(const Neon::DataView dataView) const
    -> Neon::set::LaunchParameters
{
    return m_gpu.getLaunchInfo(dataView);
}


template <typename T, int C>
auto dField<T, C>::getPartition(const Neon::DeviceType& devType,
                                const Neon::SetIdx&     idx,
                                const Neon::DataView&   dataView) const
    -> const Partition&
{
    mData.partitionTable.getPartition()
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
auto dField<T, C>::getPartition(Neon::Execution       execution,
                                Neon::SetIdx          setIdx,
                                const Neon::DataView& dataView)
    const
    -> const Partition&
{
    const Neon::DataUse dataUse = this->getDataUse();
    bool                isOk = Neon::ExecutionUtils::checkCompatibility(dataUse, execution);
    if (isOk) {
        mData.partitionTable.getPartition(execution, setIdx, dataView);
    }
    std::stringstream message;
    message << "The requested execution mode ( " << execution << " ) is not compatible with the field DataUse (" << dataUse << ")";
    NEON_THROW_UNSUPPORTED_OPERATION(message.str());
}

template <typename T, int C>
auto dField<T, C>::getPartition(Neon::Execution       execution,
                                Neon::SetIdx          setIdx,
                                const Neon::DataView& dataView)
    -> Partition&
{
    const auto dataUse = this->getDataUse();
    bool       isOk = Neon::ExecutionUtils::checkCompatibility(dataUse, execution);
    if (isOk) {
        mData.partitionTable.getPartition(execution, setIdx, dataView);
    }
    std::stringstream message;
    message << "The requested execution mode ( " << execution << " ) is not compatible with the field DataUse (" << dataUse << ")";
    NEON_THROW_UNSUPPORTED_OPERATION(message.str());
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
    NEON_TRACE("haloUpdate stream {} transferMode {} ", opt.streamSetIdx(), Neon::set::TransferModeUtils::toString(opt.transferMode()));
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
auto dField<T, C>::hostHaloUpdate() const
    -> void
{
    Neon::set::HuOptions opt(Neon::set::TransferMode::get, false);
    NEON_TRACE("haloUpdate stream {} transferMode {} ", opt.streamSetIdx(), Neon::set::TransferModeUtils::toString(opt.transferMode()));
    auto& bk = self().getBackend();
    auto  fieldDev = field(Neon::DeviceType::CPU);

    fieldDev.template haloUpdate<Neon::set::TransferMode::get>(bk, -1, opt.startWithBarrier(), opt.streamSetIdx());
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
            NEON_TRACE("TRACE haloUpdate PUT setIdx {} stream {} transferMode {} ", setIdx.idx(), opt.streamSetIdx(), Neon::set::TransferModeUtils::toString(opt.transferMode()));

            fieldDev.template haloUpdate<Neon::set::TransferMode::put>(setIdx, bk, -1, opt.startWithBarrier(), opt.streamSetIdx());
            break;
        case Neon::set::TransferMode::get:
            NEON_TRACE("TRACE haloUpdate GET setIdx {} stream {} transferMode {} ", setIdx.idx(), opt.streamSetIdx(), Neon::set::TransferModeUtils::toString(opt.transferMode()));

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
    NEON_TRACE("haloUpdate stream {} transferMode {} ", opt.streamSetIdx(), Neon::set::TransferModeUtils::toString(opt.transferMode()));

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
#pragma omp critical
        {
            NEON_TRACE("TRACE haloUpdate PUT setIdx {} stream {} transferMode {} ", setIdx.idx(), opt.streamSetIdx(), Neon::set::TransferModeUtils::toString(opt.transferMode()));
        }
            fieldDev.template haloUpdate<Neon::set::TransferMode::put>(setIdx, bk, -1, opt.startWithBarrier(), opt.streamSetIdx());
            break;
        case Neon::set::TransferMode::get:
#pragma omp critical
        {
            NEON_TRACE("TRACE haloUpdate GET setIdx {} stream {} transferMode {} ", setIdx.idx(), opt.streamSetIdx(), Neon::set::TransferModeUtils::toString(opt.transferMode()));
        }
            fieldDev.template haloUpdate<Neon::set::TransferMode::get>(setIdx, bk, -1, opt.startWithBarrier(), opt.streamSetIdx());
            break;
        default:
            NEON_THROW_UNSUPPORTED_OPTION();
            break;
    }
}


template <typename T, int C>
auto dField<T, C>::
    haloUpdateContainer(Neon::set::TransferMode    transferMode,
                        Neon::set::StencilSemantic stencilSemantic)
        const -> Neon::set::Container
{
    // We need to define a graph of Containers
    // One for the actual memory transfer
    // One for the synchronization
    // The order depends on the transfer mode: put or get
    Neon::set::Container dataTransferContainer =
        Neon::set::Container::factoryDataTransfer(
            *this,
            transferMode,
            stencilSemantic);

    Neon::set::Container SyncContainer =
        Neon::set::Container::factorySynchronization(
            *this,
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

template <typename T, int C>
auto dField<T, C>::getData() -> typename Self::Data&
{
    return std::ref(mData);
}


}  // namespace Neon::domain::internal::exp::dGrid
