#pragma once

namespace Neon::domain::details::mGrid {

template <typename T, int C, typename SBlock>
xField<T, C, SBlock>::xField(const std::string&         name,
                             const Grid&                grid,
                             int                        cardinality,
                             T                          outsideVal,
                             Neon::DataUse              dataUse,
                             const Neon::MemoryOptions& memoryOptions)
    : Neon::domain::interface::FieldBaseTemplate<T, C, Grid, Partition, int>(&grid,
                                                                             name,
                                                                             "xbField",
                                                                             cardinality,
                                                                             outsideVal,
                                                                             dataUse,
                                                                             memoryOptions,
                                                                             Neon::domain::haloStatus_et::ON)
{
    mData = std::make_shared<Data>();
    mData->field = grid.newField(name, cardinality, outsideVal, dataUse, memoryOptions);
}


template <typename T, int C, typename SBlock>
auto xField<T, C, SBlock>::isInsideDomain(const Neon::index_3d& idx) const -> bool
{
    return this->mData->field.isInsideDomain(idx);
}

template <typename T, int C, typename SBlock>
auto xField<T, C, SBlock>::getReference(const Neon::index_3d& idx, const int& cardinality) -> T&
{
    return this->operator()(idx, cardinality);
}

template <typename T, int C, typename SBlock>
auto xField<T, C, SBlock>::operator()(const Neon::index_3d& idx, const int& cardinality) const -> T
{
    return mData->field.getReference(idx, cardinality);
}

template <typename T, int C, typename SBlock>
auto xField<T, C, SBlock>::operator()(const Neon::index_3d& idx,
                              const int&            cardinality) -> T&
{
    return mData->field.getReference(idx, cardinality);
}


template <typename T, int C, typename SBlock>
auto xField<T, C, SBlock>::getPartition(const Neon::DeviceType& devType,
                                const Neon::SetIdx&     idx,
                                const Neon::DataView&   dataView) const -> const Partition&
{
    if (devType == Neon::DeviceType::CUDA) {
        return mData->mPartitions[PartitionBackend::gpu][Neon::DataViewUtil::toInt(dataView)][idx];
    } else {
        return mData->mPartitions[PartitionBackend::cpu][Neon::DataViewUtil::toInt(dataView)][idx];
    }
}

template <typename T, int C, typename SBlock>
auto xField<T, C, SBlock>::getPartition(const Neon::DeviceType& devType,
                                const Neon::SetIdx&     idx,
                                const Neon::DataView&   dataView) -> Partition&
{
    if (devType == Neon::DeviceType::CUDA) {
        return mData->mPartitions[PartitionBackend::gpu][Neon::DataViewUtil::toInt(dataView)][idx];
    } else {
        return mData->mPartitions[PartitionBackend::cpu][Neon::DataViewUtil::toInt(dataView)][idx];
    }
}

template <typename T, int C, typename SBlock>
auto xField<T, C, SBlock>::getPartition(Neon::Execution       exec,
                                Neon::SetIdx          idx,
                                const Neon::DataView& dataView) const -> const Partition&
{

    if (exec == Neon::Execution::host) {
        return getPartition(Neon::DeviceType::CPU, idx, dataView);
    } else {
        if (mData->field.getBackend().runtime() == Neon::Runtime::openmp) {
            return getPartition(Neon::DeviceType::CPU, idx, dataView);
        } else {
            return getPartition(Neon::DeviceType::CUDA, idx, dataView);
        }
    }

    NEON_THROW_UNSUPPORTED_OPERATION("xField::getPartition() unsupported Execution");
}


template <typename T, int C, typename SBlock>
auto xField<T, C, SBlock>::getPartition(Neon::Execution       exec,
                                Neon::SetIdx          idx,
                                const Neon::DataView& dataView) -> Partition&
{
    if (exec == Neon::Execution::host) {
        return getPartition(Neon::DeviceType::CPU, idx, dataView);
    } else {
        if (mData->field.getBackend().runtime() == Neon::Runtime::openmp) {
            return getPartition(Neon::DeviceType::CPU, idx, dataView);
        } else {
            return getPartition(Neon::DeviceType::CUDA, idx, dataView);
        }
    }

    NEON_THROW_UNSUPPORTED_OPERATION("xField::getPartition() unsupported Execution");
}

template <typename T, int C, typename SBlock>
auto xField<T, C, SBlock>::updateHostData(int streamId) -> void
{
    mData->field.updateHostData(streamId);
}

template <typename T, int C, typename SBlock>
auto xField<T, C, SBlock>::updateDeviceData(int streamId) -> void
{
    mData->field.updateDeviceData(streamId);
}


}  // namespace Neon::domain::details::mGrid