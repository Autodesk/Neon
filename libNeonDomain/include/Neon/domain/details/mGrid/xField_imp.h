#pragma once

#include "Neon/domain/details/mGrid/xField.h"

namespace Neon::domain::details::mGrid {

template <typename T, int C>
xField<T, C>::xField(const std::string&         name,
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


template <typename T, int C>
auto xField<T, C>::isInsideDomain(const Neon::index_3d& idx) const -> bool
{
    return this->mData->field.isInsideDomain(idx);
}

template <typename T, int C>
auto xField<T, C>::getReference(const Neon::index_3d& idx, const int& cardinality) -> T&
{
    return this->operator()(idx, cardinality);
}


template <typename T, int C>
auto xField<T, C>::haloUpdate(Neon::set::HuOptions& opt) const -> void
{
    mData->field.haloUpdate(opt);
}

template <typename T, int C>
auto xField<T, C>::haloUpdate(Neon::set::HuOptions& opt) -> void
{
    mData->field.haloUpdate(opt);
}

template <typename T, int C>
auto xField<T, C>::operator()(const Neon::index_3d& idx, const int& cardinality) const -> T
{
    return mData->field.getReference(idx, cardinality);
}

template <typename T, int C>
auto xField<T, C>::operator()(const Neon::index_3d& idx,
                              const int&            cardinality) -> T&
{
    return mData->field.getReference(idx, cardinality);
}


template <typename T, int C>
auto xField<T, C>::getPartition(const Neon::DeviceType& devType,
                                const Neon::SetIdx&     idx,
                                const Neon::DataView&   dataView) const -> const Partition&
{
    if (devType == Neon::DeviceType::CUDA) {
        return mData->mPartitions[PartitionBackend::gpu][Neon::DataViewUtil::toInt(dataView)][idx];
    } else {
        return mData->mPartitions[PartitionBackend::cpu][Neon::DataViewUtil::toInt(dataView)][idx];
    }
}

template <typename T, int C>
auto xField<T, C>::getPartition(const Neon::DeviceType& devType,
                                const Neon::SetIdx&     idx,
                                const Neon::DataView&   dataView) -> Partition&
{
    if (devType == Neon::DeviceType::CUDA) {
        return mData->mPartitions[PartitionBackend::gpu][Neon::DataViewUtil::toInt(dataView)][idx];
    } else {
        return mData->mPartitions[PartitionBackend::cpu][Neon::DataViewUtil::toInt(dataView)][idx];
    }
}

template <typename T, int C>
auto xField<T, C>::getPartition(Neon::Execution       exec,
                                Neon::SetIdx          idx,
                                const Neon::DataView& dataView) const -> const Partition&
{

    if (exec == Neon::Execution::device) {
        return getPartition(Neon::DeviceType::CUDA, idx, dataView);
    }
    if (exec == Neon::Execution::host) {
        return getPartition(Neon::DeviceType::CPU, idx, dataView);
    }

    NEON_THROW_UNSUPPORTED_OPERATION("xField::getPartition() unsupported Execution");
}


template <typename T, int C>
auto xField<T, C>::getPartition(Neon::Execution       exec,
                                Neon::SetIdx          idx,
                                const Neon::DataView& dataView) -> Partition&
{
    if (exec == Neon::Execution::device) {
        return getPartition(Neon::DeviceType::CUDA, idx, dataView);
    }
    if (exec == Neon::Execution::host) {
        return getPartition(Neon::DeviceType::CPU, idx, dataView);
    }

    NEON_THROW_UNSUPPORTED_OPERATION("xField::getPartition() unsupported Execution");
}

template <typename T, int C>
auto xField<T, C>::updateIO(int streamId) -> void
{
    mData->field.updateIO(streamId);
}

template <typename T, int C>
auto xField<T, C>::updateCompute(int streamId) -> void
{
    mData->field.updateCompute(streamId);
}


}  // namespace Neon::domain::details::mGrid