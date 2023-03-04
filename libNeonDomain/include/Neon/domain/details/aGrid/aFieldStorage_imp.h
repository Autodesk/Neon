#pragma once
#include "aFieldStorage.h"

namespace Neon::domain::internal::aGrid {

template <typename T, int C>
Storage<T, C>::Storage()
{
}

template <typename T, int C>
auto Storage<T, C>::getPartition(Neon::Execution execution,
                                 Neon::DataView  dataView,
                                 Neon::SetIdx    setIdx) -> Partition&
{
    return partitions[Neon::ExecutionUtils::toInt(execution)][Neon::DataViewUtil::toInt(dataView)][setIdx];
}

template <typename T, int C>
auto Storage<T, C>::getPartition(Neon::Execution execution,
                                 Neon::DataView  dataView,
                                 Neon::SetIdx    setIdx) const -> const Partition&
{
    return partitions[Neon::ExecutionUtils::toInt(execution)][Neon::DataViewUtil::toInt(dataView)][setIdx];
}


template <typename T, int C>
auto Storage<T, C>::getPartitionSet(Neon::Execution execution, Neon::DataView dataView) -> Neon::set::DataSet<Partition>&
{
    return partitions[Neon::ExecutionUtils::toInt(execution)][Neon::DataViewUtil::toInt(dataView)];
}

}  // namespace Neon::domain::arrayNew::internal
