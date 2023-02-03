#pragma once
#include "Neon/domain/internal/sGrid/sFieldStorage.h"

namespace Neon::domain::internal::sGrid {

template <typename OuterGridT, typename T, int C>
sFieldStorage<OuterGridT, T, C>::sFieldStorage()
{
}

template <typename OuterGridT, typename T, int C>
sFieldStorage<OuterGridT, T, C>::sFieldStorage(const Neon::domain::interface::GridBase& gb)
{
    for (const auto& exec : PlaceUtils::getAllOptions()) {
        for (const auto& dw : DataViewUtil::validOptions()) {
            partitions[Neon::PlaceUtils::toInt(exec)][Neon::DataViewUtil::toInt(dw)] = gb.getDevSet().template newDataSet<Partition>();
        }
    }
}

template <typename OuterGridT, typename T, int C>
auto sFieldStorage<OuterGridT, T, C>::getPartition(Neon::Place execution,
                                                   Neon::DataView  dataView,
                                                   Neon::SetIdx    setIdx)
    -> Partition&
{
    return partitions[Neon::PlaceUtils::toInt(execution)][Neon::DataViewUtil::toInt(dataView)][setIdx];
}

template <typename OuterGridT, typename T, int C>
auto sFieldStorage<OuterGridT, T, C>::getPartition(Neon::Place execution,
                                                   Neon::DataView  dataView,
                                                   Neon::SetIdx    setIdx) const -> const Partition&
{
    return partitions[Neon::PlaceUtils::toInt(execution)][Neon::DataViewUtil::toInt(dataView)][setIdx];
}


template <typename OuterGridT, typename T, int C>
auto sFieldStorage<OuterGridT, T, C>::getPartitionSet(Neon::Place execution, Neon::DataView dataView) -> Neon::set::DataSet<Partition>&
{
    return partitions[Neon::PlaceUtils::toInt(execution)][Neon::DataViewUtil::toInt(dataView)];
}

}  // namespace Neon::domain::internal::sGrid
