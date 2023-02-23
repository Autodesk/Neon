#pragma once

#include "Neon/domain/tools/PartitionTable.h"
#include "Neon/set/Backend.h"

namespace Neon::domain::tool {

template <typename Partition,
          typename UserData>
PartitionTable<Partition, UserData>::PartitionTable(Neon::Backend& bk)
{  // Setting up the mask for supported executions (i.e host and device | host only | device only)
    for (Neon::Execution execution : Neon::ExecutionUtils::getAllOptions()) {
        for (auto dw : Neon::DataViewUtil::validOptions()) {
            mPartitions[Neon::ExecutionUtils::toInt(execution)]
                       [Neon::DataViewUtil::toInt(dw)] =
                           bk.devSet().template newDataSet<Partition>();
        }
    }
    for (Neon::Execution execution : Neon::ExecutionUtils::getAllOptions()) {
        for (auto dw : Neon::DataViewUtil::validOptions()) {
            mUserData[Neon::ExecutionUtils::toInt(execution)]
                     [Neon::DataViewUtil::toInt(dw)] =
                         bk.devSet().template newDataSet<UserData>();
        }
    }
    mSetSize = bk.getDeviceCount();
}

template <typename Partition,
          typename UserData>
auto PartitionTable<Partition, UserData>::
    getPartition(Neon::Execution execution,
                 Neon::SetIdx    setIdx,
                 Neon::DataView  dw)
        -> Partition&
{
    int const dwInt = Neon::DataViewUtil::toInt(dw);
    int const executionInt = Neon::ExecutionUtils::toInt(execution);
    auto&     output = mPartitions[executionInt][dwInt][setIdx.idx()];
    return output;
}

template <typename Partition,
          typename UserData>
auto PartitionTable<Partition, UserData>::
    getPartition(Neon::Execution execution,
                 Neon::SetIdx    setIdx,
                 Neon::DataView  dw)
        const -> const Partition&
{
    int const dwInt = Neon::DataViewUtil::toInt(dw);
    int const executionInt = Neon::ExecutionUtils::toInt(execution);
    auto&     output = mPartitions[executionInt][dwInt][setIdx.idx()];
    return output;
}

template <typename Partition,
          typename UserData>
auto PartitionTable<Partition, UserData>::
    getUserData(Neon::Execution execution,
                Neon::SetIdx    setIdx,
                Neon::DataView  dw)
        const -> const UserData&
{
    int const dwInt = Neon::DataViewUtil::toInt(dw);
    int const executionInt = Neon::ExecutionUtils::toInt(execution);
    auto&     output = mUserData[executionInt][dwInt][setIdx.idx()];
    return output;
}

template <typename Partition,
          typename UserData>
auto PartitionTable<Partition, UserData>::
    getUserData(Neon::Execution execution,
                Neon::SetIdx    setIdx,
                Neon::DataView  dw)
        -> UserData&
{
    int const dwInt = Neon::DataViewUtil::toInt(dw);
    int const executionInt = Neon::ExecutionUtils::toInt(execution);
    auto&     output = mUserData[executionInt][dwInt][setIdx.idx()];
    return output;
}

template <typename Partition,
          typename UserData>
template <class Lambda>
auto PartitionTable<Partition, UserData>::
    forEachConfiguration(Lambda const& lambda)
        -> void
{
    for (auto execution : Neon::ExecutionUtils::getAllOptions()) {
        for (auto setIdx = 0; setIdx < mSetSize; setIdx++) {
            for (auto dw : Neon::DataViewUtil::validOptions()) {
                lambda(execution, setIdx, dw, getPartition(execution, setIdx, dw));
            }
        }
    }
}

template <typename Partition,
          typename UserData>
template <class Lambda>
auto PartitionTable<Partition, UserData>::
    forEachConfigurationWithUserData(Lambda const& lambda)
        -> void
{
    for (auto execution : Neon::ExecutionUtils::getAllOptions()) {
        for (auto setIdx = 0; setIdx < mSetSize; setIdx++) {
            for (auto dw : Neon::DataViewUtil::validOptions()) {
                lambda(execution, setIdx, dw,
                       getPartition(execution, setIdx, dw),
                       getUserData(execution, setIdx, dw));
            }
        }
    }
}

}  // namespace Neon::domain::tool