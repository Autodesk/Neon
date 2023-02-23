#pragma once
#include "Neon/set/Backend.h"
#include "Neon/set/DataSet.h"
namespace Neon::domain::tool {

/**
 * A helper class to storage and access partitions parametrically w.r.t Neon::DataView and Neon::Executions
 */
template <typename Partition, typename UserData = int>
struct PartitionTable
{
    PartitionTable() = default;

    explicit PartitionTable(Neon::Backend& bk);

    auto getPartition(Neon::Execution execution,
                      Neon::SetIdx    setIdx,
                      Neon::DataView  dw)
        -> Partition&;

    auto getPartition(Neon::Execution execution,
                      Neon::SetIdx    setIdx,
                      Neon::DataView  dw)
        const -> const Partition&;

    auto getUserData(Neon::Execution execution,
                     Neon::SetIdx    setIdx,
                     Neon::DataView  dw)
        -> UserData&;

    auto getUserData(Neon::Execution execution,
                     Neon::SetIdx    setIdx,
                     Neon::DataView  dw)
        const -> const UserData&;

    template <class Lambda>
    auto forEachConfiguration(Lambda const& lambda)
        -> void;

    template <class Lambda>
    auto forEachConfigurationWithUserData(Lambda const& lambda)
        -> void;

   private:
    using PartitionsByDevice = Neon::set::DataSet<Partition>;
    using PartitionByDeviceByDataView = std::array<PartitionsByDevice, Neon::DataViewUtil::nConfig>;
    using PartitionByDeviceByDataViewByExecution = std::array<PartitionByDeviceByDataView, Neon::ExecutionUtils::numConfigurations>;

    using UserDataByDevice = Neon::set::DataSet<UserData>;
    using UserDataByDeviceByDataView = std::array<UserDataByDevice, Neon::DataViewUtil::nConfig>;
    using UserDataByDeviceByDataViewByExecution = std::array<UserDataByDeviceByDataView, Neon::ExecutionUtils::numConfigurations>;

    PartitionByDeviceByDataViewByExecution mPartitions;
    UserDataByDeviceByDataViewByExecution  mUserData;

    int mSetSize = 0;
};

}  // namespace Neon::domain::tool

#include "Neon/domain/tools/PartitionTable_imp.h"