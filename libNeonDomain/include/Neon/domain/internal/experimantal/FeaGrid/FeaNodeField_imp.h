#pragma once

#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"
#include "Neon/domain/interface/FieldBaseTemplate.h"
#include "Neon/set/DataConfig.h"
#include "Neon/set/DevSet.h"
#include "Neon/set/HuOptions.h"

#include "Neon/domain/internal/experimantal/FeaGrid/FeaNodePartition.h"

#include "Neon/domain/internal/experimantal/FeaGrid/FeaNodeField.h"

namespace Neon::domain::internal::experimental::FeaVoxelGrid {


template <typename BuildingBlockGridT, typename T, int C>
auto FeaNodeField<BuildingBlockGridT, T, C>::self() -> FeaNodeField::Self&
{
    return *this;
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaNodeField<BuildingBlockGridT, T, C>::self() const -> const FeaNodeField::Self&
{
    return *this;
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaNodeField<BuildingBlockGridT, T, C>::operator()(const index_3d& idx,
                                                        const int&      cardinality) const -> Type
{
    return this->getStorage().buildingBlockField.operator()(idx, cardinality);
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaNodeField<BuildingBlockGridT, T, C>::haloUpdate(set::HuOptions& opt) const -> void
{
    return this->getStorage().buildingBlockField.haloUpdate(opt);
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaNodeField<BuildingBlockGridT, T, C>::haloUpdate(SetIdx setIdx, set::HuOptions& opt) const -> void
{
    return this->getStorage().buildingBlockField.haloUpdate(setIdx, opt);
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaNodeField<BuildingBlockGridT, T, C>::haloUpdate(set::HuOptions& opt) -> void
{
    return this->getStorage().buildingBlockField.haloUpdate(opt);
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaNodeField<BuildingBlockGridT, T, C>::haloUpdate(SetIdx setIdx, set::HuOptions& opt) -> void
{
    return this->getStorage().buildingBlockField.haloUpdate(setIdx, opt);
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaNodeField<BuildingBlockGridT, T, C>::getReference(const index_3d& idx, const int& cardinality) -> Type&
{
    return this->getStorage().buildingBlockField.getReference(idx, cardinality);
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaNodeField<BuildingBlockGridT, T, C>::updateCompute(int streamSetId) -> void
{
    return this->getStorage().buildingBlockField.updateCompute(streamSetId);
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaNodeField<BuildingBlockGridT, T, C>::updateIO(int streamSetId) -> void
{
    return this->getStorage().buildingBlockField.updateIO(streamSetId);
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaNodeField<BuildingBlockGridT, T, C>::getPartition(const DeviceType& devType,
                                                          const SetIdx&     idx,
                                                          const DataView&   dataView) const -> const FeaNodeField::Partition&
{
    const Neon::Execution execution = DeviceTypeUtil::getExecution(devType);
    const auto&           partition = getPartition(execution, idx, dataView);
    return partition;
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaNodeField<BuildingBlockGridT, T, C>::getPartition(const DeviceType& devType,
                                                          const SetIdx&     idx,
                                                          const DataView&   dataView) -> FeaNodeField::Partition&
{
    const Neon::Execution execution = DeviceTypeUtil::getExecution(devType);
    auto&                 partition = getPartition(execution, idx, dataView);
    return partition;
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaNodeField<BuildingBlockGridT, T, C>::getPartition(Neon::Execution execution,
                                                          Neon::SetIdx    setIdx,
                                                          const DataView& dataView) const -> const FeaNodeField::Partition&
{
    auto& s = this->getStorage();
    if (!s.supportedExecutions[Neon::ExecutionUtils::toInt(execution)]) {
        std::stringstream message;
        message << "An execution of type " << execution << " is not supported by a " << s.dataUse;
        NEON_THROW_UNSUPPORTED_OPERATION(message.str())
    }

    const auto& partition = s.partitionsByViewAndDev[Neon::DataViewUtil::toInt(dataView)][setIdx.idx()];

    return partition;
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaNodeField<BuildingBlockGridT, T, C>::getPartition(Neon::Execution execution,
                                                          Neon::SetIdx    setIdx,
                                                          const DataView& dataView) -> FeaNodeField::Partition&
{
    auto& s = this->getStorage();
    if (!s.supportedExecutions[Neon::ExecutionUtils::toInt(execution)]) {
        std::stringstream message;
        message << "An execution of type " << execution << " is not supported by a " << s.dataUse;
        NEON_THROW_UNSUPPORTED_OPERATION(message.str())
    }

    auto& partition = s.partitionsByViewAndDev[Neon::DataViewUtil::toInt(dataView)][setIdx.idx()];

    return partition;
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaNodeField<BuildingBlockGridT, T, C>::swap(FeaNodeField& A, FeaNodeField& B) -> void
{
    (void)A;
    (void)B;
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename BuildingBlockGridT, typename T, int C>
FeaNodeField<BuildingBlockGridT, T, C>::FeaNodeField(const std::string&                   fieldUserName,
                                                     Neon::DataUse                        dataUse,
                                                     const Neon::MemoryOptions&           memoryOptions,
                                                     const Grid&                          grid,
                                                     const typename BuildingBlocks::Grid& buildingBlockGrid,
                                                     int                                  cardinality,
                                                     T                                    outsideVal,
                                                     Neon::domain::haloStatus_et::e       haloStatus)
    : Neon::domain::interface::FieldBaseTemplate<T, C, typename Self::Grid, typename Self::Partition, Storage>(&grid,
                                                                                                               fieldUserName,
                                                                                                               std::string("Fea-") + buildingBlockGrid.getImplementationName(),
                                                                                                               cardinality,
                                                                                                               outsideVal,
                                                                                                               dataUse,
                                                                                                               memoryOptions,
                                                                                                               haloStatus) {
    this->getStorage().buildingBlockField = buildingBlockGrid.template newField<T, C>(fieldUserName,
                                                                                      cardinality,
                                                                                      outsideVal,
                                                                                      dataUse,
                                                                                      memoryOptions);
    const Neon::Backend& bk = grid.getBackend();
    auto&                s = this -> getStorage();

    {  // Setting up the mask for supported executions (i.e host and device | host only | device only)
        for (Neon::Execution execution : Neon::ExecutionUtils::getAllOptions()) {
            s.supportedExecutions[ExecutionUtils::toInt(execution)] = false;
        }
        for (Neon::Execution execution : Neon::ExecutionUtils::getCompatibleOptions(dataUse)) {
            s.supportedExecutions[ExecutionUtils::toInt(execution)] = true;
        }
    }


    {  // initializing partition data
        for (Neon::Execution execution : Neon::ExecutionUtils::getCompatibleOptions(dataUse)) {
            for (auto dw : Neon::DataViewUtil::validOptions()) {
                for (Neon::SetIdx setIdx : bk.devSet().getRange()) {
                    auto& buildingBlockPartition = s.buildingBlockField.getPartition(execution, setIdx, dw);
                    s.partitionsByViewAndDev[Neon::DataViewUtil::toInt(dw)][setIdx.idx()] =
                        Partition(buildingBlockPartition);
                }
            }
        }
    }
}

}  // namespace Neon::domain::internal::experimental::FeaVoxelGrid
