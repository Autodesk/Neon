#pragma once

#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"
#include "Neon/domain/interface/FieldBaseTemplate.h"
#include "Neon/set/DataConfig.h"
#include "Neon/set/DevSet.h"
#include "Neon/set/HuOptions.h"

#include "Neon/domain/internal/experimantal/FeaGrid/FeaElementField.h"

namespace Neon::domain::internal::experimental::FeaVoxelGrid {


template <typename BuildingBlockGridT, typename T, int C>
auto FeaElementField<BuildingBlockGridT, T, C>::self() -> FeaElementField::Self&
{
    return *this;
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaElementField<BuildingBlockGridT, T, C>::self() const -> const FeaElementField::Self&
{
    return *this;
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaElementField<BuildingBlockGridT, T, C>::operator()(const index_3d& idx,
                                                           const int&      cardinality) const -> Type
{
    (void)idx;
    (void)cardinality;
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaElementField<BuildingBlockGridT, T, C>::haloUpdate(set::HuOptions& opt) const -> void
{
    return this->getStorage().buildingBlockField.haloUpdate(opt);
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaElementField<BuildingBlockGridT, T, C>::haloUpdate(SetIdx setIdx, set::HuOptions& opt) const -> void
{
    return this->getStorage().buildingBlockField.haloUpdate(setIdx, opt);
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaElementField<BuildingBlockGridT, T, C>::haloUpdate(set::HuOptions& opt) -> void
{
    return this->getStorage().buildingBlockField.haloUpdate(opt);
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaElementField<BuildingBlockGridT, T, C>::haloUpdate(SetIdx setIdx, set::HuOptions& opt) -> void
{
    return this->getStorage().buildingBlockField.haloUpdate(setIdx, opt);
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaElementField<BuildingBlockGridT, T, C>::getReference(const index_3d& idx, const int& cardinality) -> Type&
{
    return this->getStorage().buildingBlockField.getReference(idx, cardinality);
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaElementField<BuildingBlockGridT, T, C>::updateCompute(int streamSetId) -> void
{
    return this->getStorage().buildingBlockField.updateCompute(streamSetId);
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaElementField<BuildingBlockGridT, T, C>::updateIO(int streamSetId) -> void
{
    return this->getStorage().buildingBlockField.updateIO(streamSetId);
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaElementField<BuildingBlockGridT, T, C>::getPartition(const DeviceType& devType,
                                                             const SetIdx&     idx,
                                                             const DataView&   dataView) const -> const FeaElementField::Partition&
{
    (void)devType;
    (void)idx;
    (void)dataView;
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaElementField<BuildingBlockGridT, T, C>::getPartition(const DeviceType& devType,
                                                             const SetIdx&     idx,
                                                             const DataView&   dataView) -> FeaElementField::Partition&
{
    (void)devType;
    (void)idx;
    (void)dataView;

    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaElementField<BuildingBlockGridT, T, C>::getPartition(Neon::Execution execution,
                                                             Neon::SetIdx    setIdx,
                                                             const DataView& dataView) const -> const FeaElementField::Partition&
{
    (void)execution;
    (void)setIdx;
    (void)dataView;
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaElementField<BuildingBlockGridT, T, C>::getPartition(Neon::Execution execution,
                                                             Neon::SetIdx    setIdx,
                                                             const DataView& dataView) -> FeaElementField::Partition&
{
    (void)execution;
    (void)setIdx;
    (void)dataView;
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaElementField<BuildingBlockGridT, T, C>::swap(FeaElementField& A, FeaElementField& B) -> void
{
    (void)A;
    (void)B;
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename BuildingBlockGridT, typename T, int C>
FeaElementField<BuildingBlockGridT, T, C>::FeaElementField(const std::string&                  fieldUserName,
                                                           Neon::DataUse                       dataUse,
                                                           const MemoryOptions&                memoryOptions,
                                                           const FeaElementField::Grid&        grid,
                                                           const set::DataSet<Neon::index_3d>& dims,
                                                           int                                 zHaloDim,
                                                           Neon::domain::haloStatus_et::e      haloStatus,
                                                           int                                 cardinality)
{
    (void)fieldUserName;
    (void)dataUse;
    (void)memoryOptions;
    (void)grid;
    (void)dims;
    (void)zHaloDim;
    (void)haloStatus;
    (void)cardinality;
    NEON_DEV_UNDER_CONSTRUCTION("");
}

}  // namespace Neon::domain::internal::experimental::FeaVoxelGrid
