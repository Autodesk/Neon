#pragma once

#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"
#include "Neon/domain/interface/FieldBaseTemplate.h"
#include "Neon/set/DataConfig.h"
#include "Neon/set/DevSet.h"
#include "Neon/set/HuOptions.h"

#include "Neon/domain/internal/experimantal/staggeredGrid/VoxelField.h"

namespace Neon::domain::internal::experimental::staggeredGrid {


template <typename BuildingBlockGridT, typename T, int C>
auto VoxelField<BuildingBlockGridT, T, C>::self() -> VoxelField::Self&
{
    return *this;
}

template <typename BuildingBlockGridT, typename T, int C>
auto VoxelField<BuildingBlockGridT, T, C>::self() const -> const VoxelField::Self&
{
    return *this;
}

template <typename BuildingBlockGridT, typename T, int C>
auto VoxelField<BuildingBlockGridT, T, C>::operator()(const index_3d& idx,
                                                           const int&      cardinality) const -> Type
{
    (void)idx;
    (void)cardinality;
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename BuildingBlockGridT, typename T, int C>
auto VoxelField<BuildingBlockGridT, T, C>::haloUpdate(set::HuOptions& opt) const -> void
{
    return this->getStorage().buildingBlockField.haloUpdate(opt);
}

template <typename BuildingBlockGridT, typename T, int C>
auto VoxelField<BuildingBlockGridT, T, C>::haloUpdate(SetIdx setIdx, set::HuOptions& opt) const -> void
{
    return this->getStorage().buildingBlockField.haloUpdate(setIdx, opt);
}

template <typename BuildingBlockGridT, typename T, int C>
auto VoxelField<BuildingBlockGridT, T, C>::haloUpdate(set::HuOptions& opt) -> void
{
    return this->getStorage().buildingBlockField.haloUpdate(opt);
}

template <typename BuildingBlockGridT, typename T, int C>
auto VoxelField<BuildingBlockGridT, T, C>::haloUpdate(SetIdx setIdx, set::HuOptions& opt) -> void
{
    return this->getStorage().buildingBlockField.haloUpdate(setIdx, opt);
}

template <typename BuildingBlockGridT, typename T, int C>
auto VoxelField<BuildingBlockGridT, T, C>::getReference(const index_3d& idx, const int& cardinality) -> Type&
{
    return this->getStorage().buildingBlockField.getReference(idx, cardinality);
}

template <typename BuildingBlockGridT, typename T, int C>
auto VoxelField<BuildingBlockGridT, T, C>::updateCompute(int streamSetId) -> void
{
    return this->getStorage().buildingBlockField.updateCompute(streamSetId);
}

template <typename BuildingBlockGridT, typename T, int C>
auto VoxelField<BuildingBlockGridT, T, C>::updateIO(int streamSetId) -> void
{
    return this->getStorage().buildingBlockField.updateIO(streamSetId);
}

template <typename BuildingBlockGridT, typename T, int C>
auto VoxelField<BuildingBlockGridT, T, C>::getPartition(const DeviceType& devType,
                                                             const SetIdx&     idx,
                                                             const DataView&   dataView) const -> const VoxelField::Partition&
{
    (void)devType;
    (void)idx;
    (void)dataView;
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename BuildingBlockGridT, typename T, int C>
auto VoxelField<BuildingBlockGridT, T, C>::getPartition(const DeviceType& devType,
                                                             const SetIdx&     idx,
                                                             const DataView&   dataView) -> VoxelField::Partition&
{
    (void)devType;
    (void)idx;
    (void)dataView;

    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename BuildingBlockGridT, typename T, int C>
auto VoxelField<BuildingBlockGridT, T, C>::getPartition(Neon::Execution execution,
                                                             Neon::SetIdx    setIdx,
                                                             const DataView& dataView) const -> const VoxelField::Partition&
{
    (void)execution;
    (void)setIdx;
    (void)dataView;
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename BuildingBlockGridT, typename T, int C>
auto VoxelField<BuildingBlockGridT, T, C>::getPartition(Neon::Execution execution,
                                                             Neon::SetIdx    setIdx,
                                                             const DataView& dataView) -> VoxelField::Partition&
{
    (void)execution;
    (void)setIdx;
    (void)dataView;
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename BuildingBlockGridT, typename T, int C>
auto VoxelField<BuildingBlockGridT, T, C>::swap(VoxelField& A, VoxelField& B) -> void
{
    (void)A;
    (void)B;
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename BuildingBlockGridT, typename T, int C>
VoxelField<BuildingBlockGridT, T, C>::VoxelField(const std::string&                  fieldUserName,
                                                           Neon::DataUse                       dataUse,
                                                           const MemoryOptions&                memoryOptions,
                                                           const VoxelField::Grid&        grid,
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

}  // namespace Neon::domain::internal::experimental::staggeredGrid
