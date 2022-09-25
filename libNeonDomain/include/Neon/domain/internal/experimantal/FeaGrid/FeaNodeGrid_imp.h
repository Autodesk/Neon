#pragma once
#include <assert.h>

#include "Neon/core/core.h"
#include "Neon/core/types/DataUse.h"
#include "Neon/core/types/Macros.h"

#include "Neon/set/BlockConfig.h"
#include "Neon/set/Containter.h"
#include "Neon/set/DataConfig.h"
#include "Neon/set/DevSet.h"
#include "Neon/set/MemoryOptions.h"

#include "Neon/sys/memory/MemDevice.h"

#include "Neon/domain/interface/GridBaseTemplate.h"
#include "Neon/domain/interface/KernelConfig.h"
#include "Neon/domain/interface/LaunchConfig.h"
#include "Neon/domain/interface/Stencil.h"
#include "Neon/domain/interface/common.h"
#include "Neon/domain/patterns/PatternScalar.h"

#include "Neon/domain/internal/experimantal/FeaGrid/FeaNode.h"
#include "Neon/domain/internal/experimantal/FeaGrid/FeaNodeField.h"
#include "Neon/domain/internal/experimantal/FeaGrid/FeaNodePartition.h"

#include "Neon/domain/internal/experimantal/FeaGrid/FeaNodeGrid.h"


namespace Neon::domain::internal::experimental::FeaVoxelGrid {

template <typename BuildingBlockGridT>
FeaNodeGrid<BuildingBlockGridT>::FeaNodeGrid(typename BuildingBlocks::Grid& buildingBlockGrid)
{
    mStorage->buildingBlockGrid = buildingBlockGrid;
}

template <typename BuildingBlockGridT>
auto FeaNodeGrid<BuildingBlockGridT>::getPartitionIndexSpace(Neon::DeviceType devE,
                                                             SetIdx           setIdx,
                                                             Neon::DataView   dataView)
    -> const PartitionIndexSpace&
{
    return mStorage->buildingBlockGrid.getPartitionIndexSpace(devE, setIdx, dataView);
}

template <typename BuildingBlockGridT>
auto FeaNodeGrid<BuildingBlockGridT>::getLaunchParameters(Neon::DataView  dataView,
                                                          const index_3d& blockSize,
                                                          const size_t&   shareMem)
    const -> Neon::set::LaunchParameters
{
    return mStorage->buildingBlockGrid.getLaunchParameters(dataView, blockSize, shareMem);
}

template <typename BuildingBlockGridT>
FeaNodeGrid<BuildingBlockGridT>::FeaNodeGrid()
{
    mStorage = std::make_shared<Storage>();
}

template <typename BuildingBlockGridT>
auto FeaNodeGrid<BuildingBlockGridT>::isInsideDomain(const index_3d& idx)
    const -> bool
{
    bool output = mStorage->buildingBlockGrid.isInsideDomain(idx);
    return output;
}

template <typename BuildingBlockGridT>
template <typename T, int C>
auto FeaNodeGrid<BuildingBlockGridT>::newNodeField(const std::string   fieldUserName,
                                                   int                 cardinality,
                                                   T                   inactiveValue,
                                                   Neon::DataUse       dataUse,
                                                   Neon::MemoryOptions memoryOptions)
    const -> FeaNodeGrid::NodeField<T, C>
{
    FeaNodeGrid::NodeField<T, C> output = FeaNodeGrid::NodeField<T, C>(fieldUserName,
                                                                       cardinality,
                                                                       inactiveValue,
                                                                       dataUse,
                                                                       memoryOptions);

    return output;
}

template <typename BuildingBlockGridT>
auto FeaNodeGrid<BuildingBlockGridT>::setReduceEngine(Neon::sys::patterns::Engine eng)
    -> void
{
    mStorage->buildingBlockGrid.setReduceEngine(eng);
}

template <typename BuildingBlockGridT>
template <typename T>
auto FeaNodeGrid<BuildingBlockGridT>::newPatternScalar() const -> PatternScalar<T>
{
    return mStorage->buildingBlockGrid.newPatternScalar();
}
template <typename BuildingBlockGridT>
template <typename T, int C>
auto FeaNodeGrid<BuildingBlockGridT>::dot(const std::string&            name,
                                          FeaNodeGrid::NodeField<T, C>& input1,
                                          FeaNodeGrid::NodeField<T, C>& input2,
                                          PatternScalar<T>&             scalar)
    const -> Neon::set::Container
{
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename BuildingBlockGridT>
template <typename T, int C>
auto FeaNodeGrid<BuildingBlockGridT>::norm2(const std::string&            name,
                                            FeaNodeGrid::NodeField<T, C>& input,
                                            PatternScalar<T>&             scalar)
    const -> Neon::set::Container
{
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename BuildingBlockGridT>
auto FeaNodeGrid<BuildingBlockGridT>::getKernelConfig(int            streamIdx,
                                                      Neon::DataView dataView)
    -> Neon::set::KernelConfig
{
    return mStorage->buildingBlockGrid.getKernelConfig(streamIdx, dataView);
}

template <typename BuildingBlockGridT>
auto FeaNodeGrid<BuildingBlockGridT>::isInsideNodeDomain(const index_3d& idx)
    const -> bool
{
    return mStorage->buildingBlockGrid.isInsideNodeDomain(idx);
}

template <typename BuildingBlockGridT>
auto FeaNodeGrid<BuildingBlockGridT>::flattenedLengthSet(Neon::DataView dataView)
    const -> const Neon::set::DataSet<size_t>
{
    return mStorage->buildingBlockGrid.flattenedLengthSet(dataView);
}

template <typename BuildingBlockGridT>
auto FeaNodeGrid<BuildingBlockGridT>::getLaunchInfo(Neon::DataView dataView)
    const -> Neon::set::LaunchParameters
{
    return mStorage->buildingBlockGrid.getLaunchInfo(dataView);
}

template <typename BuildingBlockGridT>
auto FeaNodeGrid<BuildingBlockGridT>::flattenedPartitions(Neon::DataView dataView)
    const -> const Neon::set::DataSet<size_t>
{
    return mStorage->buildingBlockGrid.flattenedPartitions(dataView);
}

template <typename BuildingBlockGridT>
auto FeaNodeGrid<BuildingBlockGridT>::stencil()
    const -> const Neon::domain::Stencil&
{
    return mStorage->buildingBlockGrid.stencil();
}

template <typename BuildingBlockGridT>
auto FeaNodeGrid<BuildingBlockGridT>::newGpuLaunchParameters()
    const -> Neon::set::LaunchParameters
{
    return mStorage->buildingBlockGrid.newGpuLaunchParameters();
}

template <typename BuildingBlockGridT>
auto FeaNodeGrid<BuildingBlockGridT>::getProperties(const index_3d& idx)
    const -> typename GridBaseTemplate::CellProperties
{
    auto                                      boudlingBlockProperties = mStorage->buildingBlockGrid.getProperties(idx);
    typename GridBaseTemplate::CellProperties output;
    output.init(boudlingBlockProperties.getSetIdx(),
                boudlingBlockProperties.getDataView(),
                boudlingBlockProperties.getOuterCell());
    return output;
}

template <typename BuildingBlockGridT>
auto FeaNodeGrid<BuildingBlockGridT>::setKernelConfig(KernelConfig& gridKernelConfig)
    const -> void
{
    return mStorage->buildingBlockGrid.setKernelConfig(gridKernelConfig);
}

template <typename BuildingBlockGridT>
template <typename LoadingLambda>
auto FeaNodeGrid<BuildingBlockGridT>::getContainer(const std::string& name,
                                                   index_3d           blockSize,
                                                   size_t             sharedMem,
                                                   LoadingLambda      lambda)
    const -> Neon::set::Container
{
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename BuildingBlockGridT>
template <typename LoadingLambda>
auto FeaNodeGrid<BuildingBlockGridT>::getContainer(const std::string& name, LoadingLambda lambda)
    const -> Neon::set::Container
{
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename BuildingBlockGridT>
auto FeaNodeGrid<BuildingBlockGridT>::getBuildingBlockGrid()
    -> typename BuildingBlocks::Grid&
{
    return mStorage->buildingBlockGrid;
}


}  // namespace Neon::domain::internal::experimental::FeaVoxelGrid
