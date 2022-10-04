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

#include "Neon/domain/internal/experimantal/staggeredGrid/NodeField.h"
#include "Neon/domain/internal/experimantal/staggeredGrid/NodeGeneric.h"
#include "Neon/domain/internal/experimantal/staggeredGrid/NodePartition.h"

#include "Neon/domain/internal/experimantal/staggeredGrid/NodeGrid.h"


namespace Neon::domain::internal::experimental::staggeredGrid::details {

template <typename BuildingBlockGridT>
NodeGrid<BuildingBlockGridT>::NodeGrid(typename BuildingBlocks::Grid& buildingBlockGrid)
{
    mStorage = std::make_shared<Storage>();
    mStorage->buildingBlockGrid = buildingBlockGrid;
}

template <typename BuildingBlockGridT>
auto NodeGrid<BuildingBlockGridT>::getPartitionIndexSpace(Neon::DeviceType devE,
                                                             SetIdx           setIdx,
                                                             Neon::DataView   dataView)
    -> const PartitionIndexSpace&
{
    return mStorage->buildingBlockGrid.getPartitionIndexSpace(devE, setIdx, dataView);
}

template <typename BuildingBlockGridT>
auto NodeGrid<BuildingBlockGridT>::getLaunchParameters(Neon::DataView  dataView,
                                                          const index_3d& blockSize,
                                                          const size_t&   shareMem)
    const -> Neon::set::LaunchParameters
{
    return mStorage->buildingBlockGrid.getLaunchParameters(dataView, blockSize, shareMem);
}

template <typename BuildingBlockGridT>
NodeGrid<BuildingBlockGridT>::NodeGrid()
{
    mStorage = std::make_shared<Storage>();
}

template <typename BuildingBlockGridT>
auto NodeGrid<BuildingBlockGridT>::isInsideDomain(const index_3d& idx)
    const -> bool
{
    bool output = mStorage->buildingBlockGrid.isInsideDomain(idx);
    return output;
}

template <typename BuildingBlockGridT>
template <typename T, int C>
auto NodeGrid<BuildingBlockGridT>::newNodeField(const std::string   fieldUserName,
                                                   int                 cardinality,
                                                   T                   inactiveValue,
                                                   Neon::DataUse       dataUse,
                                                   Neon::MemoryOptions memoryOptions)
    const -> NodeGrid::NodeField<T, C>
{
    NodeGrid::NodeField<T, C> output = NodeGrid::NodeField<T, C>(fieldUserName,
                                                                       cardinality,
                                                                       inactiveValue,
                                                                       dataUse,
                                                                       memoryOptions);

    return output;
}

template <typename BuildingBlockGridT>
auto NodeGrid<BuildingBlockGridT>::setReduceEngine(Neon::sys::patterns::Engine eng)
    -> void
{
    mStorage->buildingBlockGrid.setReduceEngine(eng);
}

template <typename BuildingBlockGridT>
template <typename T>
auto NodeGrid<BuildingBlockGridT>::newPatternScalar() const -> PatternScalar<T>
{
    return mStorage->buildingBlockGrid.newPatternScalar();
}
template <typename BuildingBlockGridT>
template <typename T, int C>
auto NodeGrid<BuildingBlockGridT>::dot(const std::string&            name,
                                          NodeGrid::NodeField<T, C>& input1,
                                          NodeGrid::NodeField<T, C>& input2,
                                          PatternScalar<T>&             scalar)
    const -> Neon::set::Container
{
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename BuildingBlockGridT>
template <typename T, int C>
auto NodeGrid<BuildingBlockGridT>::norm2(const std::string&            name,
                                            NodeGrid::NodeField<T, C>& input,
                                            PatternScalar<T>&             scalar)
    const -> Neon::set::Container
{
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename BuildingBlockGridT>
auto NodeGrid<BuildingBlockGridT>::getKernelConfig(int            streamIdx,
                                                      Neon::DataView dataView)
    -> Neon::set::KernelConfig
{
    return mStorage->buildingBlockGrid.getKernelConfig(streamIdx, dataView);
}

template <typename BuildingBlockGridT>
auto NodeGrid<BuildingBlockGridT>::isInsideNodeDomain(const index_3d& idx)
    const -> bool
{
    return mStorage->buildingBlockGrid.isInsideNodeDomain(idx);
}

template <typename BuildingBlockGridT>
auto NodeGrid<BuildingBlockGridT>::flattenedLengthSet(Neon::DataView dataView)
    const -> const Neon::set::DataSet<size_t>
{
    return mStorage->buildingBlockGrid.flattenedLengthSet(dataView);
}

template <typename BuildingBlockGridT>
auto NodeGrid<BuildingBlockGridT>::getLaunchInfo(Neon::DataView dataView)
    const -> Neon::set::LaunchParameters
{
    return mStorage->buildingBlockGrid.getLaunchInfo(dataView);
}

template <typename BuildingBlockGridT>
auto NodeGrid<BuildingBlockGridT>::flattenedPartitions(Neon::DataView dataView)
    const -> const Neon::set::DataSet<size_t>
{
    return mStorage->buildingBlockGrid.flattenedPartitions(dataView);
}

template <typename BuildingBlockGridT>
auto NodeGrid<BuildingBlockGridT>::stencil()
    const -> const Neon::domain::Stencil&
{
    return mStorage->buildingBlockGrid.stencil();
}

template <typename BuildingBlockGridT>
auto NodeGrid<BuildingBlockGridT>::newGpuLaunchParameters()
    const -> Neon::set::LaunchParameters
{
    return mStorage->buildingBlockGrid.newGpuLaunchParameters();
}

template <typename BuildingBlockGridT>
auto NodeGrid<BuildingBlockGridT>::getProperties(const index_3d& idx)
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
auto NodeGrid<BuildingBlockGridT>::setKernelConfig(KernelConfig& gridKernelConfig)
    const -> void
{
    return mStorage->buildingBlockGrid.setKernelConfig(gridKernelConfig);
}

template <typename BuildingBlockGridT>
template <typename LoadingLambda>
auto NodeGrid<BuildingBlockGridT>::getContainerOnNodes(const std::string& name,
                                                   index_3d           blockSize,
                                                   size_t             sharedMem,
                                                   LoadingLambda      lambda)
    const -> Neon::set::Container
{
mStorage->buildingBlockGrid.getContainer(name, blockSize, sharedMem, lambda);
}

template <typename BuildingBlockGridT>
template <typename LoadingLambda>
auto NodeGrid<BuildingBlockGridT>::getContainerOnNodes(const std::string& name, LoadingLambda lambda)
    const -> Neon::set::Container
{
    mStorage->buildingBlockGrid.getContainer(name, lambda);
}

template <typename BuildingBlockGridT>
auto NodeGrid<BuildingBlockGridT>::getBuildingBlockGrid()
    -> typename BuildingBlocks::Grid&
{
    return mStorage->buildingBlockGrid;
}


}  // namespace Neon::domain::internal::experimental::staggeredGrid
