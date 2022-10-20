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

#include "NodeField.h"
#include "NodeGeneric.h"
#include "NodePartition.h"

#include "NodeGrid.h"


namespace Neon::domain::internal::experimental::staggeredGrid::details {

template <typename BuildingBlockGridT>
NodeGrid<BuildingBlockGridT>::NodeGrid(typename BuildingBlocks::Grid& buildingBlockGrid)
{
    const Neon::Backend& bk = buildingBlockGrid.getBackend();
    auto&                dev = bk.devSet();
    mStorage = std::make_shared<Storage>();
    mStorage->buildingBlockGrid = buildingBlockGrid;

    for (auto dw : Neon::DataViewUtil::validOptions()) {
        const auto dwIdx = Neon::DataViewUtil::toInt(dw);
        mStorage->partitionIndexSpace[dwIdx] = dev.newDataSet<PartitionIndexSpace>();
        for (auto setId : dev.getRange()) {
            mStorage->partitionIndexSpace[dwIdx][setId.idx()] =
                PartitionIndexSpace(buildingBlockGrid.getPartitionIndexSpace(Neon::DeviceType::NONE, setId, dw));
        }
    }

    Self::GridBase::init(std::string("NodeGird-") + mStorage->buildingBlockGrid.getImplementationName(),
                         bk,
                         buildingBlockGrid.getDimension(),
                         buildingBlockGrid.getStencil(),
                         buildingBlockGrid.getNumActiveCellsPerPartition(),
                         buildingBlockGrid.getDefaultBlock(),
                         buildingBlockGrid.getSpacing(),
                         buildingBlockGrid.getOrigin());
}

template <typename BuildingBlockGridT>
auto NodeGrid<BuildingBlockGridT>::getPartitionIndexSpace(Neon::DeviceType,
                                                          SetIdx         setIdx,
                                                          Neon::DataView dataView)
    -> const PartitionIndexSpace&
{
    const auto dwIdx = Neon::DataViewUtil::toInt(dataView);
    return mStorage->partitionIndexSpace[dwIdx][setIdx.idx()];
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
                                                                 dataUse,
                                                                 memoryOptions,
                                                                 *this,
                                                                 mStorage->buildingBlockGrid,
                                                                 cardinality,
                                                                 inactiveValue,
                                                                 Neon::domain::haloStatus_et::ON);

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
auto NodeGrid<BuildingBlockGridT>::dot(const std::string&         name,
                                       NodeGrid::NodeField<T, C>& input1,
                                       NodeGrid::NodeField<T, C>& input2,
                                       PatternScalar<T>&          scalar)
    const -> Neon::set::Container
{
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename BuildingBlockGridT>
template <typename T, int C>
auto NodeGrid<BuildingBlockGridT>::norm2(const std::string&         name,
                                         NodeGrid::NodeField<T, C>& input,
                                         PatternScalar<T>&          scalar)
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
    Neon::set::Container kContainer = Neon::set::Container::factory(name,
                                                                    Neon::set::internal::ContainerAPI::DataViewSupport::on,
                                                                    *this,
                                                                    lambda,
                                                                    blockSize,
                                                                    [&](const Neon::index_3d&) { return size_t(sharedMem); });
    return kContainer;
}

template <typename BuildingBlockGridT>
template <typename LoadingLambda>
auto NodeGrid<BuildingBlockGridT>::getContainerOnNodes(const std::string& name, LoadingLambda lambda)
    const -> Neon::set::Container
{
    const Neon::index_3d& defaultBlockSize = this->getDefaultBlock();
    Neon::set::Container  kContainer = Neon::set::Container::factory(name,
                                                                     Neon::set::internal::ContainerAPI::DataViewSupport::on,
                                                                     *this,
                                                                     lambda,
                                                                     defaultBlockSize,
                                                                     [](const Neon::index_3d&) { return size_t(0); });
    return kContainer;
}

template <typename BuildingBlockGridT>
auto NodeGrid<BuildingBlockGridT>::getBuildingBlockGrid()
    -> typename BuildingBlocks::Grid&
{
    return mStorage->buildingBlockGrid;
}

template <typename BuildingBlockGridT>
auto NodeGrid<BuildingBlockGridT>::getNodeToVoxelMaskField() -> typename BuildingBlocks::template Field<NodeToVoxelMask, 1>&
{
    return mStorage->nodeToVoxelMaskField;
}


}  // namespace Neon::domain::internal::experimental::staggeredGrid::details
