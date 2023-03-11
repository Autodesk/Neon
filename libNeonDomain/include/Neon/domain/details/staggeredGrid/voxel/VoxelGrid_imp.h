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

#include "VoxelField.h"
#include "VoxelGeneric.h"
#include "VoxelPartition.h"

#include "VoxelGrid.h"


namespace Neon::domain::details::experimental::staggeredGrid::details {

template <typename BuildingBlockGridT>
VoxelGrid<BuildingBlockGridT>::
    VoxelGrid(typename BuildingBlocks::Grid&                                     buildingBlockGrid,
              const typename BuildingBlocks::template Field<uint8_t, 1>&         mask,
              const typename BuildingBlocks::template Field<NodeToVoxelMask, 1>& nodeToVoxelMaskField)
{
    const Neon::Backend& bk = buildingBlockGrid.getBackend();
    auto&                dev = bk.devSet();

    mStorage = std::make_shared<Storage>();
    mStorage->buildingBlockGrid = buildingBlockGrid;
    mStorage->mask = mask;
    mStorage->nodeToVoxelMaskField = nodeToVoxelMaskField;

    // Initializing the data set for all configurations for a PartitionIndexSpace
    for (auto devType : Neon::DeviceTypeUtil::getOptions()) {
        for (auto dw : Neon::DataViewUtil::validOptions()) {
            const auto dwIdx = Neon::DataViewUtil::toInt(dw);
            const auto devTypeIdx = Neon::DeviceTypeUtil::toInt(devType);

            mStorage->partitionIndexSpace[devTypeIdx][dwIdx] = dev.newDataSet<PartitionIndexSpace>();
        }
    }

    const std::vector<DeviceType> targetDevType = [&] {
        std::vector<DeviceType> output;
        output.push_back(bk.devType());
        return output;
    }();

    for (auto devType : targetDevType) {
        for (auto dw : Neon::DataViewUtil::validOptions()) {
            const auto devTypeIdx = Neon::DeviceTypeUtil::toInt(devType);
            const auto dwIdx = Neon::DataViewUtil::toInt(dw);
            for (auto setId : dev.getRange()) {
                mStorage->partitionIndexSpace[devTypeIdx][dwIdx][setId.idx()] =
                    PartitionIndexSpace(buildingBlockGrid.getPartitionIndexSpace(devType, setId, dw),
                                        mStorage->mask.getPartition(devType, setId, dw));
            }
        }
    }
    {
        auto spacing = buildingBlockGrid.getSpacing();
        auto origin = buildingBlockGrid.getOrigin();  // + (spacing / 2.0);
        auto dim = buildingBlockGrid.getDimension() - 1;

        Self::GridBase::init(std::string("NodeGird-") + mStorage->buildingBlockGrid.getImplementationName(),
                             bk,
                             dim,
                             buildingBlockGrid.getStencil(),
                             buildingBlockGrid.getNumActiveCellsPerPartition(),
                             buildingBlockGrid.getDefaultBlock(),
                             spacing,
                             origin);
    }
}

template <typename BuildingBlockGridT>
auto VoxelGrid<BuildingBlockGridT>::
    getPartitionIndexSpace(Neon::DeviceType deviceType,
                           SetIdx           setIdx,
                           Neon::DataView   dataView)
        -> const PartitionIndexSpace&
{
    const auto dwIdx = Neon::DataViewUtil::toInt(dataView);
    const auto devTypeIdx = Neon::DeviceTypeUtil::toInt(deviceType);

    return mStorage->partitionIndexSpace[devTypeIdx][dwIdx][setIdx.idx()];
}

template <typename BuildingBlockGridT>
auto VoxelGrid<BuildingBlockGridT>::
    getLaunchParameters(Neon::DataView  dataView,
                        const index_3d& blockSize,
                        const size_t&   shareMem)
        const -> Neon::set::LaunchParameters
{
    return mStorage->buildingBlockGrid.getLaunchParameters(dataView, blockSize, shareMem);
}

template <typename BuildingBlockGridT>
VoxelGrid<BuildingBlockGridT>::
    VoxelGrid()
{
    mStorage = std::make_shared<Storage>();
}

template <typename BuildingBlockGridT>
auto VoxelGrid<BuildingBlockGridT>::
    isInsideDomain(const index_3d& idx)
        const -> bool
{
    bool output = mStorage->buildingBlockGrid.isInsideDomain(idx);
    return output;
}

template <typename BuildingBlockGridT>
template <typename T, int C>
auto VoxelGrid<BuildingBlockGridT>::
    newVoxelField(const std::string   fieldUserName,
                  int                 cardinality,
                  T                   inactiveValue,
                  Neon::DataUse       dataUse,
                  Neon::MemoryOptions memoryOptions)
        const -> VoxelField<T, C>
{
    VoxelField<T, C> output = VoxelField<T, C>(fieldUserName,
                                               dataUse,
                                               memoryOptions,
                                               *this,
                                               mStorage->nodeToVoxelMaskField,
                                               mStorage->buildingBlockGrid,
                                               cardinality,
                                               inactiveValue,
                                               Neon::domain::haloStatus_et::ON);

    return output;
}

template <typename BuildingBlockGridT>
auto VoxelGrid<BuildingBlockGridT>::
    setReduceEngine(Neon::sys::patterns::Engine eng)
        -> void
{
    mStorage->buildingBlockGrid.setReduceEngine(eng);
}

template <typename BuildingBlockGridT>
template <typename T>
auto VoxelGrid<BuildingBlockGridT>::
    newPatternScalar() const -> PatternScalar<T>
{
    return mStorage->buildingBlockGrid.newPatternScalar();
}
template <typename BuildingBlockGridT>
template <typename T, int C>
auto VoxelGrid<BuildingBlockGridT>::dot(const std::string& name,
                                        VoxelField<T, C>&  input1,
                                        VoxelField<T, C>&  input2,
                                        PatternScalar<T>&  scalar)
    const -> Neon::set::Container
{
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename BuildingBlockGridT>
template <typename T, int C>
auto VoxelGrid<BuildingBlockGridT>::
    norm2(const std::string& name,
          VoxelField<T, C>&  input,
          PatternScalar<T>&  scalar)
        const -> Neon::set::Container
{
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename BuildingBlockGridT>
auto VoxelGrid<BuildingBlockGridT>::
    getKernelConfig(int            streamIdx,
                    Neon::DataView dataView)
        -> Neon::set::KernelConfig
{
    return mStorage->buildingBlockGrid.getKernelConfig(streamIdx, dataView);
}

template <typename BuildingBlockGridT>
auto VoxelGrid<BuildingBlockGridT>::
    isInsideNodeDomain(const index_3d& idx)
        const -> bool
{
    return mStorage->buildingBlockGrid.isInsideNodeDomain(idx);
}

template <typename BuildingBlockGridT>
auto VoxelGrid<BuildingBlockGridT>::
    flattenedLengthSet(Neon::DataView dataView)
        const -> const Neon::set::DataSet<size_t>
{
    return mStorage->buildingBlockGrid.getNumActiveCellsPerPartition(dataView);
}

template <typename BuildingBlockGridT>
auto VoxelGrid<BuildingBlockGridT>::
    getLaunchInfo(Neon::DataView dataView)
        const -> Neon::set::LaunchParameters
{
    return mStorage->buildingBlockGrid.getLaunchInfo(dataView);
}

template <typename BuildingBlockGridT>
auto VoxelGrid<BuildingBlockGridT>::
    flattenedPartitions(Neon::DataView dataView)
        const -> const Neon::set::DataSet<size_t>
{
    return mStorage->buildingBlockGrid.flattenedPartitions(dataView);
}

template <typename BuildingBlockGridT>
auto VoxelGrid<BuildingBlockGridT>::
    stencil()
        const -> const Neon::domain::Stencil&
{
    return mStorage->buildingBlockGrid.stencil();
}

template <typename BuildingBlockGridT>
auto VoxelGrid<BuildingBlockGridT>::
    newGpuLaunchParameters()
        const -> Neon::set::LaunchParameters
{
    return mStorage->buildingBlockGrid.newGpuLaunchParameters();
}

template <typename BuildingBlockGridT>
auto VoxelGrid<BuildingBlockGridT>::
    getProperties(const index_3d& idx)
        const -> typename Self::GridBaseTemplate::CellProperties
{
    auto boudlingBlockProperties = mStorage->buildingBlockGrid.getProperties(idx);

    typename GridBaseTemplate::CellProperties output;
    output.init(boudlingBlockProperties.getSetIdx(),
                boudlingBlockProperties.getDataView(),
                boudlingBlockProperties.getOuterCell());

    if (boudlingBlockProperties.isInside() && mStorage->mask(idx, 0) == 1) {
        output.setIsInside(true);
    } else {
        output.setIsInside(false);
    }
    return output;
}

template <typename BuildingBlockGridT>
auto VoxelGrid<BuildingBlockGridT>::
    setKernelConfig(KernelConfig& gridKernelConfig)
        const -> void
{
    return mStorage->buildingBlockGrid.setKernelConfig(gridKernelConfig);
}

template <typename BuildingBlockGridT>
template <typename LoadingLambda>
auto VoxelGrid<BuildingBlockGridT>::
    getContainerOnVoxels(const std::string& name,
                         index_3d           blockSize,
                         size_t             sharedMem,
                         LoadingLambda      lambda)
        const -> Neon::set::Container
{

    const Neon::index_3d& defaultBlockSize = this->getDefaultBlock();
    Neon::set::Container  kContainer = Neon::set::Container::factory(name,
                                                                     Neon::set::internal::ContainerAPI::DataViewSupport::on,
                                                                     *this,
                                                                     lambda,
                                                                     blockSize,
                                                                     [&](const Neon::index_3d&) { return size_t(sharedMem); });
    return kContainer;
}

template <typename BuildingBlockGridT>
template <typename LoadingLambda>
auto VoxelGrid<BuildingBlockGridT>::
    getContainerOnVoxels(const std::string& name,
                         LoadingLambda      lambda)
        const -> Neon::set::Container
{
    Neon::domain::KernelConfig kernelConfig(0);

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
auto VoxelGrid<BuildingBlockGridT>::getBuildingBlockGrid()
    -> typename BuildingBlocks::Grid&
{
    return mStorage->buildingBlockGrid;
}


}  // namespace Neon::domain::details::experimental::staggeredGrid::details
