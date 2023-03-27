#pragma once

#include "Neon/set/Containter.h"

#include "Neon/domain/interface/GridBaseTemplate.h"
#include "Neon/domain/interface/KernelConfig.h"
#include "Neon/domain/interface/LaunchConfig.h"
#include "Neon/domain/interface/Stencil.h"
#include "Neon/domain/interface/common.h"
#include "Neon/domain/tools/gridTransformer/tGrid.h"

namespace Neon::domain::tool::details {


template <typename GridTransformation>
tGrid<GridTransformation>::tGrid(FoundationGrid& foundationGrid)
{
    Neon::Backend& bk = foundationGrid.getBackend();
    mStorage = std::make_shared<Storage>(bk);
    mStorage.foundationGrid = foundationGrid;

    GridTransformation::initPartitionIndexSpace(mStorage->foundationGrid,
                                                NEON_OUT mStorage->indexSpace);
}

template <typename GridTransformation>
tGrid<GridTransformation>::tGrid()
{
    mStorage = std::make_shared<Storage>();
}

template <typename GridTransformation>
template <typename ActiveCellLambda>
tGrid<GridTransformation>::
    tGrid(const Neon::Backend&         backend,
          const Neon::int32_3d&        dimension,
          const ActiveCellLambda&      activeCellLambda,
          const Neon::domain::Stencil& stencil,
          const Vec_3d<double>&        spacingData,
          const Vec_3d<double>&        origin)
{
    mStorage = std::make_shared<Storage>(backend);
    mStorage->foundationGrid = FoundationGrid(backend,
                                              dimension,
                                              activeCellLambda,
                                              stencil,
                                              spacingData,
                                              origin);
    GridTransformation::initPartitionIndexSpace(mStorage->foundationGrid,
                                                NEON_OUT mStorage->indexSpace);
}

template <typename GridTransformation>
auto tGrid<GridTransformation>::
    getLaunchParameters(Neon::DataView        dataView,
                        const Neon::index_3d& blockSize,
                        const size_t&         shareMem) const
    -> Neon::set::LaunchParameters
{
    auto output = GridTransformation::getLaunchParameters(mStorage->foundationGrid,
                                                          dataView, blockSize, shareMem);
    return output;
}

template <typename GridTransformation>
auto tGrid<GridTransformation>::getPartitionIndexSpace(Neon::DeviceType devE,
                                                       SetIdx           setIdx,
                                                       Neon::DataView   dataView)
    -> const PartitionIndexSpace&
{
    return mStorage->indexSpace[DataViewUtil::toInt(dataView)].local(devE, setIdx);
}

template <typename GridTransformation>
template <typename T, int C>
auto tGrid<GridTransformation>::newField(const std::string&  fieldUserName,
                                         int                 cardinality,
                                         T                   inactiveValue,
                                         Neon::DataUse       dataUse,
                                         Neon::MemoryOptions memoryOptions) const
    -> Field<T, C>
{
    Neon::Backend& bk = mStorage->foundationGrid.getBackend();
    memoryOptions = bk.devSet().sanitizeMemoryOption(memoryOptions);

    const auto haloStatus = Neon::domain::haloStatus_et::ON;

    if (C != 0 && cardinality != C) {
        NeonException exception("tGrid::newField Dynamic and static cardinality do not match.");
        NEON_THROW(exception);
    }

    Field<T, C> field(fieldUserName,
                      dataUse,
                      memoryOptions,
                      *this,
                      cardinality);

    return field;
}

template <typename GridTransformation>
template <typename T, int C>
auto tGrid<GridTransformation>::newField(typename FoundationGrid::template Field<T, C>& foundationField)
    const -> tGrid::Field<T, C>
{
    Neon::Backend& bk = mStorage->foundationGrid.getBackend();

    Field<T, C> field(foundationField);

    return field;
}

template <typename GridTransformation>
template <typename LoadingLambda>
auto tGrid<GridTransformation>::getContainer(const std::string& name,
                                             index_3d           blockSize,
                                             size_t             sharedMem,
                                             LoadingLambda      lambda) const
    -> Neon::set::Container
{
    const Neon::index_3d&                              defaultBlockSize = GridTransformation::getDefaultBlock(mStorage->foundationGrid);
    Neon::set::internal::ContainerAPI::DataViewSupport dataViewSupport =
        GridTransformation::dataViewSupport;

    Neon::set::Container kContainer = Neon::set::Container::factory(name,
                                                                    dataViewSupport,
                                                                    *this,
                                                                    lambda,
                                                                    blockSize,
                                                                    [sharedMem](const Neon::index_3d&) { return sharedMem; });
    return kContainer;
}

template <typename GridTransformation>
template <typename LoadingLambda>
auto tGrid<GridTransformation>::getContainer(const std::string& name,
                                             LoadingLambda      lambda)
    const
    -> Neon::set::Container
{
    const Neon::index_3d&                              defaultBlockSize = GridTransformation::getDefaultBlock(mStorage->foundationGrid);
    Neon::set::internal::ContainerAPI::DataViewSupport dataViewSupport =
        GridTransformation::dataViewSupport;

    Neon::set::Container kContainer = Neon::set::Container::factory(name,
                                                                    dataViewSupport,
                                                                    *this,
                                                                    lambda,
                                                                    defaultBlockSize,
                                                                    [](const Neon::index_3d&) { return size_t(0); });
    return kContainer;
}

template <typename GridTransformation>
auto tGrid<GridTransformation>::isInsideDomain(const Neon::index_3d& idx) const
    -> bool
{
    return mStorage->foundationGrid.isInsideDomain();
}

template <typename GridTransformation>
auto tGrid<GridTransformation>::getProperties(const Neon::index_3d& idx) const
    -> typename GridBaseTemplate::CellProperties
{
    return mStorage->foundationGrid.getProperties();
}

template <typename GridTransformation>
tGrid<GridTransformation>::~tGrid()
{
    mStorage = std::make_shared<Storage>();
}

template <typename GridTransformation>
tGrid<GridTransformation>::tGrid(const tGrid& other)
{
    mStorage = other.mStorage;
}

template <typename GridTransformation>
tGrid<GridTransformation>::tGrid(tGrid&& other) noexcept
{
    mStorage = std::move(other.mStorage);
}

template <typename GridTransformation>
auto tGrid<GridTransformation>::operator=(const tGrid& other)
    -> tGrid&
{
    mStorage = other.mStorage;
}

template <typename GridTransformation>
auto tGrid<GridTransformation>::operator=(tGrid&& other) noexcept
    -> tGrid&
{
    mStorage = std::move(other.mStorage);
}
}  // namespace Neon::domain::tool::details