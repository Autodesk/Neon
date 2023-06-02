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
    mData = std::make_shared<Data>(bk);
    mData->foundationGrid = foundationGrid;
    GridTransformation::initSpan(mData->foundationGrid,
                                 NEON_OUT mData->spanTable);
    tGrid::GridBase::init("tGrid",
                          bk,
                          foundationGrid.getDimension(),
                          foundationGrid.getStencil(),
                          foundationGrid.getNumActiveCellsPerPartition(),
                          foundationGrid.getDefaultBlock(),
                          foundationGrid.getSpacing(),
                          foundationGrid.getOrigin());
}

template <typename GridTransformation>
tGrid<GridTransformation>::tGrid()
{
    mData = std::make_shared<Data>();
}

template <typename GridTransformation>
auto tGrid<GridTransformation>::
    getLaunchParameters(Neon::DataView        dataView,
                        const Neon::index_3d& blockSize,
                        const size_t&         shareMem) const
    -> Neon::set::LaunchParameters
{
    auto output = GridTransformation::initLaunchParameters(mData->foundationGrid,
                                                           dataView, blockSize, shareMem);
    return output;
}

template <typename GridTransformation>
auto tGrid<GridTransformation>::
    getSpan(Neon::Execution execution,
            SetIdx          setIdx,
            Neon::DataView  dataView)
        -> const Span&
{
    return mData->spanTable.getSpan(execution, setIdx, dataView);
}

template <typename GridTransformation>
auto tGrid<GridTransformation>::getSetIdx(const Neon::index_3d& idx) const
    -> int32_t
{
    return mData->foundationGrid.getSetIdx(idx);
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
    Neon::Backend& bk = mData->foundationGrid.getBackend();
    memoryOptions = bk.devSet().sanitizeMemoryOption(memoryOptions);

    if (C != 0 && cardinality != C) {
        NeonException exception("tGrid::newField Dynamic and static cardinality do not match.");
        NEON_THROW(exception);
    }

    typename FoundationGrid::template Field<T, C> fieldFoundation = mData->foundationGrid.template newField<T, C>(fieldUserName,
                                                                                                                  cardinality,
                                                                                                                  inactiveValue,
                                                                                                                  dataUse,
                                                                                                                  memoryOptions);
    Field<T, C>                                   field(fieldFoundation, *this);
    return field;
}

template <typename GridTransformation>
template <Neon::Execution execution,
          typename LoadingLambda>
auto tGrid<GridTransformation>::newContainer(const std::string& name,
                                             index_3d           blockSize,
                                             size_t             sharedMem,
                                             LoadingLambda      lambda) const
    -> Neon::set::Container
{
    const Neon::index_3d&                              defaultBlockSize = GridTransformation::getDefaultBlock(mData->foundationGrid);
    Neon::set::internal::ContainerAPI::DataViewSupport dataViewSupport =
        GridTransformation::dataViewSupport;

    Neon::set::Container kContainer = Neon::set::Container::factory<execution>(name,
                                                                               dataViewSupport,
                                                                               *this,
                                                                               lambda,
                                                                               blockSize,
                                                                               [sharedMem](const Neon::index_3d&) { return sharedMem; });
    return kContainer;
}

template <typename GridTransformation>
template <Neon::Execution execution,
          typename LoadingLambda>
auto tGrid<GridTransformation>::newContainer(const std::string& name,
                                             LoadingLambda      lambda)
    const
    -> Neon::set::Container
{
    const Neon::index_3d&                              defaultBlockSize = GridTransformation::getDefaultBlock(mData->foundationGrid);
    Neon::set::internal::ContainerAPI::DataViewSupport dataViewSupport = GridTransformation::dataViewSupport;
    Neon::set::Container                               kContainer = Neon::set::Container::factory<execution>(name,
                                                                               dataViewSupport,
                                                                               *this,
                                                                               lambda,
                                                                               defaultBlockSize,
                                                                               [](const Neon::index_3d&) { return 0; });
    return kContainer;
}

template <typename GridTransformation>
auto tGrid<GridTransformation>::isInsideDomain(const Neon::index_3d& idx) const
    -> bool
{
    return mData->foundationGrid.isInsideDomain(idx);
}

template <typename GridTransformation>
auto tGrid<GridTransformation>::getProperties(const Neon::index_3d& idx) const
    -> typename GridBaseTemplate::CellProperties
{
    return mData->foundationGrid.getProperties(idx);
}

template <typename GridTransformation>
tGrid<GridTransformation>::~tGrid()
{
    mData = std::make_shared<Data>();
}

template <typename GridTransformation>
tGrid<GridTransformation>::tGrid(const tGrid& other)
{
    mData = other.mData;
    tGrid::GridBase::operator=(other);
}

template <typename GridTransformation>
tGrid<GridTransformation>::tGrid(tGrid&& other) noexcept
{
    mData = std::move(other.mData);
    tGrid::GridBase::operator=(other);
}

template <typename GridTransformation>
auto tGrid<GridTransformation>::operator=(const tGrid& other)
    -> tGrid&
{
    mData = other.mData;
    tGrid::GridBase::operator=(other);
    return *this;
}

template <typename GridTransformation>
auto tGrid<GridTransformation>::operator=(tGrid&& other) noexcept
    -> tGrid&
{
    mData = std::move(other.mData);
    tGrid::GridBase::operator=(other);
    return *this;
}
}  // namespace Neon::domain::tool::details