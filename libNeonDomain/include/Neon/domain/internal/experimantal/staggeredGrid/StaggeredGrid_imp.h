#pragma once

#include "Neon/domain/internal/experimantal/staggeredGrid/StaggeredGrid.h"


namespace Neon::domain::internal::experimental::staggeredGrid {

template <typename BuildingBlockGridT>
template <typename ActiveNodesLambda>
StaggeredGrid<BuildingBlockGridT>::StaggeredGrid(const Backend&                            backend,
                                                 const int32_3d&                           dimension,
                                                 ActiveNodesLambda                         nodeActiveLambda,
                                                 const std::vector<Neon::domain::Stencil>& optionalExtraStencil,
                                                 const Vec_3d<double>&                     spacingData,
                                                 const Vec_3d<double>&                     origin)
{
    mStorage = std::make_shared<Storage>();

    Neon::domain::Stencil feaVoxelFirstOrderStencil({
                                                        {-1, 0, 0},
                                                        {-1, -1, 0},
                                                        {0, -1, 0},
                                                        {0, 0, -1},
                                                        {-1, 0, -1},
                                                        {-1, -1, -1},
                                                        {0, -1, -1},
                                                        /*----*/
                                                        {0, 0, 0},
                                                        /*----*/
                                                        {1, 0, 0},
                                                        {1, 1, 0},
                                                        {0, 1, 0},
                                                        {0, 0, 1},
                                                        {1, 0, 1},
                                                        {1, 1, 1},
                                                        {0, 1, 1},
                                                    },
                                                    false);
    if (!optionalExtraStencil.empty()) {
        NEON_DEV_UNDER_CONSTRUCTION("");
    }

    mStorage->buildingBlockGrid = typename BuildingBlocks::Grid(backend,
                                                                dimension,
                                                                nodeActiveLambda,
                                                                feaVoxelFirstOrderStencil,
                                                                spacingData,
                                                                origin);

    mStorage->nodeGrid = Self::NodeGrid(mStorage->buildingBlockGrid);

    Self::GridBase::init(std::string("FeaNodeGird-") + mStorage->buildingBlockGrid.getImplementationName(),
                         backend,
                         dimension,
                         feaVoxelFirstOrderStencil,
                         mStorage->buildingBlockGrid.getNumActiveCellsPerPartition(),
                         mStorage->buildingBlockGrid.getDefaultBlock(),
                         spacingData,
                         origin);
}

template <typename BuildingBlockGridT>
template <typename T, int C>
auto StaggeredGrid<BuildingBlockGridT>::newNodeField(const std::string&  fieldUserName,
                                                     int                 cardinality,
                                                     T                   inactiveValue,
                                                     Neon::DataUse       dataUse,
                                                     Neon::MemoryOptions memoryOptions) const -> NodeField<T, C>
{

    auto output = NodeField<T, C>(fieldUserName,
                                  dataUse,
                                  memoryOptions,
                                  *this,
                                  mStorage->buildingBlockGrid,
                                  cardinality,
                                  inactiveValue,
                                  Neon::domain::haloStatus_et::e::ON);
    return output;
}

template <typename BuildingBlockGridT>
auto StaggeredGrid<BuildingBlockGridT>::isInsideDomain(const index_3d& idx) const -> bool
{
    return mStorage->buildingBlockGrid.isInsideDomain(idx);
}

template <typename BuildingBlockGridT>
StaggeredGrid<BuildingBlockGridT>::StaggeredGrid()
{
    mStorage = std::make_shared<Storage>();
}

template <typename BuildingBlockGridT>
auto StaggeredGrid<BuildingBlockGridT>::getProperties(const index_3d& /*idx*/)
    const -> typename GridBaseTemplate::CellProperties
{
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename BuildingBlockGridT>
template <typename LoadingLambda>
auto StaggeredGrid<BuildingBlockGridT>::getContainerOnNodes(const std::string& name,
                                                            index_3d           blockSize,
                                                            size_t             sharedMem,
                                                            LoadingLambda      lambda) const -> Neon::set::Container
{
     Neon::set::Container output;
     output = mStorage->nodeGrid.getContainer(name, blockSize, sharedMem, lambda);
     return output;
}


}  // namespace Neon::domain::internal::experimental::staggeredGrid


#include "Neon/domain/internal/experimantal/staggeredGrid/NodeField_imp.h"
#include "Neon/domain/internal/experimantal/staggeredGrid/NodeGrid_imp.h"