#pragma once

#include "Neon/domain/internal/experimantal/FeaGrid/FeaVoxelGrid.h"


namespace Neon::domain::internal::experimental::FeaVoxelGrid {

template <typename BuildingBlockGridT>
template <typename ActiveNodesLambda>
FeaVoxelGrid<BuildingBlockGridT>::FeaVoxelGrid(const Backend&                            backend,
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

    mStorage->nodeGrid = FeaNodeGrid(mStorage->buildingBlockGrid);

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
auto FeaVoxelGrid<BuildingBlockGridT>::newNodeField(const std::string&  fieldUserName,
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
auto FeaVoxelGrid<BuildingBlockGridT>::isInsideDomain(const index_3d& idx) const -> bool
{
    return mStorage->buildingBlockGrid.isInsideDomain(idx);
}

template <typename BuildingBlockGridT>
FeaVoxelGrid<BuildingBlockGridT>::FeaVoxelGrid()
{
    mStorage = std::make_shared<Storage>();
}

template <typename BuildingBlockGridT>
auto FeaVoxelGrid<BuildingBlockGridT>::getProperties(const index_3d& /*idx*/)
    const -> typename GridBaseTemplate::CellProperties
{
    NEON_DEV_UNDER_CONSTRUCTION("");
}
// template <typename BuildingBlockGridT>
// template <typename T, int C>
// auto FeaVoxelGrid<BuildingBlockGridT>::newElementField(const std::string fieldUserName, int cardinality, T inactiveValue, Neon::DataUse dataUse, Neon::MemoryOptions memoryOptions) const -> FeaVoxelGrid::ElementField<T>
//{
//     return FeaVoxelGrid::ElementField<T>();
// }

}  // namespace Neon::domain::internal::experimental::FeaVoxelGrid


#include "Neon/domain/internal/experimantal/FeaGrid/FeaNodeField_imp.h"
#include "Neon/domain/internal/experimantal/FeaGrid/FeaNodeGrid_imp.h"