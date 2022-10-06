#pragma once

#include "Neon/domain/internal/experimantal/staggeredGrid/StaggeredGrid.h"


namespace Neon::domain::internal::experimental::staggeredGrid {

template <typename BuildingBlockGridT>
template <typename ActiveNodesLambda>
StaggeredGrid<BuildingBlockGridT>::StaggeredGrid(const Backend&                            backend,
                                                 const int32_3d&                           voxDimension,
                                                 ActiveNodesLambda                         voxelActiveLambda,
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
    auto nodeDim = voxDimension + 1;

    std::vector<uint8_t> voxels = std::vector<uint8_t>(nodeDim.rMul(), 0);
    std::vector<uint8_t> nodes = std::vector<uint8_t>(nodeDim.rMul(), 0);

    voxDimension.forEach([&](const Neon::index_3d& queryPoint) {
        bool isVoxelActive = voxelActiveLambda(queryPoint);
        if (isVoxelActive) {
            size_t voxPitch = queryPoint.mPitch(nodeDim);
            voxels[voxPitch] = 1;
            nodes[voxPitch] = 1;

            std::vector<Neon::index_3d> nodeOffset{
                {1, 0, 0},
                {1, 1, 0},
                {0, 1, 0},
                {0, 0, 1},
                {1, 0, 1},
                {1, 1, 1},
                {0, 1, 1},
            };

            for (auto a : nodeOffset) {
                a = a + queryPoint;
                const size_t pitch = a.mPitch(nodeDim);
                nodes[pitch] = 1;
            }
        }
    });

    mStorage->buildingBlockGrid = typename BuildingBlocks::Grid(
        backend,
        nodeDim,
        [&](Neon::index_3d idx) {
            const size_t pitch = idx.mPitch(nodeDim);
            return nodes[pitch] == 1;
        },
        feaVoxelFirstOrderStencil,
        spacingData,
        origin);

    auto mask = mStorage->buildingBlockGrid.template newField<uint8_t, 1>("Voxel-mask", 1, 0, Neon::DataUse::IO_COMPUTE);
    mask.forEachActiveCell([&](const Neon::index_3d& idx, int, uint8_t& maskValue) {
        size_t voxPitch = idx.mPitch(nodeDim);
        maskValue = voxels[voxPitch];
    });
    mask.updateCompute(Neon::Backend::mainStreamIdx);

    mStorage->nodeGrid = Self::NodeGrid(mStorage->buildingBlockGrid);

    mStorage->voxelGrid = Self::VoxelGrid(mStorage->buildingBlockGrid, mask);

    Self::GridBase::init(std::string("FeaNodeGird-") + mStorage->buildingBlockGrid.getImplementationName(),
                         backend,
                         nodeDim,
                         feaVoxelFirstOrderStencil,
                         mStorage->buildingBlockGrid.getNumActiveCellsPerPartition(),
                         mStorage->buildingBlockGrid.getDefaultBlock(),
                         spacingData,
                         origin);

    backend.sync(Neon::Backend::mainStreamIdx);
}

template <typename BuildingBlockGridT>
template <typename T, int C>
auto StaggeredGrid<BuildingBlockGridT>::newNodeField(const std::string&  fieldUserName,
                                                     int                 cardinality,
                                                     T                   inactiveValue,
                                                     Neon::DataUse       dataUse,
                                                     Neon::MemoryOptions memoryOptions) const
    -> NodeField<T, C>
{

    auto& nodeGrid = mStorage->nodeGrid;
    auto  output = nodeGrid.template newNodeField<T, C>(fieldUserName,
                                                       cardinality,
                                                       inactiveValue,
                                                       dataUse,
                                                       memoryOptions);
    return output;
}

template <typename BuildingBlockGridT>
template <typename T, int C>
auto StaggeredGrid<BuildingBlockGridT>::newVoxelField(const std::string&  fieldUserName,
                                                     int                 cardinality,
                                                     T                   inactiveValue,
                                                     Neon::DataUse       dataUse,
                                                     Neon::MemoryOptions memoryOptions) const
    -> VoxelField<T, C>
{

    auto& voxelGrid = mStorage->voxelGrid;
    auto  output = voxelGrid.template newVoxelField<T, C>(fieldUserName,
                                                       cardinality,
                                                       inactiveValue,
                                                       dataUse,
                                                       memoryOptions);
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
                                                            LoadingLambda      lambda)
    const -> Neon::set::Container
{
    Neon::set::Container output;
    output = mStorage->nodeGrid.getContainerOnNodes(name, blockSize, sharedMem, lambda);
    return output;
}


template <typename BuildingBlockGridT>
template <typename LoadingLambda>
auto StaggeredGrid<BuildingBlockGridT>::getContainerOnNodes(const std::string& name,
                                                            LoadingLambda      lambda)
    const -> Neon::set::Container
{
    Neon::set::Container output;
    output = mStorage->nodeGrid.getContainerOnNodes(name, lambda);
    return output;
}


}  // namespace Neon::domain::internal::experimental::staggeredGrid


#include "Neon/domain/internal/experimantal/staggeredGrid/node/NodeField_imp.h"
#include "Neon/domain/internal/experimantal/staggeredGrid/node/NodeGrid_imp.h"