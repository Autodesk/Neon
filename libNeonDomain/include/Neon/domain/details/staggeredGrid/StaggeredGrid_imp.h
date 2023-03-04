#pragma once

#include "Neon/domain/details/staggeredGrid/node/NodeToVoxelMask.h"
#include "StaggeredGrid.h"

namespace Neon::domain::details::experimental::staggeredGrid {

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

    const Neon::domain::Stencil unionOfAllStencils = [&] {
        // Stencil that is required to move from voxels to nodes
        // Reminder, voxel@(0,0,0) is mapped into node@(0,0,0)
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

        std::vector<Neon::domain::Stencil> allStencils;
        allStencils.push_back(feaVoxelFirstOrderStencil);
        allStencils.insert(allStencils.end(),
                           optionalExtraStencil.begin(),
                           optionalExtraStencil.end());
        return Neon::domain::Stencil::getUnion(allStencils);
    }();

    auto nodeDim = voxDimension + 1;

    std::vector<uint8_t> voxels = std::vector<uint8_t>(nodeDim.rMul(), 0);
    std::vector<uint8_t> nodes = std::vector<uint8_t>(nodeDim.rMul(), 0);

    voxDimension.forEach([&](const Neon::index_3d& queryPoint) {
        bool isVoxelActive = voxelActiveLambda(queryPoint);
        if (isVoxelActive) {
            size_t voxPitch = queryPoint.mPitch(nodeDim);
            voxels[voxPitch] = 1;

            std::vector<Neon::index_3d> nodeOffset{
                {1, 1, 1},
                {1, 1, 0},
                {1, 0, 1},
                {1, 0, 0},
                {0, 1, 1},
                {0, 1, 0},
                {0, 0, 1},
                {0, 0, 0}};

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
        [&](const Neon::index_3d& idx) {
            const size_t pitch = idx.mPitch(nodeDim);
            bool         isIn = nodes[pitch] == 1;
            return isIn;
        },
        unionOfAllStencils,
        spacingData,
        origin);
    mStorage->buildingBlockGrid.setReduceEngine(Neon::sys::patterns::Engine::CUB);

    auto mask = mStorage->buildingBlockGrid.template newField<uint8_t, 1>("Voxel-mask", 1, 0, Neon::DataUse::IO_COMPUTE);
    mask.forEachActiveCell([&](const Neon::index_3d& idx, int, uint8_t& maskValue) {
        size_t voxPitch = idx.mPitch(nodeDim);
        maskValue = voxels[voxPitch];
    });
    mask.updateDeviceData(Neon::Backend::mainStreamIdx);

    using NodeToVoxelMask = Neon::domain::details::experimental::staggeredGrid::details::NodeToVoxelMask;
    auto nodeToVoxelMask = mStorage->buildingBlockGrid.template newField<NodeToVoxelMask, 1>("NodeToVoxelMask", 1, NodeToVoxelMask(), Neon::DataUse::IO_COMPUTE);

    nodeToVoxelMask.forEachActiveCell([&](const Neon::index_3d& queryPoint, int, NodeToVoxelMask& nodeToVoxelMaskValue) {
        nodeToVoxelMaskValue.reset();

        for (auto a : NodeToVoxelMask::directions) {
            Neon::int32_3d neighbourInVoxelSpace;

            for (int i = 0; i < 3; i++) {
                neighbourInVoxelSpace.v[i] = (a[i] == 1 ? a[i] - 1 : a[i]) + queryPoint.v[i];
            };
            auto voxelStatus = mask(neighbourInVoxelSpace, 0);
            if (voxelStatus == 1) {
                nodeToVoxelMaskValue.setAsValid(a[0], a[1], a[2]);
            }
        }
    });
    nodeToVoxelMask.updateDeviceData(Neon::Backend::mainStreamIdx);

    mStorage->nodeGrid = Self::NodeGrid(mStorage->buildingBlockGrid);

    mStorage->voxelGrid = Self::VoxelGrid(mStorage->buildingBlockGrid, mask, nodeToVoxelMask);

    Self::GridBase::init(std::string("FeaNodeGird-") + mStorage->buildingBlockGrid.getImplementationName(),
                         backend,
                         nodeDim,
                         unionOfAllStencils,
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
auto StaggeredGrid<BuildingBlockGridT>::isInsideDomain(const index_3d&) const -> bool
{
    NEON_THROW_UNSUPPORTED_OPERATION("");
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
auto StaggeredGrid<BuildingBlockGridT>::isNodeInsideDomain(const index_3d& idx) const -> bool
{
    return mStorage->nodeGrid.isInsideDomain(idx);
}

template <typename BuildingBlockGridT>
auto StaggeredGrid<BuildingBlockGridT>::isVoxelInsideDomain(const index_3d& idx) const -> bool
{
    return mStorage->voxelGrid.isInsideDomain(idx);
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

}  // namespace Neon::domain::details::experimental::staggeredGrid


#include "Neon/domain/details/staggeredGrid/node/NodeField_imp.h"
#include "Neon/domain/details/staggeredGrid/node/NodeGrid_imp.h"