#pragma once

#include "Neon/core/core.h"
#include "Neon/core/types/DataUse.h"
#include "Neon/core/types/Macros.h"

#include "Neon/sys/memory/MemDevice.h"

#include "Neon/set/BlockConfig.h"
#include "Neon/set/Containter.h"
#include "Neon/set/DataConfig.h"
#include "Neon/set/DevSet.h"
#include "Neon/set/MemoryOptions.h"

#include "Neon/domain/interface/GridBaseTemplate.h"
#include "Neon/domain/interface/KernelConfig.h"
#include "Neon/domain/interface/LaunchConfig.h"
#include "Neon/domain/interface/Stencil.h"
#include "Neon/domain/interface/common.h"
#include "Neon/domain/patterns/PatternScalar.h"

#include "Neon/domain/details/staggeredGrid/node/NodeGrid.h"
#include "Neon/domain/details/staggeredGrid/voxel/VoxelGrid.h"

namespace Neon::domain::internal::experimental::staggeredGrid {

template <typename BuildingBlockGridT>
struct StaggeredGrid : public Neon::domain::interface::GridBaseTemplate<StaggeredGrid<BuildingBlockGridT>,
                                                                        Neon::domain::internal::experimental::staggeredGrid::details::NodeGeneric<BuildingBlockGridT>>
{
   private:
    struct BuildingBlocks
    {
        using Grid = BuildingBlockGridT;
        template <typename TypeT, int CardinalityT = 0>
        using Field = typename BuildingBlockGridT::template Field<TypeT, CardinalityT>;
        template <typename TypeT, int CardinalityT = 0>
        using Partition = typename Field<TypeT, CardinalityT>::Partition;
        using Ngh_idx = typename Partition<int, 0>::nghIdx_t;
        using PartitionIndexSpace = typename BuildingBlocks::Grid::PartitionIndexSpace;
    };

   public:
    /**
     * Some public alias that hides uses from some low level aspects of the system
     */
    using PartitionIndexSpace = typename BuildingBlocks::PartitionIndexSpace;

    using Node = Neon::domain::internal::experimental::staggeredGrid::details::NodeGeneric<typename BuildingBlocks::Grid>;
    using Voxel = Neon::domain::internal::experimental::staggeredGrid::details::VoxelGeneric<typename BuildingBlocks::Grid>;

    using NodeGrid = typename Neon::domain::internal::experimental::staggeredGrid::details::NodeGrid<typename BuildingBlocks::Grid>;
    using VoxelGrid = typename Neon::domain::internal::experimental::staggeredGrid::details::VoxelGrid<typename BuildingBlocks::Grid>;

    template <typename TypeT, int CardinalityT = 0>
    using VoxelField = typename Neon::domain::internal::experimental::staggeredGrid::details::VoxelField<typename BuildingBlocks::Grid, TypeT, CardinalityT>;

    template <typename TypeT, int CardinalityT = 0>
    using NodeField = typename Neon::domain::internal::experimental::staggeredGrid::details::NodeField<typename BuildingBlocks::Grid, TypeT, CardinalityT>;

    using GridBase = typename Neon::domain::interface::GridBaseTemplate<StaggeredGrid<BuildingBlockGridT>,
                                                                        Neon::domain::internal::experimental::staggeredGrid::details::NodeGeneric<BuildingBlockGridT>>;

   public:
    /**
     * Constructors
     * While default constructor user-defined,
     * the remaining 4 of the 5s are still defaulted.
     */
    StaggeredGrid();
    StaggeredGrid(const StaggeredGrid&) noexcept = default;
    StaggeredGrid(StaggeredGrid&&) noexcept = default;
    StaggeredGrid& operator=(StaggeredGrid&&) noexcept = default;
    virtual ~StaggeredGrid() = default;

    /**
     * Constructor compatible with the general grid API
     */
    template <typename ActiveNodesLambda>
    StaggeredGrid(const Neon::Backend&                      backend,
                  const Neon::int32_3d&                     dimension /**< Dimension of the box containing nodes */,
                  ActiveNodesLambda                         voxelActiveLambda /**< InOrOutLambda({x,y,z}->{true, false}) */,
                  const std::vector<Neon::domain::Stencil>& optionalExtraStencil = {},
                  const Neon::double_3d&                    spacingData = Neon::double_3d(1, 1, 1) /**< Spacing, i.e. size of a voxel */,
                  const Neon::double_3d&                    origin = Neon::double_3d(0, 0, 0) /**<      Origin  */);


    /**
     * Allocates a new field over the node grid
     */
    template <typename T, int C = 0>
    auto newNodeField(const std::string&  fieldUserName /**< User defined name for the field. */,
                      int                 cardinality /**< Number of components of the field */,
                      T                   inactiveValue /**< Default background value */,
                      Neon::DataUse       dataUse = Neon::DataUse::IO_COMPUTE /** Type of use for the field */,
                      Neon::MemoryOptions memoryOptions = Neon::MemoryOptions() /**< Memory options */)
        const -> NodeField<T, C>;

    /**
     * Allocates a new field over the voxel grid
     */
    template <typename T, int C = 0>
    auto newVoxelField(const std::string&  fieldUserName /**< User defined name for the field. */,
                       int                 cardinality /**<   Number of components of the field */,
                       T                   inactiveValue /**< Default background value */,
                       Neon::DataUse       dataUse = Neon::DataUse::IO_COMPUTE /**<   Type of use for the field */,
                       Neon::MemoryOptions memoryOptions = Neon::MemoryOptions() /**< Memory options */)
        const -> VoxelField<T, C>;

    /**
     * Returns a container running over all the nodes
     */
    template <typename LoadingLambda>
    auto getContainerOnNodes(const std::string& name /**<      User define name for the computation */,
                             index_3d           blockSize /**< Define the block size for the computation */,
                             size_t             sharedMem /**< Size of required CUDA shared memory */,
                             LoadingLambda      lambda /**<    Neon Loading Lambda*/)
        const -> Neon::set::Container;

    /**
     * Returns a container running over all the voxels
     */
    template <typename LoadingLambda>
    auto getContainerOnNodes(const std::string& name /**<   User define name for the computation */,
                             LoadingLambda      lambda /**< Neon Loading Lambda*/)
        const -> Neon::set::Container;

    /**
     * Returns true if the voxel is active
     */
    auto isVoxelInsideDomain(const Neon::index_3d& idx /**< Query ponint */)
        const -> bool;

    auto isNodeInsideDomain(const Neon::index_3d& idx /**< Query ponint */)
        const -> bool;

   private:
    /**
     * Hiding this method as it is replaced by isVoxelInsideDomain and isNodeInsideDomain
     */
    auto isInsideDomain(const Neon::index_3d& idx) const
        -> bool final;

    using Self = StaggeredGrid;
    using GridBaseTemplate = Neon::domain::interface::GridBaseTemplate<StaggeredGrid<BuildingBlockGridT>, Self::Node>;

    auto getProperties(const Neon::index_3d& idx) const
        -> typename GridBaseTemplate::CellProperties final;

    struct Storage
    {
        typename BuildingBlocks::Grid buildingBlockGrid;

        Self::NodeGrid  nodeGrid;
        Self::VoxelGrid voxelGrid;
    };

    std::shared_ptr<Storage> mStorage;
};

}  // namespace Neon::domain::internal::experimental::staggeredGrid


#include "StaggeredGrid_imp.h"

#include "Neon/domain/details/staggeredGrid/node/NodeField_imp.h"
#include "Neon/domain/details/staggeredGrid/node/NodePartitionIndexSpace_imp.h"


#include "Neon/domain/details/staggeredGrid/voxel/VoxelField_imp.h"
#include "Neon/domain/details/staggeredGrid/voxel/VoxelGrid_imp.h"
#include "Neon/domain/details/staggeredGrid/voxel/VoxelPartitionIndexSpace_imp.h"
