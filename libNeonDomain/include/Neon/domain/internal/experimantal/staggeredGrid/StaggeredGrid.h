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
#include "Neon/domain/internal/experimantal/staggeredGrid/VoxelField.h"
#include "Neon/domain/internal/experimantal/staggeredGrid/VoxelGeneric.h"
#include "Neon/domain/internal/experimantal/staggeredGrid/VoxelPartition.h"


namespace Neon::domain::internal::experimental::staggeredGrid {

template <typename BuildingBlockGridT>
struct StaggeredGrid : public Neon::domain::interface::GridBaseTemplate<StaggeredGrid<BuildingBlockGridT>,
                                                                        Neon::domain::internal::experimental::staggeredGrid::details::NodeGeneric<BuildingBlockGridT>>
{
   private:
    struct BuildingBlocks
    {
        using Grid = BuildingBlockGridT;
        template <typename T_ta, int cardinality_ta = 0>
        using Field = typename BuildingBlockGridT::template Field<T_ta, cardinality_ta>;
        template <typename T_ta, int cardinality_ta = 0>
        using Partition = typename BuildingBlocks::Field<T_ta, cardinality_ta>::Partition;
        using Ngh_idx = typename BuildingBlocks::Partition<int>::nghIdx_t;
        using PartitionIndexSpace = typename BuildingBlocks::Grid::PartitionIndexSpace;
    };


   public:
    enum FeaComponents
    {
        Nodes,
        Elements
    };

    enum FeaOrder
    {
        First
    };

    using PartitionIndexSpace = typename BuildingBlocks::PartitionIndexSpace;

    using Node = Neon::domain::internal::experimental::staggeredGrid::details::NodeGeneric<typename BuildingBlocks::Grid>;
    using Voxel = Neon::domain::internal::experimental::staggeredGrid::details::VoxelGeneric<typename BuildingBlocks::Grid>;

//    using VoxelGrid = typename Neon::domain::internal::experimental::staggeredGrid::details::VexelGrid<typename BuildingBlocks::Grid, T_ta, cardinality_ta>;
//
    using NodeGrid = typename Neon::domain::internal::experimental::staggeredGrid::details::NodeGrid<typename BuildingBlocks::Grid>;

    template <typename T_ta, int cardinality_ta = 0>
    using VoxelField = typename Neon::domain::internal::experimental::staggeredGrid::details::VoxelField<typename BuildingBlocks::Grid, T_ta, cardinality_ta>;

    template <typename T_ta, int cardinality_ta = 0>
    using NodeField = typename Neon::domain::internal::experimental::staggeredGrid::details::NodeField<typename BuildingBlocks::Grid, T_ta, cardinality_ta>;

    using GridBase = typename Neon::domain::interface::GridBaseTemplate<StaggeredGrid<BuildingBlockGridT>,
                                                                        Neon::domain::internal::experimental::staggeredGrid::details::NodeGeneric<BuildingBlockGridT>>;

   public:
    StaggeredGrid();

    StaggeredGrid(const StaggeredGrid& rhs) = default;

    ~StaggeredGrid() = default;

    /**
     * Constructor compatible with the general grid API
     */
    template <typename ActiveNodesLambda>
    StaggeredGrid(const Neon::Backend&                      backend,
                  const Neon::int32_3d&                     dimension /**< Dimension of the box containing nodes */,
                  ActiveNodesLambda                         nodeActiveLambda /**< InOrOutLambda({x,y,z}->{true, false}) */,
                  const std::vector<Neon::domain::Stencil>& optionalExtraStencil = {},
                  const Vec_3d<double>&                     spacingData = Vec_3d<double>(1, 1, 1) /**< Spacing, i.e. size of a voxel */,
                  const Vec_3d<double>&                     origin = Vec_3d<double>(0, 0, 0) /**<      Origin  */);


    /**
     * Creates a new Field
     */
    template <typename T, int C = 0>
    auto newNodeField(const std::string&  fieldUserName,
                      int                 cardinality,
                      T                   inactiveValue,
                      Neon::DataUse       dataUse = Neon::DataUse::IO_COMPUTE,
                      Neon::MemoryOptions memoryOptions = Neon::MemoryOptions()) const
        -> NodeField<T, C>;

    template <typename LoadingLambda>
    auto getContainerOnNodes(const std::string& name,
                      index_3d           blockSize,
                      size_t             sharedMem,
                      LoadingLambda      lambda) const
        -> Neon::set::Container;

    template <typename T, int C = 0>
    auto newVoxelField(const std::string   fieldUserName,
                         int                 cardinality,
                         T                   inactiveValue,
                         Neon::DataUse       dataUse = Neon::DataUse::IO_COMPUTE,
                         Neon::MemoryOptions memoryOptions = Neon::MemoryOptions()) const
        -> VoxelField<T, C>;


    auto setReduceEngine(Neon::sys::patterns::Engine eng) -> void;

    auto isNodeInsideDomain(const Neon::index_3d& idx) const
        -> bool;
   private:
    auto getKernelConfig(int            streamIdx,
                         Neon::DataView dataView)
        -> Neon::set::KernelConfig;



   private:
    auto partitions() const
        -> const Neon::set::DataSet<index_3d>;

    auto flattenedLengthSet(Neon::DataView dataView = Neon::DataView::STANDARD)
        const -> const Neon::set::DataSet<size_t>;

    auto flattenedPartitions(Neon::DataView dataView = Neon::DataView::STANDARD) const
        -> const Neon::set::DataSet<size_t>;

    auto getLaunchInfo(const Neon::DataView dataView) const
        -> Neon::set::LaunchParameters;

    auto stencil() const
        -> const Neon::domain::Stencil&;

    auto newGpuLaunchParameters() const -> Neon::set::LaunchParameters;

    void setKernelConfig(Neon::domain::KernelConfig& gridKernelConfig) const;


   private:
    using Self = StaggeredGrid;
    using GridBaseTemplate = Neon::domain::interface::GridBaseTemplate<StaggeredGrid<BuildingBlockGridT>, Self::Node>;

   public:
    auto isInsideDomain(const Neon::index_3d& idx) const
        -> bool final;

    auto getProperties(const Neon::index_3d& idx) const
        -> typename GridBaseTemplate::CellProperties final;

    struct Storage
    {
        typename BuildingBlocks::Grid buildingBlockGrid;
        //  m_partitionDims indicates the size of each partition. For example,
        // given a gridDim of size 77 (in 1D for simplicity) distrusted over 5
        // device, it should be distributed as (16 16 15 15 15)
        Neon::set::DataSet<index_3d> partitionDims;

        Neon::index_3d                                       halo;
        std::vector<Neon::set::DataSet<PartitionIndexSpace>> partitionIndexSpaceVec;
        Neon::sys::patterns::Engine                          reduceEngine{Neon::sys::patterns::Engine::CUB};

        Self::NodeGrid nodeGrid;
    };

    std::shared_ptr<Storage> mStorage;
};


}  // namespace Neon::domain::internal::experimental::staggeredGrid


#include "Neon/domain/internal/experimantal/staggeredGrid/NodeField_imp.h"
#include "Neon/domain/internal/experimantal/staggeredGrid/NodeGrid_imp.h"
#include "Neon/domain/internal/experimantal/staggeredGrid/StaggeredGrid_imp.h"
