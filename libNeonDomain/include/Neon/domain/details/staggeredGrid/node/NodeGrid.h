#pragma once
#include <assert.h>

#include "Neon/core/core.h"
#include "Neon/core/types/DataUse.h"
#include "Neon/core/types/Macros.h"

#include "Neon/set/BlockConfig.h"
#include "Neon/set/Containter.h"
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
#include "NodePartitionIndexSpace.h"

#include "Neon/domain/details/staggeredGrid/voxel/VoxelGeneric.h"
#include "NodeToVoxelMask.h"


namespace Neon::domain::details::experimental::staggeredGrid::details {

template <typename BuildingBlockGridT>
struct NodeGrid : public Neon::domain::interface::GridBaseTemplate<NodeGrid<BuildingBlockGridT>,
                                                                   NodeGeneric<BuildingBlockGridT>>
{
   private:
    struct BuildingBlocks
    {
        using Grid = BuildingBlockGridT;
        template <typename T_ta, int cardinality_ta = 0>
        using Field = typename BuildingBlockGridT::template Field<T_ta, cardinality_ta>;

        template <typename T_ta, int cardinality_ta = 0>
        using Partition = typename Field<T_ta, cardinality_ta>::Partition;

        using NghIdx = typename Partition<int, 0>::NghIdx;
        using Span = typename BuildingBlocks::Grid::Span;
    };

   public:
    using PartitionIndexSpace = NodePartitionIndexSpace<typename BuildingBlocks::Grid>;
    using Grid = NodeGrid<typename BuildingBlocks::Grid>;
    using Node = Neon::domain::details::experimental::staggeredGrid::details::NodeGeneric<typename BuildingBlocks::Grid>;
    using Cell = Node;
    template <typename T_ta, int cardinality_ta>
    using NodeField = typename Neon::domain::details::experimental::staggeredGrid::details::NodeField<typename BuildingBlocks::Grid, T_ta, cardinality_ta>;

   public:
    NodeGrid();

    NodeGrid(const NodeGrid& rhs) = default;

    virtual ~NodeGrid() = default;

    /**
     * Constructor compatible with the general grid API
     */
    explicit NodeGrid(typename BuildingBlocks::Grid& buildingBlockGrid);

    auto getBuildingBlockGrid()
        -> typename BuildingBlocks::Grid&;

    /**
     * Returns a LaunchParameters configured for the specified inputs
     */
    auto getLaunchParameters(Neon::DataView        dataView,
                             const Neon::index_3d& blockSize,
                             const size_t&         shareMem)
        const -> Neon::set::LaunchParameters;

    auto getPartitionIndexSpace(Neon::DeviceType devE,
                                SetIdx           setIdx,
                                Neon::DataView   dataView)
        -> const PartitionIndexSpace&;

    template <typename T, int C = 0>
    auto newNodeField(const std::string   fieldUserName,
                      int                 cardinality,
                      T                   inactiveValue,
                      Neon::DataUse       dataUse = Neon::DataUse::HOST_DEVICE,
                      Neon::MemoryOptions memoryOptions = Neon::MemoryOptions())
        const -> NodeField<T, C>;

    template <typename LoadingLambda>
    auto getContainerOnNodes(const std::string& name,
                             index_3d           blockSize,
                             size_t             sharedMem,
                             LoadingLambda      lambda)
        const -> Neon::set::Container;

    template <typename LoadingLambda>
    auto getContainerOnNodes(const std::string& name,
                             LoadingLambda      lambda)
        const -> Neon::set::Container;

    auto setReduceEngine(Neon::sys::patterns::Engine eng)
        -> void;

    template <typename T>
    auto newPatternScalar()
        const -> Neon::template PatternScalar<T>;

    template <typename T, int C>
    auto dot(const std::string&               name,
             NodeField<T, C>&                 input1,
             NodeField<T, C>&                 input2,
             Neon::template PatternScalar<T>& scalar)
        const -> Neon::set::Container;

    template <typename T, int C>
    auto norm2(const std::string&               name,
               NodeField<T, C>&                 input,
               Neon::template PatternScalar<T>& scalar)
        const -> Neon::set::Container;

    auto getKernelConfig(int            streamIdx,
                         Neon::DataView dataView)
        -> Neon::set::KernelConfig;

    auto isInsideNodeDomain(const Neon::index_3d& idx)
        const -> bool;

   private:
    auto getNodeToVoxelMaskField()
        -> typename BuildingBlocks::template Field<NodeToVoxelMask, 1>&;

    auto flattenedLengthSet(Neon::DataView dataView = Neon::DataView::STANDARD)
        const -> const Neon::set::DataSet<size_t>;

    auto flattenedPartitions(Neon::DataView dataView = Neon::DataView::STANDARD) const
        -> const Neon::set::DataSet<size_t>;

    auto getLaunchInfo(Neon::DataView dataView)
        const -> Neon::set::LaunchParameters;

    auto stencil() const
        -> const Neon::domain::Stencil&;

    auto newGpuLaunchParameters()
        const -> Neon::set::LaunchParameters;

    auto setKernelConfig(Neon::domain::KernelConfig& gridKernelConfig)
        const -> void;


   private:
    using Self = NodeGrid;
    using GridBaseTemplate = Neon::domain::interface::GridBaseTemplate<Self::Grid, Self::Node>;

   public:
    auto isInsideDomain(const Neon::index_3d& idx)
        const -> bool final;

    auto getProperties(const Neon::index_3d& idx)
        const -> typename GridBaseTemplate::CellProperties final;

    struct Storage
    {
        typename BuildingBlocks::Grid                                                    buildingBlockGrid;
        std::array<Neon::set::DataSet<PartitionIndexSpace>, Neon::DataViewUtil::nConfig> partitionIndexSpace;
    };

    std::shared_ptr<Storage> mStorage;
};


}  // namespace Neon::domain::details::experimental::staggeredGrid::details

#include "NodeField_imp.h"
#include "NodeGeneric_imp.h"
#include "NodePartitionIndexSpace_imp.h"