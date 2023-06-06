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

#include "Neon/domain/details/staggeredGrid/node/NodeGeneric.h"

#include "Neon/domain/details/staggeredGrid/node/NodeToVoxelMask.h"
#include "VoxelField.h"
#include "VoxelGeneric.h"
#include "VoxelPartition.h"
#include "VoxelSpan.h"


namespace Neon::domain::details::experimental::staggeredGrid::details {

template <typename BuildingBlockGridT>
struct VoxelGrid : public Neon::domain::interface::GridBaseTemplate<VoxelGrid<BuildingBlockGridT>,
                                                                    VoxelGeneric<BuildingBlockGridT>>
{

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
    using Span = VoxelSpan<typename BuildingBlocks::Grid>;
    using Grid = VoxelGrid<typename BuildingBlocks::Grid>;
    using Voxel = Neon::domain::details::experimental::staggeredGrid::details::VoxelGeneric<typename BuildingBlocks::Grid>;
    using Cell = Voxel;
    template <typename TypeT, int CardinalityT>
    using VoxelField = typename Neon::domain::details::experimental::staggeredGrid::details::VoxelField<typename BuildingBlocks::Grid, TypeT, CardinalityT>;

   private:
    using Self = VoxelGrid;
    using GridBaseTemplate = Neon::domain::interface::GridBaseTemplate<typename Self::Grid, typename Self::Voxel>;

   public:
    VoxelGrid();

    VoxelGrid(const VoxelGrid& rhs) = default;

    virtual ~VoxelGrid() = default;

    /**
     * Constructor compatible with the general grid API
     */
    explicit VoxelGrid(typename BuildingBlocks::Grid&                                                                                                   buildingBlockGrid,
                       const typename BuildingBlocks::template Field<uint8_t, 1>&                                                                       mask,
                       const typename BuildingBlocks::template Field<Neon::domain::details::experimental::staggeredGrid::details::NodeToVoxelMask, 1>& nodeToVoxelMaskField);

    auto getBuildingBlockGrid()
        -> typename BuildingBlocks::Grid&;

    /**
     * Returns a LaunchParameters configured for the specified inputs
     */
    auto getLaunchParameters(Neon::DataView        dataView,
                             const Neon::index_3d& blockSize,
                             const size_t&         shareMem)
        const -> Neon::set::LaunchParameters;

    auto getSpan(Neon::DeviceType devE,
                                SetIdx           setIdx,
                                Neon::DataView   dataView)
        -> const Span&;

    template <typename T, int C = 0>
    auto newVoxelField(const std::string   fieldUserName,
                       int                 cardinality,
                       T                   inactiveValue,
                       Neon::DataUse       dataUse = Neon::DataUse::HOST_DEVICE,
                       Neon::MemoryOptions memoryOptions = Neon::MemoryOptions())
        const -> VoxelField<T, C>;

    template <typename LoadingLambda>
    auto getContainerOnVoxels(const std::string& name,
                              index_3d           blockSize,
                              size_t             sharedMem,
                              LoadingLambda      lambda)
        const -> Neon::set::Container;

    template <typename LoadingLambda>
    auto getContainerOnVoxels(const std::string& name,
                              LoadingLambda      lambda)
        const -> Neon::set::Container;

    auto isInsideDomain(const Neon::index_3d& idx)
        const -> bool final;

    auto getProperties(const Neon::index_3d& idx)
        const -> typename GridBaseTemplate::CellProperties final;

    auto setReduceEngine(Neon::sys::patterns::Engine eng)
        -> void;

    template <typename T>
    auto newPatternScalar()
        const -> Neon::template PatternScalar<T>;

    template <typename T, int C>
    auto dot(const std::string&               name,
             VoxelField<T, C>&                input1,
             VoxelField<T, C>&                input2,
             Neon::template PatternScalar<T>& scalar)
        const -> Neon::set::Container;

    template <typename T, int C>
    auto norm2(const std::string&               name,
               VoxelField<T, C>&                input,
               Neon::template PatternScalar<T>& scalar)
        const -> Neon::set::Container;

    auto getKernelConfig(int            streamIdx,
                         Neon::DataView dataView)
        -> Neon::set::KernelConfig;

    auto isInsideNodeDomain(const Neon::index_3d& idx)
        const -> bool;

   private:
    auto flattenedLengthSet(Neon::DataView dataView = Neon::DataView::STANDARD)
        const -> const Neon::set::DataSet<size_t>;

    auto flattenedPartitions(Neon::DataView dataView = Neon::DataView::STANDARD)
        const -> const Neon::set::DataSet<size_t>;

    auto getLaunchInfo(Neon::DataView dataView)
        const -> Neon::set::LaunchParameters;

    auto stencil()
        const -> const Neon::domain::Stencil&;

    auto newGpuLaunchParameters()
        const -> Neon::set::LaunchParameters;

    auto setKernelConfig(Neon::domain::KernelConfig& gridKernelConfig)
        const -> void;

    struct Storage
    {
        std::array<std::array<Neon::set::DataSet<Span>, Neon::DataViewUtil::nConfig>, Neon::DeviceTypeUtil::nConfig> partitionIndexSpace;

        typename BuildingBlocks::Grid                               buildingBlockGrid;
        typename BuildingBlocks::template Field<uint8_t, 1>         mask;
        typename BuildingBlocks::template Field<NodeToVoxelMask, 1> nodeToVoxelMaskField;
    };

    std::shared_ptr<Storage> mStorage;
};


}  // namespace Neon::domain::details::experimental::staggeredGrid::details

#include "VoxelField.h"
#include "VoxelGeneric_imp.h"
#include "VoxelGrid_imp.h"
#include "VoxelPartitionIndexSpace_imp.h"
