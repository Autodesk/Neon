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

#include "Neon/domain/internal/experimantal/FeaGrid/FeaNode.h"
#include "Neon/domain/internal/experimantal/FeaGrid/FeaNodeField.h"
#include "Neon/domain/internal/experimantal/FeaGrid/FeaNodePartition.h"

#include "Neon/domain/internal/experimantal/FeaGrid/FeaElement.h"
#include "Neon/domain/internal/experimantal/FeaGrid/FeaElementField.h"
#include "Neon/domain/internal/experimantal/FeaGrid/FeaElementPartition.h"


namespace Neon::domain::internal::experimental::FeaVoxelGrid {

template <typename BuildingBlockGridT>
struct FeaNodeGrid : public Neon::domain::interface::GridBaseTemplate<FeaNodeGrid<BuildingBlockGridT>,
                                                                      FeaNode<BuildingBlockGridT>>
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
    using PartitionIndexSpace = typename BuildingBlocks::PartitionIndexSpace;
    using Grid = FeaNodeGrid<typename BuildingBlocks::Grid>;

    using Node = Neon::domain::internal::experimental::FeaVoxelGrid::FeaNode<typename BuildingBlocks::Grid>;
    template <typename T_ta, int cardinality_ta>
    using NodeField = typename Neon::domain::internal::experimental::FeaVoxelGrid::FeaNodeField<typename BuildingBlocks::Grid, T_ta, cardinality_ta>;

   public:
    FeaNodeGrid();

    FeaNodeGrid(const FeaNodeGrid& rhs) = default;

    virtual ~FeaNodeGrid() = default;

    /**
     * Constructor compatible with the general grid API
     */
    explicit FeaNodeGrid(typename BuildingBlocks::Grid& buildingBlockGrid);

    auto getBuildingBlockGrid()
        -> typename BuildingBlocks::Grid&;

    /**
     * Returns a LaunchParameters configured for the specified inputs
     */
    auto getLaunchParameters(Neon::DataView        dataView,
                             const Neon::index_3d& blockSize,
                             const size_t&         shareMem) const
        -> Neon::set::LaunchParameters;

    auto getPartitionIndexSpace(Neon::DeviceType devE,
                                SetIdx           setIdx,
                                Neon::DataView   dataView)
        -> const PartitionIndexSpace&;

    /**
     * Creates a new Field
     */
    template <typename T, int C = 0>
    auto newNodeField(const std::string   fieldUserName,
                      int                 cardinality,
                      T                   inactiveValue,
                      Neon::DataUse       dataUse = Neon::DataUse::IO_COMPUTE,
                      Neon::MemoryOptions memoryOptions = Neon::MemoryOptions()) const
        -> NodeField<T, C>;

    template <typename LoadingLambda>
    auto getContainer(const std::string& name,
                      index_3d           blockSize,
                      size_t             sharedMem,
                      LoadingLambda      lambda) const
        -> Neon::set::Container;

    template <typename LoadingLambda>
    auto getContainer(const std::string& name,
                      LoadingLambda      lambda)
        const
        -> Neon::set::Container;

    auto setReduceEngine(Neon::sys::patterns::Engine eng) -> void;

    template <typename T>
    auto newPatternScalar() const
        -> Neon::template PatternScalar<T>;

    template <typename T, int C>
    auto dot(const std::string&               name,
             NodeField<T, C>&                 input1,
             NodeField<T, C>&                 input2,
             Neon::template PatternScalar<T>& scalar) const
        -> Neon::set::Container;

    template <typename T, int C>
    auto norm2(const std::string&               name,
               NodeField<T, C>&                 input,
               Neon::template PatternScalar<T>& scalar) const
        -> Neon::set::Container;

    auto getKernelConfig(int            streamIdx,
                         Neon::DataView dataView)
        -> Neon::set::KernelConfig;

    auto isInsideNodeDomain(const Neon::index_3d& idx) const
        -> bool;

   private:
    auto flattenedLengthSet(Neon::DataView dataView = Neon::DataView::STANDARD)
        const -> const Neon::set::DataSet<size_t>;

    auto flattenedPartitions(Neon::DataView dataView = Neon::DataView::STANDARD) const
        -> const Neon::set::DataSet<size_t>;

    auto getLaunchInfo(Neon::DataView dataView) const
        -> Neon::set::LaunchParameters;

    auto stencil() const
        -> const Neon::domain::Stencil&;

    auto newGpuLaunchParameters() const -> Neon::set::LaunchParameters;

    auto setKernelConfig(Neon::domain::KernelConfig& gridKernelConfig)
        const -> void;


   private:
    using Self = FeaNodeGrid;
    using GridBaseTemplate = Neon::domain::interface::GridBaseTemplate<FeaNodeGrid<BuildingBlockGridT>, FeaNode<BuildingBlockGridT>>;

   public:
    auto isInsideDomain(const Neon::index_3d& idx) const
        -> bool final;

    auto getProperties(const Neon::index_3d& idx) const
        -> typename GridBaseTemplate::CellProperties final;

    struct Storage
    {
        typename BuildingBlocks::Grid buildingBlockGrid;
    };

    std::shared_ptr<Storage> mStorage;
};


}  // namespace Neon::domain::internal::experimental::FeaVoxelGrid
