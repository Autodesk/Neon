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
struct FeaNodeGrid : public Neon::domain::interface::GridBaseTemplate<FeaNodeGrid<BuildingBlockGridT>, FeaNode<BuildingBlockGridT>>
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
    template <typename T_ta, int cardinality_ta = 0>
    using NodeField = typename Neon::domain::internal::experimental::FeaVoxelGrid::FeaNodeField<typename BuildingBlocks::Grid, T_ta, cardinality_ta>;

   public:
    FeaNodeGrid();

    FeaNodeGrid(const FeaNodeGrid& rhs) = default;

    virtual ~FeaNodeGrid() = default;

    /**
     * Constructor compatible with the general grid API
     */
    template <typename ActiveCellLambda>
    FeaNodeGrid(const Neon::Backend&                      backend,
                const Neon::int32_3d&                     dimension /**< Dimension of the box containing nodes */,
                ActiveCellLambda                          activeLambda /**< InOrOutLambda({x,y,z}->{true, false}) */,
                const std::vector<Neon::domain::Stencil>& optionalExtraStencil = {},
                const Vec_3d<double>&                     spacingData = Vec_3d<double>(1, 1, 1) /**< Spacing, i.e. size of a voxel */,
                const Vec_3d<double>&                     origin = Vec_3d<double>(0, 0, 0) /**<      Origin  */);

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

    template <typename T>
    auto dot(const std::string&               name,
             NodeField<T>&                    input1,
             NodeField<T>&                    input2,
             Neon::template PatternScalar<T>& scalar) const
        -> Neon::set::Container;

    template <typename T>
    auto norm2(const std::string&               name,
               NodeField<T>&                    input,
               Neon::template PatternScalar<T>& scalar) const
        -> Neon::set::Container;

    auto getKernelConfig(int            streamIdx,
                         Neon::DataView dataView)
        -> Neon::set::KernelConfig;

    auto isInsideNodeDomain(const Neon::index_3d& idx) const
        -> bool;

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

template <typename BuildingBlockGridT>
template <typename ActiveCellLambda>
FeaNodeGrid<BuildingBlockGridT>::FeaNodeGrid(const Backend&                            backend,
                                             const int32_3d&                           dimension,
                                             ActiveCellLambda                          activeNodes,
                                             const std::vector<Neon::domain::Stencil>& stencilVec,
                                             const Vec_3d<double>&                     spacingData,
                                             const Vec_3d<double>&                     origin)
{
    mStorage->buildingBlockGrid = BuildingBlocks::Grid(backend,
                                                       dimension,
                                                       activeNodes,
                                                       spacingData,
                                                       origin);
}

template <typename BuildingBlockGridT>
auto FeaNodeGrid<BuildingBlockGridT>::getPartitionIndexSpace(Neon::DeviceType devE,
                                                             SetIdx           setIdx,
                                                             Neon::DataView   dataView) -> const PartitionIndexSpace&
{
    return mStorage->buildingBlockGrid.getPartitionIndexSpace(devE, setIdx, dataView);
}
template <typename BuildingBlockGridT>
auto FeaNodeGrid<BuildingBlockGridT>::getLaunchParameters(Neon::DataView  dataView,
                                                          const index_3d& blockSize,
                                                          const size_t&   shareMem)
    const -> Neon::set::LaunchParameters
{
    return mStorage->buildingBlockGrid.getLaunchParameters(dataView, blockSize, shareMem);
}
template <typename BuildingBlockGridT>
FeaNodeGrid<BuildingBlockGridT>::FeaNodeGrid()
{
    mStorage = std::
}

}

template <typename BuildingBlockGridT>
template <typename T, int C>
auto FeaNodeGrid<BuildingBlockGridT>::newNodeField(const std::string   fieldUserName,
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
auto FeaNodeGrid<BuildingBlockGridT>::isInsideDomain(const index_3d& /*idx*/) const -> bool
{
    NEON_DEV_UNDER_CONSTRUCTION("");
}
template <typename BuildingBlockGridT>
auto FeaNodeGrid<BuildingBlockGridT>::getLaunchParameters(Neon::DataView  dataView,
                                                          const index_3d& blockSize,
                                                          const size_t&   shareMem) const -> Neon::set::LaunchParameters
{
    return mStorage->buildingBlockGrid.getLaunchParameters(dataView,
                                                           blockSize, shareMem);
}
template <typename BuildingBlockGridT>
auto FeaNodeGrid<BuildingBlockGridT>::getPartitionIndexSpace(Neon::DeviceType devE, SetIdx setIdx, Neon::DataView dataView) -> const PartitionIndexSpace&
{

    (void)devE;
    (void)setIdx;
    (void)dataView;
    NEON_DEV_UNDER_CONSTRUCTION("");
}
template <typename BuildingBlockGridT>
template <typename T, int C>
auto FeaNodeGrid<BuildingBlockGridT>::newElementField(const std::string fieldUserName, int cardinality, T inactiveValue, Neon::DataUse dataUse, Neon::MemoryOptions memoryOptions) const -> FeaNodeGrid::ElementField<T>
{
    return FeaNodeGrid::ElementField<T>();
}

}  // namespace Neon::domain::internal::experimental::FeaVoxelGrid
