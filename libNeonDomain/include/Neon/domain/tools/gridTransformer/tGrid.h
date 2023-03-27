#pragma once

#include "Neon/set/Containter.h"

#include "Neon/domain/interface/GridBaseTemplate.h"
#include "Neon/domain/interface/KernelConfig.h"
#include "Neon/domain/interface/LaunchConfig.h"
#include "Neon/domain/interface/Stencil.h"
#include "Neon/domain/interface/common.h"
#include "Neon/domain/patterns/PatternScalar.h"
#include "Neon/domain/tools/SpanTable.h"

/**
 * template <typename FoundationGrid>
 * GridTransformation {
 *      using PartitionIndexSpace
 *      using Partition
 *      using FoundationGrid
 *
 *      getLaunchParameters(Neon::DataView        dataView,
const Neon::index_3d& blockSize,
const size_t&         shareMem)
 * }
 */
namespace Neon::domain::tool::details {

template <typename T,
          int card,
          typename GridTransformation>
class tField;

template <typename GridTransformation>
class tGrid : public Neon::domain::interface::GridBaseTemplate<tGrid<GridTransformation>,
                                                               typename GridTransformation::FoundationGrid::Cell>
{
   public:
    template <class T, int card>
    using Field = tField<T, card, GridTransformation>;
    template <class T, int card>
    using Partition = typename GridTransformation::template Partition<T, card>;
    using Span = typename GridTransformation::Span;

   private:
    using GridBaseTemplate = Neon::domain::interface::GridBaseTemplate<tGrid<GridTransformation>,
                                                                       typename GridTransformation::FoundationGrid::Idx>;

    using FoundationGrid = typename GridTransformation::FoundationGrid;

   public:
    explicit tGrid(FoundationGrid& foundationGrid);

    tGrid();
    virtual ~tGrid();
    tGrid(const tGrid& other);                 // copy constructor
    tGrid(tGrid&& other) noexcept;             // move constructor
    tGrid& operator=(const tGrid& other);      // copy assignment
    tGrid& operator=(tGrid&& other) noexcept;  // move assignment

    template <typename ActiveCellLambda>
    tGrid(const Neon::Backend&         backend,
          const Neon::int32_3d&        dimension /**< Dimension of the box containing the sparse domain */,
          const ActiveCellLambda&      activeCellLambda /**< InOrOutLambda({x,y,z}->{true, false}) */,
          const Neon::domain::Stencil& stencil,
          const Vec_3d<double>&        spacingData = Vec_3d<double>(1, 1, 1) /**< Spacing, i.e. size of a voxel */,
          const Vec_3d<double>&        origin = Vec_3d<double>(0, 0, 0) /**<      Origin  */);

    auto getLaunchParameters(Neon::DataView        dataView,
                             const Neon::index_3d& blockSize,
                             const size_t&         shareMem) const
        -> Neon::set::LaunchParameters;

    auto getPartitionIndexSpace(Neon::DeviceType devE,
                                SetIdx           setIdx,
                                Neon::DataView   dataView)
        -> const Span &;

    template <typename T, int C = 0>
    auto newField(const std::string&  fieldUserName,
                  int                 cardinality,
                  T                   inactiveValue,
                  Neon::DataUse       dataUse = Neon::DataUse::HOST_DEVICE,
                  Neon::MemoryOptions memoryOptions = Neon::MemoryOptions()) const
        -> Field<T, C>;

    template <typename T, int C = 0>
    auto newField(typename FoundationGrid::template Field<T, C>& field) const
        -> Field<T, C>;

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

    auto isInsideDomain(const Neon::index_3d& idx) const
        -> bool final;

    auto getProperties(const Neon::index_3d& idx) const
        -> typename GridBaseTemplate::CellProperties final;

   private:
    struct Storage
    {
        Storage() = default;
        explicit Storage(Neon::Backend& bk)
        {
            spanTable = Neon::domain::tool::SpanTable<Span>(bk);
        }
        //using IndexSpaceInformation = std::array<Neon::set::DataSet<PartitionIndexSpace>, Neon::DataViewUtil::nConfig>;

        FoundationGrid                                           foundationGrid;
        Neon::domain::tool::SpanTable<Span> spanTable;
        // std::array<Neon::set::DataSet<PartitionIndexSpace>> partitionIndexSpaceVec;
    };

    std::shared_ptr<Storage> mStorage;
};


}  // namespace Neon::domain::tool::details