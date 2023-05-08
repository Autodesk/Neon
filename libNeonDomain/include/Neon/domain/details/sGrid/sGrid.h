#pragma once

#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"

#include "Neon/sys/memory/MemDevice.h"

#include "Neon/set/BlockConfig.h"
#include "Neon/set/Containter.h"

#include "Neon/set/DevSet.h"

#include "Neon/domain/interface/GridBaseTemplate.h"
#include "Neon/domain/interface/KernelConfig.h"
#include "Neon/domain/interface/common.h"
#include "Neon/domain/tools/PointHashTableSet.h"

#include "Neon/domain/details/sGrid/sField.h"
#include "Neon/domain/details/sGrid/sFieldStorage.h"
#include "Neon/domain/details/sGrid/sIndex.h"
#include "Neon/domain/details/sGrid/sPartition.h"
#include "Neon/domain/details/sGrid/sSpan.h"

namespace Neon::domain::details::sGrid {

template <typename OuterGridT, typename T, int C>
class sField;

template <typename OuterGridT>
class sGrid : public Neon::domain::interface::GridBaseTemplate<sGrid<OuterGridT>,
                                                               sIndex>
{
   public:
    using OuterGrid = OuterGridT;
    using Grid = sGrid;
    using Idx = sIndex;

    template <typename OuterGridTK, typename T, int C>
    friend class sField;

    template <typename T, int C>
    using Partition = sPartition<OuterGridT, T, C>; /** Type of a partition for sGrid */

    template <typename T, int C>
    using Field = Neon::domain::details::sGrid::sField<OuterGrid, T, C>; /**< Type of a field for sGrid */

    using Span = Neon::domain::details::sGrid::sSpan; /**< Type of the space is indexes for a lambda executor */

    static constexpr Neon::set::details::ExecutionThreadSpan executionThreadSpan = Neon::set::details::ExecutionThreadSpan::d1;
    using ExecutionThreadSpanIndexType = uint32_t;

    /**
     * Empty constructor
     */
    sGrid();

    /**
     * Main constructor for the grid.
     */
    sGrid(OuterGridT const&                  outerGrid /**< A reference to the target outer grid */,
          std::vector<Neon::index_3d> const& subGridPoints /**< Points of the outer grid that will be represented by sGrid */);

    /**
     * Returns a LaunchParameters object set up for this grid
     */
    auto getLaunchParameters(Neon::DataView        dataView,
                             const Neon::index_3d& blockSize = Neon::index_3d(256, 1, 1),
                             size_t                shareMem = 0) const
        -> Neon::set::LaunchParameters;

    /**
     * Returns the Partition Index Space for this grid
     */
    auto getSpan(Neon::Execution execution,
                 Neon::SetIdx    setIdx,
                 Neon::DataView  dataView) const
        -> const Span&;

    /**
     * Returns a KernelConfig set up for this grid
     */
    auto getKernelConfig(int            streamIdx,
                         Neon::DataView dataView) -> Neon::set::KernelConfig;

    /**
     * Creates a new Field
     */
    template <typename T, int C = 0>
    auto newField(const std::string fieldUserName /**< A name the user would like to associate to this field */,
                  int               cardinality /**< If 1 this is a scalar field, if higher this is a vector field,
                                                     where each cell value has a number of components equal to cardinality. */
                  ,
                  T                   inactiveValue /**< default value for point outside the domain */,
                  Neon::DataUse       dataUse = Neon::DataUse::HOST_DEVICE /**< use of the field: computation or post processing */,
                  Neon::MemoryOptions memoryOptions = Neon::MemoryOptions() /**< Memory options including layout */) const
        -> Field<T, C>;

    /**
     * Creates a container that will run on this grid
     */
    template <typename LoadingLambda>
    auto newContainer(const std::string& name,
                      LoadingLambda      lambda,
                      Neon::Execution    execution)
        const
        -> Neon::set::Container;

    /**
     * Creates a container that will run on this grid
     */
    template <typename LoadingLambda>
    auto newContainer(const std::string& name,
                      index_3d           blockSize,
                      size_t             sharedMem,
                      LoadingLambda      lambda)
        const
        -> Neon::set::Container;

    /**
     * Returns true if the specified point is in the domain
     */
    auto isInsideDomain(const Neon::index_3d& idx) const
        -> bool final;

    /**
     * Returns some properties for specified point
     */
    auto getProperties(const Neon::index_3d& idx) const
        -> typename Neon::domain::interface::GridBaseTemplate<sGrid<OuterGrid>, sIndex>::CellProperties final;

    auto getSetIdx(Neon::index_3d const&) const -> int final;

   private:
    using Self = sGrid<OuterGrid>;
    using GridBaseTemplate = Neon::domain::interface::GridBaseTemplate<Self, sIndex>;
    using Count = typename Partition<char, 0>::Count;
    using Index = typename Partition<char, 0>::Index;

    /**
     * Internal helper function to set KernelConfig structures
     */
    auto setKernelConfig(Neon::domain::KernelConfig& gridKernelConfig) const -> void;

    /**
     * Internal helper function to initialize default parameters
     */
    auto initDefaults() -> void;

    struct Meta
    {
        Meta() = default;
        Meta(int64_t                                    offset,
             typename OuterGridT::Cell::OuterIdx const& outerCell_)
            : cellOffset(offset), outerCell(outerCell_)
        {
        }
        int32_t                             cellOffset;
        typename OuterGridT::Cell::OuterIdx outerCell;
    };

    struct sStorage
    {
        /** init storage stricture */
        auto init(const OuterGrid& outerGrid_)
        {
            outerGrid = outerGrid_;

            for (auto& dw : DataViewUtil::validOptions()) {
                getCount(dw) = outerGrid.getDevSet().template newDataSet<size_t>();
                getPartitionIndexSpace(dw) = outerGrid.getDevSet().template newDataSet<sSpan>();
                map = Neon::domain::tool::PointHashTableSet<int, Meta>(outerGrid);
            }
        }

        /**
         * Returns the number of cell by data view.
         * The information is return through a DataSet:
         * One counter for each device.
         * */
        auto getCount(DataView dw) -> Neon::set::DataSet<size_t>&
        {
            return count[DataViewUtil::toInt(dw)];
        }

        /**
         * Return the index space on all GPUs based on a data view.
         * @param dw
         * @return
         */
        auto getPartitionIndexSpace(DataView dw) -> Neon::set::DataSet<sSpan>&
        {
            return partitionIndexSpace[DataViewUtil::toInt(dw)];
        }

        Neon::domain::tool::PointHashTableSet<int, Meta>      map;
        OuterGrid                                             outerGrid;
        Neon::set::MemSet<typename OuterGrid::Cell::OuterIdx> tableToOuterIdx;

       private:
        std::array<Neon::set::DataSet<size_t>, Neon::DataViewUtil::nConfig> count;
        std::array<Neon::set::DataSet<sSpan>, Neon::DataViewUtil::nConfig>  partitionIndexSpace;
    };

    std::shared_ptr<sStorage> mStorage;
};
template <typename OuterGridT>
auto sGrid<OuterGridT>::getSetIdx(const index_3d&) const -> int
{
    NEON_DEV_UNDER_CONSTRUCTION("");
    return Neon::SetIdx();
}

}  // namespace Neon::domain::details::sGrid

#include "Neon/domain/details/sGrid/sFieldStorage_imp.h"
#include "Neon/domain/details/sGrid/sField_imp.h"
#include "Neon/domain/details/sGrid/sGrid_imp.h"
#include "Neon/domain/details/sGrid/sPartitionIndexSpace_imp.h"
#include "Neon/domain/details/sGrid/sPartition_imp.h"
