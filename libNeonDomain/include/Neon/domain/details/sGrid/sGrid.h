#pragma once

#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"

#include "Neon/sys/memory/MemDevice.h"

#include "Neon/set/BlockConfig.h"
#include "Neon/set/Containter.h"
#include "Neon/set/DataConfig.h"
#include "Neon/set/DevSet.h"

#include "Neon/domain/interface/GridBaseTemplate.h"
#include "Neon/domain/interface/KernelConfig.h"
#include "Neon/domain/interface/common.h"
#include "Neon/domain/tools/PointHashTableSet.h"

#include "Neon/domain/details/sGrid/sCell.h"
#include "Neon/domain/details/sGrid/sField.h"
#include "Neon/domain/details/sGrid/sFieldStorage.h"
#include "Neon/domain/details/sGrid/sPartition.h"
#include "Neon/domain/details/sGrid/sPartitionIndexSpace.h"

namespace Neon::domain::details::sGrid {

template <typename OuterGridT, typename T, int C>
class sField;

template <typename OuterGridT>
class sGrid : public Neon::domain::interface::GridBaseTemplate<sGrid<OuterGridT>,
                                                               sCell>
{
   public:
    using OuterGrid = OuterGridT;
    using Grid = sGrid;
    using Cell = sCell;

    template <typename OuterGridTK, typename T, int C>
    friend class sField;

    template <typename T, int C>
    using Partition = sPartition<OuterGridT, T, C>; /** Type of a partition for sGrid */

    template <typename T, int C>
    using Field = Neon::domain::details::sGrid::sField<OuterGrid, T, C>; /**< Type of a field for sGrid */

    using PartitionIndexSpace = Neon::domain::details::sGrid::sPartitionIndexSpace; /**< Type of the space is indexes for a lambda executor */


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
    auto getPartitionIndexSpace(Neon::DeviceType devE,
                                SetIdx           setIdx,
                                Neon::DataView   dataView) const
        -> const PartitionIndexSpace&;

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
    auto getContainer(const std::string& name,
                      LoadingLambda      lambda)
        const
        -> Neon::set::Container;

    /**
     * Creates a container that will run on this grid
     */
    template <typename LoadingLambda>
    auto getContainer(const std::string& name,
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
        -> typename Neon::domain::interface::GridBaseTemplate<sGrid<OuterGrid>, sCell>::CellProperties final;

   private:
    using Self = sGrid<OuterGrid>;
    using GridBaseTemplate = Neon::domain::interface::GridBaseTemplate<Self, sCell>;
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
        Meta(int64_t                                     offset,
             typename OuterGridT::Cell::OuterCell const& outerCell_)
            : cellOffset(offset), outerCell(outerCell_)
        {
        }
        int32_t                              cellOffset;
        typename OuterGridT::Cell::OuterCell outerCell;
    };

    struct sStorage
    {
        /** init storage stricture */
        auto init(const OuterGrid& outerGrid_)
        {
            outerGrid = outerGrid_;

            for (auto& dw : DataViewUtil::validOptions()) {
                getCount(dw) = outerGrid.getDevSet().template newDataSet<size_t>();
                getPartitionIndexSpace(dw) = outerGrid.getDevSet().template newDataSet<sPartitionIndexSpace>();
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
        auto getPartitionIndexSpace(DataView dw) -> Neon::set::DataSet<sPartitionIndexSpace>&
        {
            return partitionIndexSpace[DataViewUtil::toInt(dw)];
        }

        Neon::domain::tool::PointHashTableSet<int, Meta>         map;
        OuterGrid                                                outerGrid;
        Neon::set::MemSet<typename OuterGrid::Cell::OuterCell> tableToOuterCell;

       private:
        std::array<Neon::set::DataSet<size_t>, Neon::DataViewUtil::nConfig>               count;
        std::array<Neon::set::DataSet<sPartitionIndexSpace>, Neon::DataViewUtil::nConfig> partitionIndexSpace;
    };

    std::shared_ptr<sStorage> mStorage;
};

}  // namespace Neon::domain::details::sGrid

#include "Neon/domain/details/sGrid/sFieldStorage_imp.h"
#include "Neon/domain/details/sGrid/sField_imp.h"
#include "Neon/domain/details/sGrid/sGrid_imp.h"
#include "Neon/domain/details/sGrid/sPartitionIndexSpace_imp.h"
#include "Neon/domain/details/sGrid/sPartition_imp.h"
