/**
//    # Neon aGrid
//
//    ---
//
//    ## Description
//
//    aGrid is an abbreviation for array-based Grid. It represents a 1D grid distributed over a set of accelerators.
//    Natively, `aGrid does not support stencil operations` as the other grids do.
//
//    ## What aGris can be used for?
//
//    aGrid is a valuable abstraction to extend the capabilities of other grids. For example, aGrid can store boundary
//    condition information associated only with a subset of elements of a domain. Such use of aGrid allows the
//    application to reduce the required memory as no space needs to be allocated for elements on the boundary. It also
//    allows the application to run specific kernels only on the boundary elements. However, the mapping between the
//    aGrid elements and the domain boundary elements is not handled automatically at this point.
//
//    ## Limitation and future plans
//
//    - aGrid does not expose any interface for stencil operation because design decisions. aGrid is more a tool to map
//    a higher dimensionality space into 1D. The user must entirely handle the mapping, and therefore aGrid does not
//    have any knowledge of it. In the future, we may consider adding a mechanism to aGrid to allow the user to share
//    the mapping information.
//
//
//    - Because aGrid does not support stencil operations, the definition of DataView for aGrid is not fully defined.
//    At the moment, only `standard` views are supported. In the future, we plan to extend aGrid with an API that can
//    allow users to manually create the DataView classification based on the relation of aGrid w.r.t. to other grids.
//
//    ## Internal design
//
//    The internal structure of aGrid is simple: it implements the Neon grid API on top of MemSet
**/
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

#include "Neon/domain/internal/aGrid/aField.h"
#include "Neon/domain/internal/aGrid/aFieldStorage.h"
#include "Neon/domain/internal/aGrid/aPartition.h"
#include "Neon/domain/internal/aGrid/aPartitionIndexSpace.h"

namespace Neon::domain::internal::aGrid {

template <typename T, int C>
class aField;

class aGrid : public Neon::domain::interface::GridBaseTemplate<aGrid, aCell>
{
   public:
    using Grid = aGrid;
    using Cell = aCell;

    template <typename T, int C>
    using Partition = aPartition<T, C>; /** Type of a partition for aGrid */

    template <typename T, int C>
    using Field = Neon::domain::internal::aGrid::aField<T, C>; /**< Type of a field for aGrid */

    using PartitionIndexSpace = Neon::domain::internal::aGrid::aPartitionIndexSpace; /**< Type of the space is indexes for a lambda executor */

    using Count = typename aPartition<char, 0>::count_t;

    /**
     * Default constructor
     */
    aGrid();

    /**
     * Constructor compatible with the general grid API
     */
    aGrid(const Neon::Backend&  backend,
          const Neon::int32_3d& dimension /**< Dimension of the box containing the sparse domain */,
          const int32_3d&       blockDim = int32_3d(256, 1, 1) /**< Default block size */,
          const Vec_3d<double>& spacingData = Vec_3d<double>(1, 1, 1) /**< Spacing, i.e. size of a voxel */,
          const Vec_3d<double>& origin = Vec_3d<double>(0, 0, 0) /**<      Origin  */);

    /**
     * Constructor specific for aGrid, where users can specify manually the size of each partition
     */
    aGrid(const Neon::Backend&              backend,
          const Neon::set::DataSet<size_t>& lenghts /**< Length of each vector stored on accelerator */,
          const int32_3d&                   blockDim = int32_3d(256, 1, 1) /**< Default block size */,
          const Vec_3d<double>&             spacingData = Vec_3d<double>(1, 1, 1) /**< Spacing, i.e. size of a voxel */,
          const Vec_3d<double>&             origin = Vec_3d<double>(0, 0, 0) /**< Origin  */);

    /**
     * Returns a LaunchParameters configured for the specified inputs
     */
    auto getLaunchParameters(Neon::DataView        dataView,
                             const Neon::index_3d& blockSize = Neon::index_3d(256, 1, 1),
                             size_t                shareMem = 0) const
        -> Neon::set::LaunchParameters;

    /**
     * Return the first index mapped to each partition
     * @return
     */
    auto getFirstIdxPerPartition() const
        -> const Neon::set::DataSet<size_t>&;

    /**
     * Returns the partition space that can be used by the lambda executor to run a Container
     */
    auto getPartitionIndexSpace(Neon::DeviceType devE,
                                SetIdx           setIdx,
                                Neon::DataView   dataView) const
        -> const aGrid::PartitionIndexSpace&;

    auto getKernelConfig(int            streamIdx,
                         Neon::DataView dataView) -> Neon::set::KernelConfig;

    /**
     * Creates a new Field
     */
    template <typename T, int C = 0>
    auto newField(const std::string   fieldUserName,
                  int                 cardinality,
                  T                   inactiveValue,
                  Neon::DataUse       dataUse = Neon::DataUse::IO_COMPUTE,
                  Neon::MemoryOptions memoryOptions = Neon::MemoryOptions()) const
        -> Field<T, C>;

    /**
     * Creates a container
     */
    template <typename LoadingLambda>
    auto getContainer(const std::string& name,
                      LoadingLambda      lambda)
        const
        -> Neon::set::Container;

    /**
     * Creates a container with the ability of specifying the block and shared memory size
     */
    template <typename LoadingLambda>
    auto getContainer(const std::string& name,
                      index_3d           blockSize,
                      size_t             sharedMem,
                      LoadingLambda      lambda)
        const
        -> Neon::set::Container;

    auto isInsideDomain(const Neon::index_3d& idx) const
        -> bool final;

    auto getProperties(const Neon::index_3d& idx) const
        -> CellProperties final;

   private:
    using GridBaseTemplate = Neon::domain::interface::GridBaseTemplate<aGrid, aCell>;

    using Self = aGrid;
    using Index = typename aPartition<char, 0>::index_t;

    /**
     * Internal helper function for initialization
     */
    auto init(const Neon::Backend&              backend,
              const Neon::int32_3d&             dimension,
              const Neon::set::DataSet<size_t>& lenghts,
              const int32_3d&                   blockDim,
              const Vec_3d<double>&             spacingData,
              const Vec_3d<double>&             origin)
        -> void;

    /**
     * Internal helper function to set KernelConfig structures
     */
    auto setKernelConfig(Neon::domain::KernelConfig& gridKernelConfig) const -> void;

    /**
     * Internal helper function to initialize default parameters
     */
    auto initDefaults() -> void;

    struct Storage
    {
        Neon::set::DataSet<PartitionIndexSpace> partitionSpaceSet;
        Neon::set::DataSet<size_t>              firstIdxPerPartition;
    };

    std::shared_ptr<Storage> mStorage;
};

#define AGRID_NEW_FIELD_EXPLICIT_INSTANTIATION(TYPE)                                       \
    extern template auto aGrid::newField<TYPE, 0>(const std::string   fieldUserName,       \
                                                  int                 cardinality,         \
                                                  TYPE                inactiveValue,       \
                                                  Neon::DataUse       dataUse,             \
                                                  Neon::MemoryOptions memoryOptions) const \
        ->Field<TYPE, 0>;

AGRID_NEW_FIELD_EXPLICIT_INSTANTIATION(double)
AGRID_NEW_FIELD_EXPLICIT_INSTANTIATION(float)
AGRID_NEW_FIELD_EXPLICIT_INSTANTIATION(int32_t)
AGRID_NEW_FIELD_EXPLICIT_INSTANTIATION(int64_t)

#undef AGRID_NEW_FIELD_EXPLICIT_INSTANTIATION
}  // namespace Neon::domain::internal::aGrid

#include "Neon/domain/internal/aGrid/aFieldStorage_imp.h"
#include "Neon/domain/internal/aGrid/aField_imp.h"
#include "Neon/domain/internal/aGrid/aGrid_imp.h"
#include "Neon/domain/internal/aGrid/aPartitionIndexSpace_imp.h"
#include "Neon/domain/internal/aGrid/aPartition_imp.h"
