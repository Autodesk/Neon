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

#include "Neon/domain/aGrid.h"

#include "Neon/domain/interface/GridBaseTemplate.h"
#include "Neon/domain/interface/GridConcept.h"
#include "Neon/domain/interface/KernelConfig.h"
#include "Neon/domain/interface/LaunchConfig.h"
#include "Neon/domain/interface/Stencil.h"
#include "Neon/domain/interface/common.h"

#include "Neon/domain/tools/SpanTable.h"

#include "Neon/domain/patterns/PatternScalar.h"

#include "dField.h"
#include "dPartition.h"
#include "dSpan.h"


namespace Neon::domain::details::dGrid {

/**
 * dGrid is the blueprint of creating dense field. It store the number of devices,
 * how data is distributed among them. User needs to create and instance of dGrid to
 * be able to create field. dGrid also manages launching kernels and exporting
 * fields to VTI
 */
class dGrid : public Neon::domain::interface::GridBaseTemplate<dGrid, dIndex>
{
   public:
    using Grid = dGrid;
    using Idx = dIndex;

    template <typename T, int CardinalityTa = 0>
    using Field = dField<T, CardinalityTa>;

    template <typename T, int CardinalityTa = 0>
    using Partition = typename Field<T, CardinalityTa>::Partition;

    using Span = dSpan;
    using NghIdx = typename Partition<int>::NghIdx;

    static constexpr Neon::set::details::ExecutionThreadSpan executionThreadSpan = Span::executionThreadSpan;
    using ExecutionThreadSpanIndexType = dSpan::ExecutionThreadSpanIndexType;

    template <typename T, int CardinalityTa>
    friend class dField;

   public:
    /**
     * Empty constructor
     */
    dGrid();

    /**
     * Copy constructor with a shallow copy semantic
     */
    dGrid(const dGrid& rhs) = default;

    /**
     * Destructor
     */
    virtual ~dGrid() = default;

    /**
     * Constructor compatible with the general grid API
     */
    template <Neon::domain::SparsityPattern SparsityPattern>
    dGrid(const Neon::Backend&         backend /**< Target for computation */,
          const Neon::int32_3d&        dimension /**< Dimension of the bounding box containing the domain */,
          const SparsityPattern&       activeCellLambda /**< InOrOutLambda({x,y,z}->{true, false}) */,
          const Neon::domain::Stencil& stencil /**< Stencil used by any computation on the grid */,
          const Vec_3d<double>&        spacing = Vec_3d<double>(1, 1, 1) /**< Spacing, i.e. size of a voxel */,
          const Vec_3d<double>&        origin = Vec_3d<double>(0, 0, 0) /**< Origin  */);

    /**
     * Returns a LaunchParameters configured for the specified inputs.
     * This methods used by the Container infrastructure.
     */
    auto getLaunchParameters(Neon::DataView        dataView,
                             const Neon::index_3d& blockSize,
                             const size_t&         shareMem) const
        -> Neon::set::LaunchParameters;

    /**
     * Method used by the Container infrastructure to retrieve the thread space
     */
    auto getSpan(Neon::Execution execution,
                 SetIdx          setIdx,
                 Neon::DataView  dataView)
        const -> const Span&;

    /**
     * Creates a new Field
     */
    template <typename T, int C = 0>
    auto newField(const std::string&  fieldUserName,
                  int                 cardinality,
                  T                   inactiveValue,
                  Neon::DataUse       dataUse = Neon::DataUse::HOST_DEVICE,
                  Neon::MemoryOptions memoryOptions = Neon::MemoryOptions()) const
        -> Field<T, C>;

    /**
     * Creates a new container running on this grid
     */
    template <Neon::Execution execution = Neon::Execution::device,
              typename LoadingLambda = void*>
    auto newContainer(const std::string& name,
                      index_3d           blockSize,
                      size_t             sharedMem,
                      LoadingLambda      lambda) const
        -> Neon::set::Container;

    /**
     * Creates a new container running on this grid
     */
    template <Neon::Execution execution = Neon::Execution::device,
              typename LoadingLambda = void*>
    auto newContainer(const std::string& name,
                      LoadingLambda      lambda)
        const
        -> Neon::set::Container;

    /**
     * Switch for different reduction engines.
     */
    auto setReduceEngine(Neon::sys::patterns::Engine eng)
        -> void;

    /**
     * Creation of a new scalar type that can store output from reduction operations
     */
    template <typename T>
    auto newPatternScalar()
        const -> Neon::template PatternScalar<T>;

    /**
     * creates a container implementing a dot product
     */
    template <typename T>
    auto dot(const std::string&               name,
             dField<T>&                       input1,
             dField<T>&                       input2,
             Neon::template PatternScalar<T>& scalar) const
        -> Neon::set::Container;

    /**
     * creates a container implementing a norm2 operation
     */
    template <typename T>
    auto norm2(const std::string&               name,
               dField<T>&                       input,
               Neon::template PatternScalar<T>& scalar,
               Neon::Execution                  execution) const
        -> Neon::set::Container;

    /**
     * Convert a list of 3d offsets for stencil operation in 1D local offsets
     */
    auto convertToNghIdx(std::vector<Neon::index_3d> const& stencilOffsets)
        const -> std::vector<NghIdx>;

    /**
     * Convert a list of 3d offsets for stencil operation in 1D local offsets
     */
    auto convertToNghIdx(Neon::index_3d const& stencilOffsets)
        const -> NghIdx;

    /**
     * The methods returns true if the the domain index has been flagged as active during initialization
     */
    auto isInsideDomain(const Neon::index_3d& idx)
        const -> bool final;

    auto getSetIdx(const Neon::index_3d& idx)
        const -> int32_t final;
    /**
     * Return the properties of a point
     */
    auto getProperties(const Neon::index_3d& idx) const
        -> GridBaseTemplate::CellProperties final;

   private:
    auto helpGetPartitionDim()
        const -> const Neon::set::DataSet<index_3d>;

    auto helpIdexPerPartition(Neon::DataView dataView = Neon::DataView::STANDARD)
        const -> const Neon::set::DataSet<int>;

    auto helpFieldMemoryAllocator()
        const -> const Neon::aGrid&;

    auto helpGetFirstZindex()
        const -> const Neon::set::DataSet<int32_t>&;

   private:
    struct Data
    {
        Data() = default;
        Data(const Neon::Backend& bk);

        //  partitionDims indicates the size of each partition. For example,
        // given a gridDim of size 77 (in 1D for simplicity) distrusted over 5
        // device, it should be distributed as (16 16 15 15 15)
        Neon::set::DataSet<index_3d>         partitionDims /** Bounding box size of each partition */;
        Neon::set::DataSet<index_t>          firstZIndex /** Lower z-index for each partition */;
        Neon::domain::tool::SpanTable<dSpan> spanTable /** Span for each data view configurations */;
        Neon::domain::tool::SpanTable<int>   elementsPerPartition /** Number of indexes for each partition */;

        Neon::index_3d              halo;
        Neon::sys::patterns::Engine reduceEngine;
        Neon::aGrid                 memoryGrid /** memory allocator for fields */;

        Neon::set::MemSet<Neon::int8_3d> stencilIdTo3dOffset;
    };

    std::shared_ptr<Data> mData;
};

}  // namespace Neon::domain::details::dGrid
#include "dField_imp.h"
#include "dGrid_imp.h"
