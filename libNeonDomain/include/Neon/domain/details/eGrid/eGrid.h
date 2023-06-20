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

#include "Neon/domain/tools/Partitioner1D.h"
#include "Neon/domain/tools/SpanTable.h"

#include "Neon/domain/patterns/PatternScalar.h"

#include "eField.h"
#include "ePartition.h"
#include "eSpan.h"


namespace Neon::domain::details::eGrid {

/**
 * dGrid is the blueprint of creating dense field. It store the number of devices,
 * how data is distributed among them. User needs to create and instance of dGrid to
 * be able to create field. dGrid also manages launching kernels and exporting
 * fields to VTI
 */
class eGrid : public Neon::domain::interface::GridBaseTemplate<eGrid, eIndex>
{
   public:
    using Grid = eGrid;
    using Idx = eIndex;

    template <typename T, int CardinalityTa = 0>
    using Field = eField<T, CardinalityTa>;

    template <typename T, int CardinalityTa = 0>
    using Partition = typename Field<T, CardinalityTa>::Partition;

    using Span = eSpan;
    using NghIdx = typename Partition<int>::NghIdx;
    static constexpr Neon::set::details::ExecutionThreadSpan executionThreadSpan = Span::executionThreadSpan;
    using ExecutionThreadSpanIndexType = Span::ExecutionThreadSpanIndexType;

    template <typename T, int CardinalityTa>
    friend class eField;

   public:
    /**
     * Empty constructor
     */
    eGrid();

    /**
     * Copy constructor with a shallow copy semantic
     */
    eGrid(const eGrid& rhs) = default;

    /**
     * Destructor
     */
    virtual ~eGrid() = default;

    /**
     * Constructor compatible with the general grid API
     */
    template <typename SparsityPattern>
    eGrid(const Neon::Backend&         backend /**< Target for computation */,
          const Neon::int32_3d&        dimension /**< Dimension of the bounding box containing the domain */,
          const SparsityPattern&       activeCellLambda /**< InOrOutLambda({x,y,z}->{true, false}) */,
          const Neon::domain::Stencil& stencil /**< Stencil used by any computation on the grid */,
          const Vec_3d<double>&        spacing = Vec_3d<double>(1, 1, 1) /**< Spacing, i.e. size of a voxel */,
          const Vec_3d<double>&        origin = Vec_3d<double>(0, 0, 0) /**< Origin  */);

    eGrid(const Neon::Backend&               backend /**< Target for computation */,
          const Neon::int32_3d&              dimension /**< Dimension of the bounding box containing the domain */,
          Neon::domain::tool::Partitioner1D& partitioner,
          const Neon::domain::Stencil&       stencil /**< Stencil used by any computation on the grid */,
          const Vec_3d<double>&              spacing,
          const Vec_3d<double>&              origin);
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
     * Convert a 3d index into a SetId and eGrid::Index
     * The returned SetIdx component is set to invalid if the user provided idx is not active
     * @param idx
     * @return
     */
    auto helpGetSetIdxAndGridIdx(Neon::index_3d idx) const -> std::tuple<Neon::SetIdx, eIndex>;

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
             eField<T>&                       input1,
             eField<T>&                       input2,
             Neon::template PatternScalar<T>& scalar) const
        -> Neon::set::Container;

    /**
     * creates a container implementing a norm2 operation
     */
    template <typename T>
    auto norm2(const std::string&               name,
               eField<T>&                       input,
               Neon::template PatternScalar<T>& scalar,
               Neon::Execution                  execution) const
        -> Neon::set::Container;

    /**
     * Convert a list of 3d offsets for stencil operation in 1D local offsets
     */
    auto convertToNghIdx(
        std::vector<Neon::index_3d> const& stencilOffsets)
        const -> std::vector<NghIdx>;

    /**
     * Convert a list of 3d offsets for stencil operation in 1D local offsets
     */
    auto convertToNghIdx(
        Neon::index_3d const& stencilOffsets)
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
    auto getMemoryGrid()
        -> Neon::aGrid&;

    auto getConnectivityField()
        -> Neon::aGrid::Field<int32_t, 0>;

    auto getGlobalMappingField()
        -> Neon::aGrid::Field<index_3d, 0>;

    auto getStencil3dTo1dOffset()
        -> Neon::set::MemSet<int8_t>;

    auto getPartitioner()
        -> const tool::Partitioner1D&;

   private:
    struct Data
    {
        Data() = default;
        Data(Neon::Backend const& bk);

        //  partitionDims indicates the size of each partition. For example,
        // given a gridDim of size 77 (in 1D for simplicity) distrusted over 5
        // device, it should be distributed as (16 16 15 15 15)
        Neon::domain::tool::SpanTable<eSpan> spanTable /** Span for each data view configurations */;
        Neon::domain::tool::SpanTable<int>   elementsPerPartition /** Number of indexes for each partition */;

        Neon::domain::tool::Partitioner1D partitioner1D;
        Stencil                           stencil;
        Neon::sys::patterns::Engine       reduceEngine;
        Neon::aGrid                       memoryGrid /** memory allocator for fields */;

        Neon::set::MemSet<int8_t>       mStencil3dTo1dOffset;
        Neon::aGrid::Field<int32_t, 0>  mConnectivityAField;
        Neon::aGrid::Field<index_3d, 0> mGlobalMappingAField;
    };

    std::shared_ptr<Data> mData;
    const Neon::aGrid&    helpFieldMemoryAllocator() const;

   public:
    auto helpGetData() -> Data&;
};


}  // namespace Neon::domain::details::eGrid
#include "eField_imp.h"
#include "eGrid_imp.h"
