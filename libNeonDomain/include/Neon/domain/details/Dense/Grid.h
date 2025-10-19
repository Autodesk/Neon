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
#include "Neon/domain/interface/Representation.h"

#include "Neon/domain/interface/GridBaseTemplate.h"
#include "Neon/domain/interface/GridConcept.h"
#include "Neon/domain/interface/KernelConfig.h"
#include "Neon/domain/interface/LaunchConfig.h"
#include "Neon/domain/interface/Stencil.h"
#include "Neon/domain/interface/common.h"
#include "Neon/domain/tools/SpaceCurves.h"
#include "Neon/domain/tools/SpanTable.h"

#include "Neon/domain/patterns/PatternScalar.h"

#include "Neon/domain/details/Dense/Field.h"
#include "Neon/domain/details/Dense/Partition.h"
#include "Neon/domain/details/Dense/Span.h"


namespace Neon::domain::details::Dense {

/**
 * Dense Grid type - satisfies the Neon::domain::Grid concept
 * Dense Grid is the blueprint for creating dense fields. It stores the number of devices,
 * how data is distributed among them. Users need to create an instance of Dense Grid to
 * be able to create fields. Dense Grid also manages launching kernels and exporting
 * fields to VTI. This implementation provides a complete dense grid abstraction with
 * full concept compliance for generic grid programming.
 */
template <int Layout>
class Grid : public Neon::domain::interface::GridBaseTemplate<Grid<Layout>, Idx>
{
   public:
    constexpr static bool alphaBetaCapabilitySupported = false;
    using Self = Grid<Layout>;
    using GridBase = Neon::domain::interface::GridBaseTemplate<Grid<Layout>, Neon::domain::details::Dense::Idx>;
    using Idx = Neon::domain::details::Dense::Idx;

    template <typename T, int C = 0>
    using Field = Neon::domain::details::Dense::Field<Layout, T, C>;

    template <typename T, int C = 0>
    using Partition = typename Field<T, C>::Partition;

    using Span = Neon::domain::details::Dense::Span<Layout>;
    using SpanTable = Neon::domain::tool::SpanTable<Span>;

    using NghIdx = typename Partition<int>::NghIdx;

    using Representation = Neon::representation::Dense;
    static constexpr Neon::set::details::ExecutionThreadSpan executionThreadSpan = Span::executionThreadSpan;
    using ExecutionThreadSpanIndexType = Neon::domain::details::Dense::Span<Layout>::ExecutionThreadSpanIndexType;

    // Friend declarations moved outside class due to template specialization issues

   public:
    /**
     * Empty constructor
     */
    Grid();

    /**
     * Copy constructor with a shallow copy semantic
     */
    Grid(const Grid& rhs) = default;

    /**
     * Destructor
     */
    virtual ~Grid() = default;

    /**
     * Constructor compatible with the general grid API
     */
    template <typename SparsityPattern>
    Grid(const Neon::Backend&                         backend /**< Target for computation */,
         const Neon::int32_3d&                        dimension /**< Dimension of the bounding box containing the domain */,
         const SparsityPattern&                       activeCellLambda /**< InOrOutLambda({x,y,z}->{true, false}) */,
         const Neon::domain::Stencil&                 stencil /**< Stencil used by any computation on the grid */,
         const Vec_3d<double>&                        spacing = Vec_3d<double>(1, 1, 1) /**< Spacing, i.e. size of a voxel */,
         const Vec_3d<double>&                        origin = Vec_3d<double>(0, 0, 0) /**< Origin  */,
         Neon::domain::tool::spaceCurves::EncoderType encoderType = Neon::domain::tool::spaceCurves::EncoderType::sweep);

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
     * Returns access to the span table (required by Neon::domain::Grid concept)
     */
    auto getSpanTable() const -> const SpanTable&;

    /**
     * Creates a new Field
     */
    template <typename T, int C = 0>
    auto newField(const std::string&  fieldUserName,
                  int                 cardinality,
                  T                   inactiveValue,
                  Neon::DataUse       dataUse = Neon::DataUse::HOST_DEVICE,
                  Neon::MemoryOptions memoryOptions = Neon::MemoryOptions()) const
        -> Self::Field<T, C>;

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
        -> GridBase::CellProperties final;

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
        Neon::set::DataSet<index_3d>        partitionDims /** Bounding box size of each partition */;
        Neon::set::DataSet<index_t>         firstZIndex /** Lower z-index for each partition */;
        Neon::domain::tool::SpanTable<Span> spanTable /** Span for each data view configurations */;
        Neon::domain::tool::SpanTable<int>  elementsPerPartition /** Number of indexes for each partition */;

        Neon::index_3d              halo;
        Neon::sys::patterns::Engine reduceEngine;
        Neon::aGrid                 memoryGrid /** memory allocator for fields */;

        Neon::set::MemSet<Neon::int8_3d> stencilIdTo3dOffset;
    };

    std::shared_ptr<Data> mData;
};

}  // namespace Neon::domain::details::Dense
#include "Neon/domain/details/Dense/Field.imp.h"
#include "Neon/domain/details/Dense/Grid.imp.h"
