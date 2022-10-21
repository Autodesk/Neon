#pragma once
#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"
#include "Neon/core/types/memSetOptions.h"

#include "Neon/sys/memory/MemDevice.h"

#include "Neon/set/Capture.h"
#include "Neon/set/Containter.h"
#include "Neon/set/DataConfig.h"
#include "Neon/set/DevSet.h"

#include "Neon/domain/interface/GridBaseTemplate.h"
#include "Neon/domain/interface/KernelConfig.h"
#include "Neon/domain/interface/LaunchConfig.h"
#include "Neon/domain/interface/Stencil.h"
#include "Neon/domain/interface/common.h"
#include "Neon/domain/internal/eGrid/eInternals/dsBuilder.h"
#include "Neon/domain/patterns/PatternScalar.h"
#include "ePartition.h"

namespace Neon::domain::internal::eGrid {

template <typename T, int C>
class eField;
struct eStorage;

class eGrid : public Neon::domain::interface::GridBaseTemplate<eGrid, eCell>
{
   public:
    using Self = eGrid;
    using Cell = eCell;

    template <typename T, int C = 0>
    using Field = eField<T, C>;

    template <typename T, int C = 0>
    using Partition = Neon::domain::internal::eGrid::ePartition<T, C>;

    using PartitionIndexSpace = ePartitionIndexSpace;
    using ngh_idx = typename Partition<int>::nghIdx_t;

   public:
    /**
     * Default constructor
     */
    eGrid();

    /**
     * Constructor for an eGrid object
     * TODO: include a list of stencil instead of only one
     *
     * @param backend
     * @param cellDomain: size of the dense background grid
     * @param implicitF: a sparsity map defined over the background grid. True means the element is active
     * @param stencil: stencil that will be used over the grid.
     * @param includeInveseMappingField
     */
    template <typename ActiveCellLambda>
    eGrid(const Neon::Backend&         backend,
          const Neon::index_3d&        cellDomain,
          const ActiveCellLambda&      activeCellLambda,
          const Neon::domain::Stencil& stencil,
          const Vec_3d<double>&        spacingData = Vec_3d<double>(1, 1, 1) /**< Spacing, i.e. size of a voxel */,
          const Vec_3d<double>&        origin = Vec_3d<double>(0, 0, 0) /**<      Origin  */,
          bool                         includeInveseMappingField = false);

    /**
     * Returns a LaunchParameters configured for the specified inputs
     */
    auto getLaunchParameters(Neon::DataView        dataView,
                             const Neon::index_3d& blockSize = Neon::index_3d(256, 1, 1),
                             const size_t&         shareMem = 0) const
        -> Neon::set::LaunchParameters;

    auto getPartitionIndexSpace(Neon::DeviceType devE,
                                SetIdx           setIdx,
                                Neon::DataView   dataView) -> const PartitionIndexSpace&;

    /**
     * Creates a new Field
     */
    template <typename T, int C = 0>
    auto newField(const std::string&  fieldUserName,
                  int                 cardinality,
                  T                   inactiveValue,
                  Neon::DataUse       dataUse = Neon::DataUse::IO_COMPUTE,
                  Neon::MemoryOptions memoryOptions = Neon::MemoryOptions()) const
        -> Field<T, C>;


    template <typename LoadingLambda>
    auto getContainer(const std::string& name,
                      index_3d           blockSize,
                      size_t             sharedMem,
                      LoadingLambda      lambda)
        const
        -> Neon::set::Container;

    template <typename LoadingLambda>
    auto getContainer(const std::string& name, LoadingLambda lambda)
        const
        -> Neon::set::Container;

    template <typename T>
    auto newPatternScalar() const
        -> Neon::template PatternScalar<T>;

    template <typename T, int C>
    auto dot(const std::string& /*name*/,
             eField<T, C>& /*input1*/,
             eField<T, C>& /*input2*/,
             Neon::template PatternScalar<T>& /*scalar*/) const -> Neon::set::Container;

    template <typename T, int C>
    auto norm2(const std::string& /*name*/,
               eField<T, C>& /*input*/,
               Neon::template PatternScalar<T>& /*scalar*/) const -> Neon::set::Container;
       

    auto convertToNgh(const std::vector<Neon::index_3d>& stencilOffsets)
        -> std::vector<ngh_idx>;

    auto convertToNgh(const Neon::index_3d& stencilOffset)
        -> ngh_idx;

    auto getKernelConfig(int            streamIdx,
                         Neon::DataView dataView) -> Neon::set::KernelConfig;

    auto isInsideDomain(const Neon::index_3d& idx) const
        -> bool final;

    auto getProperties(const Neon::index_3d& idx)
        const -> GridBaseTemplate::CellProperties final;

    auto setReduceEngine(Neon::sys::patterns::Engine eng)
        -> void;

   private:
    using GridBaseTemplate = Neon::domain::interface::GridBaseTemplate<eGrid, eCell>;

    /**
     * Return a reference to this object
     * @return
     */
    inline auto cself() const -> const Self&;


    /**
     * Returns for each partition the number of elements based on the requested indexing schema
     * @tparam indexing_ta
     * @return
     */
    auto nElements(Neon::DataView dataView = Neon::DataView::STANDARD)
        const
        -> Neon::set::DataSet<count_t>;
    /**
     * Returns the number of active elements
     * @return
     */
    auto numElements() -> int64_t;

    /**
     * Return the size of elements stores by each partition
     * @return
     */
    auto flattenedLengthSet(Neon::DataView dataView);

    /**
     * Returns the frame of the grid
     * @return
     */
    auto frame() const
        -> const internals::dsFrame_t*;

    auto helpGetMds() -> eStorage&;

    auto helpGetMds() const -> const eStorage&;


    /**
     * Returns a new StreamSet distributed across the devices
     **/
    auto newStreamSet() const
        -> Neon::set::StreamSet;

    /**
     * Returning a new StreamSet distributed across the devices
     * @return
     **/
    auto newLaunchParameters() const
        -> Neon::set::LaunchParameters;

    auto setKernelConfig(Neon::set::KernelConfig& kernelConfig) const
        -> void;

    auto setKernelConfig(Neon::domain::KernelConfig& gridKernelConfig) const
        -> void;

    /**
     * Private function to set the default size of CUDA blocks
     * @param blockDim
     */
    auto helpSetDefaultBlock()
        -> void;

    std::shared_ptr<eStorage> m_ds;


};  // namespace eGrid

}  // namespace Neon::domain::internal::eGrid

#include "ePartitionIndexSpace.h"
#include "ePartitionIndexSpace_imp.h"

//#include "Neon/domain/sparse/eGrid/ePartition_imp.h"
#include "eField.h"
#include "eField_imp.h"
#include "eGridStorage.h"
#include "eGrid_imp.h"
