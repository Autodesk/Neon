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

#include "Neon/domain/internal/dGrid/dField.h"
#include "Neon/domain/internal/dGrid/dFieldDev.h"
#include "Neon/domain/internal/dGrid/dPartition.h"
#include "Neon/domain/internal/dGrid/dPartitionIndexSpace.h"


namespace Neon::domain::internal::dGrid {

/**
 * dGrid is the blueprint of creating dense field. It store the number of devices,
 * how data is distributed among them. User needs to create and instance of dGrid to
 * be able to create field. dGrid also manages launching kernels and exporting
 * fields to VTI
 */
class dGrid : public Neon::domain::interface::GridBaseTemplate<dGrid, dCell>
{
   public:
    using Grid = dGrid;
    using Cell = dCell;

    template <typename T_ta, int cardinality_ta = 0>
    using Field = dField<T_ta, cardinality_ta>;

    template <typename T_ta, int cardinality_ta = 0>
    using Partition = typename Field<T_ta, cardinality_ta>::Partition;

    using PartitionIndexSpace = dPartitionIndexSpace;

    using ngh_idx = typename Partition<int>::nghIdx_t;

    template <typename T, int C>
    friend class dFieldDev;

   public:
    dGrid();

    dGrid(const dGrid& rhs) = default;

    ~dGrid() = default;

    /**
     * Constructor compatible with the general grid API
     */
    template <typename ActiveCellLambda>
    dGrid(const Neon::Backend&         backend,
          const Neon::int32_3d&        dimension /**< Dimension of the box containing the sparse domain */,
          const ActiveCellLambda       activeCellLambda /**< InOrOutLambda({x,y,z}->{true, false}) */,
          const Neon::domain::Stencil& stencil,
          const Vec_3d<double>&        spacingData = Vec_3d<double>(1, 1, 1) /**< Spacing, i.e. size of a voxel */,
          const Vec_3d<double>&        origin = Vec_3d<double>(0, 0, 0) /**<      Origin  */);

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
    auto newField(const std::string   fieldUserName,
                  int                 cardinality,
                  T                   inactiveValue,
                  Neon::DataUse       dataUse = Neon::DataUse::IO_COMPUTE,
                  Neon::MemoryOptions memoryOptions = Neon::MemoryOptions()) const
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

    auto setReduceEngine(Neon::sys::patterns::Engine eng) -> void;

    template <typename T>
    auto newPatternScalar() const
        -> Neon::template PatternScalar<T>;

    template <typename T>
    auto dot(const std::string&               name,
             dField<T>&                       input1,
             dField<T>&                       input2,
             Neon::template PatternScalar<T>& scalar) const
        -> Neon::set::Container;

    template <typename T>
    auto norm2(const std::string&               name,
               dField<T>&                       input,
               Neon::template PatternScalar<T>& scalar) const
        -> Neon::set::Container;

    auto convertToNgh(const std::vector<Neon::index_3d>& stencilOffsets)
        -> std::vector<ngh_idx>;

    auto convertToNgh(const Neon::index_3d stencilOffsets) -> ngh_idx;

    auto getKernelConfig(int            streamIdx,
                         Neon::DataView dataView)
        -> Neon::set::KernelConfig;

    auto isInsideDomain(const Neon::index_3d& idx) const
        -> bool final;

    auto getProperties(const Neon::index_3d& idx) const
        -> GridBaseTemplate::CellProperties final;

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

    template <typename T_ta, int cardinality_ta = 0>
    auto newFieldDev(Neon::sys::memConf_t           memConf,
                     int                            cardinality,
                     [[maybe_unused]] T_ta          inactiveValue,
                     Neon::domain::haloStatus_et::e haloStatus)
        -> dFieldDev<T_ta, cardinality_ta>;

    void setKernelConfig(Neon::domain::KernelConfig& gridKernelConfig) const;


   private:
    using Self = dGrid;

    struct data_t
    {
        //  m_partitionDims indicates the size of each partition. For example,
        // given a gridDim of size 77 (in 1D for simplicity) distrusted over 5
        // device, it should be distributed as (16 16 15 15 15)
        Neon::set::DataSet<index_3d> partitionDims;

        Neon::index_3d                                       halo;
        std::vector<Neon::set::DataSet<PartitionIndexSpace>> partitionIndexSpaceVec;
        Neon::sys::patterns::Engine                          reduceEngine;
    };
    std::shared_ptr<data_t> m_data;
};

}  // namespace Neon::domain::internal::dGrid
#include "dFieldDev_imp.h"
#include "dField_imp.h"
#include "dGrid_imp.h"
