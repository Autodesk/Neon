#pragma once

#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"
#include "Neon/domain/interface/FieldBaseTemplate.h"
#include "Neon/set/DataConfig.h"
#include "Neon/set/DevSet.h"
#include "Neon/set/HuOptions.h"
#include "dFieldDev.h"
#include "dPartition.h"

namespace Neon::domain::internal::dGrid {

class dGrid;

/**
 * Create and manage a dense field on both GPU and CPU. dField also manages updating
 * the GPU->CPU and CPU-GPU as well as updating the halo. User can use dField to populate
 * the field with data as well was exporting it to VTI. To create a new dField,
 * use the newField function in dGrid.
 */

template <typename T, int C = 0>
class dField : public Neon::domain::interface::FieldBaseTemplate<T,
                                                                 C,
                                                                 dGrid,
                                                                 dPartition<T, C>,
                                                                 int>
{
    friend dGrid;

   public:
    static constexpr int Cardinality = C;
    using Type = T;
    using Self = dField<Type, Cardinality>;

    using Grid = dGrid;
    using Field = dField;
    using Partition = dPartition<T, C>;
    using FieldDev = dFieldDev<T, C>;
    using Cell = typename Partition::Cell;
    using ngh_idx = typename Partition::nghIdx_t;  // for compatibility with eGrid

    dField() = default;

    virtual ~dField() = default;

    auto self() -> Self&;

    auto self() const -> const Self&;

    /**
     * Returns the metadata associated with the element in location idx.
     * If the element is not active (it does not belong to the voxelized domain),
     * then the default outside value is returned.
     */
    auto operator()(const Neon::index_3d& idx,
                    const int&            cardinality) const
        -> Type final;

    auto haloUpdate(Neon::set::HuOptions& opt) const
        -> void final;

    auto haloUpdateContainer(Neon::set::TransferMode,
                             Neon::set::StencilSemantic)
        const -> Neon::set::Container final;

    auto haloUpdate(SetIdx setIdx, Neon::set::HuOptions& opt) const
        -> void;  // TODO add this function to the API if performance boost is reasonable -> void final;

    auto haloUpdate(Neon::set::HuOptions& opt)
        -> void final;

    auto haloUpdate(SetIdx setIdx, Neon::set::HuOptions& opt)
        -> void;  // TODO add this function to the API if performance boost is reasonable -> void final;

    virtual auto getReference(const Neon::index_3d& idx,
                              const int&            cardinality)
        -> Type& final;

    auto updateCompute(int streamSetId)
        -> void;

    auto updateIO(int streamSetId)
        -> void;

    auto getPartition(const Neon::DeviceType& devType,
                      const Neon::SetIdx&     idx,
                      const Neon::DataView&   dataView = Neon::DataView::STANDARD)
        const
        -> const Partition&;

    auto getPartition(const Neon::DeviceType& devType,
                      const Neon::SetIdx&     idx,
                      const Neon::DataView&   dataView = Neon::DataView::STANDARD)
        -> Partition&;

    /**
     * Return a constant reference to a specific partition based on a set of parameters:
     * execution type, target device, dataView
     */
    auto getPartition(Neon::Place  execution,
                      Neon::SetIdx          setIdx,
                      const Neon::DataView& dataView = Neon::DataView::STANDARD) const
        -> const Partition& final;
    /**
     * Return a reference to a specific partition based on a set of parameters:
     * execution type, target device, dataView
     */
    auto getPartition(Neon::Place  execution,
                      Neon::SetIdx          setIdx,
                      const Neon::DataView& dataView = Neon::DataView::STANDARD)
        -> Partition& final;

    auto dot(Neon::set::patterns::BlasSet<T>& blasSet,
             const dField<T>&                 input,
             Neon::set::MemDevSet<T>&         output,
             const Neon::DataView&            dataView = Neon::DataView::STANDARD) -> T;

    auto dotCUB(Neon::set::patterns::BlasSet<T>& blasSet,
                const dField<T>&                 input,
                Neon::set::MemDevSet<T>&         output,
                const Neon::DataView&            dataView = Neon::DataView::STANDARD) -> void;

    auto norm2(Neon::set::patterns::BlasSet<T>& blasSet,
               Neon::set::MemDevSet<T>&         output,
               const Neon::DataView&            dataView = Neon::DataView::STANDARD) -> T;

    auto norm2CUB(Neon::set::patterns::BlasSet<T>& blasSet,
                  Neon::set::MemDevSet<T>&         output,
                  const Neon::DataView&            dataView = Neon::DataView::STANDARD) -> void;

    static auto swap(Field& A, Field& B) -> void;

   private:
    auto deviceField(const Neon::Backend& backendConfig) -> FieldDev&;

    template <Neon::run_et::et runMode_ta = Neon::run_et::et::async>
    auto update(const Neon::set::StreamSet& streamSet, const Neon::DeviceType& devEt) -> void;

    auto updateCompute(const Neon::set::StreamSet& streamSet) -> void;


    auto updateIO(const Neon::set::StreamSet& streamSet)
        -> void;

    auto field(const Neon::DeviceType& devType) const -> const FieldDev&;

    auto field(Neon::Backend& backendConfig) const -> const FieldDev&;

    auto field(const Neon::DeviceType& devType) -> FieldDev&;

    auto field(Neon::Backend& backendConfig) -> FieldDev&;

    FieldDev& cpu();

    FieldDev& gpu();

    auto getLaunchInfo(const Neon::DataView dataView) const -> Neon::set::LaunchParameters;

    const FieldDev& ccpu() const;

    const FieldDev& cgpu() const;


    template <Neon::set::TransferMode transferMode_ta>
    auto haloUpdate(const Neon::Backend& bk,
                    bool                 startWithBarrier = true,
                    int                  streamSetIdx = 0)
        -> void;

    template <Neon::set::TransferMode transferMode_ta>
    auto haloUpdate(const Neon::Backend& bk,
                    int                  cardinality,
                    bool                 startWithBarrier = true,
                    int                  streamSetIdx = 0)
        -> void;


    dField(const std::string&                        fieldUserName,
           Neon::DataUse                             dataUse,
           const Neon::MemoryOptions&                memoryOptions,
           const Grid&                               grid,
           const Neon::set::DataSet<Neon::index_3d>& dims,
           int                                       zHaloDim,
           Neon::domain::haloStatus_et::e            haloStatus,
           int                                       cardinality);

    FieldDev            m_cpu;
    FieldDev            m_gpu;
    Neon::DataUse       mDataUse;
    Neon::MemoryOptions mMemoryOptions;
};



}  // namespace Neon::domain::internal::dGrid
