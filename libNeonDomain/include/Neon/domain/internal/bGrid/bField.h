#pragma once
#include "Neon/domain/interface/FieldBaseTemplate.h"
#include "Neon/domain/internal/bGrid/bPartition.h"
#include "Neon/set/patterns/BlasSet.h"

namespace Neon::domain::internal::bGrid {
class bGrid;


template <typename T, int C = 0>
class bField : public Neon::domain::interface::FieldBaseTemplate<T,
                                                                 C,
                                                                 bGrid,
                                                                 bPartition<T, C>,
                                                                 int>
{
    friend bGrid;

   public:
    using Type = T;
    using Grid = bGrid;
    using Field = bField;
    using Partition = bPartition<T, C>;
    using Cell = bCell;
    using ngh_idx = typename Partition::nghIdx_t;

    bField() = default;

    virtual ~bField() = default;

    virtual auto getPartition(const Neon::DeviceType& devType,
                              const Neon::SetIdx&     idx,
                              const Neon::DataView&   dataView = Neon::DataView::STANDARD) const -> const Partition& final;

    virtual auto getPartition(const Neon::DeviceType& devType,
                              const Neon::SetIdx&     idx,
                              const Neon::DataView&   dataView = Neon::DataView::STANDARD) -> Partition& final;

    auto getPartition(Neon::Execution,
                      Neon::SetIdx,
                      const Neon::DataView& dataView) const -> const Partition& final;

    auto getPartition(Neon::Execution,
                      Neon::SetIdx          idx,
                      const Neon::DataView& dataView) -> Partition& final;

    //TODO this should be implemented in FieldBase
    auto isInsideDomain(const Neon::index_3d& idx) const -> bool final;

    auto operator()(const Neon::index_3d& idx,
                    const int&            cardinality) const -> T final;

    auto getReference(const Neon::index_3d& idx,
                      const int&            cardinality) -> T& final;

    auto haloUpdate(Neon::set::HuOptions& opt) const -> void final;

    auto haloUpdate(Neon::set::HuOptions& opt) -> void final;

    auto updateIO(int streamId = 0) -> void final;

    auto updateCompute(int streamId = 0) -> void final;

    auto getSharedMemoryBytes(const int32_t stencilRadius) const -> size_t;

    auto dot(Neon::set::patterns::BlasSet<T>& blasSet,
             const bField<T>&                 input,
             Neon::set::MemDevSet<T>&         output,
             const Neon::DataView&            dataView) -> void;

    auto norm2(Neon::set::patterns::BlasSet<T>& blasSet,
               Neon::set::MemDevSet<T>&         output,
               const Neon::DataView&            dataView) -> void;


   private:
    bField(const std::string&             name,
           const bGrid&                   grid,
           int                            cardinality,
           T                              outsideVal,
           Neon::DataUse                  dataUse,
           const Neon::MemoryOptions&     memoryOptions,
           Neon::domain::haloStatus_et::e haloStatus);

    auto getRef(const Neon::index_3d& idx, const int& cardinality) const -> T&;

    enum PartitionBackend
    {
        cpu = 0,
        gpu = 1,
    };

    struct Data
    {
        std::shared_ptr<bGrid> mGrid;
        Neon::set::MemSet_t<T> mMem;
        int                    mCardinality;

        std::array<
            std::array<
                Neon::set::DataSet<Partition>,
                Neon::DataViewUtil::nConfig>,
            2>  //2 for host and device
            mPartitions;
    };
    std::shared_ptr<Data> mData;
};
}  // namespace Neon::domain::internal::bGrid

#include "Neon/domain/internal/bGrid/bField_imp.h"