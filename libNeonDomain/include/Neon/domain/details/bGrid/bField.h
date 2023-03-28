#pragma once
#include "Neon/domain/details/bGrid/bPartition.h"
#include "Neon/domain/interface/FieldBaseTemplate.h"
#include "Neon/set/patterns/BlasSet.h"

namespace Neon::domain::details::bGrid {

class bGrid;

template <typename T, int C>
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
    using Field = bField<T, C>;
    using Partition = bPartition<T, C>;
    using Idx = bIndex;

    using NghIdx = typename Partition::NghIdx;
    using Ngh3DIdx = typename Partition::Ngh3DIdx;
    using Ngh1DIdx = typename Partition::Ngh1DIdx;
    using NghData = typename Partition::NghData;

    bField(const std::string&             name,
           const bGrid&                   grid,
           int                            cardinality,
           T                              outsideVal,
           Neon::DataUse                  dataUse,
           const Neon::MemoryOptions&     memoryOptions,
           Neon::domain::haloStatus_et::e haloStatus);

    bField() = default;

    virtual ~bField() = default;

    auto getPartition(Neon::Execution,
                      Neon::SetIdx,
                      const Neon::DataView& dataView) const -> const Partition& final;

    auto getPartition(Neon::Execution,
                      Neon::SetIdx,
                      const Neon::DataView& dataView) -> Partition& final;

    auto isInsideDomain(const Neon::index_3d& idx) const -> bool;


    auto operator()(const Neon::index_3d& idx,
                    const int&            cardinality) const -> T final;

    auto getReference(const Neon::index_3d& idx,
                      const int&            cardinality) -> T& final;


    auto haloUpdate(Neon::set::HuOptions& opt) const -> void final;

    auto haloUpdate(Neon::set::HuOptions& opt) -> void final;

    auto updateHostData(int streamId = 0) -> void final;

    auto updateDeviceData(int streamId = 0) -> void final;

    auto getSharedMemoryBytes(const int32_t stencilRadius) const -> size_t;

    auto getMem() -> Neon::set::MemSet<T>&;

    auto forEachActiveCell(const std::function<void(const Neon::index_3d&,
                                                    const int& cardinality,
                                                    T&)>&     fun,
                           Neon::computeMode_t::computeMode_e mode = Neon::computeMode_t::computeMode_e::par) -> void override;


   private:
    auto getRef(const Neon::index_3d& idx, const int& cardinality) const -> T&;


    enum PartitionBackend
    {
        cpu = 0,
        gpu = 1,
    };

    struct Data
    {

        std::shared_ptr<Grid> grid;

        Neon::set::MemSet<T> mem;

        int mCardinality;

        std::array<
            std::array<
                Neon::set::DataSet<Partition>,
                Neon::DataViewUtil::nConfig>,
            2>  // 2 for host and device
            partitions;
    };
    std::shared_ptr<Data> mData;
};
}  // namespace Neon::domain::details::bGrid

#include "Neon/domain/details/bGrid/bField_imp.h"