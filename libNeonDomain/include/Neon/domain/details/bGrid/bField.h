#pragma once
#include "Neon/domain/details/bGrid/bPartition.h"
#include "Neon/domain/interface/FieldBaseTemplate.h"
#include "Neon/set/patterns/BlasSet.h"

#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"

#include "Neon/set/DevSet.h"
#include "Neon/set/HuOptions.h"
#include "Neon/set/MemoryTransfer.h"

#include "Neon/domain/interface/FieldBaseTemplate.h"
#include "Neon/domain/tools/HaloUpdateTable1DPartitioning.h"
#include "Neon/domain/tools/PartitionTable.h"
#include "bPartition.h"

namespace Neon::domain::details::bGrid {


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

    bField(const std::string&         fieldUserName,
           Neon::DataUse              dataUse,
           const Neon::MemoryOptions& memoryOptions,
           const Grid&                grid,
           int                        cardinality,
           T                          inactiveValue);

    bField();

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

    auto updateHostData(int streamId = 0) -> void final;

    auto updateDeviceData(int streamId = 0) -> void final;

    auto newHaloUpdate(Neon::set::StencilSemantic semantic,
                       Neon::set::TransferMode    transferMode,
                       Neon::Execution            execution)
        const -> Neon::set::Container;

   private:
    auto getRef(const Neon::index_3d& idx, const int& cardinality) const -> T&;

    auto initHaloUpdateTable() -> void;


    //
    //    enum PartitionBackend
    //    {
    //        cpu = 0,
    //        gpu = 1,
    //    };

    struct Data
    {
        Data() = default;
        Data(Neon::Backend const& bk)
        {
            partitionTable.init(bk);
        }

        std::shared_ptr<Grid>      grid;
        BlockViewGrid::Field<T, C> memoryField;

        int mCardinality;

        Neon::domain::tool::HaloTable1DPartitioning   latticeHaloUpdateTable;
        Neon::domain::tool::HaloTable1DPartitioning   soaHaloUpdateTable;
        Neon::domain::tool::HaloTable1DPartitioning   aosHaloUpdateTable;
        Neon::domain::tool::PartitionTable<Partition> partitionTable;
    };
    std::shared_ptr<Data> mData;
};

}  // namespace Neon::domain::details::bGrid
