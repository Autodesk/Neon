#pragma once
#include "Neon/domain/details/bGridDisg/bDisgPartition.h"
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
#include "bDisgPartition.h"

namespace Neon::domain::details::disaggregated::bGridDisg {


template <typename T, int C, typename SBlock>
class bFieldDisg : public Neon::domain::interface::FieldBaseTemplate<T,
                                                                 C,
                                                                 bGridDisg<SBlock>,
                                                                 bDisgPartition<T, C, SBlock>,
                                                                 int>
{
    friend bGridDisg<SBlock>;

   public:
    using Type = T;
    using Grid = bGridDisg<SBlock>;
    using Field = bFieldDisg<T, C, SBlock>;
    using Partition = bDisgPartition<T, C, SBlock>;
    using Idx = bDisgIndex<SBlock>;
    using BlockViewGrid = Neon::domain::tool::GridTransformer<details::GridTransformation>::Grid;
    template <typename TT, int CC = 0>
    using BlockViewField = BlockViewGrid::template Field<TT, CC>;

    using NghIdx = typename Partition::NghIdx;
    using NghData = typename Partition::NghData;

    bFieldDisg(const std::string&  fieldUserName,
           Neon::DataUse       dataUse,
           Neon::MemoryOptions memoryOptions,
           const Grid&         grid,
           int                 cardinality,
           T                   inactiveValue);

    bFieldDisg();

    virtual ~bFieldDisg() = default;

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

    auto getMemoryField() -> BlockViewGrid::Field<T, C>&;


   private:
    auto getRef(const Neon::index_3d& idx, const int& cardinality) const -> T&;

    auto initHaloUpdateTable() -> void;


    struct Data
    {
        Data() = default;
        Data(Neon::Backend const& bk)
        {
            partitionTable.init(bk);
        }

        enum EndPoints
        {
            src = 1,
            dst = 0
        };

        struct EndPointsUtils
        {
            static constexpr int nConfigs = 2;
        };

        std::shared_ptr<Grid> grid;
        BlockViewField<T, 0>  memoryField;
        int                   cardinality;

        Neon::domain::tool::HaloTable1DPartitioning   mStandardHaloUpdateTable;
        Neon::domain::tool::PartitionTable<Partition> partitionTable;
    };
    std::shared_ptr<Data> mData;
};


}  // namespace Neon::domain::details::disaggregated::bGrid