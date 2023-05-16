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


template <typename T, int C, int8_t dataBlockSizeX, int8_t dataBlockSizeY, int8_t dataBlockSizeZ>
class bField : public Neon::domain::interface::FieldBaseTemplate<T,
                                                                 C,
                                                                 bGrid<dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ>,
                                                                 bPartition<T, C, dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ>,
                                                                 int>
{
    friend bGrid<dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ>;

   public:
    using Type = T;
    using Grid = bGrid<dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ>;
    using Field = bField<T, C, dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ>;
    using Partition = bPartition<T, C, dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ>;
    using Idx = bIndex<dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ>;

    using NghIdx = typename Partition::NghIdx;
    using NghData = typename Partition::NghData;

    static constexpr Neon::index_3d dataBlockSize3D = Neon::index_3d(dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ);

    static constexpr Neon::int8_3d DataBlockSize = Neon::int8_3d(dataBlockSizeX,
                                                                 dataBlockSizeY,
                                                                 dataBlockSizeZ);


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

        enum EndPoints
        {
            src = 1,
            dst = 0
        };

        struct EndPointsUtils
        {
            static constexpr int nConfigs = 2;
        };

        std::shared_ptr<Grid>      grid;
        BlockViewGrid::Field<T, C> memoryField;

        int mCardinality;

        //        Neon::domain::tool::HaloTable1DPartitioning   latticeHaloUpdateTable;
        Neon::domain::tool::HaloTable1DPartitioning soaHaloUpdateTable;
        //        Neon::domain::tool::HaloTable1DPartitioning   aosHaloUpdateTable;
        Neon::domain::tool::PartitionTable<Partition> partitionTable;
    };
    std::shared_ptr<Data> mData;
};


}  // namespace Neon::domain::details::bGrid