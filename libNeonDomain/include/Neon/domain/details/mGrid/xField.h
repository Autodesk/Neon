#pragma once

#include "Neon/domain/details/mGrid/mPartition.h"
#include "Neon/domain/interface/FieldBaseTemplate.h"

namespace Neon::domain::details::bGrid {
template <typename SBlock>
class bGrid;
}

namespace Neon::domain::details::mGrid {
/**
 *We have to define this field because we want a field that is a bField but gives a mPartition 
*/
class mGrid;

template <typename T, int C = 0>
class xField : public Neon::domain::interface::FieldBaseTemplate<T,
                                                                 C,
                                                                 Neon::domain::details::bGrid::bGrid<kStaticBlock>,
                                                                 mPartition<T, C>,
                                                                 int>

{
   public:
    using Partition = mPartition<T, C>;
    using Grid = Neon::domain::details::bGrid::bGrid<kStaticBlock>;
    using Field = typename Grid::Field<T, C>;


    xField() = default;

    xField(const std::string&         name,
           const Grid&                grid,
           int                        cardinality,
           T                          outsideVal,
           Neon::DataUse              dataUse,
           const Neon::MemoryOptions& memoryOptions);


    auto isInsideDomain(const Neon::index_3d& idx) const -> bool final;

    auto getReference(const Neon::index_3d& idx, const int& cardinality) -> T& final;

    auto operator()(const Neon::index_3d& idx, const int& cardinality) const -> T final;

    auto operator()(const Neon::index_3d& idx,
                    const int&            cardinality) -> T&;


    auto getPartition(const Neon::DeviceType& devType,
                      const Neon::SetIdx&     idx,
                      const Neon::DataView&   dataView) const -> const Partition&;

    auto getPartition(const Neon::DeviceType& devType,
                      const Neon::SetIdx&     idx,
                      const Neon::DataView&   dataView) -> Partition&;

    auto getPartition(Neon::Execution       exec,
                      Neon::SetIdx          idx,
                      const Neon::DataView& dataView) const -> const Partition& final;


    auto getPartition(Neon::Execution       exec,
                      Neon::SetIdx          idx,
                      const Neon::DataView& dataView) -> Partition& final;

    auto updateHostData(int streamId) -> void;

    auto updateDeviceData(int streamId) -> void;

    virtual ~xField() = default;


    enum PartitionBackend
    {
        cpu = 0,
        gpu = 1,
    };
    struct Data
    {
        Field field;

        std::array<
            std::array<
                Neon::set::DataSet<Partition>,
                Neon::DataViewUtil::nConfig>,
            Neon::ExecutionUtils::numConfigurations>
            mPartitions;
    };

    std::shared_ptr<Data> mData;
};
}  // namespace Neon::domain::details::mGrid

#include "Neon/domain/details/mGrid/xField_imp.h"