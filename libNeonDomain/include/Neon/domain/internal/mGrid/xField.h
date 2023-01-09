#pragma once

#include "Neon/domain/interface/FieldBaseTemplate.h"
#include "Neon/domain/internal/mGrid/mPartition.h"

namespace Neon::domain::internal::bGrid {
class bGrid;
}

namespace Neon::domain::internal::mGrid {
template <typename T, int C = 0>
class xField : public Neon::domain::interface::FieldBaseTemplate<T,
                                                                 C,
                                                                 Neon::domain::internal::bGrid::bGrid,
                                                                 mPartition<T, C>,
                                                                 int>

{
   public:
    using Field = typename Neon::domain::internal::bGrid::bField<T, C>;
    using Partition = typename Neon::domain::internal::mGrid::mPartition<T, C>;
    using Grid = Neon::domain::internal::bGrid::bGrid;


    xField() = default;

    xField(const std::string&         name,
           const Grid&                grid,
           int                        cardinality,
           T                          outsideVal,
           Neon::DataUse              dataUse,
           const Neon::MemoryOptions& memoryOptions);


    auto isInsideDomain(const Neon::index_3d& idx) const -> bool final;

    auto getReference(const Neon::index_3d& idx, const int& cardinality) -> T& final;


    auto haloUpdate(Neon::set::HuOptions& opt) const -> void final;

    auto haloUpdate(Neon::set::HuOptions& opt) -> void final;

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

    auto updateIO(int streamId) -> void;

    auto updateCompute(int streamId) -> void;

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
            2>  //2 for host and device
            mPartitions;
    };

    std::shared_ptr<Data> mData;
};
}  // namespace Neon::domain::internal::mGrid

#include "Neon/domain/internal/mGrid/xField_imp.h"