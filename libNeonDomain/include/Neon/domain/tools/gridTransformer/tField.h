#pragma once
#include "Neon/domain/interface/FieldBaseTemplate.h"
#include "Neon/domain/tools/PartitionTable.h"

namespace Neon::domain::tool::details {

template <typename GridTransformation>
class tGrid;


template <typename T, int C, typename GridTransformation>
class tField : public Neon::domain::interface::FieldBaseTemplate<T,
                                                                 C,
                                                                 tGrid<GridTransformation>,
                                                                 typename GridTransformation::template Partition<T, C>,
                                                                 int>
{
    friend tGrid<GridTransformation>;

   public:
    static constexpr int Cardinality = C;
    using Type = T;
    using Self = tField<Type, Cardinality, GridTransformation>;
    using Grid = tGrid<GridTransformation>;
    using Field = tField<Type, Cardinality, GridTransformation>;
    using Partition = typename GridTransformation::template Partition<T, C>;
    using Idx = typename Partition::Idx;
    using NghIdx = typename Partition::NghIdx;  // for compatibility with eGrid

   private:
    using FoundationGrid = typename GridTransformation::FoundationGrid;
    using FoundationField = typename GridTransformation::FoundationGrid::template Field<T, C>;
    using FieldBaseTemplate = Neon::domain::interface::FieldBaseTemplate<T,
                                                                         C,
                                                                         tGrid<GridTransformation>,
                                                                         typename GridTransformation::template Partition<T, C>,
                                                                         int>;

   public:
    tField()
    {
        mData = std::make_shared<Data>();
    }
    ~tField() = default;

   private:
    explicit tField(FoundationField foundationField, const Grid& grid)
        : Neon::domain::interface::FieldBaseTemplate<T, C, Grid, Partition, int>(&grid,
                                                                                 foundationField.getName(),
                                                                                 "tField_" + foundationField.getClassName(),
                                                                                 foundationField.getCardinality(),
                                                                                 foundationField.getOutsideValue(),
                                                                                 foundationField.getDataUse(),
                                                                                 foundationField.getMemoryOptions(),
                                                                                 Neon::domain::haloStatus_et::e::ON) {
        mData = std::make_shared<Data>();
        const Neon::Backend& bk = foundationField.getGrid().getBackend();
        mData->init(bk);
        mData->foundationField = foundationField;
        GridTransformation::initFieldPartition(mData->foundationField, NEON_OUT mData->partitions);
    }

    public :
        /**
         * Returns the metadata associated with the element in location idx.
         * If the element is not active (it does not belong to the voxelized domain),
         * then the default outside value is returned.
         */
        auto
        operator()(const Neon::index_3d& idx,
                   const int&            cardinality) const
        -> Type final
    {
        return mData->foundationField.operator()(idx, cardinality);
    }

    auto newHaloUpdate(Neon::set::StencilSemantic semantic,
                       Neon::set::TransferMode    transferMode,
                       Neon::Execution            execution)
        const -> Neon::set::Container
    {
        return mData->foundationField.newHaloUpdate(semantic, transferMode, execution);
    }

    virtual auto getReference(const Neon::index_3d& idx,
                              const int&            cardinality)
        -> Type& final
    {
        return mData->foundationField.getReference(idx, cardinality);
    }

    auto updateDeviceData(int streamSetId)
        -> void
    {
        mData->foundationField.updateDeviceData(streamSetId);
    }

    auto updateHostData(int streamSetId)
        -> void
    {
        mData->foundationField.updateHostData(streamSetId);
    }

    /**
     * Return a constant reference to a specific partition based on a set of parameters:
     * execution type, target device, dataView
     */
    auto getPartition(Neon::Execution       execution,
                      Neon::SetIdx          setIdx,
                      const Neon::DataView& dataView = Neon::DataView::STANDARD) const
        -> const Partition& final
    {
        const auto& partition = mData->partitions.getPartition(execution, setIdx, dataView);
        return partition;
    }
    /**
     * Return a reference to a specific partition based on a set of parameters:
     * execution type, target device, dataView
     */
    auto getPartition(Neon::Execution       execution,
                      Neon::SetIdx          setIdx,
                      const Neon::DataView& dataView = Neon::DataView::STANDARD)
        -> Partition& final
    {
        auto& partition = mData->partitions.getPartition(execution, setIdx, dataView);
        return partition;
    }

    static auto swap(Field& A, Field& B) -> void
    {
        FieldBaseTemplate::swapUIDBeforeFullSwap(A, B);
        std::swap(A, B);
    }

   private:
    struct Data
    {
        Data() = default;
        auto init(Neon::Backend const& bk) -> void
        {
            partitions.init(bk);
        }
        FoundationField                               foundationField;
        Neon::domain::tool::PartitionTable<Partition> partitions;
    };

    std::shared_ptr<Data> mData;
};

}  // namespace Neon::domain::tool::details