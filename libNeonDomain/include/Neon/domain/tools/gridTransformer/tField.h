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
        mStroage = std::make_shared<Storage>();
    }
    ~tField() = default;

   private:
    tField(const std::string&                        fieldUserName,
           Neon::DataUse                             dataUse,
           const Neon::MemoryOptions&                memoryOptions,
           const Grid&                               grid,
           const Neon::set::DataSet<Neon::index_3d>& dims,
           int                                       cardinality)
    {
        Neon::Backend& bk = grid.getBackend();
        mStroage = std::make_shared<Storage>(bk);
        auto& foundationGrid = grid.getFoundationGrid();
        mStroage->foundationField = foundationGrid.template newField<T, C>(fieldUserName, dataUse, memoryOptions, dims, cardinality);
        GridTransformation::initFieldPartition(mStroage->foundationField, NEON_OUT mStroage->partitions);
        NEON_DEV_UNDER_CONSTRUCTION("");
    }

    explicit tField(FoundationField foundationField)
    {
        Neon::Backend& bk = foundationField.getGrid.getBackend();
        mStroage->foundationField = foundationField;
        GridTransformation::initFieldPartition(mStroage->foundationField, NEON_OUT mStroage->partitions);
        NEON_DEV_UNDER_CONSTRUCTION("");
    }

   public:
    /**
     * Returns the metadata associated with the element in location idx.
     * If the element is not active (it does not belong to the voxelized domain),
     * then the default outside value is returned.
     */
    auto operator()(const Neon::index_3d& idx,
                    const int&            cardinality) const
        -> Type final;

    auto haloUpdate(Neon::set::HuOptions& opt) const
        -> void final
    {
        mStroage->foundationField.haloUpdate(opt);
    }

    auto haloUpdate(SetIdx setIdx, Neon::set::HuOptions& opt) const
        -> void
    {
        mStroage->foundationField.haloUpdate(setIdx, opt);
    }

    auto haloUpdate(Neon::set::HuOptions& opt)
        -> void final
    {
        mStroage->foundationField.haloUpdate(opt);
    }

    auto haloUpdate(SetIdx setIdx, Neon::set::HuOptions& opt)
        -> void
    {
        mStroage->foundationField.haloUpdate(setIdx, opt);
    }

    virtual auto getReference(const Neon::index_3d& idx,
                              const int&            cardinality)
        -> Type& final
    {
        return mStroage->foundationField.getReference(idx, cardinality);
    }

    auto updateCompute(int streamSetId)
        -> void
    {
        mStroage->foundationField.updateCompute(streamSetId);
    }

    auto updateIO(int streamSetId)
        -> void
    {
        mStroage->foundationField.updateIO(streamSetId);
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
        const auto& partition = mStroage->partitions.getPartition(execution, dataView, setIdx);
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
        auto& partition = mStroage->partitions.getPartition(execution, dataView, setIdx);
        return partition;
    }

    static auto swap(Field& A, Field& B) -> void
    {
        FieldBaseTemplate::swapUIDBeforeFullSwap(A, B);
        std::swap(A, B);
    }

   private:
    struct Storage
    {
        Storage() = default;
        explicit Storage(Neon::Backend& bk)
        {
            partitions = Neon::domain::tool::PartitionTable<Partition>(bk);
        }
        FoundationField                               foundationField;
        Neon::domain::tool::PartitionTable<Partition> partitions;
    };

    std::shared_ptr<Storage> mStroage;
};

}  // namespace Neon::domain::tool::details