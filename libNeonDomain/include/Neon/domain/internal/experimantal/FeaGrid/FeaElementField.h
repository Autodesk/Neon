#pragma once

#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"
#include "Neon/domain/interface/FieldBaseTemplate.h"
#include "Neon/set/DataConfig.h"
#include "Neon/set/DevSet.h"
#include "Neon/set/HuOptions.h"

#include "Neon/domain/internal/experimantal/FeaGrid/FeaNodePartition.h"

namespace Neon::domain::internal::experimental::FeaVoxelGrid {

template <typename BuildingBlockGridT, typename T, int C = 0>
struct ElementStorage
{
    struct BuildingBlocks
    {
        using Grid = BuildingBlockGridT;
        using Field = typename Grid::template Field<T, C>;
        using Partition = typename BuildingBlockGridT::template Partition<T, C>;
    };

    typename BuildingBlocks::Field buildingBlockField;
};

template <typename BuildingBlockGridT>
struct FeaVoxelGrid;

/**
 * Create and manage a dense field on both GPU and CPU. FeaElementField also manages updating
 * the GPU->CPU and CPU-GPU as well as updating the halo. User can use FeaElementField to populate
 * the field with data as well was exporting it to VTI. To create a new FeaElementField,
 * use the newField function in dGrid.
 */

template <typename BuildingBlockGridT, typename T, int C = 0>
class FeaElementField : public Neon::domain::interface::FieldBaseTemplate<T,
                                                                          C,
                                                                          BuildingBlockGridT,
                                                                          FeaNodePartition<BuildingBlockGridT, T, C>,
                                                                          ElementStorage<BuildingBlockGridT, T, C>>
{

   public:
    struct BuildingBlocks
    {
        using Grid = BuildingBlockGridT;
        using Partition = typename BuildingBlockGridT::template Partition<T, C>;
    };


    static constexpr int Cardinality = C;
    using Type = T;
    using Self = FeaElementField<typename BuildingBlocks::Grid, Type, Cardinality>;

    using Grid = FeaVoxelGrid<typename BuildingBlocks::Grid>;
    using Partition = FeaNodePartition<typename BuildingBlocks::Grid, T, C>;
    using Node = FeaNode<typename BuildingBlocks::Grid>;
    using Element = FeaElement<typename BuildingBlocks::Grid>;

    friend Grid;

    FeaElementField() = default;

    virtual ~FeaElementField() = default;

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

    auto haloUpdate(SetIdx                setIdx,
                    Neon::set::HuOptions& opt) const
        -> void;

    auto haloUpdate(Neon::set::HuOptions& opt)
        -> void final;

    auto haloUpdate(SetIdx                setIdx,
                    Neon::set::HuOptions& opt)
        -> void;

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
    auto getPartition(Neon::Execution       execution,
                      Neon::SetIdx          setIdx,
                      const Neon::DataView& dataView = Neon::DataView::STANDARD) const
        -> const Partition& final;
    /**
     * Return a reference to a specific partition based on a set of parameters:
     * execution type, target device, dataView
     */
    auto getPartition(Neon::Execution       execution,
                      Neon::SetIdx          setIdx,
                      const Neon::DataView& dataView = Neon::DataView::STANDARD)
        -> Partition& final;


    static auto swap(FeaElementField& A, FeaElementField& B) -> void;

   private:
    FeaElementField(const std::string&                        fieldUserName,
                    Neon::DataUse                             dataUse,
                    const Neon::MemoryOptions&                memoryOptions,
                    const Grid&                               grid,
                    const Neon::set::DataSet<Neon::index_3d>& dims,
                    int                                       zHaloDim,
                    Neon::domain::haloStatus_et::e            haloStatus,
                    int                                       cardinality);

    typename BuildingBlocks::Grid::template Field<T, C> mBluidBlockFiled;
};


template <typename BuildingBlockGridT, typename T, int C>
auto FeaElementField<BuildingBlockGridT, T, C>::self() -> FeaElementField::Self&
{
    return *this;
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaElementField<BuildingBlockGridT, T, C>::self() const -> const FeaElementField::Self&
{
    return *this;
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaElementField<BuildingBlockGridT, T, C>::operator()(const index_3d& idx,
                                                           const int&      cardinality) const -> Type
{
    (void)idx;
    (void)cardinality;
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaElementField<BuildingBlockGridT, T, C>::haloUpdate(set::HuOptions& opt) const -> void
{
    return this->getStorage().buildingBlockField.haloUpdate(opt);
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaElementField<BuildingBlockGridT, T, C>::haloUpdate(SetIdx setIdx, set::HuOptions& opt) const -> void
{
    return this->getStorage().buildingBlockField.haloUpdate(setIdx, opt);
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaElementField<BuildingBlockGridT, T, C>::haloUpdate(set::HuOptions& opt) -> void
{
    return this->getStorage().buildingBlockField.haloUpdate(opt);
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaElementField<BuildingBlockGridT, T, C>::haloUpdate(SetIdx setIdx, set::HuOptions& opt) -> void
{
    return this->getStorage().buildingBlockField.haloUpdate(setIdx, opt);
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaElementField<BuildingBlockGridT, T, C>::getReference(const index_3d& idx, const int& cardinality) -> Type&
{
    return this->getStorage().buildingBlockField.getReference(idx, cardinality);
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaElementField<BuildingBlockGridT, T, C>::updateCompute(int streamSetId) -> void
{
    return this->getStorage().buildingBlockField.updateCompute(streamSetId);
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaElementField<BuildingBlockGridT, T, C>::updateIO(int streamSetId) -> void
{
    return this->getStorage().buildingBlockField.updateIO(streamSetId);
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaElementField<BuildingBlockGridT, T, C>::getPartition(const DeviceType& devType,
                                                             const SetIdx&     idx,
                                                             const DataView&   dataView) const -> const FeaElementField::Partition&
{
    (void)devType;
    (void)idx;
    (void)dataView;
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaElementField<BuildingBlockGridT, T, C>::getPartition(const DeviceType& devType,
                                                             const SetIdx&     idx,
                                                             const DataView&   dataView) -> FeaElementField::Partition&
{
    (void)devType;
    (void)idx;
    (void)dataView;

    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaElementField<BuildingBlockGridT, T, C>::getPartition(Neon::Execution execution,
                                                             Neon::SetIdx    setIdx,
                                                             const DataView& dataView) const -> const FeaElementField::Partition&
{
    (void)execution;
    (void)setIdx;
    (void)dataView;
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaElementField<BuildingBlockGridT, T, C>::getPartition(Neon::Execution execution,
                                                             Neon::SetIdx    setIdx,
                                                             const DataView& dataView) -> FeaElementField::Partition&
{
    (void)execution;
    (void)setIdx;
    (void)dataView;
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename BuildingBlockGridT, typename T, int C>
auto FeaElementField<BuildingBlockGridT, T, C>::swap(FeaElementField& A, FeaElementField& B) -> void
{
    (void)A;
    (void)B;
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename BuildingBlockGridT, typename T, int C>
FeaElementField<BuildingBlockGridT, T, C>::FeaElementField(const std::string&                  fieldUserName,
                                                           Neon::DataUse                       dataUse,
                                                           const MemoryOptions&                memoryOptions,
                                                           const FeaElementField::Grid&        grid,
                                                           const set::DataSet<Neon::index_3d>& dims,
                                                           int                                 zHaloDim,
                                                           Neon::domain::haloStatus_et::e      haloStatus,
                                                           int                                 cardinality)
{
    (void)fieldUserName;
    (void)dataUse;
    (void)memoryOptions;
    (void)grid;
    (void)dims;
    (void)zHaloDim;
    (void)haloStatus;
    (void)cardinality;
    NEON_DEV_UNDER_CONSTRUCTION("");
}

}  // namespace Neon::domain::internal::experimental::FeaVoxelGrid
