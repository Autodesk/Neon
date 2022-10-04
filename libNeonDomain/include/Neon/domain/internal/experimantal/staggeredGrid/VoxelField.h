#pragma once

#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"
#include "Neon/domain/interface/FieldBaseTemplate.h"
#include "Neon/set/DataConfig.h"
#include "Neon/set/DevSet.h"
#include "Neon/set/HuOptions.h"

#include "Neon/domain/internal/experimantal/staggeredGrid/NodePartition.h"

namespace Neon::domain::internal::experimental::staggeredGrid::details {

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
struct StaggeredGrid;

/**
 * Create and manage a dense field on both GPU and CPU. VoxelField also manages updating
 * the GPU->CPU and CPU-GPU as well as updating the halo. User can use VoxelField to populate
 * the field with data as well was exporting it to VTI. To create a new VoxelField,
 * use the newField function in dGrid.
 */

template <typename BuildingBlockGridT, typename T, int C = 0>
class VoxelField : public Neon::domain::interface::FieldBaseTemplate<T,
                                                                          C,
                                                                          BuildingBlockGridT,
                                                                          NodePartition<BuildingBlockGridT, T, C>,
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
    using Self = VoxelField<typename BuildingBlocks::Grid, Type, Cardinality>;

    using Grid = StaggeredGrid<typename BuildingBlocks::Grid>;
    using Partition = NodePartition<typename BuildingBlocks::Grid, T, C>;
    using Node = NodeGeneric<typename BuildingBlocks::Grid>;
    using Element = VoxelGeneric<typename BuildingBlocks::Grid>;

    friend Grid;

    VoxelField() = default;

    virtual ~VoxelField() = default;

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

    auto getReference(const Neon::index_3d& idx,
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


    static auto swap(VoxelField& A, VoxelField& B) -> void;

   private:
    VoxelField(const std::string&                        fieldUserName,
                    Neon::DataUse                             dataUse,
                    const Neon::MemoryOptions&                memoryOptions,
                    const Grid&                               grid,
                    const Neon::set::DataSet<Neon::index_3d>& dims,
                    int                                       zHaloDim,
                    Neon::domain::haloStatus_et::e            haloStatus,
                    int                                       cardinality);

    typename BuildingBlocks::Grid::template Field<T, C> mBluidBlockFiled;
};



}  // namespace Neon::domain::internal::experimental::staggeredGrid
