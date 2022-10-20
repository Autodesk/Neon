#pragma once

#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"
#include "Neon/domain/interface/FieldBaseTemplate.h"
#include "Neon/set/DataConfig.h"
#include "Neon/set/DevSet.h"
#include "Neon/set/HuOptions.h"

#include "Neon/domain/internal/experimental/staggeredGrid/voxel/VoxelGrid.h"
#include "Neon/domain/internal/experimental/staggeredGrid/voxel/VoxelPartition.h"


namespace Neon::domain::internal::experimental::staggeredGrid::details {

template <typename BuildingBlockGridT>
struct VoxelGrid;

template <typename BuildingBlockGridT, typename T, int C = 0>
struct VoxelStorage
{
   private:
    struct BuildingBlocks
    {
        using Grid = BuildingBlockGridT;
        using Field = typename Grid::template Field<T, C>;
        using Partition = typename BuildingBlockGridT::template Partition<T, C>;
        using FieldNodeToVoxelMask = typename BuildingBlockGridT::template Field<NodeToVoxelMask, 1>;
        using PartitionNodeToVoxelMask = typename BuildingBlockGridT::template Partition<NodeToVoxelMask, 1>;
    };
    using Partition = VoxelPartition<BuildingBlockGridT, T, C>;

   public:
    VoxelStorage() = default;

    VoxelStorage(typename BuildingBlocks::Field&                      buildingBlocksField,
                 const typename BuildingBlocks::FieldNodeToVoxelMask& nodeToVoxelMaskField,
                 Neon::DataUse                                        dataUse)
        : mBuildingBlockField(buildingBlocksField), mNodeToVoxelMaskField(nodeToVoxelMaskField), mDataUse(dataUse)
    {
        auto& bk = buildingBlocksField.getGrid().getBackend();
        {  // Setting up the mask for supported executions (i.e host and device | host only | device only)
            for (Neon::Execution execution : Neon::ExecutionUtils::getAllOptions()) {
                mSupportedExecutions[ExecutionUtils::toInt(execution)] = false;
            }
            for (Neon::Execution execution : Neon::ExecutionUtils::getCompatibleOptions(dataUse)) {
                mSupportedExecutions[ExecutionUtils::toInt(execution)] = true;
            }
        }

        {  // Setting up the mask for supported executions (i.e host and device | host only | device only)
            for (Neon::Execution execution : Neon::ExecutionUtils::getCompatibleOptions(dataUse)) {
                for (auto dw : Neon::DataViewUtil::validOptions()) {
                    getPartitionDataSet(execution, dw) = bk.devSet().template newDataSet<VoxelPartition<BuildingBlockGridT, T, C>>();
                    for (Neon::SetIdx setIdx : bk.devSet().getRange()) {
                        const typename BuildingBlocks::Partition&                buildingBlocksPartition = mBuildingBlockField.getPartition(execution, setIdx.idx(), dw);
                        const typename BuildingBlocks::PartitionNodeToVoxelMask& nodeToVoxelMaskPartition = mNodeToVoxelMaskField.getPartition(execution, setIdx.idx(), dw);


                        this->getPartition(execution, dw, setIdx.idx()) =
                            VoxelPartition<BuildingBlockGridT, T, C>(buildingBlocksPartition,
                                                                     nodeToVoxelMaskPartition);
                    }
                }
            }
        }
    }

    auto getPartition(Neon::Execution execution, Neon::DataView dw, Neon::SetIdx setIdx)
        -> VoxelPartition<BuildingBlockGridT, T, C>&
    {
        int   dwInt = Neon::DataViewUtil::toInt(dw);
        int   executionInt = Neon::ExecutionUtils::toInt(execution);
        auto& output = mPartitionsByExecutionViewAndDevIdx[executionInt][dwInt][setIdx.idx()];
        return output;
    }

    auto getPartition(Neon::Execution execution,
                      Neon::DataView  dw,
                      Neon::SetIdx    setIdx)
        const -> const VoxelPartition<BuildingBlockGridT, T, C>&
    {
        int         dwInt = Neon::DataViewUtil::toInt(dw);
        int         executionInt = Neon::ExecutionUtils::toInt(execution);
        const auto& output = mPartitionsByExecutionViewAndDevIdx[executionInt][dwInt][setIdx.idx()];
        return output;
    }

    auto isSupported(Neon::Execution ex)
        const -> bool
    {
        int  exInt = Neon::ExecutionUtils::toInt(ex);
        bool output = mSupportedExecutions[exInt];
        return output;
    }

    auto getDataUse() const -> Neon::DataUse
    {
        return mDataUse;
    }

    auto getBuildingBlockField()
        -> typename BuildingBlocks::Field&
    {
        return mBuildingBlockField;
    }

    auto getBuildingBlockField()
        const -> const typename BuildingBlocks::Field&
    {
        return mBuildingBlockField;
    }

   private:
    auto getPartitionDataSet(Neon::Execution execution, Neon::DataView dw)
        -> Neon::set::DataSet<VoxelPartition<BuildingBlockGridT, T, C>>&
    {

        int dwInt = Neon::DataViewUtil::toInt(dw);
        int executionInt = Neon::ExecutionUtils::toInt(execution);
        return mPartitionsByExecutionViewAndDevIdx[executionInt][dwInt];
    }

    typename BuildingBlocks::Field                            mBuildingBlockField;
    typename BuildingBlocks::FieldNodeToVoxelMask             mNodeToVoxelMaskField;
    std::array<bool, Neon::ExecutionUtils::numConfigurations> mSupportedExecutions;
    Neon::DataUse                                             mDataUse;

    std::array<std::array<Neon::set::DataSet<VoxelPartition<BuildingBlockGridT, T, C>>,
                          Neon::DataViewUtil::nConfig>,
               Neon::ExecutionUtils::numConfigurations>
        mPartitionsByExecutionViewAndDevIdx;
};


/**
 * Create and manage a dense field on both GPU and CPU. VoxelField also manages updating
 * the GPU->CPU and CPU-GPU as well as updating the halo. User can use VoxelField to populate
 * the field with data as well was exporting it to VTI. To create a new VoxelField,
 * use the newField function in dGrid.
 */

template <typename BuildingBlockGridT, typename T, int C = 0>
class VoxelField : public Neon::domain::interface::FieldBaseTemplate<T,
                                                                     C,
                                                                     VoxelGrid<BuildingBlockGridT>,
                                                                     VoxelPartition<BuildingBlockGridT, T, C>,
                                                                     VoxelStorage<BuildingBlockGridT, T, C>>
{

   public:
    struct BuildingBlocks
    {
        using Grid = BuildingBlockGridT;
        using Partition = typename BuildingBlockGridT::template Partition<T, C>;
        using PartitionNodeToVoxelMask = typename BuildingBlockGridT::template Partition<NodeToVoxelMask, 1>;
        using FieldNodeToVoxelMask = typename BuildingBlockGridT::template Field<NodeToVoxelMask, 1>;
    };


    static constexpr int Cardinality = C;
    using Type = T;
    using Self = VoxelField<typename BuildingBlocks::Grid, Type, Cardinality>;

    using Grid = VoxelGrid<typename BuildingBlocks::Grid>;
    using Partition = VoxelPartition<typename BuildingBlocks::Grid, T, C>;
    using Node = NodeGeneric<typename BuildingBlocks::Grid>;
    using Voxel = VoxelGeneric<typename BuildingBlocks::Grid>;
    using Storage = VoxelStorage<BuildingBlockGridT, T, C>;

    friend Grid;

    VoxelField() = default;

    virtual ~VoxelField() = default;

    auto self() -> Self&;

    auto self() const -> const Self&;

    /**
     * Returns the metadata associated with the VoxelGeneric in location idx.
     * If the VoxelGeneric is not active (it does not belong to the voxelized domain),
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


    static auto swap(VoxelField& A, VoxelField& B) -> void;

   private:
    using Neon::domain::interface::FieldBaseTemplate<T,
                                                     C,
                                                     VoxelGrid<BuildingBlockGridT>,
                                                     VoxelPartition<BuildingBlockGridT, T, C>,
                                                     VoxelStorage<BuildingBlockGridT, T, C>>::ioToVtk;

    VoxelField(const std::string&                                   fieldUserName,
               Neon::DataUse                                        dataUse,
               const Neon::MemoryOptions&                           memoryOptions,
               const Grid&                                          grid,
               const typename BuildingBlocks::FieldNodeToVoxelMask& partitionNodeToVoxelMask,
               const typename BuildingBlocks::Grid&                 buildingBlockGrid,
               int                                                  cardinality,
               T                                                    inactiveValue,
               Neon::domain::haloStatus_et::e                       haloStatus);


   public:
    template <typename VtiExportType = Type>
    auto ioToVtk(const std::string& fileName,
                 const std::string& FieldName,
                 bool               includeDomain = false,
                 Neon::IoFileType   ioFileType = Neon::IoFileType::ASCII,
                 bool               isNodeSpace = false) const -> void;
};


}  // namespace Neon::domain::internal::experimental::staggeredGrid::details
