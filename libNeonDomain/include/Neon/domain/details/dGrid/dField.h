#pragma once

#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"

#include "Neon/set/DataConfig.h"
#include "Neon/set/DevSet.h"
#include "Neon/set/HuOptions.h"


#include "Neon/domain/aGrid.h"
#include "Neon/domain/interface/FieldBaseTemplate.h"
#include "Neon/domain/tools/PartitionTable.h"

#include "Neon/domain/tools/HaloUpdateTable1DPartitioning.h"
#include "dPartition.h"

namespace Neon::domain::details::dGrid {


/**
 * Create and manage a dense field on both GPU and CPU. dField also manages updating
 * the GPU->CPU and CPU-GPU as well as updating the halo. User can use dField to populate
 * the field with data as well was exporting it to VTI. To create a new dField,
 * use the newField function in dGrid.
 */
template <typename T, int C = 0>
class dField : public Neon::domain::interface::FieldBaseTemplate<T,
                                                                 C,
                                                                 dGrid,
                                                                 dPartition<T, C>,
                                                                 int>
{
    friend dGrid;

   public:
    static constexpr int Cardinality = C;
    using Type = T;
    using Self = dField<Type, Cardinality>;
    using Grid = dGrid;
    using Field = dField;
    using Partition = dPartition<T, C>;
    using Idx = typename Partition::Idx;
    using NghIdx = typename Partition::NghIdx;
    using NghData = typename Partition::NghData;

    /**
     * Empty constructor
     */
    dField();

    /**
     * Destructor
     */
    virtual ~dField() = default;

    /**
     * Self operator
     */
    auto self() -> Self&;

    /**
     * Self operator
     */
    auto self() const -> const Self&;

    /**
     * Returns the metadata associated with the element in location idx.
     * If the element is not active (it does not belong to the voxelized domain),
     * then the default outside value is returned.
     */
    auto operator()(const Neon::index_3d& idx,
                    const int&            cardinality) const
        -> Type final;

    auto ioVtiAllocator(std::string name)
    {
        mData->memoryField.ioToVtk(name, name);
    }
    /**
     * Creates a container that executes a halo update operation on host or device
     */
    auto newHaloUpdate(Neon::set::StencilSemantic semantic,
                       Neon::set::TransferMode    transferMode,
                       Neon::Execution            execution)
        const -> Neon::set::Container;

    virtual auto
    getReference(const Neon::index_3d& idx,
                 const int&            cardinality)
        -> Type& final;

    /**
     * It copies host data to the device
     * @param streamSetId
     */
    auto updateDeviceData(int streamSetId)
        -> void;

    /**
     * It copies device data to the host
     * @param streamSetId
     */
    auto updateHostData(int streamSetId)
        -> void;

    /**
     * Returns a constant reference to a specific partition based on a set of parameters:
     * execution type, target device, dataView
     */
    auto getPartition(Neon::Execution       execution,
                      Neon::SetIdx          setIdx,
                      const Neon::DataView& dataView) const
        -> const Partition& final;

    /**
     * Return a reference to a specific partition based on a set of parameters:
     * execution type, target device, dataView
     */
    auto getPartition(Neon::Execution       execution,
                      Neon::SetIdx          setIdx,
                      const Neon::DataView& dataView)
        -> Partition& final;

    static auto swap(Field& A, Field& B)
        -> void;

   private:
    auto initHaloUpdateTable()
        -> void;


    /** Convert a global 3d index into a Partition local offset */
    auto helpGlobalIdxToPartitionIdx(Neon::index_3d const& index)
        const -> std::pair<Neon::index_3d, int>;

    dField(const std::string&                        fieldUserName,
           Neon::DataUse                             dataUse,
           const Neon::MemoryOptions&                memoryOptions,
           const Grid&                               grid,
           const Neon::set::DataSet<Neon::index_3d>& dims,
           int                                       zHaloRadius,
           Neon::domain::haloStatus_et::e            haloStatus,
           int                                       cardinality,
           Neon::set::MemSet<Neon::int8_3d>&         stencilIdTo3dOffset);

    struct Data
    {
        Data() = default;
        Data(Neon::Backend const& bk)
        {
            partitionTable.init(bk);
            pitch = bk.newDataSet<size_4d>();
        }

        enum EndPoints {
            src = 1,
            dst = 0
        };

        struct EndPointsUtils {
            static constexpr int  nConfigs = 2;
        };

        struct ReductionInformation
        {
            std::vector<int> startIDByView /* one entry for each cardinality */;
            std::vector<int> nElementsByView /* one entry for each cardinality */;
        };

        Neon::domain::tool::PartitionTable<Partition, ReductionInformation> partitionTable;
        Neon::domain::tool::HaloTable1DPartitioning                         latticeHaloUpdateTable;
        Neon::domain::tool::HaloTable1DPartitioning                         soaHaloUpdateTable;
        Neon::domain::tool::HaloTable1DPartitioning                         aosHaloUpdateTable;
        Neon::aGrid::Field<T, C>                                            memoryField;

        Neon::DataUse                     dataUse;
        Neon::MemoryOptions               memoryOptions;
        int                               cardinality;
        Neon::set::DataSet<Neon::size_4d> pitch;

        std::shared_ptr<Grid>          grid;
        int                            zHaloDim;
        Neon::domain::haloStatus_et::e haloStatus;
        bool                           periodic_z;

        Neon::set::MemSet<NghIdx> stencilNghIndex;
    };

    std::shared_ptr<Data> mData;
    auto                  getData() -> Data&;
};


}  // namespace Neon::domain::details::dGrid
