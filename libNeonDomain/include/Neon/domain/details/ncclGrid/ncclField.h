#pragma once

#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"
#include "ncclGrid.h"

#include "Neon/set/DevSet.h"
#include "Neon/set/HuOptions.h"


#include "Neon/domain/aGrid.h"
#include "Neon/domain/interface/FieldBaseTemplate.h"
#include "Neon/domain/tools/PartitionTable.h"

#include "Neon/domain/tools/HaloUpdateTable1DPartitioning.h"
#include "ncclPartition.h"

namespace Neon::domain::details::ncclGrid {

struct RankBaseGrid : interface::GridBase
{
    RankBaseGrid(const std::string&                           gridImplementationName,
                 const Neon::Backend&                         backend,
                 const Neon::index_3d&                        dim,
                 const Neon::domain::Stencil&                 stencil,
                 const Neon::set::DataSet<size_t>&            nPartitionElements /**< Number of element per partition */,
                 const Neon::index_3d&                        defaultBlockSize,
                 const Vec_3d<double>&                        spacingData /*! Spacing, i.e. size of a voxel */,
                 const Vec_3d<double>&                        origin /*!      Origin  */,
                 Neon::domain::tool::spaceCurves::EncoderType spaceCurve,
                 Neon::index_3d                               memoryBlock)
        : RankBaseGrid::GridBase(gridImplementationName,
                                 backend,
                                 dim,
                                 stencil,
                                 nPartitionElements,
                                 defaultBlockSize,
                                 spacingData,
                                 origin,
                                 spaceCurve,
                                 memoryBlock)
    {
    }

    auto isInsideDomain(const Neon::index_3d& /*idx*/) const
        -> bool
    {
        return true;
    }

    auto getSetIdx(const Neon::index_3d& /*idx*/) const
        -> int32_t
    {
        return 0;
    }
};


/**
 * Create and manage a dense field on both GPU and CPU. ncclField also manages updating
 * the GPU->CPU and CPU-GPU as well as updating the halo. User can use ncclField to populate
 * the field with data as well was exporting it to VTI. To create a new ncclField,
 * use the newField function in ncclGrid.
 */
template <typename T, int C = 0>
class ncclField : public Neon::domain::interface::FieldBaseTemplate<T,
                                                                    C,
                                                                    ncclGrid,
                                                                    ncclPartition<T, C>,
                                                                    int>
{
    friend ncclGrid;

   public:
    static constexpr int Cardinality = C;
    using Type = T;
    using Self = ncclField<Type, Cardinality>;
    using Grid = ncclGrid;
    using Field = ncclField;
    using Partition = ncclPartition<T, C>;
    using Idx = typename Partition::Idx;
    using NghIdx = typename Partition::NghIdx;
    using NghData = typename Partition::NghData;

    /**
     * Empty constructor
     */
    ncclField();

    /**
     * Destructor
     */
    virtual ~ncclField() = default;

    /**
     * Self operator
     */
    auto self() -> Self&;

    /**
     * Self operator
     */
    auto self() const -> const Self&;

    auto constSelf() const -> const Self&;

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

    auto ioToVtiPartitions(const std::string& fname) const -> void;

    static auto swap(Field& A, Field& B)
        -> void;

    auto forEachActiveCell(const std::function<void(const Neon::index_3d&,
                                                    std::vector<T*>&)>&        fun,
                           [[maybe_unused]] Neon::computeMode_t::computeMode_e mode = Neon::computeMode_t::computeMode_e::par)
        -> void
    {
        const int       cardinality = this->getCardinality();
        std::vector<T*> vec(cardinality, nullptr);
        assert(this->getBackend().devSet().setCardinality() == 1);
        auto        partition = this->getPartition(Neon::Execution::host, 0, Neon::DataView::STANDARD);
        auto const  span = this->getGrid().getSpan(Neon::Execution::host, 0, Neon::DataView::STANDARD);
        auto const& spanDim = span.helpGetDim();

        for (int z = 0; z < spanDim.z; z++) {
            for (int y = 0; y < spanDim.y; y++) {
                for (int x = 0; x < spanDim.x; x++) {
                    Idx gIdx;
                    span.setAndValidate(gIdx, x, y, z);
                    auto const cartesianPoint = partition.getGlobalIndex(gIdx);
                    for (int c = 0; c < cardinality; c++) {
                        vec[c] = &partition.operator()(gIdx, c);
                    }
                    fun(cartesianPoint, vec);
                }
            }
        }
    }

    template <typename VtiExportType = T>
    auto ioToVtk(const std::string& fileName_,
                 const std::string& FieldName,
                 Neon::IoFileType   ioFileType = Neon::IoFileType::ASCII,
                 bool               isNodeSpace = false) const -> void
    {
        std::string fileName = fileName_;
        auto&       bk = this->getBackend();
        int         rank = bk.getNccl().getWorldRank();
        fileName = fileName + "_" + std::to_string(rank);


        auto span = this->getGrid().getSpan(Neon::Execution::host, 0, Neon::DataView::STANDARD);
        auto baseDim = span.helpGetDim();

        auto partition = this->getPartition(Neon::Execution::host, 0, Neon::DataView::STANDARD);
        Idx  gIDxOrigin;
        span.setAndValidate(gIDxOrigin, 0, 0, 0);
        auto partitionOrigin = partition.getGlobalIndex(gIDxOrigin).template newType<double>() *
                                   this->getGrid().getSpacing() +
                               this->getGrid().getOrigin();
        std::cout << "partitionOrigin: " << partitionOrigin << std::endl;
        std::cout << "partition.getGlobalIndex(gIDxOrigin).template newType<double>(): " << partition.getGlobalIndex(gIDxOrigin).template newType<double>() << std::endl;

        std::cout << "this->getGrid().getOrigin(): " << this->getGrid().getOrigin() << std::endl;
        std::cout << "this->getGrid().getSpacing(): " << this->getGrid().getSpacing() << std::endl;

        auto& worldGrid = this->getBaseGridTool();
        auto  gbase = RankBaseGrid(
            worldGrid.getImplementationName(),
            bk,
            baseDim,
            worldGrid.getStencil(),
            worldGrid.getNumAllCells(),
            worldGrid.getDefaultBlock(),
            worldGrid.getSpacing(),
            partitionOrigin,
            worldGrid.getSpaceCurve(),
            worldGrid.getMemoryBlock());


        auto iovtk = Neon::domain::IOGridVTK<VtiExportType>(gbase, fileName, isNodeSpace, ioFileType);
        // iovtk.addField(*this, FieldName);

        Neon::IODense<VtiExportType, int32_t> domain(baseDim, this->getCardinality(), [&](const Neon::index_3d& idx, int c) {
            Idx gIdx;
            span.setAndValidate(gIdx, idx.x, idx.y, idx.z);
            auto val = partition.operator()(gIdx, c);
            return val;
        });
        iovtk.addIODenseField(domain, FieldName);

        // if (includeDomain) {
        //     iovtk.addIODenseField(domain, "Domain");
        // }
        iovtk.flushAndClear();
        return;
    }

   private:
    auto initHaloUpdateTable()
        -> void;


    /** Convert a global 3d index into a Partition local offset */
    auto helpGlobalIdxToPartitionIdx(Neon::index_3d const& index)
        const -> std::pair<Neon::index_3d, int>;

    ncclField(const std::string&                        fieldUserName,
              Neon::DataUse                             dataUse,
              const Neon::MemoryOptions&                memoryOptions,
              const Grid&                               grid,
              const Neon::set::DataSet<Neon::index_3d>& dimsRank,
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

        enum EndPoints
        {
            src = 1,
            dst = 0
        };

        struct EndPointsUtils
        {
            static constexpr int nConfigs = 2;
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


}  // namespace Neon::domain::details::ncclGrid
