#pragma once

#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"

#include "Neon/sys/memory/MemDevice.h"

#include "Neon/set/BlockConfig.h"

#include "Neon/set/DevSet.h"

#include "Neon/domain/interface/GridBase.h"
#include "Neon/domain/interface/KernelConfig.h"

#include "Neon/domain/details/sGrid/sField.h"
#include "Neon/domain/details/sGrid/sGrid.h"
#include "Neon/domain/details/sGrid/sPartition.h"


namespace Neon::domain::details::sGrid {

template <typename OuterGridT>
sGrid<OuterGridT>::sGrid()
    : Neon::domain::interface::GridBaseTemplate<Self, Idx>() {
    mStorage = std::make_shared<sStorage>();
};

template <typename OuterGridT>
sGrid<OuterGridT>::sGrid(const OuterGridT&                  outerGrid,
                         const std::vector<Neon::index_3d>& subGridPoints)
{
    // We do an initialization with nElementsPerPartition to zero,
    // then we reset to the computed number.
    auto                 nElementsPerPartition = outerGrid.getDevSet().template newDataSet<size_t>(0);
    const Neon::index_3d defaultsBlockDim(512, 1, 1);

    Self ::GridBaseTemplate::init("sGrid-" + outerGrid.getImplementationName(),
                                  outerGrid.getBackend(),
                                  outerGrid.getDimension(),
                                  outerGrid.getStencil(),
                                  nElementsPerPartition,
                                  defaultsBlockDim,
                                  outerGrid.getSpacing(),
                                  outerGrid.getOrigin());

    mStorage = std::make_shared<sStorage>();
    mStorage->init(outerGrid);

    auto initMapTable = [&]() {
        // Add points into hashTable
        // .A retrieve cell information
        // .B add point and metadata to the hashtable

        for (auto const& point : subGridPoints) {
            // A
            auto const& cellProperties = outerGrid.getProperties(point);
            bool        isInside = cellProperties.isInside();
            if (!isInside) {
                NEON_THROW_UNSUPPORTED_OPERATION("sGrid");
            }

            // B
            Meta meta(-1, cellProperties.getOuterIdx());
            mStorage->map.addPoint(point,
                                   meta,
                                   cellProperties.getSetIdx(),
                                   cellProperties.getDataView());
        }
    };


    auto initCellClassification = [this]() -> void {
        // Associating each point to an idx
        // A. target internals
        // B. target boundaries
        for (const auto& setIdx : this->getDevSet().getRange()) {
            int32_t idxInPartition = 0;

            mStorage->map.forEach(setIdx,
                                  Neon::DataView::INTERNAL,
                                  [&](const Neon::index_3d&, Meta& m) {
                                      m.cellOffset = idxInPartition;
                                      idxInPartition++;
                                  });
            const int32_t nInternal = idxInPartition + 1;
            mStorage->map.forEach(setIdx,
                                  Neon::DataView::BOUNDARY,
                                  [&](const Neon::index_3d&, Meta& m) {
                                      m.cellOffset = idxInPartition;
                                      idxInPartition++;
                                  });
            const int32_t cellsPerPartition = idxInPartition;
            const int32_t nBoundary = cellsPerPartition - nInternal;

            mStorage->getCount(Neon::DataView::STANDARD)[setIdx.idx()] = cellsPerPartition;
            mStorage->getCount(Neon::DataView::INTERNAL)[setIdx.idx()] = nInternal;
            mStorage->getCount(Neon::DataView::BOUNDARY)[setIdx.idx()] = nBoundary;
        }
    };

    auto initDefaultLaunchParameters = [this]() -> void {
        if (this->getDefaultBlock().y != 1 || this->getDefaultBlock().z != 1) {
            NeonException exc("sGrid");
            exc << "CUDA block size should be 1D\n";
            NEON_THROW(exc);
        }

        for (int i = 0; i < this->getDevSet().setCardinality(); i++) {
            for (auto indexing : DataViewUtil::validOptions()) {

                auto gridMode = Neon::sys::GpuLaunchInfo::mode_e::domainGridMode;
                auto gridDim = mStorage->getCount(indexing)[i];
                this->getDefaultLaunchParameters(indexing)[i].set(gridMode, gridDim, this->getDefaultBlock(), 0);
            }
        }
    };

    auto initToOuterMappingField = [&]() -> void {
        // Init mapping from sGrid to OuterGrid
        std::string mappingFieldName = "sGrid->" + outerGrid.getImplementationName();

        using OGCell = typename OuterGrid::Cell;
        using OGCellOGCell = typename OGCell::OuterIdx;
        OGCellOGCell oc;

        Neon::MemoryOptions memoryOptions;
        this->mStorage->tableToOuterIdx = this->getDevSet().template newMemSet<OGCellOGCell>(Neon::DataUse::HOST_DEVICE,
                                                                                             1,
                                                                                             Neon::MemoryOptions(),
                                                                                             mStorage->getCount(Neon::DataView::STANDARD));
        for (const auto& setIdx : this->getDevSet().getRange()) {
            mStorage->map.forEach(setIdx,
                                  Neon::DataView::INTERNAL,
                                  [&](const Neon::index_3d&, Meta& m) {
                                      this->mStorage->tableToOuterIdx.eRef(setIdx, m.cellOffset) = m.outerCell;
                                  });
            mStorage->map.forEach(setIdx,
                                  Neon::DataView::BOUNDARY,
                                  [&](const Neon::index_3d&, Meta& m) {
                                      this->mStorage->tableToOuterIdx.eRef(setIdx, m.cellOffset) = m.outerCell;
                                  });
        }

        if (this->getDevSet().type() != Neon::DeviceType::CPU && this->getDevSet().type() != Neon::DeviceType::OMP) {
            this->mStorage->tableToOuterIdx.updateDeviceData(this->getBackend(), 0);
        }
        this->getBackend().sync(0);
    };

    auto initPartitionIndexSpace = [this]() {
        for (auto& dw : Neon::DataViewUtil::validOptions()) {
            mStorage->getPartitionIndexSpace(dw) = this->getDevSet().template newDataSet<sSpan>();

            for (int gpuIdx = 0; gpuIdx < this->getDevSet().setCardinality(); gpuIdx++) {

                mStorage->getPartitionIndexSpace(dw)[gpuIdx].helpGetBoundaryOffset() = mStorage->getCount(DataView::INTERNAL)[gpuIdx];
                mStorage->getPartitionIndexSpace(dw)[gpuIdx].helpGetGhostOffset() = mStorage->getCount(DataView::STANDARD)[gpuIdx];
                mStorage->getPartitionIndexSpace(dw)[gpuIdx].helpGetDataView() = dw;
            }
        }
    };

    initMapTable();
    initCellClassification();
    initDefaultLaunchParameters();
    initToOuterMappingField();
    initPartitionIndexSpace();

    sGrid::GridBase::init("sGrid-" + outerGrid.getImplementationName(),
                          outerGrid.getBackend(),
                          outerGrid.getDimension(),
                          outerGrid.getStencil(),
                          mStorage->getCount(Neon::DataView::STANDARD),
                          defaultsBlockDim,
                          outerGrid.getSpacing(),
                          outerGrid.getOrigin());
}

template <typename OuterGridT>
template <typename T, int C>
auto sGrid<OuterGridT>::newField(const std::string   fieldUserName,
                                 int                 cardinality,
                                 T                   inactiveValue,
                                 Neon::DataUse       dataUse,
                                 Neon::MemoryOptions memoryOptions) const
    -> sGrid::Field<T, C>
{
    memoryOptions = this->getDevSet().sanitizeMemoryOption(memoryOptions);
    constexpr Neon::domain::haloStatus_et::e haloStatus = Neon::domain::haloStatus_et::e::ON;

    if (C != 0 && cardinality != C) {
        NeonException exception("Dynamic and static cardinality values do not match.");
        NEON_THROW(exception);
    }
    sField<OuterGridT, T, C> field(fieldUserName, *this, cardinality, inactiveValue,
                                   haloStatus, dataUse, memoryOptions, mStorage->tableToOuterIdx);
    return field;
}
template <typename OuterGridT>
template <Neon::Execution execution,
          typename LoadingLambda>
auto sGrid<OuterGridT>::newContainer(const std::string& name,
                                     LoadingLambda      lambda)
    const
    -> Neon::set::Container
{
    Neon::domain::KernelConfig kernelConfig(0);

    const Neon::index_3d& defaultBlockSize = this->getDefaultBlock();
    Neon::set::Container  kContainer = Neon::set::Container::factory<execution>(name,

                                                                               Neon::set::internal::ContainerAPI::DataViewSupport::on,
                                                                               *this,
                                                                               lambda,
                                                                               defaultBlockSize,
                                                                               [](const Neon::index_3d&) { return 0; });
    return kContainer;
}
template <typename OuterGridT>
template <Neon::Execution execution,
          typename LoadingLambda>
auto sGrid<OuterGridT>::newContainer(const std::string& name,
                                     index_3d           blockSize,
                                     size_t             sharedMem,
                                     LoadingLambda      lambda)
    const
    -> Neon::set::Container
{
    Neon::domain::KernelConfig kernelConfig(0);

    const Neon::index_3d& defaultBlockSize = this->getDefaultBlock();
    Neon::set::Container  kContainer = Neon::set::Container::factory<execution>(name,
                                                                               Neon::set::internal::ContainerAPI::DataViewSupport::on,
                                                                               *this,
                                                                               lambda,
                                                                               blockSize,
                                                                               [sharedMem](const Neon::index_3d&) { return sharedMem; });
    return kContainer;
}

template <typename OuterGridT>
auto sGrid<OuterGridT>::isInsideDomain(const index_3d& idx) const -> bool
{
    SetIdx   sId;
    DataView dw;
    auto*    meta = mStorage->map.getMetadata(idx, sId, dw);

    if (meta == nullptr)
        return false;
    return true;
}

template <typename OuterGridT>
auto sGrid<OuterGridT>::getProperties(const index_3d&) const
    -> typename GridBaseTemplate::CellProperties
{
    NEON_THROW_UNSUPPORTED_OPERATION("");
}

template <typename OuterGridT>
auto sGrid<OuterGridT>::getLaunchParameters(Neon::DataView  dataView,
                                            const index_3d& blockDim,
                                            size_t          shareMem) const -> Neon::set::LaunchParameters
{
    if (blockDim.y != 1 || blockDim.z != 1) {
        NeonException exc("sGrid");
        exc << "CUDA block size should be 1D\n";
        NEON_THROW(exc);
    }

    auto newLaunchParameters = this->getDevSet().newLaunchParameters();

    for (int i = 0; i < this->getDevSet().setCardinality(); i++) {

        auto    gridMode = Neon::sys::GpuLaunchInfo::mode_e::domainGridMode;
        int32_t gridDim = int32_t(mStorage->getCount(dataView)[i]);
        newLaunchParameters[i].set(gridMode, gridDim, blockDim, shareMem);
    }
    return newLaunchParameters;
}

template <typename OuterGridT>
auto sGrid<OuterGridT>::getSpan(Neon::Execution execution,
                                Neon::SetIdx    setIdx,
                                Neon::DataView  dataView) const -> const sGrid::Span&
{
    return mStorage->getPartitionIndexSpace(dataView).getPartition(execution, setIdx, dataView);
}
}  // namespace Neon::domain::details::sGrid
