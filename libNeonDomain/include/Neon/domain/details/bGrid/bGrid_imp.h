#include "Neon/domain/details/bGrid/bGrid.h"

namespace Neon::domain::details::bGrid {


template <typename ActiveCellLambda>
bGrid::bGrid(const Neon::Backend&         backend,
             const Neon::int32_3d&        domainSize,
             const ActiveCellLambda       activeCellLambda,
             const Neon::domain::Stencil& stencil,
             const double_3d&             spacingData,
             const double_3d&             origin)
    : bGrid(backend, domainSize, activeCellLambda, stencil, 8, 1, spacingData, origin)
{
}

template <typename ActiveCellLambda>
bGrid::bGrid(const Neon::Backend&         backend,
             const Neon::int32_3d&        domainSize,
             const ActiveCellLambda       activeCellLambda,
             const Neon::domain::Stencil& stencil,
             const int                    dataBlockSize,
             const int                    voxelSpacing,
             const double_3d&             spacingData,
             const double_3d&             origin)
{
    if (backend.devSet().setCardinality() > 1) {
        NeonException exp("bGrid");
        exp << "bGrid only supported on a single GPU";
        NEON_THROW(exp);
    }

    mData = std::make_shared<Data>();
    mData->dataBlockSize = dataBlockSize;
    mData->voxelSpacing = voxelSpacing;
    mData->stencil = stencil;
    const index_3d defaultKernelBlockSize = [&] {
        return index_3d(dataBlockSize, dataBlockSize, dataBlockSize);
    }();

    {
        auto nElementsPerPartition = backend.devSet().template newDataSet<size_t>(0);
        // We do an initialization with nElementsPerPartition to zero,
        // then we reset to the computed number.
        bGrid::GridBase::init("bGrid",
                              backend,
                              domainSize,
                              stencil,
                              nElementsPerPartition,
                              defaultKernelBlockSize,
                              voxelSpacing,
                              origin);
    }

    {  // Initialization of the partitioner

        mData->partitioner1D = Neon::domain::tool::Partitioner1D(
            backend,
            activeCellLambda,
            [](Neon::index_3d /*idx*/) { return false; },
            helpGetDataBlockSize(),
            domainSize,
            stencil,
            1);

        mData->mDataBlockOriginField = mData->partitioner1D.getGlobalMapping();
        mData->mStencil3dTo1dOffset = mData->partitioner1D.getStencil3dTo1dOffset();
        mData->memoryGrid = mData->partitioner1D.getMemoryGrid();
        mData->partitioner1D.getDenseMeta(mData->denseMeta);
    }

    {  // BlockViewGrid
        Neon::domain::details::eGrid::eGrid egrid(
            backend,
            mData->partitioner1D.getBlockSpan(),
            mData->partitioner1D,
            stencil,
            spacingData * helpGetDataBlockSize(),
            origin);

        mData->blockViewGrid = BlockViewGrid(egrid);
    }

    {  // Active bitmask
        Neon::domain::details::eGrid::eGrid egrid(
            backend,
            mData->partitioner1D.getBlockSpan(),
            mData->partitioner1D,
            stencil,
            spacingData * helpGetDataBlockSize(),
            origin);
        int requiredWords = bSpan::getRequiredWords(helpGetDataBlockSize());
        mData->activeBitMask = mData->blockViewGrid.newField<bSpan::bitMaskWordType>("BitMask",
                                                                                     requiredWords,
                                                                                     0,
                                                                                     Neon::DataUse::HOST_DEVICE, backend.getMemoryOptions(bSpan::activeMaskMemoryLayout));

        mData->mNumActiveVoxel = backend.devSet().template newDataSet<uint64_t>();

        mData->activeBitMask
            .getGrid()
            .newContainer(
                "activeBitMaskInit",
                [&](Neon::set::Loader& loader) {
                    auto bitMask = loader.load(mData->activeBitMask);
                    return [&](const auto& bitMaskIdx) {
                        auto           prtIdx = bitMask.prtID();
                        int            coutActive = 0;
                        auto const     idx3d = bitMask.getGlobalIndex(bitMaskIdx);
                        Neon::index_3d blockSize3d(dataBlockSize, dataBlockSize, dataBlockSize);
                        auto const     blockOrigin = idx3d * dataBlockSize;
                        for (int k = 0; k < dataBlockSize; k++) {
                            for (int j = 0; j < dataBlockSize; j++) {
                                for (int i = 0; i < dataBlockSize; i++) {
                                    for (int c = 0; c < bitMask.cardinality(); c++) {
                                        bitMask(bitMaskIdx, c) = 0;
                                    }
                                    Neon::int32_3d point(i, j, k);
                                    point = point + blockOrigin;
                                    auto const pitch = point.mPitch(blockSize3d);
                                    auto const targetCard = pitch / bSpan::bitMaskStorageBitWidth;
                                    auto const targetBit = pitch % bSpan::bitMaskStorageBitWidth;
                                    uint64_t   mask = uint64_t(1) << targetBit;
                                    bool       isActive = activeCellLambda(blockSize3d);
                                    if (isActive) {
                                        coutActive++;
                                        bitMask(bitMaskIdx, targetCard) |= mask;
                                    }
                                }
                            }
                        }
#pragma omp critical
                        {
                            mData->mNumActiveVoxel[prtIdx] += coutActive;
                        }
                    };
                },
                Neon::Execution::host)
            .run();

        mData->activeBitMask.updateDeviceData(Neon::Backend::mainStreamIdx);
        mData->activeBitMask.newHaloUpdate(Neon::set::StencilSemantic::standard,
                                           Neon::set::TransferMode::put,
                                           Neon::Execution::device);
    }


    {  // Neighbor blocks
        mData->blockConnectivity = mData->blockViewGrid.newField<BlockIdx, 27>("blockConnectivity",
                                                                               27,
                                                                               bSpan::getInvalidBlockId(),
                                                                               Neon::DataUse::HOST_DEVICE,
                                                                               Neon::MemoryLayout::arrayOfStructs);

        mData->blockConnectivity.getGrid().newContainer(
                                              "blockConnectivityInit",
                                              [&](Neon::set::Loader& loader) {
                                                  auto blockConnectivity = loader.load(mData->blockConnectivity);
                                                  return [&](auto const& idx) {
                                                      for (int8_t k = 0; k < 3; k++) {
                                                          for (int8_t j = 0; j < 3; j++) {
                                                              for (int8_t i = 0; i < 3; i++) {
                                                                  auto                                      targetDirection = i + 3 * j + 3 * 3 * k;
                                                                  BlockIdx                                  blockNghIdx = bSpan::getInvalidBlockId();
                                                                  typename decltype(blockConnectivity)::Idx nghIdx;
                                                                  bool                                      isValid = blockConnectivity.getNghIndex(idx, {i, j, k}, nghIdx);
                                                                  if (isValid) {
                                                                      blockNghIdx = nghIdx.helpGet();
                                                                  }
                                                                  blockConnectivity(idx, targetDirection) = nghIdx;
                                                              }
                                                          }
                                                      }
                                                  };
                                              },
                                              Neon::Execution::host)
            .run(Neon::Backend::mainStreamIdx);
        mData->blockConnectivity.updateDeviceData(Neon::Backend::mainStreamIdx);
    }

    // Initialization of the SPAN table
    mData->spanTable.forEachConfiguration([&](Neon::Execution execution,
                                              Neon::SetIdx    setIdx,
                                              Neon::DataView  dw,
                                              bSpan&          span) {
        span.mDataView = dw;
        switch (dw) {
            case Neon::DataView::STANDARD: {
                span.mFirstDataBlockOffset = 0;
                span.mDataView = dw;
                span.mDataBlockSize = helpGetDataBlockSize();
                span.mActiveMask = mData->activeBitMask.getPartition(execution, setIdx, dw).mem();
                break;
            }
            case Neon::DataView::BOUNDARY: {
                span.mFirstDataBlockOffset = mData->partitioner1D.getSpanClassifier().countInternal(setIdx);
                span.mDataView = dw;
                span.mDataBlockSize = helpGetDataBlockSize();
                span.mActiveMask = mData->activeBitMask.getPartition(execution, setIdx, dw).mem();

                break;
            }
            case Neon::DataView::INTERNAL: {
                span.mFirstDataBlockOffset = 0;
                span.mDataView = dw;
                span.mDataBlockSize = helpGetDataBlockSize();
                span.mActiveMask = mData->activeBitMask.getPartition(execution, setIdx, dw).mem();
                break;
            }
            default: {
                NeonException exc("dFieldDev");
                NEON_THROW(exc);
            }
        }
    });

    // Init the base grid
    bGrid::GridBase::init("bGrid",
                          backend,
                          domainSize,
                          Neon::domain::Stencil(),
                          mData->mNumActiveVoxel,
                          Neon::int32_3d(dataBlockSize, dataBlockSize, dataBlockSize),
                          spacingData,
                          origin);
}


template <typename T, int C>
auto bGrid::newField(const std::string          name,
                     int                        cardinality,
                     T                          inactiveValue,
                     Neon::DataUse              dataUse,
                     const Neon::MemoryOptions& memoryOptions) const -> Field<T, C>
{
    Field<T, C> field(name, *this, cardinality, inactiveValue, dataUse, memoryOptions, Neon::domain::haloStatus_et::ON);
    return field;
}


template <typename LoadingLambda>
auto bGrid::getContainer(const std::string& name,
                         index_3d           blockSize,
                         size_t             sharedMem,
                         LoadingLambda      lambda) const -> Neon::set::Container
{
    Neon::set::Container kContainer = Neon::set::Container::factory(name,
                                                                    Neon::set::internal::ContainerAPI::DataViewSupport::on,
                                                                    *this,
                                                                    lambda,
                                                                    blockSize,
                                                                    [sharedMem](const Neon::index_3d&) { return sharedMem; });
    return kContainer;
}


template <typename LoadingLambda>
auto bGrid::getContainer(const std::string& name,
                         LoadingLambda      lambda) const -> Neon::set::Container
{
    const Neon::index_3d& defaultBlockSize = this->getDefaultBlock();
    Neon::set::Container  kContainer = Neon::set::Container::factory(name,
                                                                     Neon::set::internal::ContainerAPI::DataViewSupport::on,
                                                                     *this,
                                                                     lambda,
                                                                     defaultBlockSize,
                                                                     [](const Neon::index_3d&) { return size_t(0); });
    return kContainer;
}


}  // namespace Neon::domain::details::bGrid