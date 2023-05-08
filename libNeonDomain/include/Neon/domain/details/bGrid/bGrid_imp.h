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

    mData = std::make_shared<Data>();
    mData->init(backend);

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
            Neon::domain::Stencil::s27_t(false),
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
            Neon::domain::Stencil::s27_t(false),
            spacingData * helpGetDataBlockSize(),
            origin);

        mData->blockViewGrid = BlockViewGrid(egrid);
    }

    {  // Active bitmask
        int requiredWords = bSpan::getRequiredWordsForBlockBitMask(helpGetDataBlockSize());
        mData->activeBitMask = mData->blockViewGrid.newField<bSpan::BitMaskWordType>("BitMask",
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
                        auto       prtIdx = bitMask.prtID();
                        int        coutActive = 0;
                        auto const blockOrigin = bitMask.getGlobalIndex(bitMaskIdx);
                        //                        Neon::index_3d blockSize3d(dataBlockSize, dataBlockSize, dataBlockSize);
                        //                        auto const     blockOrigin = idx3d * dataBlockSize;

                        for (int c = 0; c < bitMask.cardinality(); c++) {
                            bitMask(bitMaskIdx, c) = 0;
                        }

                        for (int k = 0; k < dataBlockSize; k++) {
                            for (int j = 0; j < dataBlockSize; j++) {
                                for (int i = 0; i < dataBlockSize; i++) {

                                    Neon::int32_3d         localPosition(i, j, k);
                                    bSpan::BitMaskWordType mask;
                                    uint32_t               wordIdx;

                                    bSpan::getMaskAndWordIdforBlockBitMask(i, j, k, dataBlockSize, NEON_OUT mask, NEON_OUT wordIdx);
                                    auto globalPosition = localPosition + blockOrigin;
                                    bool isInDomain = globalPosition < domainSize;
                                    bool isActive = activeCellLambda(globalPosition);
                                    if (isActive && isInDomain) {
                                        coutActive++;
                                        auto value = bitMask(bitMaskIdx, wordIdx);
                                        value = value | mask;
                                        bitMask(bitMaskIdx, wordIdx) = value;
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
            .run(Neon::Backend::mainStreamIdx);

        mData->activeBitMask.updateDeviceData(Neon::Backend::mainStreamIdx);
        mData->activeBitMask.newHaloUpdate(Neon::set::StencilSemantic::standard,
                                           Neon::set::TransferMode::put,
                                           Neon::Execution::device)
            .run(Neon::Backend::mainStreamIdx);
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
                                                                  Neon::int8_3d                             stencilPoint(i - int8_t(1),
                                                                                                                         j - int8_t(1),
                                                                                                                         k - int8_t(1));
                                                                  bool                                      isValid = blockConnectivity.getNghIndex(idx, stencilPoint, nghIdx);
                                                                  if (isValid) {
                                                                      blockNghIdx = nghIdx.helpGet();
                                                                  }
                                                                  blockConnectivity(idx, targetDirection) = blockNghIdx;
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

    {  // Stencil Idx to 3d offset
        auto nPoints = backend.devSet().newDataSet<uint64_t>(stencil.nNeighbours());
        mData->stencilIdTo3dOffset = backend.devSet().template newMemSet<int8_3d>(Neon::DataUse::HOST_DEVICE,
                                                                                  1,
                                                                                  backend.getMemoryOptions(),
                                                                                  nPoints);
        for (int i = 0; i < stencil.nNeighbours(); ++i) {
            for (int devIdx = 0; devIdx < backend.devSet().setCardinality(); devIdx++) {
                index_3d      pLong = stencil.neighbours()[i];
                Neon::int8_3d pShort(pLong.x, pLong.y, pLong.z);
                mData->stencilIdTo3dOffset.eRef(devIdx, i) = pShort;
            }
        }
        mData->stencilIdTo3dOffset.updateDeviceData(backend, Neon::Backend::mainStreamIdx);
    }
    // Init the base grid
    bGrid::GridBase::init("bGrid",
                          backend,
                          domainSize,
                          Neon::domain::Stencil(),
                          mData->mNumActiveVoxel,
                          Neon::int32_3d(dataBlockSize, dataBlockSize, dataBlockSize),
                          spacingData,
                          origin);
    {  // setting launchParameters
        Neon::int32_3d blockSize(dataBlockSize, dataBlockSize, dataBlockSize);
        mData->launchParametersTable.forEachSeq([&](Neon::DataView               dw,
                                                    Neon::set::LaunchParameters& bLaunchParameters) {
            auto defEGridBlock = mData->blockViewGrid.getDefaultBlock();
            auto eGridParams = mData->blockViewGrid.getLaunchParameters(dw, defEGridBlock, 0);
            eGridParams.forEachSeq([&](Neon::SetIdx setIdx, Neon::sys::GpuLaunchInfo const& launchSingleDev) {
                auto eDomainGridSize = launchSingleDev.domainGrid();
                assert(eDomainGridSize.y == 1);
                assert(eDomainGridSize.z == 1);
                int nBlocks = eDomainGridSize.x;
                bLaunchParameters.get(setIdx).set(Neon::sys::GpuLaunchInfo::mode_e::cudaGridMode,
                                                  nBlocks, blockSize, 0);
            });
        });
    }
}


template <typename T, int C>
auto bGrid::newField(const std::string          name,
                     int                        cardinality,
                     T                          inactiveValue,
                     Neon::DataUse              dataUse,
                     const Neon::MemoryOptions& memoryOptions) const -> Field<T, C>
{
    Field<T, C> field(name, dataUse, memoryOptions, *this, cardinality, inactiveValue);
    return field;
}


template <typename LoadingLambda>
auto bGrid::newContainer(const std::string& name,
                         index_3d           blockSize,
                         size_t             sharedMem,
                         LoadingLambda      lambda,
                         Neon::Execution    execution) const -> Neon::set::Container
{
    Neon::set::Container kContainer = Neon::set::Container::factory(name,
                                                                    execution,
                                                                    Neon::set::internal::ContainerAPI::DataViewSupport::on,
                                                                    *this,
                                                                    lambda,
                                                                    blockSize,
                                                                    [sharedMem](const Neon::index_3d&) { return sharedMem; });
    return kContainer;
}


template <typename LoadingLambda>
auto bGrid::newContainer(const std::string& name,
                         LoadingLambda      lambda,
                         Neon::Execution    execution) const -> Neon::set::Container
{
    const Neon::index_3d& defaultBlockSize = this->getDefaultBlock();
    Neon::set::Container  kContainer = Neon::set::Container::factory(name,
                                                                     execution,
                                                                     Neon::set::internal::ContainerAPI::DataViewSupport::on,
                                                                     *this,
                                                                     lambda,
                                                                     defaultBlockSize,
                                                                     [](const Neon::index_3d&) { return size_t(0); });
    return kContainer;
}


}  // namespace Neon::domain::details::bGrid