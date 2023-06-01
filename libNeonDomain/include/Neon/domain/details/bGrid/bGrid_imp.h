#include "Neon/domain/details/bGrid/bGrid.h"

namespace Neon::domain::details::bGrid {

template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
template <typename ActiveCellLambda>
bGrid<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::bGrid(const Neon::Backend&         backend,
                                                                                                          const Neon::int32_3d&        domainSize,
                                                                                                          const ActiveCellLambda       activeCellLambda,
                                                                                                          const Neon::domain::Stencil& stencil,
                                                                                                          const double_3d&             spacingData,
                                                                                                          const double_3d&             origin)
    : bGrid(backend, domainSize, activeCellLambda, stencil, 1, spacingData, origin)
{
}

template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
template <typename ActiveCellLambda>
bGrid<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::bGrid(const Neon::Backend&         backend,
                                                                                                          const Neon::int32_3d&        domainSize,
                                                                                                          const ActiveCellLambda       activeCellLambda,
                                                                                                          const Neon::domain::Stencil& stencil,
                                                                                                          const int                    voxelSpacing,
                                                                                                          const double_3d&             spacingData,
                                                                                                          const double_3d&             origin)
{
    static_assert(memBlockSizeX >= userBlockSizeX);
    static_assert(memBlockSizeY >= userBlockSizeY);
    static_assert(memBlockSizeZ >= userBlockSizeZ);

    static_assert(memBlockSizeX % userBlockSizeX == 0);
    static_assert(memBlockSizeY % userBlockSizeY == 0);
    static_assert(memBlockSizeZ % userBlockSizeZ == 0);

    mData = std::make_shared<Data>();
    mData->init(backend);

    mData->voxelSpacing = voxelSpacing;
    mData->stencil = stencil;
    const index_3d defaultKernelBlockSize(memBlockSizeX, memBlockSizeY, memBlockSizeZ);

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
            dataBlockSize3D,
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
            spacingData * dataBlockSize3D,
            origin);

        mData->blockViewGrid = BlockViewGrid(egrid);
    }

    {  // Active bitmask
        int requiredWords = Span::getRequiredWordsForBlockBitMask();
        mData->activeBitMask = mData->blockViewGrid.template newField<typename Span::BitMaskWordType>("BitMask",
                                                                                                      requiredWords,
                                                                                                      0,
                                                                                                      Neon::DataUse::HOST_DEVICE, backend.getMemoryOptions(Span::activeMaskMemoryLayout));

        mData->mNumActiveVoxel = backend.devSet().template newDataSet<uint64_t>();

        mData->activeBitMask
            .getGrid()
            .template newContainer<Neon::Execution::host>(
                "activeBitMaskInit",
                [&](Neon::set::Loader& loader) {
                    auto bitMask = loader.load(mData->activeBitMask);
                    return [&, bitMask](const auto& bitMaskIdx) mutable {
                        auto       prtIdx = bitMask.prtID();
                        int        coutActive = 0;
                        auto const blockOrigin = bitMask.getGlobalIndex(bitMaskIdx);

                        for (int c = 0; c < bitMask.cardinality(); c++) {
                            bitMask(bitMaskIdx, c) = 0;
                        }

                        for (int k = 0; k < dataBlockSize3D.z; k++) {
                            for (int j = 0; j < dataBlockSize3D.y; j++) {
                                for (int i = 0; i < dataBlockSize3D.x; i++) {

                                    Neon::int32_3d                 localPosition(i, j, k);
                                    typename Span::BitMaskWordType mask;
                                    uint32_t                       wordIdx;

                                    Span::getMaskAndWordIdforBlockBitMask(i, j, k, NEON_OUT mask, NEON_OUT wordIdx);
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
                })
            .run(Neon::Backend::mainStreamIdx);

        mData->activeBitMask.updateDeviceData(Neon::Backend::mainStreamIdx);
        mData->activeBitMask.newHaloUpdate(Neon::set::StencilSemantic::standard,
                                           Neon::set::TransferMode::put,
                                           Neon::Execution::device)
            .run(Neon::Backend::mainStreamIdx);
    }


    {  // Neighbor blocks
        mData->blockConnectivity = mData->blockViewGrid.template newField<BlockIdx, 27>("blockConnectivity",
                                                                                        27,
                                                                                        Span::getInvalidBlockId(),
                                                                                        Neon::DataUse::HOST_DEVICE,
                                                                                        Neon::MemoryLayout::arrayOfStructs);

        mData->blockConnectivity.getGrid().template newContainer<Neon::Execution::host>(
                                              "blockConnectivityInit",
                                              [&](Neon::set::Loader& loader) {
                                                  auto blockConnectivity = loader.load(mData->blockConnectivity);
                                                  return [&, blockConnectivity](auto const& idx) mutable {
                                                      for (int8_t k = 0; k < 3; k++) {
                                                          for (int8_t j = 0; j < 3; j++) {
                                                              for (int8_t i = 0; i < 3; i++) {
                                                                  auto                                      targetDirection = i + 3 * j + 3 * 3 * k;
                                                                  BlockIdx                                  blockNghIdx = Span::getInvalidBlockId();
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
                                              })
            .run(Neon::Backend::mainStreamIdx);
        mData->blockConnectivity.updateDeviceData(Neon::Backend::mainStreamIdx);
    }

    // Initialization of the SPAN table
    mData->spanTable.forEachConfiguration([&](Neon::Execution execution,
                                              Neon::SetIdx    setIdx,
                                              Neon::DataView  dw,
                                              Span&           span) {
        span.mDataView = dw;
        switch (dw) {
            case Neon::DataView::STANDARD: {
                span.mFirstDataBlockOffset = 0;
                span.mDataView = dw;
                span.mActiveMask = mData->activeBitMask.getPartition(execution, setIdx, dw).mem();
                break;
            }
            case Neon::DataView::BOUNDARY: {
                span.mFirstDataBlockOffset = mData->partitioner1D.getSpanClassifier().countInternal(setIdx);
                span.mDataView = dw;
                span.mActiveMask = mData->activeBitMask.getPartition(execution, setIdx, dw).mem();

                break;
            }
            case Neon::DataView::INTERNAL: {
                span.mFirstDataBlockOffset = 0;
                span.mDataView = dw;
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
                          dataBlockSize3D,
                          spacingData,
                          origin);
    {  // setting launchParameters
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
                                                  nBlocks, dataBlockSize3D, 0);
            });
        });
    }
}

template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
template <typename T, int C>
auto bGrid<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::newField(const std::string   name,
                                                                                                                  int                 cardinality,
                                                                                                                  T                   inactiveValue,
                                                                                                                  Neon::DataUse       dataUse,
                                                                                                                  Neon::MemoryOptions memoryOptions) const -> Field<T, C>
{
    memoryOptions = this->getDevSet().sanitizeMemoryOption(memoryOptions);
    Field<T, C> field(name, dataUse, memoryOptions, *this, cardinality, inactiveValue);
    return field;
}

template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
template <typename T, int C>
auto bGrid<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::newBlockViewField(const std::string   name,
                                                                                                                           int                 cardinality,
                                                                                                                           T                   inactiveValue,
                                                                                                                           Neon::DataUse       dataUse,
                                                                                                                           Neon::MemoryOptions memoryOptions) const -> BlockViewGrid::Field<T, C>
{
    memoryOptions = this->getDevSet().sanitizeMemoryOption(memoryOptions);
    BlockViewGrid::Field<T, C> blockViewField = mData->blockViewGrid.template newField<T, C>(name, cardinality, inactiveValue, dataUse, memoryOptions);
    return blockViewField;
}

template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
template <Neon::Execution execution,
          typename LoadingLambda>
auto bGrid<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::newContainer(const std::string& name,
                                                                                                                      index_3d           blockSize,
                                                                                                                      size_t             sharedMem,
                                                                                                                      LoadingLambda      lambda) const -> Neon::set::Container
{
    Neon::set::Container kContainer = Neon::set::Container::factory<execution>(name,
                                                                               Neon::set::internal::ContainerAPI::DataViewSupport::on,
                                                                               *this,
                                                                               lambda,
                                                                               blockSize,
                                                                               [sharedMem](const Neon::index_3d&) { return sharedMem; });
    return kContainer;
}

template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
template <Neon::Execution execution,
          typename LoadingLambda>
auto bGrid<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::newContainer(const std::string& name,
                                                                                                                      LoadingLambda      lambda) const -> Neon::set::Container
{
    const Neon::index_3d& defaultBlockSize = this->getDefaultBlock();
    Neon::set::Container  kContainer = Neon::set::Container::factory<execution>(name,
                                                                               Neon::set::internal::ContainerAPI::DataViewSupport::on,
                                                                               *this,
                                                                               lambda,
                                                                               defaultBlockSize,
                                                                               [](const Neon::index_3d&) { return size_t(0); });
    return kContainer;
}

template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
auto bGrid<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::
    helpGetBlockViewGrid()
        const -> BlockViewGrid&
{
    return mData->blockViewGrid;
}

template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
auto bGrid<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::
    helpGetActiveBitMask()
        const -> BlockViewGrid::Field<uint64_t, 0>&
{
    return mData->activeBitMask;
}

template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
auto bGrid<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::
    helpGetBlockConnectivity()
        const -> BlockViewGrid::Field<BlockIdx, 27>&
{
    return mData->blockConnectivity;
}
template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
auto bGrid<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::
    helpGetDataBlockOriginField()
        const -> Neon::aGrid::Field<index_3d, 0>&
{
    return mData->mDataBlockOriginField;
}
template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
auto bGrid<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::getSpan(Neon::Execution execution,
                                                                                                                 SetIdx          setIdx,
                                                                                                                 Neon::DataView  dataView) -> const bGrid::Span&
{
    return mData->spanTable.getSpan(execution, setIdx, dataView);
}

template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
bGrid<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::~bGrid()
{
}
template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
auto bGrid<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::getSetIdx(const index_3d& idx) const -> int32_t
{
    typename GridBaseTemplate::CellProperties cellProperties;

    cellProperties.setIsInside(this->isInsideDomain(idx));
    if (!cellProperties.isInside()) {
        return -1;
    }
    Neon::SetIdx setIdx = cellProperties.getSetIdx();
    return setIdx;
}
template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
auto bGrid<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::getLaunchParameters(Neon::DataView dataView,
                                                                                                                             const index_3d&,
                                                                                                                             const size_t& sharedMem) const -> Neon::set::LaunchParameters
{
    auto res = mData->launchParametersTable.get(dataView);
    res.forEachSeq([&](SetIdx const& /*setIdx*/,
                       Neon::set::LaunchParameters::launchInfo_e& launchParams) -> void {
        launchParams.setShm(sharedMem);
    });
    return res;
}

template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
auto bGrid<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::
    helpGetStencilIdTo3dOffset()
        const -> Neon::set::MemSet<Neon::int8_3d>&
{
    return mData->stencilIdTo3dOffset;
}

template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
auto bGrid<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::isInsideDomain(const index_3d& idx) const -> bool
{
    // 1. check if the block is active
    const index_3d blockIdx3d = idx / dataBlockSize3D;
    auto           blockProperties = mData->blockViewGrid.getProperties(blockIdx3d);

    if (!blockProperties.isInside()) {
        return false;
    }
    // 2. The block is active, check the element on the block
    uint32_t                       wordCardinality;
    typename Span::BitMaskWordType mask;
    Span::getMaskAndWordIdforBlockBitMask(idx.x % dataBlockSize3D.x,
                                          idx.y % dataBlockSize3D.y,
                                          idx.z % dataBlockSize3D.z,
                                          NEON_OUT mask,
                                          NEON_OUT wordCardinality);
    auto activeBits = mData->activeBitMask.getReference(blockIdx3d, int(wordCardinality));
    return (activeBits & mask) != 0;
}

template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
auto bGrid<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::getProperties(const index_3d& idx)
    const -> typename GridBaseTemplate::CellProperties
{
    typename GridBaseTemplate::CellProperties cellProperties;

    cellProperties.setIsInside(this->isInsideDomain(idx));
    if (!cellProperties.isInside()) {
        return cellProperties;
    }

    if (this->getDevSet().setCardinality() == 1) {
        cellProperties.init(0, DataView::INTERNAL);
    } else {
        const index_3d blockIdx3d = idx / dataBlockSize3D;
        auto           blockViewProperty = mData->blockViewGrid.getProperties(blockIdx3d);

        cellProperties.init(blockViewProperty.getSetIdx(),
                            blockViewProperty.getDataView());
    }
    return cellProperties;
}

template <uint32_t memBlockSizeX, uint32_t memBlockSizeY, uint32_t memBlockSizeZ, uint32_t userBlockSizeX, uint32_t userBlockSizeY, uint32_t userBlockSizeZ>
auto bGrid<memBlockSizeX, memBlockSizeY, memBlockSizeZ, userBlockSizeX, userBlockSizeY, userBlockSizeZ>::helpGetSetIdxAndGridIdx(Neon::index_3d idx)
    const -> std::tuple<Neon::SetIdx, Idx>
{
    const index_3d blockIdx3d = idx / dataBlockSize3D;
    auto [setIdx, bvGridIdx] = mData->blockViewGrid.helpGetSetIdxAndGridIdx(blockIdx3d);
    Idx bIdx;
    bIdx.mDataBlockIdx = bvGridIdx.helpGet();
    bIdx.mInDataBlockIdx.x = idx.x % dataBlockSize3D.x;
    bIdx.mInDataBlockIdx.y = idx.y % dataBlockSize3D.y;
    bIdx.mInDataBlockIdx.z = idx.z % dataBlockSize3D.z;

    return {setIdx, bIdx};
}


}  // namespace Neon::domain::details::bGrid