#include "Neon/domain/details/bGrid/bGrid.h"

namespace Neon::domain::details::bGrid {

template <typename SBlock>
template <typename ActiveCellLambda>
bGrid<SBlock>::bGrid(const Neon::Backend&         backend,
                     const Neon::int32_3d&        domainSize,
                     const ActiveCellLambda       activeCellLambda,
                     const Neon::domain::Stencil& stencil,
                     const double_3d&             spacingData,
                     const double_3d&             origin)
    : bGrid(backend, domainSize, activeCellLambda, stencil, 1, spacingData, origin)
{
}

template <typename SBlock>
template <typename ActiveCellLambda>
bGrid<SBlock>::bGrid(const Neon::Backend&         backend,
                     const Neon::int32_3d&        domainSize,
                     const ActiveCellLambda       activeCellLambda,
                     const Neon::domain::Stencil& stencil,
                     const int                    multiResDiscreteIdxSpacing,
                     const double_3d&             spacingData,
                     const double_3d&             origin)
{


    mData = std::make_shared<Data>();
    mData->init(backend);

    mData->mMultiResDiscreteIdxSpacing = multiResDiscreteIdxSpacing;
    mData->stencil = stencil;
    const index_3d defaultKernelBlockSize(SBlock::memBlockSizeX,
                                          SBlock::memBlockSizeY,
                                          SBlock::memBlockSizeZ);

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
                              multiResDiscreteIdxSpacing,
                              origin);
    }

    {  // Initialization of the partitioner

        mData->partitioner1D = Neon::domain::tool::Partitioner1D(
            backend,
            activeCellLambda,
            [](Neon::index_3d /*idx*/) { return false; },
            SBlock::memBlockSize3D.template newType<int32_t>(),
            domainSize,
            Neon::domain::Stencil::s27_t(false),
            multiResDiscreteIdxSpacing);

        mData->mDataBlockOriginField = mData->partitioner1D.getGlobalMapping();
        mData->mStencil3dTo1dOffset = mData->partitioner1D.getStencil3dTo1dOffset();
        mData->memoryGrid = mData->partitioner1D.getMemoryGrid();
    }

    {  // BlockViewGrid
        Neon::domain::details::eGrid::eGrid egrid(
            backend,
            mData->partitioner1D.getBlockSpan(),
            mData->partitioner1D,
            Neon::domain::Stencil::s27_t(false),
            spacingData * SBlock::memBlockSize3D,
            origin);

        mData->blockViewGrid = BlockView::Grid(egrid);
    }

    {  // Active bitmask
        mData->activeBitField = mData->blockViewGrid.template newField<typename SBlock::BitMask, 1>(
            "BlockViewBitMask",
            1,
            [] {
                typename SBlock::BitMask outsideBitMask;
                outsideBitMask.reset();
                return outsideBitMask;
            }(),
            Neon::DataUse::HOST_DEVICE, backend.getMemoryOptions(BlockView::layout));

        mData->mNumActiveVoxel = backend.devSet().template newDataSet<uint64_t>();

        mData->activeBitField
            .getGrid()
            .template newContainer<Neon::Execution::host>(
                "activeBitMaskInit",
                [&, this](Neon::set::Loader& loader) {
                    auto bitMaskPartition = loader.load(mData->activeBitField);
                    return [&, bitMaskPartition](const auto& bitMaskIdx) mutable {
                        auto                      prtIdx = bitMaskPartition.prtID();
                        int                       countActive = 0;
                        auto const                blockOrigin = bitMaskPartition.getGlobalIndex(bitMaskIdx);
                        typename SBlock::BitMask& bitMask = bitMaskPartition(bitMaskIdx, 0);
                        bitMask.reset();

                        for (int k = 0; k < SBlock::memBlockSize3D.template newType<int32_t>().z; k++) {
                            for (int j = 0; j < SBlock::memBlockSize3D.template newType<int32_t>().y; j++) {
                                for (int i = 0; i < SBlock::memBlockSize3D.template newType<int32_t>().x; i++) {
                                    auto       globalPosition = blockOrigin + Neon::int32_3d(i * this->mData->mMultiResDiscreteIdxSpacing,
                                                                                       j * this->mData->mMultiResDiscreteIdxSpacing,
                                                                                       k * this->mData->mMultiResDiscreteIdxSpacing);
                                    bool const isInDomain = globalPosition < domainSize * this->mData->mMultiResDiscreteIdxSpacing;
                                    bool const isActive = activeCellLambda(globalPosition);
                                    if (isActive && isInDomain) {
                                        countActive++;
                                        bitMask.setActive(i, j, k);
                                    }
                                }
                            }
                        }
#pragma omp critical
                        {
                            this->mData->mNumActiveVoxel[prtIdx] += countActive;
                        }
                    };
                })
            .run(Neon::Backend::mainStreamIdx);


        mData->activeBitField.updateDeviceData(Neon::Backend::mainStreamIdx);
        this->getBackend().sync(Neon::Backend::mainStreamIdx);
        mData->activeBitField.newHaloUpdate(Neon::set::StencilSemantic::standard,
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
                                                                      blockNghIdx = static_cast<BlockIdx>(nghIdx.helpGet());
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
                span.mActiveMask = mData->activeBitField.getPartition(execution, setIdx, dw).mem();
                break;
            }
            case Neon::DataView::BOUNDARY: {
                span.mFirstDataBlockOffset = mData->partitioner1D.getSpanClassifier().countInternal(setIdx);
                span.mDataView = dw;
                span.mActiveMask = mData->activeBitField.getPartition(execution, setIdx, dw).mem();

                break;
            }
            case Neon::DataView::INTERNAL: {
                span.mFirstDataBlockOffset = 0;
                span.mDataView = dw;
                span.mActiveMask = mData->activeBitField.getPartition(execution, setIdx, dw).mem();
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
                Neon::int8_3d pShort = pLong.newType<int8_t>();
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
                          SBlock::memBlockSize3D.template newType<int32_t>(),
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
                int nBlocks = static_cast<int>(eDomainGridSize.x);
                bLaunchParameters.get(setIdx).set(Neon::sys::GpuLaunchInfo::mode_e::cudaGridMode,
                                                  nBlocks, SBlock::memBlockSize3D.template newType<int32_t>(), 0);
            });
        });
    }
}

template <typename SBlock>
template <typename T, int C>
auto bGrid<SBlock>::newField(const std::string   name,
                             int                 cardinality,
                             T                   inactiveValue,
                             Neon::DataUse       dataUse,
                             Neon::MemoryOptions memoryOptions) const -> Field<T, C>
{
    memoryOptions = this->getDevSet().sanitizeMemoryOption(memoryOptions);
    Field<T, C> field(name, dataUse, memoryOptions, *this, cardinality, inactiveValue);
    return field;
}

template <typename SBlock>
template <typename T, int C>
auto bGrid<SBlock>::newBlockViewField(const std::string   name,
                                      int                 cardinality,
                                      T                   inactiveValue,
                                      Neon::DataUse       dataUse,
                                      Neon::MemoryOptions memoryOptions) const -> BlockView::Field<T, C>
{
    memoryOptions = this->getDevSet().sanitizeMemoryOption(memoryOptions);
    BlockView::Field<T, C> blockViewField = mData->blockViewGrid.template newField<T, C>(name, cardinality, inactiveValue, dataUse, memoryOptions);
    return blockViewField;
}

template <typename SBlock>
template <Neon::Execution execution,
          typename LoadingLambda>
auto bGrid<SBlock>::newContainer(const std::string& name,
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

template <typename SBlock>
template <Neon::Execution execution,
          typename LoadingLambda>
auto bGrid<SBlock>::newContainer(const std::string& name,
                                 LoadingLambda      lambda) const -> Neon::set::Container
{
    const Neon::index_3d& defaultBlockSize = this->getDefaultBlock();
    Neon::set::Container  kContainer = Neon::set::Container::factory<execution>(name,
                                                                               Neon::set::internal::ContainerAPI::DataViewSupport::on,
                                                                               *this,
                                                                               lambda,
                                                                               defaultBlockSize,
                                                                               [](const Neon::index_3d&) { return 0; });
    return kContainer;
}

template <typename SBlock>
auto bGrid<SBlock>::
    getBlockViewGrid()
        const -> BlockView::Grid&
{
    return mData->blockViewGrid;
}

template <typename SBlock>
auto bGrid<SBlock>::
    getActiveBitMask()
        const -> BlockView::Field<typename SBlock::BitMask, 1>&
{
    return mData->activeBitField;
}

/**
 * Helper function to retrieve the discrete index spacing used for the multi-resolution
 */
template <typename SBlock>
template <int dummy>
auto bGrid<SBlock>::helGetMultiResDiscreteIdxSpacing() const
    -> std::enable_if_t<dummy == 1, int>
{
    return mData->mMultiResDiscreteIdxSpacing;
}

template <typename SBlock>
auto bGrid<SBlock>::
    helpGetBlockConnectivity()
        const -> BlockView::Field<BlockIdx, 27>&
{
    return mData->blockConnectivity;
}
template <typename SBlock>
auto bGrid<SBlock>::
    helpGetDataBlockOriginField()
        const -> Neon::aGrid::Field<index_3d, 0>&
{
    return mData->mDataBlockOriginField;
}
template <typename SBlock>
auto bGrid<SBlock>::getSpan(Neon::Execution execution,
                            SetIdx          setIdx,
                            Neon::DataView  dataView) -> const bGrid::Span&
{
    return mData->spanTable.getSpan(execution, setIdx, dataView);
}

template <typename SBlock>
bGrid<SBlock>::~bGrid()
{
}
template <typename SBlock>
auto bGrid<SBlock>::getSetIdx(const index_3d& idx) const -> int32_t
{
    typename GridBaseTemplate::CellProperties cellProperties;

    cellProperties.setIsInside(this->isInsideDomain(idx));
    if (!cellProperties.isInside()) {
        return -1;
    }
    Neon::SetIdx setIdx = cellProperties.getSetIdx();
    return setIdx;
}
template <typename SBlock>
auto bGrid<SBlock>::getLaunchParameters(Neon::DataView dataView,
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

template <typename SBlock>
auto bGrid<SBlock>::
    helpGetStencilIdTo3dOffset()
        const -> Neon::set::MemSet<Neon::int8_3d>&
{
    return mData->stencilIdTo3dOffset;
}

template <typename SBlock>
auto bGrid<SBlock>::isInsideDomain(const index_3d& idx) const -> bool
{
    // 1. check if the block is active
    const BlockView::index_3d blockIdx3d = idx / (SBlock::memBlockSize3D.template newType<int32_t>() * mData->mMultiResDiscreteIdxSpacing);
    auto                      blockProperties = mData->blockViewGrid.getProperties(blockIdx3d);

    if (!blockProperties.isInside()) {
        return false;
    }
    // 2. The block is active, check the element in the block
    typename SBlock::BitMask const& bitMask = mData->activeBitField.getReference(blockIdx3d, 0);

    bool isActive = bitMask.isActive((idx.x / mData->mMultiResDiscreteIdxSpacing) % SBlock::memBlockSize3D.x,
                                     (idx.y / mData->mMultiResDiscreteIdxSpacing) % SBlock::memBlockSize3D.y,
                                     (idx.z / mData->mMultiResDiscreteIdxSpacing) % SBlock::memBlockSize3D.z);
    return isActive;
}

template <typename SBlock>
auto bGrid<SBlock>::getProperties(const index_3d& idx)
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
        const index_3d blockIdx3d = idx / SBlock::memBlockSize3D.template newType<int32_t>();
        auto           blockViewProperty = mData->blockViewGrid.getProperties(blockIdx3d);

        cellProperties.init(blockViewProperty.getSetIdx(),
                            blockViewProperty.getDataView());
    }
    return cellProperties;
}

template <typename SBlock>
auto bGrid<SBlock>::helpGetSetIdxAndGridIdx(Neon::index_3d idx)
    const -> std::tuple<Neon::SetIdx, Idx>
{
    const index_3d blockIdx3d = idx / (SBlock::memBlockSize3D.template newType<int32_t>() * mData->mMultiResDiscreteIdxSpacing);
    auto [setIdx, bvGridIdx] = mData->blockViewGrid.helpGetSetIdxAndGridIdx(blockIdx3d);
    Idx bIdx;
    bIdx.mDataBlockIdx = bvGridIdx.helpGet();
    bIdx.mInDataBlockIdx.x = static_cast<typename Idx::InDataBlockIdx::Integer>((idx.x / mData->mMultiResDiscreteIdxSpacing) % SBlock::memBlockSize3D.x);
    bIdx.mInDataBlockIdx.y = static_cast<typename Idx::InDataBlockIdx::Integer>((idx.y / mData->mMultiResDiscreteIdxSpacing) % SBlock::memBlockSize3D.y);
    bIdx.mInDataBlockIdx.z = static_cast<typename Idx::InDataBlockIdx::Integer>((idx.z / mData->mMultiResDiscreteIdxSpacing) % SBlock::memBlockSize3D.z);

    return {setIdx, bIdx};
}

template <typename SBlock>
auto bGrid<SBlock>::helpGetPartitioner1D() -> Neon::domain::tool::Partitioner1D&
{
    return mData->partitioner1D;
}

}  // namespace Neon::domain::details::bGrid