#include "Neon/domain/details/bGridDisgMask/bGridMask.h"
#include "Neon/domain/tools/SpaceCurves.h"

namespace Neon::domain::details::disaggregated::bGridMask {

template <typename SBlock>
template <typename ActiveCellLambda>
bGridMask<SBlock>::bGridMask(const Neon::Backend&                         backend,
                     const Neon::int32_3d&                        domainSize,
                     const ActiveCellLambda                       activeCellLambda,
                     const Neon::domain::Stencil&                 stencil,
                     const double_3d&                             spacingData,
                     const double_3d&                             origin,
                     Neon::domain::tool::spaceCurves::EncoderType encoderType)
    : bGridMask(backend, domainSize, activeCellLambda, stencil, 1, spacingData, origin, encoderType)
{
}

template <typename SBlock>
template <typename ActiveCellLambda>
bGridMask<SBlock>::bGridMask(const Neon::Backend&                         backend,
                     const Neon::int32_3d&                        domainSize,
                     const ActiveCellLambda                       activeCellLambda,
                     const Neon::domain::Stencil&                 stencil,
                     const int                                    multiResDiscreteIdxSpacing,
                     const double_3d&                             spacingData,
                     const double_3d&                             origin,
                     Neon::domain::tool::spaceCurves::EncoderType encoderType)
{


    mData = std::make_shared<Data>();
    mData->init(backend);

    mData->mMultiResDiscreteIdxSpacing = multiResDiscreteIdxSpacing;
    mData->stencil = stencil;
    const index_3d defaultKernelBlockSize(SBlock::memBlockSizeX,
                                          SBlock::memBlockSizeY,
                                          SBlock::memBlockSizeZ);

    std::stringstream gridName;
    gridName << "bGridMask_" << SBlock::memBlockSizeX << "_"
             << SBlock::memBlockSizeY << "_"
             << SBlock::memBlockSizeZ;
    {
        auto nElementsPerPartition = backend.devSet().template newDataSet<size_t>(0);
        // We do an initialization with nElementsPerPartition to zero,
        // then we reset to the computed number.

        bGridMask::GridBase::init(gridName.str(),
                              backend,
                              domainSize,
                              stencil,
                              nElementsPerPartition,
                              defaultKernelBlockSize,
                              multiResDiscreteIdxSpacing,
                              origin,
                              encoderType,
                              defaultKernelBlockSize);
    }

    {  // Initialization of the partitioner
        using returTypeOfLambda = typename std::invoke_result<ActiveCellLambda, Neon::index_3d>::type;
        if constexpr (std::is_same_v<returTypeOfLambda, bool>) {
            mData->partitioner1D = Neon::domain::tool::Partitioner1D(
                backend,
                activeCellLambda,
                nullptr,
                SBlock::memBlockSize3D.template newType<int32_t>(),
                domainSize,
                Neon::domain::Stencil::s27_t(false),
                encoderType,
                multiResDiscreteIdxSpacing);
        } else if constexpr (std::is_same_v<returTypeOfLambda, ClassSelector>) {
            mData->partitioner1D = Neon::domain::tool::Partitioner1D(
                backend,
                [&](Neon::index_3d idx) {
                    return activeCellLambda(idx) != ClassSelector::outside;
                },
                //                [&](Neon::index_3d idx) {
                //                    return (activeCellLambda(idx) == details::cGrid::ClassSelector::beta)
                //                               ? Neon::domain::tool::partitioning::ByDomain::bc
                //                               : Neon::domain::tool::partitioning::ByDomain::bulk;
                //                },
                nullptr,
                SBlock::memBlockSize3D.template newType<int32_t>(),
                domainSize,
                Neon::domain::Stencil::s27_t(false),
                encoderType,
                multiResDiscreteIdxSpacing);
        } else {
            NEON_THROW_UNSUPPORTED_OPERATION("The user defined lambda must return a bool or a ClassSelector");
        }

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
    bGridMask::GridBase::init(gridName.str(),
                          backend,
                          domainSize,
                          Neon::domain::Stencil(),
                          mData->mNumActiveVoxel,
                          SBlock::memBlockSize3D.template newType<int32_t>(),
                          spacingData,
                          origin,
                          encoderType,
                          defaultKernelBlockSize);
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

    using returTypeOfLambda = typename std::invoke_result<ActiveCellLambda, Neon::index_3d>::type;
    this->mData->classFilterEnable = std::is_same_v<returTypeOfLambda, Neon::ClassSelector>;
    if (this->mData->classFilterEnable) {
        this->init_mask_field<ActiveCellLambda>(activeCellLambda);
        mData->alphaGrid = AlphaGrid(*this);
        mData->betaGrid = BetaGrid(*this);
    }
}

template <typename SBlock>
template <typename T, int C>
auto bGridMask<SBlock>::newField(const std::string   name,
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
auto bGridMask<SBlock>::newBlockViewField(const std::string   name,
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
auto bGridMask<SBlock>::newContainer(const std::string& name,
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
auto bGridMask<SBlock>::newContainer(const std::string& name,
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
template <Neon::Execution execution,
          typename LoadingLambda>
auto bGridMask<SBlock>::newAlphaContainer(const std::string& name,
                                      LoadingLambda      lambda) const -> Neon::set::Container
{
    if (this->mData->classFilterEnable) {
        auto kContainer = mData->alphaGrid.newContainer(name,
                                                        lambda);
        return kContainer;
    }
    Neon::NeonException ex("bGridMask");
    ex << "Alpha-Beta container has been disable as the class distribution was not providded diring the bGridMask contruction";
    NEON_THROW(ex);
}

template <typename SBlock>
template <Neon::Execution execution,
          typename LoadingLambda>
auto bGridMask<SBlock>::newBetaContainer(const std::string& name,
                                     LoadingLambda      lambda) const -> Neon::set::Container
{
    if (this->mData->classFilterEnable) {
        auto kContainer = mData->betaGrid.newContainer(name,
                                                       lambda);
        return kContainer;
    }
    Neon::NeonException ex("bGridMask");
    ex << "Alpha-Beta container has been disable as the class distribution was not providded diring the bGridMask contruction";
    NEON_THROW(ex);
}
template <typename SBlock>

template <Neon::Execution execution,
          typename LoadingLambdaAlpha,
          typename LoadingLambdaBeta>
auto bGridMask<SBlock>::newAlphaBetaContainer(const std::string& name,
                                          LoadingLambdaAlpha lambdaAlpha,
                                          LoadingLambdaBeta  lambdaBeta) const -> Neon::set::Container
{
    if (this->mData->classFilterEnable) {
        std::vector<Neon::set::Container> sequence;
        auto                              containerAlpha = mData->alphaGrid.newContainer(name + "Alpha",
                                                                                         lambdaAlpha);
        auto                              containerBeta = mData->betaGrid.newContainer(name + "Beta",
                                                                                       lambdaBeta);

        sequence.push_back(containerAlpha);
        sequence.push_back(containerBeta);

        Neon::set::Container exec = Neon::set::Container::factorySequence(name + "Sequence", sequence);
        return exec;
    }
    Neon::NeonException ex("bGridMask");
    ex << "Alpha-Beta container has been disable as the class distribution was not providded diring the bGridMask contruction";
    NEON_THROW(ex);
}

template <typename SBlock>
auto bGridMask<SBlock>::
    getBlockViewGrid()
        const -> BlockView::Grid&
{
    return mData->blockViewGrid;
}

template <typename SBlock>
auto bGridMask<SBlock>::
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
auto bGridMask<SBlock>::helGetMultiResDiscreteIdxSpacing() const
    -> std::enable_if_t<dummy == 1, int>
{
    return mData->mMultiResDiscreteIdxSpacing;
}

template <typename SBlock>
auto bGridMask<SBlock>::
    helpGetBlockConnectivity()
        const -> BlockView::Field<BlockIdx, 27>&
{
    return mData->blockConnectivity;
}
template <typename SBlock>
auto bGridMask<SBlock>::
    helpGetDataBlockOriginField()
        const -> Neon::aGrid::Field<index_3d, 0>&
{
    return mData->mDataBlockOriginField;
}
template <typename SBlock>
auto bGridMask<SBlock>::getSpan(Neon::Execution execution,
                            SetIdx          setIdx,
                            Neon::DataView  dataView) -> const bGridMask::Span&
{
    return mData->spanTable.getSpan(execution, setIdx, dataView);
}

template <typename SBlock>
bGridMask<SBlock>::~bGridMask()
{
}
template <typename SBlock>
auto bGridMask<SBlock>::getSetIdx(const index_3d& idx) const -> int32_t
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
auto bGridMask<SBlock>::getLaunchParameters(Neon::DataView dataView,
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
auto bGridMask<SBlock>::
    helpGetStencilIdTo3dOffset()
        const -> Neon::set::MemSet<Neon::int8_3d>&
{
    return mData->stencilIdTo3dOffset;
}

template <typename SBlock>
auto bGridMask<SBlock>::isInsideDomain(const index_3d& idx) const -> bool
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
auto bGridMask<SBlock>::getProperties(const index_3d& idx)
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
auto bGridMask<SBlock>::helpGetSetIdxAndGridIdx(Neon::index_3d idx)
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
auto bGridMask<SBlock>::helpGetPartitioner1D() -> Neon::domain::tool::Partitioner1D&
{
    return mData->partitioner1D;
}

template <typename SBlock>
template <typename ActiveCellLambda>
auto bGridMask<SBlock>::init_mask_field(ActiveCellLambda activeCellLambda) -> void
{
    std::cout << "THIS IS A TEST" << std::endl;
    using returTypeOfLambda = typename std::invoke_result<ActiveCellLambda, Neon::index_3d>::type;
    if constexpr (std::is_same_v<returTypeOfLambda, Neon::ClassSelector>) {


        auto maskField = this->newField<uint8_t, 1>("maskField", 1, 0, Neon::DataUse::HOST_DEVICE);
        maskField.getGrid().template newContainer<Neon::Execution::host>(
                               "maskFieldInit",
                               [&](Neon::set::Loader& loader) {
                                   auto maskFieldPartition = loader.load(maskField);
                                   return [activeCellLambda, maskFieldPartition](const auto& gIdx) mutable {
                                       auto                          globalPosition = maskFieldPartition.getGlobalIndex(gIdx);
                                       Neon::ClassSelector voxelClass = activeCellLambda(globalPosition);
                                       maskFieldPartition(gIdx, 0) = static_cast<uint8_t>(voxelClass);
                                   };
                               })
            .run(Neon::Backend::mainStreamIdx);
        this->getBackend().sync(Neon::Backend::mainStreamIdx);
        maskField.updateDeviceData(Neon::Backend::mainStreamIdx);
        this->getBackend().sync(Neon::Backend::mainStreamIdx);
        maskField.template ioToVtk<int>("maskField", "maskField");
        this->mData->maskClassField = maskField;
        return;
    }
}

template <typename SBlock>
auto bGridMask<SBlock>::helpGetClassField() -> Field<uint8_t, 1>&
{
    if (this->mData->classFilterEnable) {
        return mData->maskClassField;
    }
    Neon::NeonException ex("bGridMask");
    ex << "Alpha-Beta container has been disable as the class distribution was not providded diring the bGridMask contruction";
    NEON_THROW(ex);
}

}  // namespace Neon::domain::details::disaggregated::bGridMask