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
            getBlockSize(),
            domainSize,
            stencil,
            1);

        mData->mConnectivityAField = mData->partitioner1D.getConnectivity();
        mData->mDataBlockToGlobalMappingAField = mData->partitioner1D.getGlobalMapping();
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
            spacingData * getBlockSize(),
            origin);

        mData->blockViewGrid = BlockViewGrid(egrid);
    }

    {  // Active bitmask
        Neon::domain::details::eGrid::eGrid egrid(
            backend,
            mData->partitioner1D.getBlockSpan(),
            mData->partitioner1D,
            stencil,
            spacingData * getBlockSize(),
            origin);
        int requiredWords = bSpan::getRequiredWords(getBlockSize());
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
                    return [&](const auto& idx) {
                        auto           prtIdx = bitMask.prtID();
                        int            coutActive = 0;
                        auto const     idx3d = bitMask.getGlobalIndex(idx);
                        Neon::index_3d blockSize3d(dataBlockSize, dataBlockSize, dataBlockSize);
                        auto const     blockOrigin = idx3d * dataBlockSize;
                        for (int k = 0; k < dataBlockSize; k++) {
                            for (int j = 0; j < dataBlockSize; j++) {
                                for (int i = 0; i < dataBlockSize; i++) {
                                    for (int c = 0; c < bitMask.cardinality(); c++) {
                                        bitMask(idx3d, c) = 0;
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
                                        bitMask(idx3d, targetCard) |= mask;
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
                                                                               getInvalidBlockId(),
                                                                               Neon::DataUse::HOST_DEVICE,
                                                                               Neon::MemoryLayout::arrayOfStructs);

        mData->blockViewGrid.newContainer(
                                "blockConnectivityInit",
                                [&](Neon::set::Loader& loader) {
                                    auto blockConnectivity = loader.load(mData->blockConnectivity);
                                    return [&](auto const& idx) {
                                        for (int k = 0; k < 3; k++) {
                                            for (int j = 0; j < 3; j++) {
                                                for (int i = 0; i < 3; i++) {
                                                    auto     targetDirection = i + 3 * j + 3 * 3 * k;
                                                    BlockIdx nghIdx = getInvalidBlockId();
                                                    blockConnectivity.getNghIndex(idx, {i, j, k}, nghIdx);
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


template <typename T>
auto bGrid::newPatternScalar() const -> Neon::template PatternScalar<T>
{
    // TODO this sets the numBlocks for only Standard dataView.
    auto pattern = Neon::PatternScalar<T>(this->getBackend(), Neon::sys::patterns::Engine::CUB);
    for (SetIdx id = 0; id < mData->mNumBlocks.cardinality(); id++) {
        pattern.getBlasSet(Neon::DataView::STANDARD).getBlas(id.idx()).setNumBlocks(uint32_t(mData->mNumBlocks[id]));
    }
    return pattern;
}


template <typename T>
auto bGrid::dot(const std::string&               name,
                Field<T>&                        input1,
                Field<T>&                        input2,
                Neon::template PatternScalar<T>& scalar) const -> Neon::set::Container
{
    return Neon::set::Container::factoryOldManaged(
        name,
        Neon::set::internal::ContainerAPI::DataViewSupport::on,
        Neon::set::ContainerPatternType::reduction,
        *this, [&](Neon::set::Loader& loader) {
            loader.load(input1);
            if (input1.getUid() != input2.getUid()) {
                loader.load(input2);
            }

            return [&](int streamIdx, Neon::DataView dataView) mutable -> void {
                if (dataView != Neon::DataView::STANDARD) {
                    NeonException exc("bGrid");
                    exc << "Reduction operation on bGrid works only on standard dataview";
                    exc << "Input dataview is" << Neon::DataViewUtil::toString(dataView);
                    NEON_THROW(exc);
                }

                if (dataView != Neon::DataView::STANDARD && this->getBackend().devSet().setCardinality() == 1) {
                    NeonException exc("bGrid");
                    exc << "Reduction operation can only run on standard data view when the number of partitions/GPUs is 1";
                    NEON_THROW(exc);
                }

                if (this->getBackend().devType() == Neon::DeviceType::CUDA) {
                    scalar.setStream(streamIdx, dataView);

                    // calc dot product and store results on device
                    input1.dot(scalar.getBlasSet(dataView),
                               input2,
                               scalar.getTempMemory(dataView, Neon::DeviceType::CUDA),
                               dataView);

                    // move to results to host
                    scalar.getTempMemory(dataView,
                                         Neon::DeviceType::CPU)
                        .template updateFrom<Neon::run_et::et::async>(
                            scalar.getBlasSet(dataView).getStream(),
                            scalar.getTempMemory(dataView, Neon::DeviceType::CUDA));

                    // sync
                    scalar.getBlasSet(dataView).getStream().sync();

                    // read the results
                    scalar() = scalar.getTempMemory(dataView, Neon::DeviceType::CPU).elRef(0, 0, 0);
                } else {

                    scalar() = 0;
                    input1.forEachActiveCell(
                        [&](const Neon::index_3d& idx,
                            const int&            cardinality,
                            T&                    in1) {
                            scalar() += in1 * input2(idx, cardinality);
                        },
                        Neon::computeMode_t::computeMode_e::seq);
                }
            };
        });
}


template <typename T>
auto bGrid::norm2(const std::string&               name,
                  Field<T>&                        input,
                  Neon::template PatternScalar<T>& scalar) const -> Neon::set::Container
{
    return Neon::set::Container::factoryOldManaged(
        name,
        Neon::set::internal::ContainerAPI::DataViewSupport::on,
        Neon::set::ContainerPatternType::reduction,
        *this, [&](Neon::set::Loader& loader) {
            loader.load(input);


            return [&](int streamIdx, Neon::DataView dataView) mutable -> void {
                if (dataView != Neon::DataView::STANDARD) {
                    NeonException exc("bGrid");
                    exc << "Reduction operation on bGrid works only on standard dataview";
                    exc << "Input dataview is" << Neon::DataViewUtil::toString(dataView);
                    NEON_THROW(exc);
                }

                if (dataView != Neon::DataView::STANDARD && this->getBackend().devSet().setCardinality() == 1) {
                    NeonException exc("bGrid");
                    exc << "Reduction operation can only run on standard data view when the number of partitions/GPUs is 1";
                    NEON_THROW(exc);
                }

                if (this->getBackend().devType() == Neon::DeviceType::CUDA) {
                    scalar.setStream(streamIdx, dataView);

                    // calc dot product and store results on device
                    input.norm2(scalar.getBlasSet(dataView),
                                scalar.getTempMemory(dataView, Neon::DeviceType::CUDA),
                                dataView);

                    // move to results to host
                    scalar.getTempMemory(dataView,
                                         Neon::DeviceType::CPU)
                        .template updateFrom<Neon::run_et::et::async>(
                            scalar.getBlasSet(dataView).getStream(),
                            scalar.getTempMemory(dataView, Neon::DeviceType::CUDA));

                    // sync
                    scalar.getBlasSet(dataView).getStream().sync();

                    // read the results
                    scalar() = scalar.getTempMemory(dataView, Neon::DeviceType::CPU).elRef(0, 0, 0);
                } else {

                    scalar() = 0;
                    input.forEachActiveCell(
                        [&]([[maybe_unused]] const Neon::index_3d& idx,
                            [[maybe_unused]] const int&            cardinality,
                            T&                                     in) {
                            scalar() += in * in;
                        },
                        Neon::computeMode_t::computeMode_e::seq);
                }
                scalar() = std::sqrt(scalar());
            };
        });
}
}  // namespace Neon::domain::details::bGrid