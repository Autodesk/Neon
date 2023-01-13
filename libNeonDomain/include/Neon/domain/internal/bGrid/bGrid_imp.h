#include "Neon/domain/internal/bGrid/bGrid.h"

namespace Neon::domain::internal::bGrid {

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
             const int                    blockSize,
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
    mData->blockSize = blockSize;
    mData->voxelSpacing = voxelSpacing;

    mData->mBlockOriginTo1D = Neon::domain::tool::PointHashTable<int32_t, uint32_t>(domainSize * voxelSpacing);

    mData->mNumBlocks = backend.devSet().template newDataSet<uint64_t>();


    mData->mNumActiveVoxel = backend.devSet().template newDataSet<uint64_t>();
    mData->mNumActiveVoxel[0] = 0;


    Neon::int32_3d numBlockInDomain(NEON_DIVIDE_UP(domainSize.x, blockSize),
                                    NEON_DIVIDE_UP(domainSize.y, blockSize),
                                    NEON_DIVIDE_UP(domainSize.z, blockSize));


    for (int bz = 0; bz < numBlockInDomain.z; bz++) {
        for (int by = 0; by < numBlockInDomain.y; by++) {
            for (int bx = 0; bx < numBlockInDomain.x; bx++) {

                int numVoxelsInBlock = 0;

                Neon::int32_3d blockOrigin(bx * blockSize * voxelSpacing,
                                           by * blockSize * voxelSpacing,
                                           bz * blockSize * voxelSpacing);

                for (int z = 0; z < blockSize; z++) {
                    for (int y = 0; y < blockSize; y++) {
                        for (int x = 0; x < blockSize; x++) {

                            const Neon::int32_3d id(blockOrigin.x + x * voxelSpacing,
                                                    blockOrigin.y + y * voxelSpacing,
                                                    blockOrigin.z + z * voxelSpacing);

                            if (id < domainSize * voxelSpacing && activeCellLambda(id)) {
                                numVoxelsInBlock++;
                            }
                        }
                    }
                }

                mData->mNumActiveVoxel[0] += numVoxelsInBlock;

                if (numVoxelsInBlock > 0) {
                    mData->mNumBlocks[0]++;
                    mData->mBlockOriginTo1D.addPoint(blockOrigin,
                                                     uint32_t(mData->mBlockOriginTo1D.size()));
                }
            }
        }
    }


    // Init the base grid
    bGrid::GridBase::init("bGrid",
                          backend,
                          domainSize,
                          Neon::domain::Stencil(),
                          mData->mNumActiveVoxel,
                          Neon::int32_3d(blockSize, blockSize, blockSize),
                          spacingData,
                          origin);


    Neon::MemoryOptions memOptionsAoS(Neon::DeviceType::CPU,
                                      Neon::Allocator::MALLOC,
                                      Neon::DeviceType::CUDA,
                                      ((backend.devType() == Neon::DeviceType::CUDA) ? Neon::Allocator::CUDA_MEM_DEVICE : Neon::Allocator::NULL_MEM),
                                      Neon::MemoryLayout::arrayOfStructs);

    //origin
    mData->mOrigin = backend.devSet().template newMemSet<Neon::int32_3d>({Neon::DataUse::IO_COMPUTE},
                                                                         1,
                                                                         memOptionsAoS,
                                                                         mData->mNumBlocks);


    //Stencil linear/relative index
    auto stencilNghSize = backend.devSet().template newDataSet<uint64_t>();
    for (int32_t c = 0; c < stencilNghSize.cardinality(); ++c) {
        stencilNghSize[c] = stencil.neighbours().size();
    }
    mData->mStencilNghIndex = backend.devSet().template newMemSet<nghIdx_t>({Neon::DataUse::IO_COMPUTE},
                                                                            1,
                                                                            memOptionsAoS,
                                                                            stencilNghSize);

    for (int32_t c = 0; c < mData->mStencilNghIndex.cardinality(); ++c) {
        SetIdx devID(c);
        for (uint64_t s = 0; s < stencil.neighbours().size(); ++s) {
            mData->mStencilNghIndex.eRef(c, s).x = static_cast<nghIdx_t::Integer>(stencil.neighbours()[s].x);
            mData->mStencilNghIndex.eRef(c, s).y = static_cast<nghIdx_t::Integer>(stencil.neighbours()[s].y);
            mData->mStencilNghIndex.eRef(c, s).z = static_cast<nghIdx_t::Integer>(stencil.neighbours()[s].z);
        }
    }


    // block bitmask
    mData->mActiveMaskSize = backend.devSet().template newDataSet<uint64_t>();
    for (int64_t i = 0; i < mData->mActiveMaskSize.size(); ++i) {
        mData->mActiveMaskSize[i] = mData->mNumBlocks[i] *
                                    NEON_DIVIDE_UP(blockSize * blockSize * blockSize,
                                                   Cell::sMaskSize);
    }

    mData->mActiveMask = backend.devSet().template newMemSet<uint32_t>({Neon::DataUse::IO_COMPUTE},
                                                                       1,
                                                                       memOptionsAoS,
                                                                       mData->mActiveMaskSize);


    // init bitmask to zero
    for (int32_t c = 0; c < mData->mActiveMask.cardinality(); ++c) {
        SetIdx devID(c);
        for (size_t i = 0; i < mData->mActiveMaskSize[c]; ++i) {
            mData->mActiveMask.eRef(devID, i) = 0;
        }
    }


    // Neighbor blocks
    mData->mNeighbourBlocks = backend.devSet().template newMemSet<uint32_t>({Neon::DataUse::IO_COMPUTE},
                                                                            26,
                                                                            memOptionsAoS,
                                                                            mData->mNumBlocks);
    // init neighbor blocks to invalid block id
    for (int32_t c = 0; c < mData->mNeighbourBlocks.cardinality(); ++c) {
        //TODO
        SetIdx devID(c);
        for (uint64_t i = 0; i < mData->mNumBlocks[c]; ++i) {
            for (int n = 0; n < 26; ++n) {
                mData->mNeighbourBlocks.eRef(devID, i, n) = std::numeric_limits<uint32_t>::max();
            }
        }
    }


    // loop over active blocks to populate the block origins, neighbors, and bitmask
    mData->mBlockOriginTo1D.forEach([&](const Neon::int32_3d blockOrigin, const uint32_t blockIdx) {
        //TODO need to figure out which device owns this block
        SetIdx devID(0);

        mData->mOrigin.eRef(devID, blockIdx) = blockOrigin;


        auto setCellActiveMask = [&](Cell::Location::Integer x, Cell::Location::Integer y, Cell::Location::Integer z) {
            Cell cell(x, y, z);
            cell.mBlockID = blockIdx;
            cell.mBlockSize = blockSize;
            mData->mActiveMask.eRef(devID, cell.getBlockMaskStride() + cell.getMaskLocalID(), 0) |= 1 << cell.getMaskBitPosition();
        };


        //set active mask and child ID
        for (Cell::Location::Integer z = 0; z < blockSize; z++) {
            for (Cell::Location::Integer y = 0; y < blockSize; y++) {
                for (Cell::Location::Integer x = 0; x < blockSize; x++) {

                    const Neon::int32_3d id(blockOrigin.x + x * voxelSpacing,
                                            blockOrigin.y + y * voxelSpacing,
                                            blockOrigin.z + z * voxelSpacing);

                    if (id < domainSize * voxelSpacing && activeCellLambda(id)) {
                        setCellActiveMask(x, y, z);
                    }
                }
            }
        }


        //set neighbor blocks
        for (int16_t k = -1; k < 2; k++) {
            for (int16_t j = -1; j < 2; j++) {
                for (int16_t i = -1; i < 2; i++) {
                    if (i == 0 && j == 0 && k == 0) {
                        continue;
                    }

                    Neon::int32_3d neighbourBlockOrigin;
                    neighbourBlockOrigin.x = i * blockSize * voxelSpacing + blockOrigin.x;
                    neighbourBlockOrigin.y = j * blockSize * voxelSpacing + blockOrigin.y;
                    neighbourBlockOrigin.z = k * blockSize * voxelSpacing + blockOrigin.z;

                    auto neighbour_it = mData->mBlockOriginTo1D.getMetadata(neighbourBlockOrigin);

                    if (neighbour_it) {
                        int16_3d block_offset(i, j, k);
                        mData->mNeighbourBlocks.eRef(devID,
                                                     blockIdx,
                                                     Cell::getNeighbourBlockID(block_offset)) = *neighbour_it;
                    }
                }
            }
        }
    });


    if (backend.devType() == Neon::DeviceType::CUDA) {
        mData->mActiveMask.updateCompute(backend, 0);
        mData->mOrigin.updateCompute(backend, 0);
        mData->mNeighbourBlocks.updateCompute(backend, 0);
        mData->mStencilNghIndex.updateCompute(backend, 0);
    }


    for (const auto& dv : {Neon::DataView::STANDARD,
                           Neon::DataView::INTERNAL,
                           Neon::DataView::BOUNDARY}) {

        int dv_id = DataViewUtil::toInt(dv);
        if (dv_id > 2) {
            NeonException exp("bGrid");
            exp << "Inconsistent enumeration for DataView_t";
            NEON_THROW(exp);
        }

        mData->mPartitionIndexSpace[dv_id] = backend.devSet().template newDataSet<PartitionIndexSpace>();

        for (int gpuIdx = 0; gpuIdx < backend.devSet().setCardinality(); gpuIdx++) {
            mData->mPartitionIndexSpace[dv_id][gpuIdx].mDataView = dv;
            mData->mPartitionIndexSpace[dv_id][gpuIdx].mDomainSize = domainSize * voxelSpacing;
            mData->mPartitionIndexSpace[dv_id][gpuIdx].mBlockSize = blockSize;
            mData->mPartitionIndexSpace[dv_id][gpuIdx].mSpacing = voxelSpacing;
            mData->mPartitionIndexSpace[dv_id][gpuIdx].mNumBlocks = static_cast<uint32_t>(mData->mNumBlocks[gpuIdx]);
            mData->mPartitionIndexSpace[dv_id][gpuIdx].mHostActiveMask = mData->mActiveMask.rawMem(gpuIdx, Neon::DeviceType::CPU);
            mData->mPartitionIndexSpace[dv_id][gpuIdx].mDeviceActiveMask = mData->mActiveMask.rawMem(gpuIdx, Neon::DeviceType::CUDA);
            mData->mPartitionIndexSpace[dv_id][gpuIdx].mHostBlockOrigin = mData->mOrigin.rawMem(gpuIdx, Neon::DeviceType::CPU);
            mData->mPartitionIndexSpace[dv_id][gpuIdx].mDeviceBlockOrigin = mData->mOrigin.rawMem(gpuIdx, Neon::DeviceType::CUDA);
        }
    }
}

template <typename T, int C>
auto bGrid::newField(const std::string          name,
                     int                        cardinality,
                     T                          inactiveValue,
                     Neon::DataUse              dataUse,
                     const Neon::MemoryOptions& memoryOptions) const -> Field<T, C>
{
    bField<T, C> field(name, *this, cardinality, inactiveValue, dataUse, memoryOptions, Neon::domain::haloStatus_et::ON);

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
    const Neon::index_3d& defaultBlockSize = getDefaultBlock();
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
    auto pattern = Neon::PatternScalar<T>(getBackend(), Neon::sys::patterns::Engine::CUB);
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

                if (dataView != Neon::DataView::STANDARD && getBackend().devSet().setCardinality() == 1) {
                    NeonException exc("bGrid");
                    exc << "Reduction operation can only run on standard data view when the number of partitions/GPUs is 1";
                    NEON_THROW(exc);
                }

                if (getBackend().devType() == Neon::DeviceType::CUDA) {
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

                if (dataView != Neon::DataView::STANDARD && getBackend().devSet().setCardinality() == 1) {
                    NeonException exc("bGrid");
                    exc << "Reduction operation can only run on standard data view when the number of partitions/GPUs is 1";
                    NEON_THROW(exc);
                }

                if (getBackend().devType() == Neon::DeviceType::CUDA) {
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
}  // namespace Neon::domain::internal::bGrid