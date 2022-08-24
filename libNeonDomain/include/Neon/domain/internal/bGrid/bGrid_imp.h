#include "Neon/domain/internal/bGrid/bGrid.h"

namespace Neon::domain::internal::bGrid {

template <typename ActiveCellLambda>
bGrid::bGrid(const Neon::Backend&         backend,
             const Neon::int32_3d&        domainSize,
             const ActiveCellLambda       activeCellLambda,
             const Neon::domain::Stencil& stencil,
             const double_3d&             spacingData,
             const double_3d&             origin)
    : bGrid(backend, domainSize, {activeCellLambda}, stencil, sBGridDefaultDescriptor, spacingData, origin)
{
}

template <typename Descriptor>
bGrid::bGrid(const Neon::Backend&                                    backend,
             const Neon::int32_3d&                                   domainSize,
             std::vector<std::function<bool(const Neon::index_3d&)>> activeCellLambda,
             const Neon::domain::Stencil&                            stencil,
             const Descriptor                                        descriptor,
             const double_3d&                                        spacingData,
             const double_3d&                                        origin)
{

    if (backend.devSet().setCardinality() > 1) {
        NeonException exp("bGrid");
        exp << "bGrid only supported on a single GPU";
        NEON_THROW(exp);
    }

    mData = std::make_shared<Data>();


    mData->descriptor.resize(descriptor.getDepth());
    for (int l = 0; l < descriptor.getDepth(); ++l) {
        mData->descriptor[l] = descriptor.getLevelRefFactor(l);
    }

    mData->mBlockOriginTo1D.resize(descriptor.getDepth());
    mData->mBlockOriginTo1D[0] = Neon::domain::tool::PointHashTable<int32_t, uint32_t>(domainSize);

    mData->mNumBlocks.resize(descriptor.getDepth());
    for (auto& nb : mData->mNumBlocks) {
        nb = backend.devSet().template newDataSet<uint64_t>();
    }

    mData->mNumActiveVoxel.resize(descriptor.getDepth());
    for (auto& av : mData->mNumActiveVoxel) {
        av = backend.devSet().template newDataSet<uint64_t>();
    }
    mData->mNumActiveVoxel[0][0] = 0;


    std::vector<Neon::int32_3d> numBlockInDomain(descriptor.getDepth());
    numBlockInDomain[0].set(NEON_DIVIDE_UP(domainSize.x, descriptor.get0LevelRefFactor()),
                            NEON_DIVIDE_UP(domainSize.y, descriptor.get0LevelRefFactor()),
                            NEON_DIVIDE_UP(domainSize.z, descriptor.get0LevelRefFactor()));

    for (int i = 1; i < descriptor.getDepth(); ++i) {
        mData->mNumActiveVoxel[i][0] = 0;

        numBlockInDomain[i].set(NEON_DIVIDE_UP(numBlockInDomain[i - 1].x, descriptor.getLevelRefFactor(i)),
                                NEON_DIVIDE_UP(numBlockInDomain[i - 1].y, descriptor.getLevelRefFactor(i)),
                                NEON_DIVIDE_UP(numBlockInDomain[i - 1].z, descriptor.getLevelRefFactor(i)));

        mData->mBlockOriginTo1D[i] = Neon::domain::tool::PointHashTable<int32_t, uint32_t>(domainSize);
    }

    // Number of active voxels per partition
    // Loop over all blocks and voxels in blocks to count the number of active
    // voxels and active blocks for allocation
    //Start by calculating the number of blocks on the finest level (level 0)
    //then for level>0, since we know how many blocks we need at the finest level,
    // we can calculate the number of blocks we need at every other level starting with level 1 and going up the grid/tree
    //TODO we may need to add more blocks in intermediate for strongly-balanced grid
    for (int l = 0; l < descriptor.getDepth(); ++l) {
        uint32_t  blockId = 0;
        const int ref_factor = descriptor.getLevelRefFactor(l);
        const int ref_factor_recurse = descriptor.getRefFactorRecurse(l);
        const int prv_ref_factor_recurse = descriptor.getRefFactorRecurse(l - 1);

        for (int bz = 0; bz < numBlockInDomain[l].z; bz++) {
            for (int by = 0; by < numBlockInDomain[l].y; by++) {
                for (int bx = 0; bx < numBlockInDomain[l].x; bx++) {

                    bool isActiveBlock = false;

                    Neon::int32_3d blockOrigin(bx * ref_factor_recurse,
                                               by * ref_factor_recurse,
                                               bz * ref_factor_recurse);

                    for (int z = 0; z < ref_factor; z++) {
                        for (int y = 0; y < ref_factor; y++) {
                            for (int x = 0; x < ref_factor; x++) {

                                if (l == 0) {
                                    const Neon::int32_3d id(blockOrigin.x + x,
                                                            blockOrigin.y + y,
                                                            blockOrigin.z + z);

                                    if (id < domainSize) {
                                        if (activeCellLambda[l](id)) {
                                            isActiveBlock = true;
                                            mData->mNumActiveVoxel[l][0]++;
                                        }
                                    }
                                } else {
                                    //This is the corresponding block origin in the previous level
                                    const Neon::int32_3d id(blockOrigin.x + x * prv_ref_factor_recurse,
                                                            blockOrigin.y + y * prv_ref_factor_recurse,
                                                            blockOrigin.z + z * prv_ref_factor_recurse);

                                    if (mData->mBlockOriginTo1D[l - 1].getMetadata(id) || activeCellLambda[l](id)) {
                                        isActiveBlock = true;
                                        mData->mNumActiveVoxel[l][0]++;
                                    }
                                }
                            }
                        }
                    }


                    if (isActiveBlock) {
                        mData->mNumBlocks[l][0]++;
                        mData->mBlockOriginTo1D[l].addPoint(blockOrigin, blockId);
                        blockId++;
                    }
                }
            }
        }
    }


    // Init the base grid
    bGrid::GridBase::init("bGrid",
                          backend,
                          domainSize,
                          Neon::domain::Stencil(),
                          mData->mNumActiveVoxel[0],  //passing active voxels on level 0 as the number of active grid in base grid (????)
                          Neon::int32_3d(Cell::sBlockSizeX, Cell::sBlockSizeY, Cell::sBlockSizeZ),
                          spacingData,
                          origin);


    Neon::MemoryOptions memOptions(Neon::DeviceType::CPU,
                                   Neon::Allocator::MALLOC,
                                   Neon::DeviceType::CUDA,
                                   ((backend.devType() == Neon::DeviceType::CUDA) ? Neon::Allocator::CUDA_MEM_DEVICE : Neon::Allocator::NULL_MEM),
                                   Neon::MemoryLayout::arrayOfStructs);

    mData->mOrigin.resize(descriptor.getDepth());
    for (int l = 0; l < descriptor.getDepth(); ++l) {
        mData->mOrigin[l] = backend.devSet().template newMemSet<Neon::int32_3d>({Neon::DataUse::IO_COMPUTE},
                                                                                1,
                                                                                memOptions,
                                                                                mData->mNumBlocks[l]);
    }

    //Stencil linear/relative index
    auto stencilNghSize = backend.devSet().template newDataSet<uint64_t>();
    for (int32_t c = 0; c < stencilNghSize.cardinality(); ++c) {
        stencilNghSize[c] = stencil.neighbours().size();
    }
    mData->mStencilNghIndex = backend.devSet().template newMemSet<nghIdx_t>({Neon::DataUse::IO_COMPUTE},
                                                                            1,
                                                                            memOptions,
                                                                            stencilNghSize);
    for (int32_t c = 0; c < mData->mStencilNghIndex.cardinality(); ++c) {
        SetIdx devID(c);
        for (uint64_t s = 0; s < stencil.neighbours().size(); ++s) {
            mData->mStencilNghIndex.eRef(c, s).x = static_cast<nghIdx_t::Integer>(stencil.neighbours()[s].x);
            mData->mStencilNghIndex.eRef(c, s).y = static_cast<nghIdx_t::Integer>(stencil.neighbours()[s].y);
            mData->mStencilNghIndex.eRef(c, s).z = static_cast<nghIdx_t::Integer>(stencil.neighbours()[s].z);
        }
    }


    // bitmask
    mData->mActiveMaskSize.resize(descriptor.getDepth());
    mData->mActiveMask.resize(descriptor.getDepth());
    for (int l = 0; l < descriptor.getDepth(); ++l) {
        mData->mActiveMaskSize[l] = backend.devSet().template newDataSet<uint64_t>();
        for (int64_t i = 0; i < mData->mActiveMaskSize[l].size(); ++i) {
            mData->mActiveMaskSize[l][i] = mData->mNumBlocks[l][i] *
                                           NEON_DIVIDE_UP(descriptor.getLevelRefFactor(l) * descriptor.getLevelRefFactor(l) * descriptor.getLevelRefFactor(l),
                                                          Cell::sMaskSize);
        }

        mData->mActiveMask[l] = backend.devSet().template newMemSet<uint32_t>({Neon::DataUse::IO_COMPUTE},
                                                                              1,
                                                                              memOptions,
                                                                              mData->mActiveMaskSize[l]);
    }


    // init bitmask to zero
    for (int l = 0; l < descriptor.getDepth(); ++l) {
        for (int32_t c = 0; c < mData->mActiveMask[l].cardinality(); ++c) {
            SetIdx devID(c);
            for (size_t i = 0; i < mData->mActiveMaskSize[l][c]; ++i) {
                mData->mActiveMask[l].eRef(devID, i) = 0;
            }
        }
    }


    // Neighbor blocks
    mData->mNeighbourBlocks.resize(descriptor.getDepth());
    for (int l = 0; l < descriptor.getDepth(); ++l) {
        mData->mNeighbourBlocks[l] = backend.devSet().template newMemSet<uint32_t>({Neon::DataUse::IO_COMPUTE},
                                                                                   26,
                                                                                   memOptions,
                                                                                   mData->mNumBlocks[l]);
        // init neighbor blocks to invalid block id
        for (int32_t c = 0; c < mData->mNeighbourBlocks[l].cardinality(); ++c) {
            SetIdx devID(c);
            for (uint64_t i = 0; i < mData->mNumBlocks[l][c]; ++i) {
                for (int n = 0; n < 26; ++n) {
                    mData->mNeighbourBlocks[l].eRef(devID, i, n) = std::numeric_limits<uint32_t>::max();
                }
            }
        }
    }


    // Second loop over active blocks to populate the block origins, neighbors, and bitmask
    for (int l = 0; l < descriptor.getDepth(); ++l) {
        const int ref_factor = descriptor.getLevelRefFactor(l);
        const int prv_ref_factor_recurse = descriptor.getRefFactorRecurse(l - 1);
        const int ref_factor_recurse = descriptor.getRefFactorRecurse(l);

        mData->mBlockOriginTo1D[l].forEach([&](const Neon::int32_3d blockOrigin, const uint32_t blockIdx) {
            // TODO need to figure out which device owns this block
            SetIdx devID(0);

            mData->mOrigin[l].eRef(devID, blockIdx) = blockOrigin;

            //set active mask
            for (int z = 0; z < ref_factor; z++) {
                for (int y = 0; y < ref_factor; y++) {
                    for (int x = 0; x < ref_factor; x++) {

                        bool is_active = false;

                        if (l == 0) {
                            const Neon::int32_3d id(blockOrigin.x + x,
                                                    blockOrigin.y + y,
                                                    blockOrigin.z + z);
                            is_active = id < domainSize && activeCellLambda[l](id);
                        } else {
                            //This is the corresponding block origin in the previous level
                            const Neon::int32_3d id(blockOrigin.x + x * prv_ref_factor_recurse,
                                                    blockOrigin.y + y * prv_ref_factor_recurse,
                                                    blockOrigin.z + z * prv_ref_factor_recurse);
                            if (mData->mBlockOriginTo1D[l - 1].getMetadata(id) || activeCellLambda[l](id)) {
                                is_active = true;
                            }
                        }

                        if (is_active) {
                            Cell cell(static_cast<Cell::Location::Integer>(x),
                                      static_cast<Cell::Location::Integer>(y),
                                      static_cast<Cell::Location::Integer>(z));
                            cell.mBlockID = blockIdx;
                            mData->mActiveMask[l].eRef(devID, cell.getBlockMaskStride(ref_factor) + cell.getMaskLocalID(ref_factor), 0) |= 1 << cell.getMaskBitPosition(ref_factor);
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

                        Neon::int32_3d neighbourBlockOrigin(i, j, k);
                        if (l == 0) {
                            neighbourBlockOrigin.x = neighbourBlockOrigin.x * ref_factor + blockOrigin.x;
                            neighbourBlockOrigin.y = neighbourBlockOrigin.y * ref_factor + blockOrigin.y;
                            neighbourBlockOrigin.z = neighbourBlockOrigin.z * ref_factor + blockOrigin.z;
                        } else {
                            neighbourBlockOrigin.x = neighbourBlockOrigin.x * ref_factor_recurse + blockOrigin.x;
                            neighbourBlockOrigin.y = neighbourBlockOrigin.y * ref_factor_recurse + blockOrigin.y;
                            neighbourBlockOrigin.z = neighbourBlockOrigin.z * ref_factor_recurse + blockOrigin.z;
                        }

                        if (neighbourBlockOrigin >= 0 && neighbourBlockOrigin < domainSize) {

                            auto neighbour_it = mData->mBlockOriginTo1D[l].getMetadata(neighbourBlockOrigin);

                            if (neighbour_it) {
                                int16_3d block_offset(i, j, k);
                                mData->mNeighbourBlocks[l].eRef(devID,
                                                                blockIdx,
                                                                Cell::getNeighbourBlockID(block_offset)) = *neighbour_it;
                            }
                        }
                    }
                }
            }
        });
    }

    if (backend.devType() == Neon::DeviceType::CUDA) {
        for (int l = 0; l < descriptor.getDepth(); ++l) {
            mData->mActiveMask[l].updateCompute(backend, 0);
            mData->mOrigin[l].updateCompute(backend, 0);
            mData->mNeighbourBlocks[l].updateCompute(backend, 0);
        }
        mData->mStencilNghIndex.updateCompute(backend, 0);
    }


    mData->mPartitionIndexSpace = std::vector<Neon::set::DataSet<PartitionIndexSpace>>(3);

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
            mData->mPartitionIndexSpace[dv_id][gpuIdx].mDomainSize = domainSize;
            mData->mPartitionIndexSpace[dv_id][gpuIdx].mNumBlocks = static_cast<uint32_t>(mData->mNumBlocks[0][gpuIdx]);
            mData->mPartitionIndexSpace[dv_id][gpuIdx].mHostActiveMask = mData->mActiveMask[0].rawMem(gpuIdx, Neon::DeviceType::CPU);
            mData->mPartitionIndexSpace[dv_id][gpuIdx].mDeviceActiveMask = mData->mActiveMask[0].rawMem(gpuIdx, Neon::DeviceType::CUDA);
            mData->mPartitionIndexSpace[dv_id][gpuIdx].mHostBlockOrigin = mData->mOrigin[0].rawMem(gpuIdx, Neon::DeviceType::CPU);
            mData->mPartitionIndexSpace[dv_id][gpuIdx].mDeviceBlockOrigin = mData->mOrigin[0].rawMem(gpuIdx, Neon::DeviceType::CUDA);
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
    const Neon::index_3d& defaultBlockSize = getDefaultBlock();
    Neon::set::Container  kContainer = Neon::set::Container::factory(name,
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
    for (SetIdx id = 0; id < mData->mNumBlocks[0].cardinality(); id++) {
        pattern.getBlasSet(Neon::DataView::STANDARD).getBlas(id.idx()).setNumBlocks(uint32_t(mData->mNumBlocks[0][id]));
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
                    input1.template forEachActiveCell<Neon::computeMode_t::computeMode_e::seq>(
                        [&](const Neon::index_3d& idx,
                            const int&            cardinality,
                            T&                    in1) {
                            scalar() += in1 * input2(idx, cardinality);
                        });
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
                    input.template forEachActiveCell<Neon::computeMode_t::computeMode_e::seq>(
                        [&]([[maybe_unused]] const Neon::index_3d& idx,
                            [[maybe_unused]] const int&            cardinality,
                            T&                                     in) {
                            scalar() += in * in;
                        });
                }
                scalar() = std::sqrt(scalar());
            };
        });
}
}  // namespace Neon::domain::internal::bGrid