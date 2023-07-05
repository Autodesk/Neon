
#include "Neon/domain/details//mGrid/mGrid.h"
#include "Neon/domain/details/mGrid/mPartition.h"


namespace Neon::domain::details::mGrid {
mGrid::mGrid(
    const Neon::Backend&                                    backend,
    const Neon::int32_3d&                                   domainSize,
    std::vector<std::function<bool(const Neon::index_3d&)>> activeCellLambda,
    [[maybe_unused]] const Neon::domain::Stencil&           stencil,
    const Descriptor                                        descriptor,
    bool                                                    isStrongBalanced,
    [[maybe_unused]] const double_3d&                       spacingData,
    [[maybe_unused]] const double_3d&                       origin)
{

    if (backend.devSet().setCardinality() > 1) {
        NeonException exp("mGrid");
        exp << "mGrid only supported on a single GPU";
        NEON_THROW(exp);
    }

    static_assert(kUserBlockSizeX == 2 && kUserBlockSizeY == 2 && kUserBlockSizeZ == 2, "mGird only supports octree!");

    for (int l = 0; l < descriptor.getDepth(); ++l) {
        if (descriptor.getRefFactor(l) != kUserBlockSizeX ||
            descriptor.getRefFactor(l) != kUserBlockSizeY ||
            descriptor.getRefFactor(l) != kUserBlockSizeZ) {
            NeonException exp("mGrid");
            exp << "Mismatch between the grid descriptor and the userBlockSize";
            exp << "Level = " << l << " refinement factor = " << descriptor.getRefFactor(l) << " userBlockSize= " << kUserBlockSizeX << ", " << kUserBlockSizeY << ", " << kUserBlockSizeZ;
            NEON_THROW(exp);
        }
    }

    mData = std::make_shared<Data>();

    mData->backend = backend;
    mData->domainSize = domainSize;
    mData->mStrongBalanced = isStrongBalanced;
    mData->mDescriptor = descriptor;
    int top_level_spacing = 1;
    for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {
        if (l > 0) {
            top_level_spacing *= mData->mDescriptor.getRefFactor(l);
            if (mData->mDescriptor.getRefFactor(l) < mData->mDescriptor.getRefFactor(l - 1)) {
                NeonException exp("mGrid::mGrid");
                exp << "The grid refinement factor should only go up from one level to another starting with Level 0 the leaf/finest level\n";
                exp << "Level " << l - 1 << " refinement factor= " << mData->mDescriptor.getRefFactor(l - 1) << "\n";
                exp << "Level " << l << " refinement factor= " << mData->mDescriptor.getRefFactor(l) << "\n";
                NEON_THROW(exp);
            }
        }
    }


    if (domainSize.x < top_level_spacing || domainSize.y < top_level_spacing || domainSize.z < top_level_spacing) {
        NeonException exp("mGrid::mGrid");
        exp << "The spacing of the top level of the multi-resolution grid is bigger than the domain size";
        exp << " This may create problems. Please consider increasing the domain size or decrease the branching factor or depth of the grid\n";
        exp << "DomainSize= " << domainSize << "\n";
        exp << "Top level spacing= " << top_level_spacing << "\n";
        NEON_THROW(exp);
    }

    mData->mTotalNumBlocks.resize(mData->mDescriptor.getDepth());

    constexpr uint32_t MaskSize = 32;

    for (int i = 0; i < mData->mDescriptor.getDepth(); ++i) {

        const int refFactor = mData->mDescriptor.getRefFactor(i);

        const int spacing = mData->mDescriptor.getSpacing(i);

        mData->mTotalNumBlocks[i].set(NEON_DIVIDE_UP(domainSize.x, spacing),
                                      NEON_DIVIDE_UP(domainSize.y, spacing),
                                      NEON_DIVIDE_UP(domainSize.z, spacing));

        std::vector<uint32_t> msk(NEON_DIVIDE_UP(refFactor * refFactor * refFactor * mData->mTotalNumBlocks[i].rMul(), MaskSize),
                                  0);
        mData->denseLevelsBitmask.push_back(msk);
    }

    //Each block loops over its voxels and check the lambda function and activate its voxels correspondingly
    //If a block contain an active voxel, it activates itself as well
    //This loop only sets the bitmask
    for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {
        const int refFactor = mData->mDescriptor.getRefFactor(l);

        for (int bz = 0; bz < mData->mTotalNumBlocks[l].z; bz++) {
            for (int by = 0; by < mData->mTotalNumBlocks[l].y; by++) {
                for (int bx = 0; bx < mData->mTotalNumBlocks[l].x; bx++) {

                    Neon::index_3d blockOrigin = mData->mDescriptor.toBaseIndexSpace({bx, by, bz}, l + 1);

                    bool containVoxels = false;
                    for (int z = 0; z < refFactor; z++) {
                        for (int y = 0; y < refFactor; y++) {
                            for (int x = 0; x < refFactor; x++) {

                                const Neon::int32_3d voxel = mData->mDescriptor.parentToChild(blockOrigin, l, {x, y, z});

                                if (voxel < domainSize) {
                                    //if it is already active
                                    if (levelBitMaskIsSet(l, {bx, by, bz}, {x, y, z})) {
                                        containVoxels = true;
                                    } else {
                                        if (activeCellLambda[l](voxel)) {
                                            containVoxels = true;
                                            setLevelBitMask(l, {bx, by, bz}, {x, y, z});
                                        }
                                    }
                                }
                            }
                        }
                    }


                    if (containVoxels) {
                        //if the block contains voxels, it should activate itself
                        //find its corresponding index within the next level
                        //i.e., blockOrigin is the parent block that contains refFactor^3 voxels (sparse)
                        //so it tries to find the block in the next level where blockOrigin is a child
                        //which requires finding the block in this next level and the local index within this block

                        if (l < mData->mDescriptor.getDepth() - 1) {
                            Neon::int32_3d parentBlock = mData->mDescriptor.childToParent(blockOrigin, l + 1);

                            Neon::int32_3d indexInParentBlock = mData->mDescriptor.toLocalIndex(blockOrigin, l + 1);

                            setLevelBitMask(l + 1, parentBlock, indexInParentBlock);
                        }
                    }
                }
            }
        }
    }

    //Impose the strong balance condition
    if (mData->mStrongBalanced) {
        bool again = true;
        while (again) {
            again = false;

            for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {
                const int refFactor = mData->mDescriptor.getRefFactor(l);
                const int childSpacing = mData->mDescriptor.getSpacing(l - 1);

                for (int bz = 0; bz < mData->mTotalNumBlocks[l].z; bz++) {
                    for (int by = 0; by < mData->mTotalNumBlocks[l].y; by++) {
                        for (int bx = 0; bx < mData->mTotalNumBlocks[l].x; bx++) {

                            for (int z = 0; z < refFactor; z++) {
                                for (int y = 0; y < refFactor; y++) {
                                    for (int x = 0; x < refFactor; x++) {


                                        if (levelBitMaskIsSet(l, {bx, by, bz}, {x, y, z})) {

                                            const Neon::int32_3d voxel(bx * refFactor + x,
                                                                       by * refFactor + y,
                                                                       bz * refFactor + z);

                                            for (int k = -1; k < 2; k++) {
                                                for (int j = -1; j < 2; j++) {
                                                    for (int i = -1; i < 2; i++) {
                                                        if (i == 0 && j == 0 && k == 0) {
                                                            continue;
                                                        }

                                                        Neon::int32_3d proxyVoxel(voxel.x + i,
                                                                                  voxel.y + j,
                                                                                  voxel.z + k);

                                                        const Neon::int32_3d proxyVoxelLocation(proxyVoxel.x * childSpacing,
                                                                                                proxyVoxel.y * childSpacing,
                                                                                                proxyVoxel.z * childSpacing);

                                                        if (proxyVoxelLocation < domainSize && proxyVoxelLocation >= 0) {

                                                            Neon::int32_3d prv_nVoxelBlockOrigin, prv_nVoxelLocalID;
                                                            for (int l_n = l; l_n < mData->mDescriptor.getDepth(); ++l_n) {
                                                                const int l_n_ref_factor = mData->mDescriptor.getRefFactor(l_n);


                                                                //find the block origin of n_voxel which live at level l_n
                                                                const Neon::int32_3d nVoxelBlockOrigin(proxyVoxel.x / l_n_ref_factor,
                                                                                                       proxyVoxel.y / l_n_ref_factor,
                                                                                                       proxyVoxel.z / l_n_ref_factor);

                                                                const Neon::int32_3d nVoxelLocalID(proxyVoxel.x % l_n_ref_factor,
                                                                                                   proxyVoxel.y % l_n_ref_factor,
                                                                                                   proxyVoxel.z % l_n_ref_factor);

                                                                //find if this block origin is active
                                                                if (levelBitMaskIsSet(l_n, nVoxelBlockOrigin, nVoxelLocalID)) {

                                                                    //if this neighbor is at the same level or +1 level, then there is nothing else we should check on
                                                                    if (l_n == l || l_n == l + 1) {
                                                                        break;
                                                                    } else {
                                                                        //otherwise, we should refine the previous block and voxel

                                                                        setLevelBitMask(l_n - 1, prv_nVoxelBlockOrigin, prv_nVoxelLocalID);

                                                                        again = true;
                                                                    }
                                                                }

                                                                //promote the proxy voxel to the next level
                                                                proxyVoxel = nVoxelBlockOrigin;

                                                                //cache the voxel and block at this level because we might need to activate them
                                                                prv_nVoxelBlockOrigin = nVoxelBlockOrigin;
                                                                prv_nVoxelLocalID = nVoxelLocalID;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }


    mData->grids.resize(mData->mDescriptor.getDepth());
    for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {

        int blockSize = mData->mDescriptor.getRefFactor(l);
        int voxelSpacing = mData->mDescriptor.getSpacing(l - 1);

        Neon::int32_3d levelDomainSize(mData->mTotalNumBlocks[l].x * blockSize,
                                       mData->mTotalNumBlocks[l].y * blockSize,
                                       mData->mTotalNumBlocks[l].z * blockSize);

        mData->grids[l] =
            InternalGrid(
                backend,
                levelDomainSize,
                [&](Neon::int32_3d id) {
                    if (id < domainSize) {
                        Neon::index_3d blockID = mData->mDescriptor.childToParent(id, l);
                        Neon::index_3d localID = mData->mDescriptor.toLocalIndex(id, l);
                        return levelBitMaskIsSet(l, blockID, localID);
                    } else {
                        return false;
                    }
                },
                stencil,
                voxelSpacing,
                spacingData,
                origin);
    }

    Neon::MemoryOptions memOptionsAoS(Neon::DeviceType::CPU,
                                      Neon::Allocator::MALLOC,
                                      Neon::DeviceType::CUDA,
                                      ((backend.devType() == Neon::DeviceType::CUDA) ? Neon::Allocator::CUDA_MEM_DEVICE : Neon::Allocator::NULL_MEM),
                                      Neon::MemoryLayout::arrayOfStructs);
    Neon::MemoryOptions memOptionsSoA(Neon::DeviceType::CPU,
                                      Neon::Allocator::MALLOC,
                                      Neon::DeviceType::CUDA,
                                      ((backend.devType() == Neon::DeviceType::CUDA) ? Neon::Allocator::CUDA_MEM_DEVICE : Neon::Allocator::NULL_MEM),
                                      Neon::MemoryLayout::structOfArrays);


    //parent block ID
    mData->mParentBlockID.resize(mData->mDescriptor.getDepth() - 1);
    for (int l = 0; l < mData->mDescriptor.getDepth() - 1; ++l) {
        mData->mParentBlockID[l] = backend.devSet().template newMemSet<Idx::DataBlockIdx>({Neon::DataUse::HOST_DEVICE},
                                                                                          1,
                                                                                          memOptionsAoS,
                                                                                          mData->grids[l].getBlockViewGrid().getNumActiveCellsPerPartition());
    }

    //child block ID


    std::vector<Neon::set::DataSet<uint64_t>> childAllocSize(mData->mDescriptor.getDepth());
    for (int l = 0; l < descriptor.getDepth(); ++l) {
        childAllocSize[l] = backend.devSet().template newDataSet<uint64_t>();
        for (int64_t i = 0; i < childAllocSize[l].size(); ++i) {
            if (l > 0) {
                childAllocSize[l][i] = mData->grids[l].helpGetPartitioner1D().getStandardCount()[0] *
                                       kMemBlockSizeX * kMemBlockSizeY * kMemBlockSizeZ;
            } else {
                //we actually don't need to store anything at level 0
                childAllocSize[l][i] = 1;
            }
        }
    }

    mData->mChildBlockID.resize(mData->mDescriptor.getDepth());
    for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {
        mData->mChildBlockID[l] = backend.devSet().template newMemSet<Idx::DataBlockIdx>({Neon::DataUse::HOST_DEVICE},
                                                                                         1,
                                                                                         memOptionsSoA,
                                                                                         childAllocSize[l]);
        for (int32_t c = 0; c < childAllocSize[l].cardinality(); ++c) {
            SetIdx devID(c);
            for (size_t i = 0; i < childAllocSize[l][c]; ++i) {
                mData->mChildBlockID[l].eRef(devID, i) = std::numeric_limits<Idx::DataBlockIdx>::max();
            }
        }
    }


    //descriptor
    auto descriptorSize = backend.devSet().template newDataSet<uint64_t>();
    for (int32_t c = 0; c < descriptorSize.cardinality(); ++c) {
        descriptorSize[c] = mData->mDescriptor.getDepth();
    }
    mData->mRefFactors = backend.devSet().template newMemSet<int>({Neon::DataUse::HOST_DEVICE},
                                                                  1,
                                                                  memOptionsAoS,
                                                                  descriptorSize);
    for (int32_t c = 0; c < mData->mRefFactors.cardinality(); ++c) {
        SetIdx devID(c);
        for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {
            mData->mRefFactors.eRef(c, l) = mData->mDescriptor.getRefFactor(l);
        }
    }

    mData->mSpacing = backend.devSet().template newMemSet<int>({Neon::DataUse::HOST_DEVICE},
                                                               1,
                                                               memOptionsAoS,
                                                               descriptorSize);
    for (int32_t c = 0; c < mData->mSpacing.cardinality(); ++c) {
        SetIdx devID(c);
        for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {
            mData->mSpacing.eRef(c, l) = mData->mDescriptor.getSpacing(l);
        }
    }

    SetIdx devID(0);


    // loop over active blocks to populate the block parents and child info
    for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {
        const int refFactor = mData->mDescriptor.getRefFactor(l);
        const int spacing = mData->mDescriptor.getSpacing(l);
        int       voxelSpacing = mData->mDescriptor.getSpacing(l - 1);


        mData->grids[l].helpGetPartitioner1D().forEachSeq(devID, [&](int blockIdx, Neon::index_3d memBlockOrigin, auto /*byPartition*/) {
            Neon::index_3d blockOrigin = memBlockOrigin;
            blockOrigin.x *= kMemBlockSizeX * voxelSpacing;
            blockOrigin.y *= kMemBlockSizeY * voxelSpacing;
            blockOrigin.z *= kMemBlockSizeZ * voxelSpacing;

            if (l > 0) {
                //loop over user block
                for (uint32_t k = 0; k < kNumUserBlockPerMemBlockZ; ++k) {
                    for (uint32_t j = 0; j < kNumUserBlockPerMemBlockY; ++j) {
                        for (uint32_t i = 0; i < kNumUserBlockPerMemBlockX; ++i) {

                            const Neon::index_3d userBlockOrigin(i * kUserBlockSizeX * voxelSpacing + blockOrigin.x,
                                                                 j * kUserBlockSizeY * voxelSpacing + blockOrigin.y,
                                                                 k * kUserBlockSizeZ * voxelSpacing + blockOrigin.z);

                            const Neon::int32_3d block3DIndex = userBlockOrigin / spacing;

                            //loop over each voxel in the user block
                            for (int32_t z = 0; z < refFactor; z++) {
                                for (int32_t y = 0; y < refFactor; y++) {
                                    for (int32_t x = 0; x < refFactor; x++) {
                                        Neon::index_3d localChild(x, y, z);

                                        const Neon::index_3d voxelGlobalID(x * voxelSpacing + userBlockOrigin.x,
                                                                           y * voxelSpacing + userBlockOrigin.y,
                                                                           z * voxelSpacing + userBlockOrigin.z);
                                        if (voxelGlobalID.x >= domainSize.x || voxelGlobalID.y >= domainSize.y || voxelGlobalID.z >= domainSize.z) {
                                            continue;
                                        }


                                        //set child ID
                                        if (levelBitMaskIsSet(l, block3DIndex, localChild)) {

                                            Neon::index_3d childId = mData->mDescriptor.parentToChild(userBlockOrigin, l, localChild);

                                            auto [setIdx, childBlockID] = mData->grids[l - 1].helpGetSetIdxAndGridIdx(childId);

                                            uint32_t pitch = blockIdx * kMemBlockSizeX * kMemBlockSizeY * kMemBlockSizeZ +
                                                             (i * kUserBlockSizeX + x) +
                                                             (j * kUserBlockSizeY + y) * kMemBlockSizeY +
                                                             (k * kUserBlockSizeZ + z) * kMemBlockSizeY * kMemBlockSizeZ;

                                            if (setIdx.idx() == -1) {
                                                mData->mChildBlockID[l].eRef(devID, pitch) = std::numeric_limits<Idx::DataBlockIdx>::max();
                                            } else {
                                                mData->mChildBlockID[l].eRef(devID, pitch) = childBlockID.getDataBlockIdx();
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }


            //set the parent info
            if (l < mData->mDescriptor.getDepth() - 1) {
                Neon::index_3d parentOrigin = mData->mDescriptor.toBaseIndexSpace(mData->mDescriptor.childToParent(blockOrigin, l + 1), l + 2);

                auto [setIdx, parentID] = mData->grids[l + 1].helpGetSetIdxAndGridIdx(parentOrigin);

                if (setIdx.idx() == -1) {
                    NeonException exp("mGrid::mGrid");
                    exp << "Something went wrong during constructing mGrid. Can not find the right parent of a block\n";
                    NEON_THROW(exp);
                }
                mData->mParentBlockID[l].eRef(devID, blockIdx) = parentID.getDataBlockIdx();
            }
        });
    }

    if (backend.devType() == Neon::DeviceType::CUDA) {
        for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {
            if (l < mData->mDescriptor.getDepth() - 1) {
                mData->mParentBlockID[l].updateDeviceData(backend, 0);
            }
            if (l > 0) {
                mData->mChildBlockID[l].updateDeviceData(backend, 0);
            }
        }
        mData->mRefFactors.updateDeviceData(backend, 0);
        mData->mSpacing.updateDeviceData(backend, 0);
    }
}


auto mGrid::levelBitMaskIndex(int l, const Neon::index_3d& blockID, const Neon::index_3d& localChild) const -> std::pair<int, int>
{
    constexpr uint32_t MaskSize = 32;
    const int          index1D = mData->mDescriptor.flattened1DIndex(blockID, l, mData->mTotalNumBlocks[l], localChild);
    const int          mask = index1D / MaskSize;
    const int          bitPosition = index1D % MaskSize;
    return std::pair<int, int>(mask, bitPosition);
};

auto mGrid::levelBitMaskIsSet(int l, const Neon::index_3d& blockID, const Neon::index_3d& localChild) const -> bool
{
    auto id = levelBitMaskIndex(l, blockID, localChild);
    return mData->denseLevelsBitmask[l][id.first] & (1 << id.second);
};


auto mGrid::setLevelBitMask(int l, const Neon::index_3d& blockID, const Neon::index_3d& localChild) -> void
{
    auto id = levelBitMaskIndex(l, blockID, localChild);
    mData->denseLevelsBitmask[l][id.first] |= (1 << id.second);
};

auto mGrid::isInsideDomain(const Neon::index_3d& idx, int level) const -> bool
{
    return mData->grids[level].isInsideDomain(idx);
}


auto mGrid::operator()(int level) -> InternalGrid&
{
    return mData->grids[level];
}

auto mGrid::operator()(int level) const -> const InternalGrid&
{
    return mData->grids[level];
}


auto mGrid::getOriginBlock3DIndex(const Neon::int32_3d idx, int level) const -> Neon::int32_3d
{
    //round n to nearest multiple of m
    auto roundDownToNearestMultiple = [](int32_t n, int32_t m) -> int32_t {
        return (n / m) * m;
    };

    Neon::int32_3d block_origin(roundDownToNearestMultiple(idx.x, mData->mDescriptor.getSpacing(level)),
                                roundDownToNearestMultiple(idx.y, mData->mDescriptor.getSpacing(level)),
                                roundDownToNearestMultiple(idx.z, mData->mDescriptor.getSpacing(level)));
    return block_origin;
}

auto mGrid::setReduceEngine(Neon::sys::patterns::Engine eng) -> void
{
    if (eng != Neon::sys::patterns::Engine::CUB) {
        NeonException exp("mGrid::setReduceEngine");
        exp << "mGrid only work on CUB engine for reduction";
        NEON_THROW(exp);
    }
}

auto mGrid::getParentsBlockID(int level) const -> Neon::set::MemSet<uint32_t>&
{
    if (level >= mData->mDescriptor.getDepth() - 1) {
        NeonException exp("mGrid::getParentsBlockID");
        exp << "There is no parent for level " << level << " since the tree depth is " << mData->mDescriptor.getDepth();
        NEON_THROW(exp);
    }

    return mData->mParentBlockID[level];
}
auto mGrid::getChildBlockID(int level) const -> const Neon::set::MemSet<uint32_t>&
{
    return mData->mChildBlockID[level];
}

auto mGrid::getRefFactors() const -> const Neon::set::MemSet<int>&
{
    return mData->mRefFactors;
}

auto mGrid::getLevelSpacing() const -> const Neon::set::MemSet<int>&
{
    return mData->mSpacing;
}

auto mGrid::getDescriptor() const -> const Descriptor&
{
    return mData->mDescriptor;
}

auto mGrid::getDimension(int level) const -> const Neon::index_3d
{
    return mData->mTotalNumBlocks[level] * mData->mDescriptor.getRefFactor(level);
}

auto mGrid::getDimension() const -> const Neon::index_3d
{
    return mData->domainSize;
}

auto mGrid::getNumBlocks(int level) const -> const Neon::index_3d&
{
    return mData->mTotalNumBlocks[level];
}

auto mGrid::getBackend() const -> const Backend&
{
    return mData->backend;
}
auto mGrid::getBackend() -> Backend&
{
    return mData->backend;
}

}  // namespace Neon::domain::details::mGrid
