#include "Neon/domain/internal/mGrid/mGrid.h"


namespace Neon::domain::internal::mGrid {

mGrid::mGrid(const Neon::Backend&                                    backend,
             const Neon::int32_3d&                                   domainSize,
             std::vector<std::function<bool(const Neon::index_3d&)>> activeCellLambda,
             const Neon::domain::Stencil&                            stencil,
             const mGridDescriptor                                   descriptor,
             bool                                                    isStrongBalanced,
             const double_3d&                                        spacingData,
             const double_3d&                                        origin)
{


    if (backend.devSet().setCardinality() > 1) {
        NeonException exp("mGrid");
        exp << "mGrid only supported on a single GPU";
        NEON_THROW(exp);
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


    for (int i = 0; i < mData->mDescriptor.getDepth(); ++i) {

        const int refFactor = mData->mDescriptor.getRefFactor(i);

        const int spacing = mData->mDescriptor.getSpacing(i);

        mData->mTotalNumBlocks[i].set(NEON_DIVIDE_UP(domainSize.x, spacing),
                                      NEON_DIVIDE_UP(domainSize.y, spacing),
                                      NEON_DIVIDE_UP(domainSize.z, spacing));

        std::vector<uint32_t> msk(NEON_DIVIDE_UP(refFactor * refFactor * refFactor * mData->mTotalNumBlocks[i].rMul(),
                                                 Cell::sMaskSize),
                                  0);
        mData->denseLevelsBitmask.push_back(msk);
    }

    //check if the bitmask is set assuming a dense domain
    auto levelBitMaskIndex = [&](int l, const Neon::index_3d& blockID, const Neon::index_3d& localChild) {
        const int index1D = mData->mDescriptor.flattened1DIndex(blockID, l, mData->mTotalNumBlocks[l], localChild);
        const int mask = index1D / Cell::sMaskSize;
        const int bitPosition = index1D % Cell::sMaskSize;
        return std::pair<int, int>(mask, bitPosition);
    };
    auto levelBitMaskIsSet = [&](int l, const Neon::index_3d& blockID, const Neon::index_3d& localChild) {
        auto id = levelBitMaskIndex(l, blockID, localChild);
        return mData->denseLevelsBitmask[l][id.first] & (1 << id.second);
    };

    //set the bitmask assuming a dense domain
    auto setLevelBitMask = [&](int l, const Neon::index_3d& blockID, const Neon::index_3d& localChild) {
        auto id = levelBitMaskIndex(l, blockID, localChild);
        mData->denseLevelsBitmask[l][id.first] |= (1 << id.second);
    };

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
        int blockSpacing = mData->mDescriptor.getSpacing(l - 1);

        Neon::int32_3d levelDomainSize(mData->mTotalNumBlocks[l].x * blockSize,
                                       mData->mTotalNumBlocks[l].y * blockSize,
                                       mData->mTotalNumBlocks[l].z * blockSize);

        mData->grids[l] =
            InternalGrid(
                backend,
                levelDomainSize,
                [&](Neon::int32_3d id) {
                    Neon::index_3d blockID = mData->mDescriptor.childToParent(id, l);
                    Neon::index_3d localID = mData->mDescriptor.toLocalIndex(id, l);
                    return levelBitMaskIsSet(l, blockID, localID);
                },
                stencil,
                blockSize,
                blockSpacing,
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
    mData->mParentBlockID.resize(mData->mDescriptor.getDepth());
    for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {
        mData->mParentBlockID[l] = backend.devSet().template newMemSet<uint32_t>({Neon::DataUse::IO_COMPUTE},
                                                                                 1,
                                                                                 memOptionsAoS,
                                                                                 mData->grids[l].getNumBlocks());
    }

    //child block ID
    std::vector<Neon::set::DataSet<uint64_t>> childAllocSize(mData->mDescriptor.getDepth());
    for (int l = 0; l < descriptor.getDepth(); ++l) {
        childAllocSize[l] = backend.devSet().template newDataSet<uint64_t>();
        for (int64_t i = 0; i < childAllocSize[l].size(); ++i) {
            if (l > 0) {
                childAllocSize[l][i] = mData->grids[l].getNumBlocks()[i] *
                                       descriptor.getRefFactor(l) * descriptor.getRefFactor(l) * descriptor.getRefFactor(l);
            } else {
                //we actually don't need to store anything at level 0
                childAllocSize[l][i] = 1;
            }
        }
    }

    mData->mChildBlockID.resize(mData->mDescriptor.getDepth());
    for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {
        mData->mChildBlockID[l] = backend.devSet().template newMemSet<uint32_t>({Neon::DataUse::IO_COMPUTE},
                                                                                1,
                                                                                memOptionsSoA,
                                                                                childAllocSize[l]);
    }


    //parent local index
    mData->mParentLocalID.resize(mData->mDescriptor.getDepth());
    for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {
        mData->mParentLocalID[l] = backend.devSet().template newMemSet<Cell::Location>({Neon::DataUse::IO_COMPUTE},
                                                                                       1,
                                                                                       memOptionsAoS,
                                                                                       mData->grids[l].getNumBlocks());
    }


    //descriptor
    auto descriptorSize = backend.devSet().template newDataSet<uint64_t>();
    for (int32_t c = 0; c < descriptorSize.cardinality(); ++c) {
        descriptorSize[c] = mData->mDescriptor.getDepth();
    }
    mData->mRefFactors = backend.devSet().template newMemSet<int>({Neon::DataUse::IO_COMPUTE},
                                                                  1,
                                                                  memOptionsAoS,
                                                                  descriptorSize);
    for (int32_t c = 0; c < mData->mRefFactors.cardinality(); ++c) {
        SetIdx devID(c);
        for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {
            mData->mRefFactors.eRef(c, l) = mData->mDescriptor.getRefFactor(l);
        }
    }

    mData->mSpacing = backend.devSet().template newMemSet<int>({Neon::DataUse::IO_COMPUTE},
                                                               1,
                                                               memOptionsAoS,
                                                               descriptorSize);
    for (int32_t c = 0; c < mData->mSpacing.cardinality(); ++c) {
        SetIdx devID(c);
        for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {
            mData->mSpacing.eRef(c, l) = mData->mDescriptor.getSpacing(l);
        }
    }


    // loop over active blocks to populate the block parents and child info
    for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {
        const int refFactor = mData->mDescriptor.getRefFactor(l);
        const int spacing = mData->mDescriptor.getSpacing(l);


        mData->grids[l].getBlockOriginTo1D().forEach([&](const Neon::int32_3d blockOrigin, const uint32_t blockIdx) {
            //TODO need to figure out which device owns this block
            SetIdx devID(0);

            Neon::int32_3d block3DIndex = blockOrigin / spacing;


            //set active mask and child ID
            for (Cell::Location::Integer z = 0; z < refFactor; z++) {
                for (Cell::Location::Integer y = 0; y < refFactor; y++) {
                    for (Cell::Location::Integer x = 0; x < refFactor; x++) {

                        const Neon::index_3d localChild(x, y, z);

                        if (levelBitMaskIsSet(l, block3DIndex, localChild)) {

                            if (l > 0) {
                                Neon::index_3d childBase = mData->mDescriptor.parentToChild(blockOrigin, l, localChild);
                                auto           child_it = mData->grids[l - 1].getBlockOriginTo1D().getMetadata(childBase);
                                if (child_it) {
                                    mData->mChildBlockID[l].eRef(devID,
                                                                 blockIdx * refFactor * refFactor * refFactor +
                                                                     x + y * refFactor + z * refFactor * refFactor) = *child_it;
                                } else {
                                    mData->mChildBlockID[l].eRef(devID,
                                                                 blockIdx * refFactor * refFactor * refFactor +
                                                                     x + y * refFactor + z * refFactor * refFactor) = std::numeric_limits<uint32_t>::max();
                                }
                            }
                        }
                    }
                }
            }


            //set the parent info
            //the parent info are the block inside which the parent resides i.e., grandParentOrigin
            //and the local 3d index within the grandParentOrigin block where the parent is located
            if (l < mData->mDescriptor.getDepth() - 1) {
                Neon::index_3d grandParentOrigin = mData->mDescriptor.toBaseIndexSpace(mData->mDescriptor.childToParent(blockOrigin, l + 1), l + 2);

                auto grand_parent = mData->grids[l + 1].getBlockOriginTo1D().getMetadata(grandParentOrigin);
                if (!grand_parent) {
                    NeonException exp("mGrid::mGrid");
                    exp << "Something went wrong during constructing mGrid. Can not find the right parent of a block\n";
                    NEON_THROW(exp);
                }
                mData->mParentBlockID[l].eRef(devID, blockIdx) = *grand_parent;

                //set the parent local ID
                // loop over this grand parent block to find the local index which maps back to the parent block
                const int grandParentRefFactor = mData->mDescriptor.getRefFactor(l + 1);
                bool      found = false;
                for (Cell::Location::Integer z = 0; z < grandParentRefFactor; z++) {
                    for (Cell::Location::Integer y = 0; y < grandParentRefFactor; y++) {
                        for (Cell::Location::Integer x = 0; x < grandParentRefFactor; x++) {
                            Neon::index_3d parent = mData->mDescriptor.neighbourBlock(grandParentOrigin, l + 1, {x, y, z});
                            if (parent == blockOrigin) {
                                mData->mParentLocalID[l].eRef(devID, blockIdx) = {x, y, z};
                                found = true;
                                break;
                            }
                        }
                        if (found) {
                            break;
                        }
                    }
                    if (found) {
                        break;
                    }
                }

                if (!found) {
                    NeonException exp("mGrid::mGrid");
                    exp << "Something went wrong during constructing mGrid. Can not find the right local index of a parent of a block\n";
                    NEON_THROW(exp);
                }
            }
        });
    }

    if (backend.devType() == Neon::DeviceType::CUDA) {
        for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {
            mData->mParentBlockID[l].updateCompute(backend, 0);
            mData->mParentLocalID[l].updateCompute(backend, 0);
            if (l > 0) {
                mData->mChildBlockID[l].updateCompute(backend, 0);
            }
        }
        mData->mRefFactors.updateCompute(backend, 0);
        mData->mSpacing.updateCompute(backend, 0);
    }
}

auto mGrid::isInsideDomain(const Neon::index_3d& idx, int level) const -> bool
{

    //TODO need to figure out which device owns this block
    SetIdx devID(0);

    //We don't have to check over the domain bounds. If idx is outside the domain
    // (i.e., idx beyond the bounds of the domain) its block origin will be null

    Neon::int32_3d block_origin = getOriginBlock3DIndex(idx, level);

    auto itr = mData->grids[level].getBlockOriginTo1D().getMetadata(block_origin);
    if (itr) {
        Neon::index_3d localID = mData->mDescriptor.toLocalIndex(idx, level);

        Cell cell(static_cast<Cell::Location::Integer>(localID.x),
                  static_cast<Cell::Location::Integer>(localID.y),
                  static_cast<Cell::Location::Integer>(localID.z));

        cell.mBlockID = *itr;
        cell.mBlockSize = mData->mDescriptor.getRefFactor(level);
        cell.mIsActive = cell.computeIsActive(mData->grids[level].getActiveMask().rawMem(devID, Neon::DeviceType::CPU));
        return cell.mIsActive;
    }
    return false;
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

auto mGrid::getLaunchParameters(Neon::DataView                         dataView,
                                [[maybe_unused]] const Neon::index_3d& blockSize,
                                const size_t&                          sharedMem,
                                int                                    level) const -> Neon::set::LaunchParameters
{
    if (dataView != Neon::DataView::STANDARD) {
        NEON_WARNING("Requesting LaunchParameters on {} data view but mGrid only supports Standard data view on a single GPU",
                     Neon::DataViewUtil::toString(dataView));
    }

    const Neon::int32_3d cuda_block(mData->mDescriptor.getRefFactor(level),
                                    mData->mDescriptor.getRefFactor(level),
                                    mData->mDescriptor.getRefFactor(level));

    Neon::set::LaunchParameters ret = mData->backend.devSet().newLaunchParameters();
    for (int i = 0; i < ret.cardinality(); ++i) {
        if (mData->backend.devType() == Neon::DeviceType::CUDA) {
            ret[i].set(Neon::sys::GpuLaunchInfo::mode_e::cudaGridMode,
                       Neon::int32_3d(int32_t(mData->grids[level].getNumBlocks()[i]), 1, 1),
                       cuda_block, sharedMem);
        } else {
            ret[i].set(Neon::sys::GpuLaunchInfo::mode_e::domainGridMode,
                       Neon::int32_3d(int32_t(mData->grids[level].getNumBlocks()[i]) *
                                          mData->mDescriptor.getRefFactor(level) *
                                          mData->mDescriptor.getRefFactor(level) *
                                          mData->mDescriptor.getRefFactor(level),
                                      1, 1),
                       cuda_block, sharedMem);
        }
    }
    return ret;
}

auto mGrid::getNumBlocksPerPartition(int level) const -> const Neon::set::DataSet<uint64_t>&
{
    return mData->grids[level].getNumBlocks();
}

auto mGrid::getParentsBlockID(int level) const -> const Neon::set::MemSet_t<uint32_t>&
{
    return mData->mParentBlockID[level];
}
auto mGrid::getChildBlockID(int level) const -> const Neon::set::MemSet_t<uint32_t>&
{
    return mData->mChildBlockID[level];
}

auto mGrid::getParentLocalID(int level) const -> const Neon::set::MemSet_t<Cell::Location>&
{
    return mData->mParentLocalID[level];
}

auto mGrid::getRefFactors() const -> const Neon::set::MemSet_t<int>&
{
    return mData->mRefFactors;
}

auto mGrid::getLevelSpacing() const -> const Neon::set::MemSet_t<int>&
{
    return mData->mSpacing;
}

auto mGrid::getDescriptor() const -> const mGridDescriptor&
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

void mGrid::topologyToVTK(std::string fileName, bool filterOverlaps) const
{

    std::ofstream file(fileName);
    file << "# vtk DataFile Version 2.0\n";
    file << "mGrid\n";
    file << "ASCII\n";
    file << "DATASET UNSTRUCTURED_GRID\n";
    file << "POINTS " << (getDimension(0).rMax() + 1) * (getDimension(0).rMax() + 1) * (getDimension(0).rMax() + 1) << " float \n";
    for (int z = 0; z < getDimension(0).rMax() + 1; ++z) {
        for (int y = 0; y < getDimension(0).rMax() + 1; ++y) {
            for (int x = 0; x < getDimension(0).rMax() + 1; ++x) {
                file << x << " " << y << " " << z << "\n";
            }
        }
    }

    uint64_t num_cells = 0;

    auto mapTo1D = [&](int x, int y, int z) {
        return x +
               y * (getDimension(0).rMax() + 1) +
               z * (getDimension(0).rMax() + 1) * (getDimension(0).rMax() + 1);
    };

    enum class Op : int
    {
        Count = 0,
        OutputTopology = 1,
        OutputLevels = 2,
        OutputBlockID = 3,
        OutputVoxelID = 4,
    };

    auto loopOverActiveBlocks = [&](const Op op) {
        for (int l = 0; l < mData->mDescriptor.getDepth(); ++l) {
            const int refFactor = mData->mDescriptor.getRefFactor(l);
            const int childSpacing = mData->mDescriptor.getSpacing(l - 1);

            mData->grids[l].getBlockOriginTo1D().forEach([&](const Neon::int32_3d blockOrigin, const uint32_t blockIdx) {
                // TODO need to figure out which device owns this block
                SetIdx devID(0);


                for (int z = 0; z < refFactor; z++) {
                    for (int y = 0; y < refFactor; y++) {
                        for (int x = 0; x < refFactor; x++) {
                            Cell cell(static_cast<Cell::Location::Integer>(x),
                                      static_cast<Cell::Location::Integer>(y),
                                      static_cast<Cell::Location::Integer>(z));
                            cell.mBlockID = blockIdx;
                            cell.mBlockSize = refFactor;

                            if (cell.computeIsActive(
                                    mData->grids[l].getActiveMask().rawMem(devID, Neon::DeviceType::CPU))) {

                                Neon::int32_3d corner(blockOrigin.x + x * childSpacing,
                                                      blockOrigin.y + y * childSpacing,
                                                      blockOrigin.z + z * childSpacing);

                                bool draw = true;
                                if (filterOverlaps && l != 0) {
                                    auto cornerIDIter = mData->grids[l - 1].getBlockOriginTo1D().getMetadata(corner);
                                    if (cornerIDIter) {
                                        draw = false;
                                    }
                                }

                                if (draw) {
                                    if (op == Op::Count) {
                                        num_cells++;
                                    } else if (op == Op::OutputTopology) {

                                        file << "8 ";
                                        //x,y,z
                                        file << mapTo1D(corner.x, corner.y, corner.z) << " ";
                                        //+x,y,z
                                        file << mapTo1D(corner.x + childSpacing, corner.y, corner.z) << " ";

                                        //x,+y,z
                                        file << mapTo1D(corner.x, corner.y + childSpacing, corner.z) << " ";

                                        //+x,+y,z
                                        file << mapTo1D(corner.x + childSpacing, corner.y + childSpacing, corner.z) << " ";

                                        //x,y,+z
                                        file << mapTo1D(corner.x, corner.y, corner.z + childSpacing) << " ";

                                        //+x,y,+z
                                        file << mapTo1D(corner.x + childSpacing, corner.y, corner.z + childSpacing) << " ";

                                        //x,+y,+z
                                        file << mapTo1D(corner.x, corner.y + childSpacing, corner.z + childSpacing) << " ";

                                        //+x,+y,+z
                                        file << mapTo1D(corner.x + childSpacing, corner.y + childSpacing, corner.z + childSpacing) << " ";
                                        file << "\n";
                                    } else if (op == Op::OutputLevels) {
                                        file << l << "\n";
                                    } else if (op == Op::OutputBlockID) {
                                        file << blockIdx << "\n";
                                    } else if (op == Op::OutputVoxelID) {
                                        file << blockIdx << "\n";
                                    }
                                }
                            }
                        }
                    }
                }
            });
        }
    };

    loopOverActiveBlocks(Op::Count);

    file << "CELLS " << num_cells << " " << num_cells * 9 << " \n";

    loopOverActiveBlocks(Op::OutputTopology);

    file << "CELL_TYPES " << num_cells << " \n";
    for (uint64_t i = 0; i < num_cells; ++i) {
        file << 11 << "\n";
    }

    file << "CELL_DATA " << num_cells << " \n";

    file << "SCALARS Level int 1 \n";
    file << "LOOKUP_TABLE default \n";
    loopOverActiveBlocks(Op::OutputLevels);


    file << "SCALARS BlockID int 1 \n";
    file << "LOOKUP_TABLE default \n";
    loopOverActiveBlocks(Op::OutputBlockID);

    //file << "SCALARS VoxelID int 1 \n";
    //file << "LOOKUP_TABLE default \n";
    //loopOverActiveBlocks(Op::OutputVoxelID);


    file.close();
}


}  // namespace Neon::domain::internal::mGrid