#include "Neon/domain/internal/bGrid/bGrid.h"
#include "Neon/domain/interface/KernelConfig.h"
#include "Neon/domain/internal/bGrid/bPartitionIndexSpace.h"

namespace Neon::domain::internal::bGrid {

auto bGrid::getProperties(const Neon::index_3d& idx) const -> GridBaseTemplate::CellProperties
{
    GridBaseTemplate::CellProperties cellProperties;
    cellProperties.setIsInside(isInsideDomain(idx));
    if (!cellProperties.isInside()) {
        return cellProperties;
    }

    if (this->getDevSet().setCardinality() == 1) {
        cellProperties.init(0, DataView::INTERNAL);
    } else {
        //TODO
        NEON_DEV_UNDER_CONSTRUCTION("bGrid only support single GPU");
    }
    return cellProperties;
}

auto bGrid::isInsideDomain(const Neon::index_3d& idx) const -> bool
{
    if (this->getDevSet().setCardinality() != 1) {
        NEON_DEV_UNDER_CONSTRUCTION("bGrid only support single GPU");
    }

    //TODO need to figure out which device owns this block
    SetIdx devID(0);

    //We don't have to check over the domain bounds. If idx is outside the domain
    // (i.e., idx beyond the bounds of the domain) its block origin will be null

    Neon::int32_3d block_origin = getOriginBlock3DIndex(idx);

    auto itr = mData->mBlockOriginTo1D.getMetadata(block_origin);
    if (itr) {

        Cell cell(static_cast<Cell::Location::Integer>((idx.x / mData->voxelSpacing) % mData->blockSize),
                  static_cast<Cell::Location::Integer>((idx.y / mData->voxelSpacing) % mData->blockSize),
                  static_cast<Cell::Location::Integer>((idx.z / mData->voxelSpacing) % mData->blockSize));        

        cell.mBlockID = *itr;
        cell.mBlockSize = mData->blockSize;
        cell.mIsActive = cell.computeIsActive(mData->mActiveMask.rawMem(devID, Neon::DeviceType::CPU));
        return cell.mIsActive;
    }
    return false;
}

auto bGrid::getOriginBlock3DIndex(const Neon::int32_3d idx) const -> Neon::int32_3d
{
    //round n to nearest multiple of m
    auto roundDownToNearestMultiple = [](int32_t n, int32_t m) -> int32_t {
        return (n / m) * m;
    };

    Neon::int32_3d block_origin(roundDownToNearestMultiple(idx.x, mData->blockSize * mData->voxelSpacing),
                                roundDownToNearestMultiple(idx.y, mData->blockSize * mData->voxelSpacing),
                                roundDownToNearestMultiple(idx.z, mData->blockSize * mData->voxelSpacing));
    return block_origin;
}

auto bGrid::setReduceEngine(Neon::sys::patterns::Engine eng) -> void
{
    if (eng != Neon::sys::patterns::Engine::CUB) {
        NeonException exp("bGrid::setReduceEngine");
        exp << "bGrid only work on CUB engine for reduction";
        NEON_THROW(exp);
    }
}

auto bGrid::getLaunchParameters(Neon::DataView                         dataView,
                                [[maybe_unused]] const Neon::index_3d& blockSize,
                                const size_t&                          sharedMem) const -> Neon::set::LaunchParameters
{
    if (dataView != Neon::DataView::STANDARD) {
        NEON_WARNING("Requesting LaunchParameters on {} data view but bGrid only supports Standard data view on a single GPU",
                     Neon::DataViewUtil::toString(dataView));
    }

    const Neon::int32_3d cuda_block(mData->blockSize,
                                    mData->blockSize,
                                    mData->blockSize);

    Neon::set::LaunchParameters ret = getBackend().devSet().newLaunchParameters();
    for (int i = 0; i < ret.cardinality(); ++i) {
        if (getBackend().devType() == Neon::DeviceType::CUDA) {
            ret[i].set(Neon::sys::GpuLaunchInfo::mode_e::cudaGridMode,
                       Neon::int32_3d(int32_t(mData->mNumBlocks[i]), 1, 1),
                       cuda_block, sharedMem);
        } else {
            ret[i].set(Neon::sys::GpuLaunchInfo::mode_e::domainGridMode,
                       Neon::int32_3d(int32_t(mData->mNumBlocks[i]) *
                                          mData->blockSize *
                                          mData->blockSize *
                                          mData->blockSize,
                                      1, 1),
                       cuda_block, sharedMem);
        }
    }
    return ret;
}


auto bGrid::getPartitionIndexSpace(Neon::DeviceType dev,
                                   SetIdx           setIdx,
                                   Neon::DataView   dataView) -> const PartitionIndexSpace&
{
    return mData->mPartitionIndexSpace[Neon::DataViewUtil::toInt(dataView)].local(dev, setIdx, dataView);
}

auto bGrid::getOrigins() const -> const Neon::set::MemSet_t<Neon::int32_3d>&
{
    return mData->mOrigin;
}

auto bGrid::getStencilNghIndex() const -> const Neon::set::MemSet_t<nghIdx_t>&
{
    return mData->mStencilNghIndex;
}


auto bGrid::getNeighbourBlocks() const -> const Neon::set::MemSet_t<uint32_t>&
{
    return mData->mNeighbourBlocks;
}

auto bGrid::getActiveMask() const -> Neon::set::MemSet_t<uint32_t>&
{
    return mData->mActiveMask;
}

auto bGrid::getBlockOriginTo1D() const -> Neon::domain::tool::PointHashTable<int32_t, uint32_t>&
{
    return mData->mBlockOriginTo1D;
}


auto bGrid::getDimension() const -> const Neon::index_3d
{
    return GridBase::getDimension();
}

auto bGrid::getNumBlocks() const -> const Neon::set::DataSet<uint64_t>&
{
    return mData->mNumBlocks;
}

auto bGrid::getBlockSize() const -> int
{
    return mData->blockSize;
}

auto bGrid::getVoxelSpacing() const -> int
{
    return mData->voxelSpacing;
}

}  // namespace Neon::domain::internal::bGrid