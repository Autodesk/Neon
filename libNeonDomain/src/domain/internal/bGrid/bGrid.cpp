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

    auto itr = mData->mBlockOriginTo1D[0].getMetadata(block_origin);
    if (itr) {
        Cell cell(static_cast<Cell::Location::Integer>(idx.x % Cell::sBlockSizeX),
                  static_cast<Cell::Location::Integer>(idx.y % Cell::sBlockSizeY),
                  static_cast<Cell::Location::Integer>(idx.z % Cell::sBlockSizeZ));
        cell.mBlockID = *itr;
        cell.mIsActive = cell.computeIsActive(mData->mActiveMask[0].rawMem(devID, Neon::DeviceType::CPU));
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

    Neon::int32_3d block_origin(roundDownToNearestMultiple(idx.x, Cell::sBlockSizeX),
                                roundDownToNearestMultiple(idx.y, Cell::sBlockSizeY),
                                roundDownToNearestMultiple(idx.z, Cell::sBlockSizeZ));
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
    //TODO
    if (dataView != Neon::DataView::STANDARD) {
        NEON_WARNING("Requesting LaunchParameters on {} data view but bGrid only supports Standard data view on a single GPU",
                     Neon::DataViewUtil::toString(dataView));
    }
    const Neon::int32_3d        cuda_block(Cell::sBlockSizeX, Cell::sBlockSizeY, Cell::sBlockSizeZ);
    Neon::set::LaunchParameters ret = getBackend().devSet().newLaunchParameters();
    for (int i = 0; i < ret.cardinality(); ++i) {
        if (getBackend().devType() == Neon::DeviceType::CUDA) {
            ret[i].set(Neon::sys::GpuLaunchInfo::mode_e::cudaGridMode,
                       Neon::int32_3d(int32_t(mData->mNumBlocks[0][i]), 1, 1),
                       cuda_block, sharedMem);
        } else {
            ret[i].set(Neon::sys::GpuLaunchInfo::mode_e::domainGridMode,
                       Neon::int32_3d(int32_t(mData->mNumBlocks[0][i]) * Cell::sBlockSizeX * Cell::sBlockSizeY * Cell::sBlockSizeZ, 1, 1),
                       cuda_block, sharedMem);
        }
    }
    return ret;
}

auto bGrid::getPartitionIndexSpace(Neon::DeviceType dev,
                                   SetIdx           setIdx,
                                   Neon::DataView   dataView) -> const PartitionIndexSpace&
{
    return mData->mPartitionIndexSpace.at(Neon::DataViewUtil::toInt(dataView)).local(dev, setIdx, dataView);
}


auto bGrid::getNumBlocksPerPartition(int level) const -> const Neon::set::DataSet<uint64_t>&
{
    return mData->mNumBlocks[level];
}

auto bGrid::getOrigins(int level) const -> const Neon::set::MemSet_t<Neon::int32_3d>&
{
    return mData->mOrigin[level];
}

auto bGrid::getStencilNghIndex() const -> const Neon::set::MemSet_t<nghIdx_t>&
{
    return mData->mStencilNghIndex;
}

auto bGrid::getNeighbourBlocks(int level) const -> const Neon::set::MemSet_t<uint32_t>&
{
    return mData->mNeighbourBlocks[level];
}

auto bGrid::getActiveMask(int level) const -> const Neon::set::MemSet_t<uint32_t>&
{
    return mData->mActiveMask[level];
}

auto bGrid::getBlockOriginTo1D(int level) const -> const Neon::domain::tool::PointHashTable<int32_t, uint32_t>&
{
    return mData->mBlockOriginTo1D[level];
}

auto bGrid::getKernelConfig(int            streamIdx,
                            Neon::DataView dataView) -> Neon::set::KernelConfig
{
    Neon::domain::KernelConfig kernelConfig(streamIdx, dataView);
    if (kernelConfig.runtime() != Neon::Runtime::system) {
        NEON_DEV_UNDER_CONSTRUCTION("bGrid::getKernelConfig");
    }

    Neon::set::LaunchParameters launchInfoSet = getLaunchParameters(dataView,
                                                                    getDefaultBlock(), 0);

    kernelConfig.expertSetLaunchParameters(launchInfoSet);
    kernelConfig.expertSetBackend(getBackend());

    return kernelConfig;
}
}  // namespace Neon::domain::internal::bGrid