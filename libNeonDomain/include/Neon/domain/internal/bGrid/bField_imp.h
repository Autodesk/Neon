#pragma once

#include "Neon/domain/internal/bGrid/bField.h"

namespace Neon::domain::internal::bGrid {

template <typename T, int C>
bField<T, C>::bField(const std::string&             name,
                     const bGrid&                   grid,
                     int                            cardinality,
                     T                              outsideVal,
                     Neon::DataUse                  dataUse,
                     const Neon::MemoryOptions&     memoryOptions,
                     Neon::domain::haloStatus_et::e haloStatus)
    : Neon::domain::interface::FieldBaseTemplate<T, C, Grid, Partition, int>(&grid,
                                                                             name,
                                                                             "bField",
                                                                             cardinality,
                                                                             outsideVal,
                                                                             dataUse,
                                                                             memoryOptions,
                                                                             haloStatus)
{
    mData = std::make_shared<Data>();

    mData->grid = std::make_shared<bGrid>(grid);
    mData->mCardinality = cardinality;

    int blockSize = mData->grid->getBlockSize();

    //the allocation size is the number of blocks x block size x cardinality
    Neon::set::DataSet<uint64_t> allocSize;
    allocSize = mData->grid->getBackend().devSet().template newDataSet<uint64_t>();
    for (int64_t i = 0; i < allocSize.size(); ++i) {
        allocSize[i] = mData->grid->getNumBlocks()[i] *
                       blockSize * blockSize * blockSize *
                       cardinality;
    }


    Neon::MemoryOptions memOptions(Neon::DeviceType::CPU,
                                   Neon::Allocator::MALLOC,
                                   Neon::DeviceType::CUDA,
                                   ((mData->grid->getBackend().devType() == Neon::DeviceType::CUDA) ? Neon::Allocator::CUDA_MEM_DEVICE : Neon::Allocator::NULL_MEM),
                                   Neon::MemoryLayout::structOfArrays);


    mData->mem = mData->grid->getBackend().devSet().template newMemSet<T>({Neon::DataUse::IO_COMPUTE},
                                                                          1,
                                                                          memOptions,
                                                                          allocSize);


    auto origins = mData->grid->getOrigins();
    auto neighbours_blocks = mData->grid->getNeighbourBlocks();
    auto stencil_ngh = mData->grid->getStencilNghIndex();
    auto active_mask = mData->grid->getActiveMask();

    for (int dvID = 0; dvID < Neon::DataViewUtil::nConfig; dvID++) {
        mData->partitions[PartitionBackend::cpu][dvID] = mData->grid->getBackend().devSet().template newDataSet<Partition>();
        mData->partitions[PartitionBackend::gpu][dvID] = mData->grid->getBackend().devSet().template newDataSet<Partition>();

        for (int32_t gpuID = 0; gpuID < int32_t(mData->partitions[PartitionBackend::cpu][dvID].size()); gpuID++) {

            getPartition(Neon::DeviceType::CPU, Neon::SetIdx(gpuID), Neon::DataView(dvID)) = bPartition<T, C>(
                Neon::DataView(dvID),
                mData->mem.rawMem(gpuID, Neon::DeviceType::CPU),
                mData->mCardinality,
                neighbours_blocks.rawMem(gpuID, Neon::DeviceType::CPU),
                origins.rawMem(gpuID, Neon::DeviceType::CPU),
                active_mask.rawMem(gpuID, Neon::DeviceType::CPU),
                outsideVal,
                stencil_ngh.rawMem(gpuID, Neon::DeviceType::CPU));

            getPartition(Neon::DeviceType::CUDA, Neon::SetIdx(gpuID), Neon::DataView(dvID)) = bPartition<T, C>(
                Neon::DataView(dvID),
                mData->mem.rawMem(gpuID, Neon::DeviceType::CUDA),
                mData->mCardinality,
                neighbours_blocks.rawMem(gpuID, Neon::DeviceType::CUDA),
                origins.rawMem(gpuID, Neon::DeviceType::CUDA),
                active_mask.rawMem(gpuID, Neon::DeviceType::CUDA),
                outsideVal,
                stencil_ngh.rawMem(gpuID, Neon::DeviceType::CUDA));
        }
    }
}

template <typename T, int C>
auto bField<T, C>::forEachActiveCell(
    const std::function<void(const Neon::index_3d&,
                             const int& cardinality,
                             T&)>&                      fun,
    [[maybe_unused]] Neon::computeMode_t::computeMode_e mode)
    -> void
{

    //TODO need to figure out which device owns this block
    SetIdx devID(0);

    const int blockSize = mData->grid->getBlockSize();
    const int voxelSpacing = mData->grid->getVoxelSpacing();

    mData->grid->getBlockOriginTo1D().forEach(
        [&](const Neon::int32_3d blockOrigin, const uint32_t blockIdx) {
            for (int16_t z = 0; z < blockSize; z++) {
                for (int16_t y = 0; y < blockSize; y++) {
                    for (int16_t x = 0; x < blockSize; x++) {

                        Cell cell(x, y, z);
                        cell.mBlockID = blockIdx;
                        cell.mBlockSize = blockSize;
                        if (cell.computeIsActive(mData->grid->getActiveMask().rawMem(devID, Neon::DeviceType::CPU))) {
                            for (int c = 0; c < this->getCardinality(); c++) {
                                Neon::index_3d local(x * voxelSpacing, y * voxelSpacing, z * voxelSpacing);
                                Neon::index_3d index3D = blockOrigin + local;
                                fun(index3D, c, this->getReference(index3D, c));
                            }
                        }
                    }
                }
            }
        });
}
template <typename T, int C>
auto bField<T, C>::getPartition(const Neon::DeviceType& devType,
                                const Neon::SetIdx&     idx,
                                const Neon::DataView&   dataView) const -> const Partition&
{
    if (devType == Neon::DeviceType::CUDA) {
        return mData->partitions[PartitionBackend::gpu][Neon::DataViewUtil::toInt(dataView)][idx];
    } else {
        return mData->partitions[PartitionBackend::cpu][Neon::DataViewUtil::toInt(dataView)][idx];
    }
}

template <typename T, int C>
auto bField<T, C>::getPartition(const Neon::DeviceType& devType,
                                const Neon::SetIdx&     idx,
                                const Neon::DataView&   dataView) -> Partition&
{
    if (devType == Neon::DeviceType::CUDA) {
        return mData->partitions[PartitionBackend::gpu][Neon::DataViewUtil::toInt(dataView)][idx];
    } else {
        return mData->partitions[PartitionBackend::cpu][Neon::DataViewUtil::toInt(dataView)][idx];
    }
}


template <typename T, int C>
auto bField<T, C>::isInsideDomain(const Neon::index_3d& idx) const -> bool
{
    return mData->grid->isInsideDomain(idx);
}

template <typename T, int C>
auto bField<T, C>::getRef(const Neon::index_3d& idx,
                          const int&            cardinality) const -> T&
{
    //TODO need to figure out which device owns this block
    SetIdx devID(0);

    if (!isInsideDomain(idx)) {
        return this->getOutsideValue();
    }

    auto partition = getPartition(Neon::DeviceType::CPU, devID, Neon::DataView::STANDARD);

    Neon::int32_3d blockOrigin = mData->grid->getOriginBlock3DIndex(idx);

    auto itr = mData->grid->getBlockOriginTo1D().getMetadata(blockOrigin);

    auto blockSize = mData->grid->getBlockSize();

    Cell cell(static_cast<Cell::Location::Integer>((idx.x / mData->grid->getVoxelSpacing()) % blockSize),
              static_cast<Cell::Location::Integer>((idx.y / mData->grid->getVoxelSpacing()) % blockSize),
              static_cast<Cell::Location::Integer>((idx.z / mData->grid->getVoxelSpacing()) % blockSize));

    cell.mBlockID = *itr;
    cell.mBlockSize = blockSize;
    return partition(cell, cardinality);
}

template <typename T, int C>
auto bField<T, C>::operator()(const Neon::index_3d& idx,
                              const int&            cardinality) const -> T
{
    return getRef(idx, cardinality);
}

template <typename T, int C>
auto bField<T, C>::getReference(const Neon::index_3d& idx,
                                const int&            cardinality) -> T&
{
    return getRef(idx, cardinality);
}

template <typename T, int C>
auto bField<T, C>::haloUpdate(Neon::set::HuOptions& /*opt*/) const -> void
{
    //TODO
    NEON_DEV_UNDER_CONSTRUCTION("bField::haloUpdate");
}

template <typename T, int C>
auto bField<T, C>::haloUpdate(Neon::set::HuOptions& /*opt*/) -> void
{
    //TODO
    NEON_DEV_UNDER_CONSTRUCTION("bField::haloUpdate");
}

template <typename T, int C>
auto bField<T, C>::updateIO(int streamId) -> void
{

    if (mData->grid->getBackend().devType() == Neon::DeviceType::CUDA) {
        mData->mem.updateIO(mData->grid->getBackend(), streamId);
    }
}

template <typename T, int C>
auto bField<T, C>::updateCompute(int streamId) -> void
{

    if (mData->grid->getBackend().devType() == Neon::DeviceType::CUDA) {
        mData->mem.updateCompute(mData->grid->getBackend(), streamId);
    }
}


template <typename T, int C>
auto bField<T, C>::getPartition(Neon::Execution       exec,
                                Neon::SetIdx          idx,
                                const Neon::DataView& dataView) const -> const Partition&
{

    if (exec == Neon::Execution::device) {
        return getPartition(Neon::DeviceType::CUDA, idx, dataView);
    }
    if (exec == Neon::Execution::host) {
        return getPartition(Neon::DeviceType::CPU, idx, dataView);
    }

    NEON_THROW_UNSUPPORTED_OPERATION("bField::getPartition() unsupported Execution");
}

template <typename T, int C>
auto bField<T, C>::getPartition(Neon::Execution       exec,
                                Neon::SetIdx          idx,
                                const Neon::DataView& dataView) -> Partition&
{
    if (exec == Neon::Execution::device) {
        return getPartition(Neon::DeviceType::CUDA, idx, dataView);
    }
    if (exec == Neon::Execution::host) {
        return getPartition(Neon::DeviceType::CPU, idx, dataView);
    }

    NEON_THROW_UNSUPPORTED_OPERATION("bField::getPartition() unsupported Execution");
}

template <typename T, int C>
auto bField<T, C>::getMem() -> Neon::set::MemSet_t<T>&
{
    return mData->mem;
}

template <typename T, int C>
auto bField<T, C>::getSharedMemoryBytes(const int32_t stencilRadius) const -> size_t
{
    //This return the optimal shared memory size give a stencil radius
    //i.e., only N layers is read from neighbor blocks into shared memory in addition
    // to the block itself where N = stencilRadius
    int blockSize = mData->grid->getBlockSize();
    return sizeof(T) *
           this->getCardinality() *
           (blockSize + 2 * stencilRadius) *
           (blockSize + 2 * stencilRadius) *
           (blockSize + 2 * stencilRadius);
}

}  // namespace Neon::domain::internal::bGrid
