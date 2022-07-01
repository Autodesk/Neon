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

    mData->mGrid = std::make_shared<bGrid>(grid);
    mData->mCardinality = cardinality;

    //the allocation size is the number of blocks x block size x cardinality
    Neon::set::DataSet<uint64_t> allocSize = mData->mGrid->getBackend().devSet().template newDataSet<uint64_t>();

    for (int64_t i = 0; i < allocSize.size(); ++i) {
        allocSize[i] = mData->mGrid->getNumBlocksPerPartition()[i] * Cell::sBlockSizeX * Cell::sBlockSizeY * Cell::sBlockSizeZ * cardinality;
    }

    Neon::MemoryOptions memOptions(Neon::DeviceType::CPU,
                                   Neon::Allocator::MALLOC,
                                   Neon::DeviceType::CUDA,
                                   ((mData->mGrid->getBackend().devType() == Neon::DeviceType::CUDA) ? Neon::Allocator::CUDA_MEM_DEVICE : Neon::Allocator::NULL_MEM),
                                   Neon::MemoryLayout::structOfArrays);

    mData->mMem = mData->mGrid->getBackend().devSet().template newMemSet<T>({Neon::DataUse::IO_COMPUTE},
                                                                            1,
                                                                            memOptions,
                                                                            allocSize);
    auto origins = mData->mGrid->getOrigins();
    auto neighbours_blocks = mData->mGrid->getNeighbourBlocks();
    auto active_mask = mData->mGrid->getActiveMask();

    for (int dvID = 0; dvID < Neon::DataViewUtil::nConfig; dvID++) {
        mData->mPartitions[PartitionBackend::cpu][dvID] = mData->mGrid->getBackend().devSet().template newDataSet<Partition>();
        mData->mPartitions[PartitionBackend::gpu][dvID] = mData->mGrid->getBackend().devSet().template newDataSet<Partition>();

        for (int32_t gpuID = 0; gpuID < int32_t(mData->mPartitions[PartitionBackend::cpu][dvID].size()); gpuID++) {

            getPartition(Neon::DeviceType::CPU, Neon::SetIdx(gpuID), Neon::DataView(dvID)) = bPartition<T, C>(
                Neon::DataView(dvID),
                mData->mMem.rawMem(gpuID, Neon::DeviceType::CPU),
                mData->mGrid->getDimension(),
                mData->mCardinality,
                neighbours_blocks.rawMem(gpuID, Neon::DeviceType::CPU),
                origins.rawMem(gpuID, Neon::DeviceType::CPU),
                active_mask.rawMem(gpuID, Neon::DeviceType::CPU),
                outsideVal);

            getPartition(Neon::DeviceType::CUDA, Neon::SetIdx(gpuID), Neon::DataView(dvID)) = bPartition<T, C>(
                Neon::DataView(dvID),
                mData->mMem.rawMem(gpuID, Neon::DeviceType::CUDA),
                mData->mGrid->getDimension(),
                mData->mCardinality,
                neighbours_blocks.rawMem(gpuID, Neon::DeviceType::CUDA),
                origins.rawMem(gpuID, Neon::DeviceType::CUDA),
                active_mask.rawMem(gpuID, Neon::DeviceType::CUDA),
                outsideVal);
        }
    }
}


template <typename T, int C>
auto bField<T, C>::getPartition(const Neon::DeviceType& devType,
                                const Neon::SetIdx&     idx,
                                const Neon::DataView&   dataView) const -> const Partition&
{
    if (devType == Neon::DeviceType::CUDA) {
        return mData->mPartitions[PartitionBackend::gpu][Neon::DataViewUtil::toInt(dataView)][idx];
    } else {
        return mData->mPartitions[PartitionBackend::cpu][Neon::DataViewUtil::toInt(dataView)][idx];
    }
}

template <typename T, int C>
auto bField<T, C>::getPartition(const Neon::DeviceType& devType,
                                const Neon::SetIdx&     idx,
                                const Neon::DataView&   dataView) -> Partition&
{
    if (devType == Neon::DeviceType::CUDA) {
        return mData->mPartitions[PartitionBackend::gpu][Neon::DataViewUtil::toInt(dataView)][idx];
    } else {
        return mData->mPartitions[PartitionBackend::cpu][Neon::DataViewUtil::toInt(dataView)][idx];
    }
}

template <typename T, int C>
auto bField<T, C>::isInsideDomain(const Neon::index_3d& idx) const -> bool
{
    return mData->mGrid->isInsideDomain(idx);
}

template <typename T, int C>
auto bField<T, C>::getRef(const Neon::index_3d& idx,
                          const int&            cardinality) const -> T&
{
    //TODO need to figure out which device owns this block
    SetIdx devID(0);

    auto partition = getPartition(Neon::DeviceType::CPU, devID, Neon::DataView::STANDARD);

    Neon::int32_3d block_origin = mData->mGrid->getOriginBlock3DIndex(idx);

    auto itr = mData->mGrid->getBlockOriginTo1D().getMetadata(block_origin);
    if (!itr) {
        return this->getOutsideValue();
    }
    Cell cell(static_cast<Cell::Location::Integer>(idx.x % Cell::sBlockSizeX),
              static_cast<Cell::Location::Integer>(idx.y % Cell::sBlockSizeY),
              static_cast<Cell::Location::Integer>(idx.z % Cell::sBlockSizeZ));
    cell.mBlockID = *itr;
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
    if (mData->mGrid->getBackend().devType() == Neon::DeviceType::CUDA) {
        mData->mMem.updateIO(mData->mGrid->getBackend(), streamId);
    }
}

template <typename T, int C>
auto bField<T, C>::updateCompute(int streamId) -> void
{
    if (mData->mGrid->getBackend().devType() == Neon::DeviceType::CUDA) {
        mData->mMem.updateCompute(mData->mGrid->getBackend(), streamId);
    }
}

template <typename T, int C>
auto bField<T, C>::getPartition([[maybe_unused]] Neon::Execution,
                                [[maybe_unused]] Neon::SetIdx,
                                [[maybe_unused]] const Neon::DataView& dataView) const -> const Partition&
{
    NEON_DEV_UNDER_CONSTRUCTION("bField::getPartition");
}

template <typename T, int C>
auto bField<T, C>::getPartition([[maybe_unused]] Neon::Execution,
                                [[maybe_unused]] Neon::SetIdx          idx,
                                [[maybe_unused]] const Neon::DataView& dataView) -> Partition&
{
    NEON_DEV_UNDER_CONSTRUCTION("bField::getPartition");
}

template <typename T, int C>
auto bField<T, C>::getSharedMemoryBytes(const int32_t stencilRadius) const -> size_t
{
    //This return the optimal shared memory size give a stencil radius
    //i.e., only N layers is read from neighbor blocks into shared memory in addition
    // to the block itself where N = stencilRadius
    return sizeof(T) *
           this->getCardinality() *
           (Cell::sBlockSizeX + 2 * stencilRadius) *
           (Cell::sBlockSizeY + 2 * stencilRadius) *
           (Cell::sBlockSizeZ + 2 * stencilRadius);
}
}  // namespace Neon::domain::internal::bGrid
