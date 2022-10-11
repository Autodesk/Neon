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

    auto& descriptor = mData->mGrid->getDescriptorVector();

    //the allocation size is the number of blocks x block size x cardinality
    std::vector<Neon::set::DataSet<uint64_t>> allocSize(descriptor.size());

    for (size_t l = 0; l < descriptor.size(); ++l) {
        allocSize[l] = mData->mGrid->getBackend().devSet().template newDataSet<uint64_t>();
        for (int64_t i = 0; i < allocSize[l].size(); ++i) {
            allocSize[l][i] = mData->mGrid->getNumBlocksPerPartition(int(l))[i] *
                              descriptor[l] * descriptor[l] * descriptor[l] *
                              cardinality;
        }
    }

    Neon::MemoryOptions memOptions(Neon::DeviceType::CPU,
                                   Neon::Allocator::MALLOC,
                                   Neon::DeviceType::CUDA,
                                   ((mData->mGrid->getBackend().devType() == Neon::DeviceType::CUDA) ? Neon::Allocator::CUDA_MEM_DEVICE : Neon::Allocator::NULL_MEM),
                                   Neon::MemoryLayout::structOfArrays);

    mData->mMem.resize(descriptor.size());
    for (size_t l = 0; l < descriptor.size(); ++l) {
        mData->mMem[l] = mData->mGrid->getBackend().devSet().template newMemSet<T>({Neon::DataUse::IO_COMPUTE},
                                                                                   1,
                                                                                   memOptions,
                                                                                   allocSize[l]);
    }

    mData->mPartitions.resize(descriptor.size());

    for (int l = 0; l < int(descriptor.size()); ++l) {
        auto origins = mData->mGrid->getOrigins(l);
        auto parent = mData->mGrid->getParents(l);
        auto neighbours_blocks = mData->mGrid->getNeighbourBlocks(l);
        auto stencil_ngh = mData->mGrid->getStencilNghIndex();
        auto desct = mData->mGrid->getDescriptor();
        auto active_mask = mData->mGrid->getActiveMask(l);

        for (int dvID = 0; dvID < Neon::DataViewUtil::nConfig; dvID++) {
            mData->mPartitions[l][PartitionBackend::cpu][dvID] = mData->mGrid->getBackend().devSet().template newDataSet<Partition>();
            mData->mPartitions[l][PartitionBackend::gpu][dvID] = mData->mGrid->getBackend().devSet().template newDataSet<Partition>();

            for (int32_t gpuID = 0; gpuID < int32_t(mData->mPartitions[l][PartitionBackend::cpu][dvID].size()); gpuID++) {

                getPartition(Neon::DeviceType::CPU, Neon::SetIdx(gpuID), Neon::DataView(dvID), l) = bPartition<T, C>(
                    Neon::DataView(dvID),
                    l,
                    mData->mMem[l].rawMem(gpuID, Neon::DeviceType::CPU),
                    mData->mGrid->getDimension(),
                    mData->mCardinality,
                    neighbours_blocks.rawMem(gpuID, Neon::DeviceType::CPU),
                    origins.rawMem(gpuID, Neon::DeviceType::CPU),
                    parent.rawMem(gpuID, Neon::DeviceType::CPU),
                    active_mask.rawMem(gpuID, Neon::DeviceType::CPU),
                    outsideVal,
                    stencil_ngh.rawMem(gpuID, Neon::DeviceType::CPU),
                    desct.rawMem(gpuID, Neon::DeviceType::CPU));

                getPartition(Neon::DeviceType::CUDA, Neon::SetIdx(gpuID), Neon::DataView(dvID), l) = bPartition<T, C>(
                    Neon::DataView(dvID),
                    l,
                    mData->mMem[l].rawMem(gpuID, Neon::DeviceType::CUDA),
                    mData->mGrid->getDimension(),
                    mData->mCardinality,
                    neighbours_blocks.rawMem(gpuID, Neon::DeviceType::CUDA),
                    origins.rawMem(gpuID, Neon::DeviceType::CUDA),
                    parent.rawMem(gpuID, Neon::DeviceType::CUDA),
                    active_mask.rawMem(gpuID, Neon::DeviceType::CUDA),
                    outsideVal,
                    stencil_ngh.rawMem(gpuID, Neon::DeviceType::CUDA),
                    desct.rawMem(gpuID, Neon::DeviceType::CUDA));
            }
        }
    }
}


template <typename T, int C>
auto bField<T, C>::getPartition(const Neon::DeviceType& devType,
                                const Neon::SetIdx&     idx,
                                const Neon::DataView&   dataView,
                                const int               level) const -> const Partition&
{
    if (devType == Neon::DeviceType::CUDA) {
        return mData->mPartitions[level][PartitionBackend::gpu][Neon::DataViewUtil::toInt(dataView)][idx];
    } else {
        return mData->mPartitions[level][PartitionBackend::cpu][Neon::DataViewUtil::toInt(dataView)][idx];
    }
}

template <typename T, int C>
auto bField<T, C>::getPartition(const Neon::DeviceType& devType,
                                const Neon::SetIdx&     idx,
                                const Neon::DataView&   dataView,
                                const int               level) -> Partition&
{
    if (devType == Neon::DeviceType::CUDA) {
        return mData->mPartitions[level][PartitionBackend::gpu][Neon::DataViewUtil::toInt(dataView)][idx];
    } else {
        return mData->mPartitions[level][PartitionBackend::cpu][Neon::DataViewUtil::toInt(dataView)][idx];
    }
}

template <typename T, int C>
auto bField<T, C>::isInsideDomain(const Neon::index_3d& idx, const int level) const -> bool
{
    return mData->mGrid->isInsideDomain(idx, level);
}

template <typename T, int C>
auto bField<T, C>::getRef(const Neon::index_3d& idx,
                          const int&            cardinality,
                          const int             level) const -> T&
{
    //TODO need to figure out which device owns this block
    SetIdx devID(0);

    if (!isInsideDomain(idx, level)) {
        return this->getOutsideValue();
    }

    auto partition = getPartition(Neon::DeviceType::CPU, devID, Neon::DataView::STANDARD, level);

    Neon::int32_3d block_origin = mData->mGrid->getOriginBlock3DIndex(idx, level);

    auto itr = mData->mGrid->getBlockOriginTo1D(level).getMetadata(block_origin);
    Cell cell(static_cast<Cell::Location::Integer>(idx.x % mData->mGrid->getDescriptorVector()[level]),
              static_cast<Cell::Location::Integer>(idx.y % mData->mGrid->getDescriptorVector()[level]),
              static_cast<Cell::Location::Integer>(idx.z % mData->mGrid->getDescriptorVector()[level]));
    cell.mBlockID = *itr;
    cell.mBlockSize = mData->mGrid->getDescriptorVector()[level];
    return partition(cell, cardinality);
}

template <typename T, int C>
auto bField<T, C>::operator()(const Neon::index_3d& idx,
                              const int&            cardinality,
                              const int             level) const -> T
{
    return getRef(idx, cardinality, level);
}

template <typename T, int C>
auto bField<T, C>::getReference(const Neon::index_3d& idx,
                                const int&            cardinality,
                                const int             level) -> T&
{
    return getRef(idx, cardinality, level);
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
    for (size_t l = 0; l < mData->mMem.size(); ++l) {
        if (mData->mGrid->getBackend().devType() == Neon::DeviceType::CUDA) {
            mData->mMem[l].updateIO(mData->mGrid->getBackend(), streamId);
        }
    }
}

template <typename T, int C>
auto bField<T, C>::updateCompute(int streamId) -> void
{
    for (size_t l = 0; l < mData->mMem.size(); ++l) {
        if (mData->mGrid->getBackend().devType() == Neon::DeviceType::CUDA) {
            mData->mMem[l].updateCompute(mData->mGrid->getBackend(), streamId);
        }
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
auto bField<T, C>::getSharedMemoryBytes(const int32_t stencilRadius, int level) const -> size_t
{
    //This return the optimal shared memory size give a stencil radius
    //i.e., only N layers is read from neighbor blocks into shared memory in addition
    // to the block itself where N = stencilRadius
    return sizeof(T) *
           this->getCardinality() *
           (mData->mGrid->getDescriptorVector()[level] + 2 * stencilRadius) *
           (mData->mGrid->getDescriptorVector()[level] + 2 * stencilRadius) *
           (mData->mGrid->getDescriptorVector()[level] + 2 * stencilRadius);
}
}  // namespace Neon::domain::internal::bGrid
