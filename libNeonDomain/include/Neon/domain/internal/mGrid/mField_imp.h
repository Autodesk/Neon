#pragma once

#include "Neon/domain/internal/mGrid/mField.h"

namespace Neon::domain::internal::mGrid {

template <typename T, int C>
mField<T, C>::mField(const std::string&             name,
                     const mGrid&                   grid,
                     int                            cardinality,
                     T                              outsideVal,
                     Neon::DataUse                  dataUse,
                     const Neon::MemoryOptions&     memoryOptions,
                     Neon::domain::haloStatus_et::e haloStatus)
{
    mData = std::make_shared<Data>();

    mData->grid = std::make_shared<mGrid>(grid);
    const auto& descriptor = mData->grid->getDescriptor();
    mData->fields.resize(descriptor.getDepth());


    for (int l = 0; l < descriptor.getDepth(); ++l) {
        mData->fields[l] = xField<T, C>(name,
                                        grid(l),
                                        cardinality,
                                        outsideVal,
                                        dataUse,
                                        memoryOptions,
                                        haloStatus);
    }

    auto refFactorSet = mData->grid->getRefFactors();
    auto spacingSet = mData->grid->getLevelSpacing();

    for (int l = 0; l < descriptor.getDepth(); ++l) {

        auto mem = mData->fields[l].mData->field.getMem();

        auto origins = grid(l).getOrigins();
        auto neighbours_blocks = grid(l).getNeighbourBlocks();
        auto stencil_ngh = grid(l).getStencilNghIndex();
        auto active_mask = grid(l).getActiveMask();
        auto parent = mData->grid->getParentsBlockID(l);
        auto parentLocalID = mData->grid->getParentLocalID(l);
        auto childBlockID = mData->grid->getChildBlockID(l);


        for (int dvID = 0; dvID < Neon::DataViewUtil::nConfig; dvID++) {
            mData->fields[l].mData->mPartitions[PartitionBackend::cpu][dvID] = grid.getBackend().devSet().template newDataSet<Partition>();
            mData->fields[l].mData->mPartitions[PartitionBackend::gpu][dvID] = grid.getBackend().devSet().template newDataSet<Partition>();

            for (int32_t gpuID = 0; gpuID < int32_t(mData->fields[l].mData->mPartitions[PartitionBackend::cpu][dvID].size()); gpuID++) {

                mData->fields[l].getPartition(Neon::DeviceType::CPU, Neon::SetIdx(gpuID), Neon::DataView(dvID)) =
                    Neon::domain::internal::mGrid::mPartition<T, C>(
                        Neon::DataView(dvID),
                        l,
                        mem.rawMem(gpuID, Neon::DeviceType::CPU),
                        (l == int(descriptor.getDepth()) - 1) ? nullptr : mData->fields[l + 1].mData->field.getMem().rawMem(gpuID, Neon::DeviceType::CPU),  //parent
                        (l == 0) ? nullptr : mData->fields[l - 1].mData->field.getMem().rawMem(gpuID, Neon::DeviceType::CPU),                               //child
                        cardinality,
                        neighbours_blocks.rawMem(gpuID, Neon::DeviceType::CPU),
                        origins.rawMem(gpuID, Neon::DeviceType::CPU),
                        parent.rawMem(gpuID, Neon::DeviceType::CPU),
                        parentLocalID.rawMem(gpuID, Neon::DeviceType::CPU),
                        active_mask.rawMem(gpuID, Neon::DeviceType::CPU),
                        (l == 0) ? nullptr : grid(l - 1).getActiveMask().rawMem(gpuID, Neon::DeviceType::CPU),  //lower-level mask
                        (l == 0) ? nullptr : childBlockID.rawMem(gpuID, Neon::DeviceType::CPU),
                        outsideVal,
                        stencil_ngh.rawMem(gpuID, Neon::DeviceType::CPU),
                        refFactorSet.rawMem(gpuID, Neon::DeviceType::CPU),
                        spacingSet.rawMem(gpuID, Neon::DeviceType::CPU));

                mData->fields[l].getPartition(Neon::DeviceType::CUDA, Neon::SetIdx(gpuID), Neon::DataView(dvID)) =
                    Neon::domain::internal::mGrid::mPartition<T, C>(
                        Neon::DataView(dvID),
                        l,
                        mem.rawMem(gpuID, Neon::DeviceType::CUDA),
                        (l == int(descriptor.getDepth()) - 1) ? nullptr : mData->fields[l + 1].mData->field.getMem().rawMem(gpuID, Neon::DeviceType::CUDA),  //parent
                        (l == 0) ? nullptr : mData->fields[l - 1].mData->field.getMem().rawMem(gpuID, Neon::DeviceType::CUDA),                               //child
                        cardinality,
                        neighbours_blocks.rawMem(gpuID, Neon::DeviceType::CUDA),
                        origins.rawMem(gpuID, Neon::DeviceType::CUDA),
                        parent.rawMem(gpuID, Neon::DeviceType::CUDA),
                        parentLocalID.rawMem(gpuID, Neon::DeviceType::CUDA),
                        active_mask.rawMem(gpuID, Neon::DeviceType::CUDA),
                        (l == 0) ? nullptr : grid(l - 1).getActiveMask().rawMem(gpuID, Neon::DeviceType::CUDA),  //lower-level mask
                        (l == 0) ? nullptr : childBlockID.rawMem(gpuID, Neon::DeviceType::CUDA),
                        outsideVal,
                        stencil_ngh.rawMem(gpuID, Neon::DeviceType::CUDA),
                        refFactorSet.rawMem(gpuID, Neon::DeviceType::CUDA),
                        spacingSet.rawMem(gpuID, Neon::DeviceType::CUDA));
            }
        }
    }
}


template <typename T, int C>
template <Neon::computeMode_t::computeMode_e mode>
auto mField<T, C>::forEachActiveCell(
    const int                      level,
    const std::function<void(const Neon::index_3d&,
                             const int& cardinality,
                             T&)>& fun)
    -> void
{
    const auto& descriptor = mData->grid->getDescriptor();

    //TODO need to figure out which device owns this block
    SetIdx devID(0);

    const int refFactor = descriptor.getRefFactor(level);

    mData->grid->operator()(level).getBlockOriginTo1D().forEach(
        [&](const Neon::int32_3d blockOrigin, const uint32_t blockIdx) {
            for (int16_t z = 0; z < refFactor; z++) {
                for (int16_t y = 0; y < refFactor; y++) {
                    for (int16_t x = 0; x < refFactor; x++) {

                        Cell cell(x, y, z);
                        cell.mBlockID = blockIdx;
                        cell.mBlockSize = refFactor;
                        if (cell.computeIsActive(mData->grid->operator()(level).getActiveMask().rawMem(devID, Neon::DeviceType::CPU))) {
                            for (int c = 0; c < mData->fields[level].getCardinality(); c++) {
                                const Neon::index_3d local(x, y, z);
                                Neon::index_3d       index3D = blockOrigin + descriptor.toBaseIndexSpace(local, level);
                                fun(index3D, c, this->getReference(index3D, c, level));
                            }
                        }
                    }
                }
            }
        });
}


template <typename T, int C>
auto mField<T, C>::isInsideDomain(const Neon::index_3d& idx, const int level) const -> bool
{
    return mData->grid->isInsideDomain(idx, level);
}

template <typename T, int C>
auto mField<T, C>::getRef(const Neon::index_3d& idx,
                          const int&            cardinality,
                          const int             level) const -> T&
{
    //TODO need to figure out which device owns this block
    SetIdx devID(0);

    if (!isInsideDomain(idx, level)) {
        return mData->fields[level].getOutsideValue();
    }

    auto partition = getPartition(Neon::DeviceType::CPU, devID, Neon::DataView::STANDARD, level);

    Neon::int32_3d block_origin = mData->grid->getOriginBlock3DIndex(idx, level);

    auto itr = mData->grid->getBlockOriginTo1D(level).getMetadata(block_origin);

    Neon::index_3d localID = mData->grid->getDescriptor().toLocalIndex(idx, level);

    Cell cell(static_cast<Cell::Location::Integer>(localID.x),
              static_cast<Cell::Location::Integer>(localID.y),
              static_cast<Cell::Location::Integer>(localID.z));

    cell.mBlockID = *itr;
    cell.mBlockSize = mData->grid->getDescriptor().getRefFactor(level);
    return partition(cell, cardinality);
}

template <typename T, int C>
auto mField<T, C>::operator()(int level) -> xField<T, C>&
{
    return mData->fields[level];
}

template <typename T, int C>
auto mField<T, C>::operator()(int level) const -> const xField<T, C>&
{
    return mData->fields[level];
}


template <typename T, int C>
auto mField<T, C>::getReference(const Neon::index_3d& idx,
                                const int&            cardinality,
                                const int             level) -> T&
{
    return mData->fields[level].operator()(idx, cardinality);
}

template <typename T, int C>
auto mField<T, C>::getReference(const Neon::index_3d& idx,
                                const int&            cardinality,
                                const int             level) const -> const T&
{
    return mData->fields[level].operator()(idx, cardinality);
}

template <typename T, int C>
auto mField<T, C>::haloUpdate(Neon::set::HuOptions& /*opt*/) const -> void
{
    //TODO
    NEON_DEV_UNDER_CONSTRUCTION("mField::haloUpdate");
}

template <typename T, int C>
auto mField<T, C>::haloUpdate(Neon::set::HuOptions& /*opt*/) -> void
{
    //TODO
    NEON_DEV_UNDER_CONSTRUCTION("mField::haloUpdate");
}

template <typename T, int C>
auto mField<T, C>::updateIO(int streamId) -> void
{

    for (size_t l = 0; l < mData->fields.size(); ++l) {
        mData->fields[l].updateIO(streamId);
    }
}

template <typename T, int C>
auto mField<T, C>::updateCompute(int streamId) -> void
{
    for (size_t l = 0; l < mData->fields.size(); ++l) {
        mData->fields[l].updateCompute(streamId);
    }
}


template <typename T, int C>
auto mField<T, C>::ioToVtk(const std::string& fileName,
                           const std::string& fieldName,
                           Neon::IoFileType   ioFileType) const -> void
{
    const auto& descriptor = mData->grid->getDescriptor();

    for (int l = 0; l < descriptor.getDepth(); ++l) {

        double spacing = double(descriptor.getSpacing(l - 1));

        auto iovtk = IoToVTK<int, T>(fileName + "_level" + std::to_string(l),
                                     mData->grid->getDimension(l) + 1,
                                     {spacing, spacing, spacing},
                                     {0, 0, 0},
                                     ioFileType);


        iovtk.addField(
            [&](Neon::index_3d idx, int card) -> T {
                idx = descriptor.toBaseIndexSpace(idx, l);

                if (mData->grid->isInsideDomain(idx, l)) {
                    return getReference(idx, card, l);
                } else {
                    return mData->fields[l].getOutsideValue();
                }
            },
            mData->fields[l].getCardinality(), fieldName, ioToVTKns::VtiDataType_e::voxel);

        iovtk.flushAndClear();
    }
}


template <typename T, int C>
auto mField<T, C>::getSharedMemoryBytes(const int32_t stencilRadius, int level) const -> size_t
{
    //This return the optimal shared memory size give a stencil radius
    //i.e., only N layers is read from neighbor blocks into shared memory in addition
    // to the block itself where N = stencilRadius
    int refFactor = mData->grid->getDescriptor().getRefFactor(level);
    return sizeof(T) *
           mData->fields[l].getCardinality() *
           (refFactor + 2 * stencilRadius) *
           (refFactor + 2 * stencilRadius) *
           (refFactor + 2 * stencilRadius);
}

}  // namespace Neon::domain::internal::mGrid
