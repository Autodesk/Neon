#pragma once

#include "Neon/domain/details/mGrid/mField.h"

namespace Neon::domain::details::mGrid {

template <typename T, int C>
mField<T, C>::mField(const std::string&         name,
                     const mGrid&               grid,
                     int                        cardinality,
                     T                          outsideVal,
                     Neon::DataUse              dataUse,
                     const Neon::MemoryOptions& memoryOptions)
{
    mData = std::make_shared<Data>();

    mData->grid = std::make_shared<mGrid>(grid);
    const auto& descriptor = mData->grid->getDescriptor();
    mData->fields.resize(descriptor.getDepth());


    for (int l = 0; l < descriptor.getDepth(); ++l) {
        mData->fields[l] = xField<T, C>(name,
                                        mData->grid->operator()(l),
                                        cardinality,
                                        outsideVal,
                                        dataUse,
                                        memoryOptions);
    }

    auto refFactorSet = mData->grid->getRefFactors();
    auto spacingSet = mData->grid->getLevelSpacing();

    for (int l = 0; l < descriptor.getDepth(); ++l) {

        auto childBlockID = mData->grid->getChildBlockID(l);


        //for (int dvID = 0; dvID < Neon::DataViewUtil::nConfig; dvID++) {
        int dvID = 0;

        mData->fields[l].mData->mPartitions[PartitionBackend::cpu][dvID] = mData->grid->getBackend().devSet().template newDataSet<Partition>();
        mData->fields[l].mData->mPartitions[PartitionBackend::gpu][dvID] = mData->grid->getBackend().devSet().template newDataSet<Partition>();

        for (int32_t gpuID = 0; gpuID < int32_t(mData->fields[l].mData->mPartitions[PartitionBackend::cpu][dvID].size()); gpuID++) {

            auto setIdx = Neon::SetIdx(gpuID);

            mData->fields[l].getPartition(Neon::DeviceType::CPU, setIdx, Neon::DataView(dvID)) =
                Neon::domain::details::mGrid::mPartition<T, C>(
                    l,
                    mData->fields[l].mData->field.getMemoryField().getPartition(Neon::Execution::host, setIdx, Neon::DataView::STANDARD).mem(),
                    (l == int(descriptor.getDepth()) - 1) ? nullptr : mData->fields[l + 1].mData->field.getMemoryField().getPartition(Neon::Execution::host, setIdx, Neon::DataView::STANDARD).mem(),  //parent
                    (l == 0) ? nullptr : mData->fields[l - 1].mData->field.getMemoryField().getPartition(Neon::Execution::host, setIdx, Neon::DataView::STANDARD).mem(),                               //child
                    cardinality,
                    mData->grid->operator()(l).helpGetBlockConnectivity().getPartition(Neon::Execution::host, setIdx, Neon::DataView::STANDARD).mem(),
                    mData->grid->operator()(l).helpGetDataBlockOriginField().getPartition(Neon::Execution::host, setIdx, Neon::DataView::STANDARD).mem(),
                    (l == int(descriptor.getDepth()) - 1) ? nullptr : mData->grid->getParentsBlockID(l).rawMem(gpuID, Neon::DeviceType::CPU),
                    mData->grid->operator()(l).getActiveBitMask().getPartition(Neon::Execution::host, setIdx, Neon::DataView::STANDARD).mem(),
                    (l == 0) ? nullptr : mData->grid->operator()(l - 1).getActiveBitMask().getPartition(Neon::Execution::host, setIdx, Neon::DataView::STANDARD).mem(),                               //lower-level mask
                    (l == int(descriptor.getDepth()) - 1) ? nullptr : mData->grid->operator()(l + 1).getActiveBitMask().getPartition(Neon::Execution::host, setIdx, Neon::DataView::STANDARD).mem(),  //upper-level mask
                    (l == 0) ? nullptr : childBlockID.rawMem(gpuID, Neon::DeviceType::CPU),
                    (l == int(descriptor.getDepth()) - 1) ? nullptr : mData->grid->operator()(l + 1).helpGetBlockConnectivity().getPartition(Neon::Execution::host, setIdx, Neon::DataView::STANDARD).mem(),  //parent neighbor
                    mData->grid->operator()(l).helpGetStencilIdTo3dOffset().rawMem(Neon::Execution::host, setIdx),
                    refFactorSet.rawMem(gpuID, Neon::DeviceType::CPU),
                    spacingSet.rawMem(gpuID, Neon::DeviceType::CPU));

            mData->fields[l].getPartition(Neon::DeviceType::CUDA, setIdx, Neon::DataView(dvID)) =
                Neon::domain::details::mGrid::mPartition<T, C>(
                    l,
                    mData->fields[l].mData->field.getMemoryField().getPartition(Neon::Execution::device, setIdx, Neon::DataView::STANDARD).mem(),
                    (l == int(descriptor.getDepth()) - 1) ? nullptr : mData->fields[l + 1].mData->field.getMemoryField().getPartition(Neon::Execution::device, setIdx, Neon::DataView::STANDARD).mem(),  //parent
                    (l == 0) ? nullptr : mData->fields[l - 1].mData->field.getMemoryField().getPartition(Neon::Execution::device, setIdx, Neon::DataView::STANDARD).mem(),                               //child
                    cardinality,
                    mData->grid->operator()(l).helpGetBlockConnectivity().getPartition(Neon::Execution::device, setIdx, Neon::DataView::STANDARD).mem(),
                    mData->grid->operator()(l).helpGetDataBlockOriginField().getPartition(Neon::Execution::device, setIdx, Neon::DataView::STANDARD).mem(),
                    (l == int(descriptor.getDepth()) - 1) ? nullptr : mData->grid->getParentsBlockID(l).rawMem(gpuID, Neon::DeviceType::CUDA),
                    mData->grid->operator()(l).getActiveBitMask().getPartition(Neon::Execution::device, setIdx, Neon::DataView::STANDARD).mem(),
                    (l == 0) ? nullptr : mData->grid->operator()(l - 1).getActiveBitMask().getPartition(Neon::Execution::device, setIdx, Neon::DataView::STANDARD).mem(),                               //lower-level mask
                    (l == int(descriptor.getDepth()) - 1) ? nullptr : mData->grid->operator()(l + 1).getActiveBitMask().getPartition(Neon::Execution::device, setIdx, Neon::DataView::STANDARD).mem(),  //upper-level mask
                    (l == 0) ? nullptr : childBlockID.rawMem(gpuID, Neon::DeviceType::CUDA),
                    (l == int(descriptor.getDepth()) - 1) ? nullptr : mData->grid->operator()(l + 1).helpGetBlockConnectivity().getPartition(Neon::Execution::device, setIdx, Neon::DataView::STANDARD).mem(),  //parent neighbor
                    mData->grid->operator()(l).helpGetStencilIdTo3dOffset().rawMem(Neon::Execution::device, setIdx),
                    refFactorSet.rawMem(gpuID, Neon::DeviceType::CUDA),
                    spacingSet.rawMem(gpuID, Neon::DeviceType::CUDA));
        }
        //}
    }
}


template <typename T, int C>
auto mField<T, C>::forEachActiveCell(
    const int                                           level,
    const std::function<void(const Neon::index_3d&,
                             const int& cardinality,
                             T&)>&                      fun,
    bool                                                filterOverlaps,
    [[maybe_unused]] Neon::computeMode_t::computeMode_e mode)
    -> void
{
    auto      desc = mData->grid->getDescriptor();
    auto      card = (*this)(0).getCardinality();
    const int refFactor = desc.getRefFactor(level);
    const int voxelSpacing = desc.getSpacing(level - 1);

    // TODO need to figure out which device owns this block
    SetIdx devID(0);

    (*(mData->grid))(level).helpGetPartitioner1D().forEachSeq(devID, [&](const uint32_t blockIdx, const Neon::int32_3d memBlockOrigin, auto /*byPartition*/) {
        Neon::index_3d blockOrigin = memBlockOrigin;
        blockOrigin.x *= kMemBlockSizeX * voxelSpacing;
        blockOrigin.y *= kMemBlockSizeY * voxelSpacing;
        blockOrigin.z *= kMemBlockSizeZ * voxelSpacing;

        for (uint32_t k = 0; k < kNumUserBlockPerMemBlockZ; ++k) {
            for (uint32_t j = 0; j < kNumUserBlockPerMemBlockY; ++j) {
                for (uint32_t i = 0; i < kNumUserBlockPerMemBlockX; ++i) {

                    const Neon::index_3d userBlockOrigin(i * kUserBlockSizeX * voxelSpacing + blockOrigin.x,
                                                         j * kUserBlockSizeY * voxelSpacing + blockOrigin.y,
                                                         k * kUserBlockSizeZ * voxelSpacing + blockOrigin.z);

                    for (int32_t z = 0; z < refFactor; z++) {
                        for (int32_t y = 0; y < refFactor; y++) {
                            for (int32_t x = 0; x < refFactor; x++) {

                                const Neon::index_3d voxelGlobalID(x * voxelSpacing + userBlockOrigin.x,
                                                                   y * voxelSpacing + userBlockOrigin.y,
                                                                   z * voxelSpacing + userBlockOrigin.z);

                                if ((*(mData->grid))(level).isInsideDomain(voxelGlobalID)) {

                                    bool active = true;
                                    if (level > 0 && filterOverlaps) {
                                        active = !((*(mData->grid))(level - 1).isInsideDomain(voxelGlobalID));
                                    }

                                    if (active) {
                                        Idx idx(blockIdx, int8_t(i * kUserBlockSizeX + x), int8_t(j * kUserBlockSizeY + y), int8_t(k * kUserBlockSizeZ + z));
                                        for (int c = 0; c < card; ++c) {
                                            fun(voxelGlobalID, c, (*this)(level).getPartition(Neon::Execution::host, devID, Neon::DataView::STANDARD)(idx, c));
                                        }
                                    }
                                }
                            }
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

    return mData->fields[level].getReference(idx, cardinality);
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
auto mField<T, C>::operator()(const Neon::index_3d& idx,
                              const int&            cardinality,
                              const int             level) -> T&
{
    return getReference(idx, cardinality, level);
}


template <typename T, int C>
auto mField<T, C>::operator()(const Neon::index_3d& idx,
                              const int&            cardinality,
                              const int             level) const -> const T&
{
    return getReference(idx, cardinality, level);
}

template <typename T, int C>
auto mField<T, C>::getReference(const Neon::index_3d& idx,
                                const int&            cardinality,
                                const int             level) -> T&
{
    return mData->fields[level].getReference()(idx, cardinality);
}

template <typename T, int C>
auto mField<T, C>::getReference(const Neon::index_3d& idx,
                                const int&            cardinality,
                                const int             level) const -> const T&
{
    return mData->fields[level].getReference(idx, cardinality);
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
auto mField<T, C>::updateHostData(int streamId) -> void
{

    for (size_t l = 0; l < mData->fields.size(); ++l) {
        mData->fields[l].mData->field.updateHostData(streamId);
    }
}

template <typename T, int C>
auto mField<T, C>::updateDeviceData(int streamId) -> void
{
    for (size_t l = 0; l < mData->fields.size(); ++l) {
        mData->fields[l].mData->field.updateDeviceData(streamId);
    }
}

template <typename T, int C>
auto mField<T, C>::load(Neon::set::Loader     loader,
                        int                   level,
                        Neon::MultiResCompute compute) -> typename xField<T, C>::Partition&
{
    switch (compute) {
        case Neon::MultiResCompute::MAP: {
            return loader.load(operator()(level), Neon::Pattern::MAP);
            break;
        }
        case Neon::MultiResCompute::STENCIL: {
            return loader.load(operator()(level), Neon::Pattern::STENCIL);
            break;
        }
        case Neon::MultiResCompute::STENCIL_UP: {
            const auto& parent = operator()(level + 1);
            loader.load(parent, Neon::Pattern::STENCIL);
            return loader.load(operator()(level), Neon::Pattern::MAP);
            break;
        }
        case Neon::MultiResCompute::STENCIL_DOWN: {
            const auto& child = operator()(level - 1);
            loader.load(child, Neon::Pattern::STENCIL);
            return loader.load(operator()(level), Neon::Pattern::MAP);
            break;
        }
        default:
            break;
    }
}


template <typename T, int C>
auto mField<T, C>::load(Neon::set::Loader     loader,
                        int                   level,
                        Neon::MultiResCompute compute) const -> const typename xField<T, C>::Partition&
{
    switch (compute) {
        case Neon::MultiResCompute::MAP: {
            return loader.load(operator()(level), Neon::Pattern::MAP);
            break;
        }
        case Neon::MultiResCompute::STENCIL: {
            return loader.load(operator()(level), Neon::Pattern::STENCIL);
            break;
        }
        case Neon::MultiResCompute::STENCIL_UP: {
            const auto& parent = operator()(level + 1);
            loader.load(parent, Neon::Pattern::STENCIL);
            return loader.load(operator()(level), Neon::Pattern::MAP);
            break;
        }
        case Neon::MultiResCompute::STENCIL_DOWN: {
            const auto& child = operator()(level - 1);
            loader.load(child, Neon::Pattern::STENCIL);
            return loader.load(operator()(level), Neon::Pattern::MAP);
            break;
        }
        default:
            break;
    }
}


template <typename T, int C>
auto mField<T, C>::ioToVtk(std::string fileName,
                           bool        outputLevels,
                           bool        outputBlockID,
                           bool        outputVoxelID,
                           bool        filterOverlaps) const -> void
{
    auto l0Dim = mData->grid->getDimension(0);

    std::ofstream file(fileName + ".vtk");
    file << "# vtk DataFile Version 2.0\n";
    file << "mGrid\n";
    file << "ASCII\n";
    file << "DATASET UNSTRUCTURED_GRID\n";
    file << "POINTS " << (l0Dim.rMax() + 1) * (l0Dim.rMax() + 1) * (l0Dim.rMax() + 1) << " float \n";
    for (int z = 0; z < l0Dim.rMax() + 1; ++z) {
        for (int y = 0; y < l0Dim.rMax() + 1; ++y) {
            for (int x = 0; x < l0Dim.rMax() + 1; ++x) {
                file << x << " " << y << " " << z << "\n";
            }
        }
    }

    uint64_t num_cells = 0;

    auto mapTo1D = [&](int x, int y, int z) {
        return x +
               y * (l0Dim.rMax() + 1) +
               z * (l0Dim.rMax() + 1) * (l0Dim.rMax() + 1);
    };

    enum class Op : int
    {
        Count = 0,
        OutputTopology = 1,
        OutputLevels = 2,
        OutputBlockID = 3,
        OutputVoxelID = 4,
        OutputData = 5,
    };

    auto desc = mData->grid->getDescriptor();
    auto card = (*this)(0).getCardinality();

    auto loopOverActiveBlocks = [&](const Op op) {
        for (int l = 0; l < desc.getDepth(); ++l) {
            const int refFactor = desc.getRefFactor(l);
            const int voxelSpacing = desc.getSpacing(l - 1);

            // TODO need to figure out which device owns this block
            SetIdx devID(0);

            (*(mData->grid))(l).helpGetPartitioner1D().forEachSeq(devID, [&](const uint32_t blockIdx, const Neon::int32_3d memBlockOrigin, auto /*byPartition*/) {
                Neon::index_3d blockOrigin = memBlockOrigin;
                blockOrigin.x *= kMemBlockSizeX * voxelSpacing;
                blockOrigin.y *= kMemBlockSizeY * voxelSpacing;
                blockOrigin.z *= kMemBlockSizeZ * voxelSpacing;

                for (uint32_t k = 0; k < kNumUserBlockPerMemBlockZ; ++k) {
                    for (uint32_t j = 0; j < kNumUserBlockPerMemBlockY; ++j) {
                        for (uint32_t i = 0; i < kNumUserBlockPerMemBlockX; ++i) {

                            const Neon::index_3d userBlockOrigin(i * kUserBlockSizeX * voxelSpacing + blockOrigin.x,
                                                                 j * kUserBlockSizeY * voxelSpacing + blockOrigin.y,
                                                                 k * kUserBlockSizeZ * voxelSpacing + blockOrigin.z);

                            for (int32_t z = 0; z < refFactor; z++) {
                                for (int32_t y = 0; y < refFactor; y++) {
                                    for (int32_t x = 0; x < refFactor; x++) {

                                        const Neon::index_3d voxelGlobalID(x * voxelSpacing + userBlockOrigin.x,
                                                                           y * voxelSpacing + userBlockOrigin.y,
                                                                           z * voxelSpacing + userBlockOrigin.z);

                                        if ((*(mData->grid))(l).isInsideDomain(voxelGlobalID)) {

                                            bool draw = true;
                                            if (filterOverlaps && l != 0) {
                                                draw = !((*(mData->grid))(l - 1).isInsideDomain(voxelGlobalID));
                                            }

                                            if (draw) {
                                                if (op == Op::Count) {
                                                    num_cells++;
                                                } else if (op == Op::OutputTopology) {

                                                    file << "8 ";
                                                    //x,y,z
                                                    file << mapTo1D(voxelGlobalID.x, voxelGlobalID.y, voxelGlobalID.z) << " ";
                                                    //+x,y,z
                                                    file << mapTo1D(voxelGlobalID.x + voxelSpacing, voxelGlobalID.y, voxelGlobalID.z) << " ";

                                                    //x,+y,z
                                                    file << mapTo1D(voxelGlobalID.x, voxelGlobalID.y + voxelSpacing, voxelGlobalID.z) << " ";

                                                    //+x,+y,z
                                                    file << mapTo1D(voxelGlobalID.x + voxelSpacing, voxelGlobalID.y + voxelSpacing, voxelGlobalID.z) << " ";

                                                    //x,y,+z
                                                    file << mapTo1D(voxelGlobalID.x, voxelGlobalID.y, voxelGlobalID.z + voxelSpacing) << " ";

                                                    //+x,y,+z
                                                    file << mapTo1D(voxelGlobalID.x + voxelSpacing, voxelGlobalID.y, voxelGlobalID.z + voxelSpacing) << " ";

                                                    //x,+y,+z
                                                    file << mapTo1D(voxelGlobalID.x, voxelGlobalID.y + voxelSpacing, voxelGlobalID.z + voxelSpacing) << " ";

                                                    //+x,+y,+z
                                                    file << mapTo1D(voxelGlobalID.x + voxelSpacing, voxelGlobalID.y + voxelSpacing, voxelGlobalID.z + voxelSpacing) << " ";
                                                    file << "\n";
                                                } else if (op == Op::OutputLevels) {
                                                    file << l << "\n";
                                                } else if (op == Op::OutputBlockID) {
                                                    file << blockIdx << "\n";
                                                } else if (op == Op::OutputVoxelID) {
                                                    file << x + y * refFactor + z * refFactor * refFactor
                                                         << "\n";
                                                } else if (op == Op::OutputData) {
                                                    Idx idx(blockIdx, int8_t(i * kUserBlockSizeX + x), int8_t(j * kUserBlockSizeY + y), int8_t(k * kUserBlockSizeZ + z));
                                                    for (int c = 0; c < card; ++c) {
                                                        file << (*this)(l).getPartition(Neon::Execution::host, devID, Neon::DataView::STANDARD)(idx, c) << "\n";
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

    //data
    file << "SCALARS " << (*this)(0).getName() << " float " << card << "\n";
    file << "LOOKUP_TABLE default \n";
    loopOverActiveBlocks(Op::OutputData);

    if (outputLevels) {
        file << "SCALARS Level int 1 \n";
        file << "LOOKUP_TABLE default \n";
        loopOverActiveBlocks(Op::OutputLevels);
    }

    if (outputBlockID) {
        file << "SCALARS BlockID int 1 \n";
        file << "LOOKUP_TABLE default \n";
        loopOverActiveBlocks(Op::OutputBlockID);
    }

    if (outputVoxelID) {
        file << "SCALARS VoxelID int 1 \n";
        file << "LOOKUP_TABLE default \n";
        loopOverActiveBlocks(Op::OutputVoxelID);
    }


    file.close();
}

template <typename T, int C>
auto mField<T, C>::getSharedMemoryBytes(const int32_t stencilRadius, int level) const -> size_t
{
    //This return the optimal shared memory size give a stencil radius
    //i.e., only N layers is read from neighbor blocks into shared memory in addition
    // to the block itself where N = stencilRadius
    int refFactor = mData->grid->getDescriptor().getRefFactor(level);
    return sizeof(T) *
           mData->fields[level].getCardinality() *
           (refFactor + 2 * stencilRadius) *
           (refFactor + 2 * stencilRadius) *
           (refFactor + 2 * stencilRadius);
}

}  // namespace Neon::domain::details::mGrid
