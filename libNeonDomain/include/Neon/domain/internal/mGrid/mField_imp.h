#pragma once

#include "Neon/domain/internal/mGrid/mField.h"

namespace Neon::domain::internal::mGrid {

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

        auto mem = mData->fields[l].mData->field.getMem();

        auto origins = mData->grid->operator()(l).getOrigins();
        auto neighbours_blocks = mData->grid->operator()(l).getNeighbourBlocks();
        auto stencil_ngh = mData->grid->operator()(l).getStencilNghIndex();
        auto active_mask = mData->grid->operator()(l).getActiveMask();
        auto                            parent = mData->grid->getParentsBlockID(l);
        auto                            parentLocalID = mData->grid->getParentLocalID(l);
        auto                            childBlockID = mData->grid->getChildBlockID(l);


        for (int dvID = 0; dvID < Neon::DataViewUtil::nConfig; dvID++) {
            mData->fields[l].mData->mPartitions[PartitionBackend::cpu][dvID] = mData->grid->getBackend().devSet().template newDataSet<Partition>();
            mData->fields[l].mData->mPartitions[PartitionBackend::gpu][dvID] = mData->grid->getBackend().devSet().template newDataSet<Partition>();

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
                        (l == 0) ? nullptr : mData->grid->operator()(l - 1).getActiveMask().rawMem(gpuID, Neon::DeviceType::CPU),                               //lower-level mask
                        (l == int(descriptor.getDepth()) - 1) ? nullptr : mData->grid->operator()(l + 1).getActiveMask().rawMem(gpuID, Neon::DeviceType::CPU),  //upper-level mask
                        (l == 0) ? nullptr : childBlockID.rawMem(gpuID, Neon::DeviceType::CPU),
                        (l == int(descriptor.getDepth()) - 1) ? nullptr : mData->grid->operator()(l + 1).getNeighbourBlocks().rawMem(gpuID, Neon::DeviceType::CPU),  //parent neighbor
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
                        (l == 0) ? nullptr : mData->grid->operator()(l - 1).getActiveMask().rawMem(gpuID, Neon::DeviceType::CUDA),                               //lower-level mask
                        (l == int(descriptor.getDepth()) - 1) ? nullptr : mData->grid->operator()(l + 1).getActiveMask().rawMem(gpuID, Neon::DeviceType::CUDA),  //upper-level mask
                        (l == 0) ? nullptr : childBlockID.rawMem(gpuID, Neon::DeviceType::CUDA),
                        (l == int(descriptor.getDepth()) - 1) ? nullptr : mData->grid->operator()(l + 1).getNeighbourBlocks().rawMem(gpuID, Neon::DeviceType::CUDA),  //parent neighbor
                        outsideVal,
                        stencil_ngh.rawMem(gpuID, Neon::DeviceType::CUDA),
                        refFactorSet.rawMem(gpuID, Neon::DeviceType::CUDA),
                        spacingSet.rawMem(gpuID, Neon::DeviceType::CUDA));
            }
        }
    }
}


template <typename T, int C>
auto mField<T, C>::forEachActiveCell(
    const int                                           level,
    const std::function<void(const Neon::index_3d&,
                             const int& cardinality,
                             T&)>&                      fun,
    [[maybe_unused]] Neon::computeMode_t::computeMode_e mode)
    -> void
{
    //mData->fields[level].mData->field.forEachActiveCell(fun, mode);

    auto      desc = mData->grid->getDescriptor();
    auto      card = (*this)(0).getCardinality();
    const int refFactor = desc.getRefFactor(level);
    const int childSpacing = desc.getSpacing(level - 1);

    (*(mData->grid))(level).getBlockOriginTo1D().forEach([&](const Neon::int32_3d blockOrigin, const uint32_t blockIdx) {
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
                            (*(mData->grid))(level).getActiveMask().rawMem(devID, Neon::DeviceType::CPU))) {

                        Neon::int32_3d corner(blockOrigin.x + x * childSpacing,
                                              blockOrigin.y + y * childSpacing,
                                              blockOrigin.z + z * childSpacing);

                        bool active = true;
                        if (level != 0) {
                            auto cornerIDIter = (*(mData->grid))(level - 1).getBlockOriginTo1D().getMetadata(corner);
                            if (cornerIDIter) {
                                active = false;
                            }
                        }

                        if (active) {
                            for (int c = 0; c < card; ++c) {
                                fun(corner, c, (*this)(level).getPartition(Neon::Execution::host, devID, Neon::DataView::STANDARD)(cell, c));
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
auto mField<T, C>::updateIO(int streamId) -> void
{

    for (size_t l = 0; l < mData->fields.size(); ++l) {
        mData->fields[l].mData->field.updateIO(streamId);
    }
}

template <typename T, int C>
auto mField<T, C>::updateCompute(int streamId) -> void
{
    for (size_t l = 0; l < mData->fields.size(); ++l) {
        mData->fields[l].mData->field.updateCompute(streamId);
    }
}

template <typename T, int C>
auto mField<T, C>::load(Neon::set::Loader     loader,
                        int                   level,
                        Neon::MultiResCompute compute) -> typename xField<T, C>::Partition&
{
    switch (compute) {
        case Neon::MultiResCompute::MAP: {
            return loader.load(operator()(level), Neon::Compute::MAP);
            break;
        }
        case Neon::MultiResCompute::STENCIL: {
            return loader.load(operator()(level), Neon::Compute::STENCIL);
            break;
        }
        case Neon::MultiResCompute::STENCIL_UP: {
            const auto& parent = operator()(level + 1);
            loader.load(parent, Neon::Compute::STENCIL);
            return loader.load(operator()(level), Neon::Compute::MAP);
            break;
        }
        case Neon::MultiResCompute::STENCIL_DOWN: {
            const auto& child = operator()(level - 1);
            loader.load(child, Neon::Compute::STENCIL);
            return loader.load(operator()(level), Neon::Compute::MAP);
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
            return loader.load(operator()(level), Neon::Compute::MAP);
            break;
        }
        case Neon::MultiResCompute::STENCIL: {
            return loader.load(operator()(level), Neon::Compute::STENCIL);
            break;
        }
        case Neon::MultiResCompute::STENCIL_UP: {
            const auto& parent = operator()(level + 1);
            loader.load(parent, Neon::Compute::STENCIL);
            return loader.load(operator()(level), Neon::Compute::MAP);
            break;
        }
        case Neon::MultiResCompute::STENCIL_DOWN: {
            const auto& child = operator()(level - 1);
            loader.load(child, Neon::Compute::STENCIL);
            return loader.load(operator()(level), Neon::Compute::MAP);
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
            const int childSpacing = desc.getSpacing(l - 1);

            (*(mData->grid))(l).getBlockOriginTo1D().forEach([&](const Neon::int32_3d blockOrigin, const uint32_t blockIdx) {
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
                                    (*(mData->grid))(l).getActiveMask().rawMem(devID, Neon::DeviceType::CPU))) {

                                Neon::int32_3d corner(blockOrigin.x + x * childSpacing,
                                                      blockOrigin.y + y * childSpacing,
                                                      blockOrigin.z + z * childSpacing);

                                bool draw = true;
                                if (filterOverlaps && l != 0) {
                                    auto cornerIDIter = (*(mData->grid))(l - 1).getBlockOriginTo1D().getMetadata(corner);
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
                                        file << x + y * refFactor + z * refFactor * refFactor
                                             << "\n";
                                    } else if (op == Op::OutputData) {
                                        for (int c = 0; c < card; ++c) {
                                            file << (*this)(l).getPartition(Neon::Execution::host, devID, Neon::DataView::STANDARD)(cell, c) << "\n";
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

}  // namespace Neon::domain::internal::mGrid
