#pragma once

#include "Neon/domain/details/bGrid/bField.h"

namespace Neon::domain::details::bGrid {

template <typename T, int C>
bField<T, C>::bField()
{
    mData = std::make_shared<Data>();
}

template <typename T, int C>
bField<T, C>::bField(const std::string&         fieldUserName,
                     Neon::DataUse              dataUse,
                     const Neon::MemoryOptions& memoryOptions,
                     const Grid&                grid,
                     int                        cardinality,
                     T                          inactiveValue)
    : Neon::domain::interface::FieldBaseTemplate<T, C, Grid, Partition, int>(&grid,
                                                                             fieldUserName,
                                                                             "bField",
                                                                             cardinality,
                                                                             inactiveValue,
                                                                             dataUse,
                                                                             memoryOptions,
                                                                             Neon::domain::haloStatus_et::e::ON) {
    mData = std::make_shared<Data>(grid.getBackend());
    mData->grid = std::make_shared<Grid>(grid);


    int const blockSize = mData->grid->helpGetDataBlockSize();

    // the allocation size is the number of blocks x block size x cardinality
    mData->memoryField = mData->grid->helpGetBlockViewGrid().template newField<T, 0>(
        "BitMask",
        [&] {
            int elPerBlock = blockSize * blockSize * blockSize;
            elPerBlock = elPerBlock * cardinality;
            return elPerBlock;
        }(),
        0,
        dataUse,
        mData->grid->getBackend().getMemoryOptions(bSpan::activeMaskMemoryLayout));


    {  // Setting up partitionTable
        // const int setCardinality = mData->grid->getBackend().getDeviceCount();
        mData->partitionTable.forEachConfiguration(
            [&](Neon::Execution execution,
                Neon::SetIdx    setIdx,
                Neon::DataView,
                Partition& partition) {
                auto& memoryFieldPartition = mData->memoryField.getPartition(execution, setIdx, Neon::DataView::STANDARD);
                auto& blockConnectivity = mData->grid->helpGetBlockConnectivity().getPartition(execution, setIdx, Neon::DataView::STANDARD);
                auto& bitmask = mData->grid->helpGetActiveBitMask().getPartition(execution, setIdx, Neon::DataView::STANDARD);
                auto& dataBlockOrigins = mData->grid->helpGetDataBlockOriginField().getPartition(execution, setIdx, Neon::DataView::STANDARD);

                partition = bPartition<T, C>(setIdx,
                                             cardinality,
                                             memoryFieldPartition.mem(),
                                             grid.helpGetDataBlockSize(),
                                             blockConnectivity.mem(),
                                             bitmask.mem(),
                                             dataBlockOrigins.mem(),
                                             mData->grid->helpGetStencilIdTo3dOffset().rawMem(execution, setIdx));
            });
    }

    initHaloUpdateTable();
}

template <typename T, int C>
auto bField<T, C>::isInsideDomain(const Neon::index_3d& idx) const -> bool
{
    return mData->grid->isInsideDomain(idx);
}

template <typename T, int C>
auto bField<T, C>::getRef(const Neon::index_3d& /*idx*/,
                          const int&            /*cardinality*/) const -> T&
{
    //    // TODO need to figure out which device owns this block
    //    SetIdx devID(0);
    //
    //    if (!isInsideDomain(idx)) {
    //        return this->getOutsideValue();
    //    }
    //
    //    auto partition = getPartition(Neon::DeviceType::CPU, devID, Neon::DataView::STANDARD);
    //
    //    Neon::int32_3d blockOrigin = mData->grid->getOriginBlock3DIndex(idx);
    //
    //    auto itr = mData->grid->getBlockOriginTo1D().getMetadata(blockOrigin);
    //
    //    auto blockSize = mData->grid->getBlockSize();
    //
    //    Idx cell(static_cast<Idx::Location::Integer>((idx.x / mData->grid->getVoxelSpacing()) % blockSize),
    //             static_cast<Idx::Location::Integer>((idx.y / mData->grid->getVoxelSpacing()) % blockSize),
    //             static_cast<Idx::Location::Integer>((idx.z / mData->grid->getVoxelSpacing()) % blockSize));
    //
    //    cell.mBlockID = *itr;
    //    cell.mBlockSize = blockSize;
    //    return partition(cell, cardinality);
    NEON_DEV_UNDER_CONSTRUCTION("");
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
auto bField<T, C>::updateHostData(int streamId) -> void
{
    mData->memoryField.updateHostData(streamId);
}

template <typename T, int C>
auto bField<T, C>::updateDeviceData(int streamId) -> void
{
    mData->memoryField.updateDeviceData(streamId);
}

template <typename T, int C>
auto bField<T, C>::getPartition(Neon::Execution       execution,
                                Neon::SetIdx          setIdx,
                                const Neon::DataView& dataView) const -> const Partition&
{
    const Neon::DataUse dataUse = this->getDataUse();
    bool                isOk = Neon::ExecutionUtils::checkCompatibility(dataUse, execution);
    if (isOk) {
        Partition const& result = mData->partitionTable.getPartition(execution, setIdx, dataView);
        return result;
    }
    std::stringstream message;
    message << "The requested execution mode ( " << execution << " ) is not compatible with the field DataUse (" << dataUse << ")";
    NEON_THROW_UNSUPPORTED_OPERATION(message.str());
}

template <typename T, int C>
auto bField<T, C>::getPartition(Neon::Execution       execution,
                                Neon::SetIdx          setIdx,
                                const Neon::DataView& dataView) -> Partition&
{
    const Neon::DataUse dataUse = this->getDataUse();
    bool                isOk = Neon::ExecutionUtils::checkCompatibility(dataUse, execution);
    if (isOk) {
        Partition& result = mData->partitionTable.getPartition(execution, setIdx, dataView);
        return result;
    }
    std::stringstream message;
    message << "The requested execution mode ( " << execution << " ) is not compatible with the field DataUse (" << dataUse << ")";
    NEON_THROW_UNSUPPORTED_OPERATION(message.str());
}


template <typename T, int C>
auto bField<T, C>::initHaloUpdateTable() -> void
{
    NEON_THROW_UNSUPPORTED_OPERATION("");
#if 0
    auto& grid = this->getGrid();
    auto  bk = grid.getBackend();
    auto  getNghSetIdx = [&](SetIdx setIdx, Neon::domain::tool::partitioning::ByDirection direction) {
        int res;
        if (direction == Neon::domain::tool::partitioning::ByDirection::up) {
            res = (setIdx + 1) % bk.getDeviceCount();
        } else {
            res = (setIdx + bk.getDeviceCount() - 1) % bk.getDeviceCount();
        }
        return res;
    };

    mData->soaHaloUpdateTable.forEachPutConfiguration(
        bk, [&](Neon::SetIdx                                  setIdxSrc,
                Execution                                     execution,
                Neon::domain::tool::partitioning::ByDirection byDirection,
                std::vector<Neon::set::MemoryTransfer>&       transfersVec) {
            {
                using namespace Neon::domain::tool::partitioning;

                Neon::SetIdx setIdxDst = getNghSetIdx(setIdxSrc, byDirection);
                Neon::SetIdx setIdxVec[2];
                setIdxVec[Data::EndPoints::dst] = setIdxDst;
                setIdxVec[Data::EndPoints::src] = setIdxSrc;

                std::array<Partition*, Data::EndPointsUtils::nConfigs>                                  partitions;
                std::array<std::array<int, ByDirectionUtils::nConfigs>, Data::EndPointsUtils::nConfigs> ghostZBeginIdx;
                std::array<std::array<int, ByDirectionUtils::nConfigs>, Data::EndPointsUtils::nConfigs> boundaryZBeginIdx;
                std::array<Neon::size_4d, Data::EndPointsUtils::nConfigs>                               memPhyDim;

                partitions[Data::EndPoints::dst] = &this->getPartition(execution, setIdxDst, Neon::DataView::STANDARD);
                partitions[Data::EndPoints::src] = &this->getPartition(execution, setIdxSrc, Neon::DataView::STANDARD);

                for (auto endPoint : {Data::EndPoints::dst, Data::EndPoints::src}) {
                    for (auto direction : {ByDirection::down, ByDirection::up}) {
                        ghostZBeginIdx[endPoint][static_cast<int>(direction)] = mData->grid->mData->partitioner1D.getSpanLayout().getGhostBoundary(setIdxVec[endPoint], direction).first;
                        boundaryZBeginIdx[endPoint][static_cast<int>(direction)] = mData->grid->mData->partitioner1D.getSpanLayout().getBoundsBoundary(setIdxVec[endPoint], direction).first;
                    }

                    memPhyDim[endPoint] = Neon::size_4d(
                        1,
                        1,
                        1,
                        size_t(partitions[endPoint]->mCountAllocated));
                }

                if (ByDirection::up == byDirection && bk.isLastDevice(setIdxSrc)) {
                    return;
                }

                if (ByDirection::down == byDirection && bk.isFirstDevice(setIdxSrc)) {
                    return;
                }

                for (int j = 0; j < this->getCardinality(); j++) {

                    T* srcMem = partitions[Data::EndPoints::src]->mem();
                    T* dstMem = partitions[Data::EndPoints::dst]->mem();

                    Neon::size_4d srcBoundaryBuff(boundaryZBeginIdx[Data::EndPoints::src][static_cast<int>(byDirection)], 0, 0, j);
                    Neon::size_4d dstGhostBuff(ghostZBeginIdx[Data::EndPoints::dst][static_cast<int>(ByDirectionUtils::invert(byDirection))], 0, 0, j);
                    size_t        transferElementCount = mData->grid->mData->partitioner1D.getSpanLayout().getBoundsBoundary(setIdxVec[Data::EndPoints::src], byDirection).count;

                    //                    std::cout << "To  " << dstGhostBuff << " prt " << partitions[Data::EndPoints::dst]->prtID() << " From  " << srcBoundaryBuff  << std::endl;
                    //                    std::cout << "dst mem " << partitions[Data::EndPoints::dst]->mem() << " " << std::endl;
                    //                    std::cout << "dst transferElementCount " << transferElementCount << " " << std::endl;
                    //                    std::cout << "dst pitch " << (dstGhostBuff * memPhyDim[Data::EndPoints::dst]).rSum() << " " << std::endl;
                    //                    std::cout << "dst dstGhostBuff " << dstGhostBuff << " " << std::endl;
                    //                    std::cout << "dst pitch all" << memPhyDim[Data::EndPoints::dst] << " " << std::endl;

                    Neon::set::MemoryTransfer transfer({setIdxDst, dstMem + (dstGhostBuff * memPhyDim[Data::EndPoints::dst]).rSum(), dstGhostBuff},
                                                       {setIdxSrc, srcMem + (srcBoundaryBuff * memPhyDim[Data::EndPoints::src]).rSum(), srcBoundaryBuff},
                                                       sizeof(T) *
                                                           transferElementCount);


                    //                    std::cout << transfer.toString() << std::endl;
                    transfersVec.push_back(transfer);
                }
            }
        });

    mData->aosHaloUpdateTable.forEachPutConfiguration(
        bk, [&](Neon::SetIdx                                  setIdxSrc,
                Execution                                     execution,
                Neon::domain::tool::partitioning::ByDirection byDirection,
                std::vector<Neon::set::MemoryTransfer>&       transfersVec) {
            {
                using namespace Neon::domain::tool::partitioning;

                Neon::SetIdx setIdxDst = getNghSetIdx(setIdxSrc, byDirection);
                Neon::SetIdx setIdxVec[2];
                setIdxVec[Data::EndPoints::dst] = setIdxDst;
                setIdxVec[Data::EndPoints::src] = setIdxSrc;

                std::array<Partition*, Data::EndPointsUtils::nConfigs>                                  partitions;
                std::array<std::array<int, ByDirectionUtils::nConfigs>, Data::EndPointsUtils::nConfigs> ghostZBeginIdx;
                std::array<std::array<int, ByDirectionUtils::nConfigs>, Data::EndPointsUtils::nConfigs> boundaryZBeginIdx;
                std::array<Neon::size_4d, Data::EndPointsUtils::nConfigs>                               memPhyDim;

                partitions[Data::EndPoints::dst] = &this->getPartition(execution, setIdxDst, Neon::DataView::STANDARD);
                partitions[Data::EndPoints::src] = &this->getPartition(execution, setIdxSrc, Neon::DataView::STANDARD);

                for (auto endPoint : {Data::EndPoints::dst, Data::EndPoints::src}) {
                    for (auto direction : {ByDirection::down, ByDirection::up}) {
                        ghostZBeginIdx[endPoint][static_cast<int>(direction)] = mData->grid->mData->partitioner1D.getSpanLayout().getGhostBoundary(setIdxVec[endPoint], direction).first;
                        boundaryZBeginIdx[endPoint][static_cast<int>(direction)] = mData->grid->mData->partitioner1D.getSpanLayout().getBoundsBoundary(setIdxVec[endPoint], direction).first;
                    }

                    memPhyDim[endPoint] = Neon::size_4d(
                        this->getCardinality(),
                        1,
                        1,
                        1);
                }


                T* srcMem = partitions[Data::EndPoints::src]->mem();
                T* dstMem = partitions[Data::EndPoints::dst]->mem();

                Neon::size_4d srcBoundaryBuff(boundaryZBeginIdx[Data::EndPoints::src][static_cast<int>(byDirection)], 0, 0, 0);
                Neon::size_4d dstGhostBuff(ghostZBeginIdx[Data::EndPoints::dst][static_cast<int>(ByDirectionUtils::invert(byDirection))], 0, 0, 0);
                size_t        transferElementCount = mData->grid->mData->partitioner1D.getSpanLayout().getBoundsBoundary(setIdxVec[Data::EndPoints::src], byDirection).count;

                //                    std::cout << "To  " << dstGhostBuff << " prt " << partitions[Data::EndPoints::dst]->prtID() << " From  " << srcBoundaryBuff << "(src dim" << partitions[Data::EndPoints::src]->dim() << ")" << std::endl;
                //                    std::cout << "dst mem " << partitions[Data::EndPoints::dst]->mem() << " " << std::endl;
                //                    std::cout << "dst pitch " << (dstGhostBuff * memPhyDim[Data::EndPoints::dst]).rSum() << " " << std::endl;
                //                    std::cout << "dst dstGhostBuff " << dstGhostBuff << " " << std::endl;
                //                    std::cout << "dst pitch all" << memPhyDim[Data::EndPoints::dst] << " " << std::endl;

                Neon::set::MemoryTransfer transfer({setIdxDst, dstMem + (dstGhostBuff * memPhyDim[Data::EndPoints::dst]).rSum(), dstGhostBuff},
                                                   {setIdxSrc, srcMem + (srcBoundaryBuff * memPhyDim[Data::EndPoints::src]).rSum(), srcBoundaryBuff},
                                                   sizeof(T) *
                                                       transferElementCount * this->getCardinality());
                if (ByDirection::up == byDirection && bk.isLastDevice(setIdxSrc)) {
                    return;
                }

                if (ByDirection::down == byDirection && bk.isFirstDevice(setIdxSrc)) {
                    return;
                }

                // std::cout << transfer.toString() << std::endl;
                transfersVec.push_back(transfer);
            }
        });
    //
    //    mData->latticeHaloUpdateTable.forEachPutConfiguration(
    //        bk, [&](Neon::SetIdx                                  setIdxSrc,
    //                Execution                                     execution,
    //                Neon::domain::tool::partitioning::ByDirection byDirection,
    //                std::vector<Neon::set::MemoryTransfer>&       transfersVec) {
    //            {
    //                using namespace Neon::domain::tool::partitioning;
    //
    //                Neon::SetIdx setIdxDst = getNghSetIdx(setIdxSrc, byDirection);
    //                auto&        srcPartition = this->getPartition(execution, setIdxSrc, Neon::DataView::STANDARD);
    //                auto&        dstPartition = this->getPartition(execution, setIdxDst, Neon::DataView::STANDARD);
    //
    //                int r = grid.getStencil().getRadius();
    //
    //                int ghostZBeginIdx[2];
    //                int boundaryZBeginIdx[2];
    //
    //                ghostZBeginIdx[static_cast<int>(ByDirection::down)] = 0;
    //                ghostZBeginIdx[static_cast<int>(ByDirection::up)] = grid.getDimension().z + r;
    //
    //                boundaryZBeginIdx[static_cast<int>(ByDirection::down)] = r;
    //                boundaryZBeginIdx[static_cast<int>(ByDirection::up)] = grid.getDimension().z;
    //
    //                Neon::size_4d memPitch(1,
    //                                       grid.getDimension().x,
    //                                       grid.getDimension().x * grid.getDimension().y,
    //                                       grid.getDimension().x * grid.getDimension().y * (grid.getDimension().z + 2 * r));
    //
    //                for (int j = 0; j < this->getCardinality(); j++) {
    //
    //                    T* srcMem = srcPartition.mem();
    //                    T* dstMem = dstPartition.mem();
    //
    //                    Neon::size_4d srcBoundaryBuff(0, 0, boundaryZBeginIdx[static_cast<int>(byDirection)], j);
    //                    Neon::size_4d dstGhostBuff(0, 0, ghostZBeginIdx[static_cast<int>(byDirection)], j);
    //
    //                    Neon::set::MemoryTransfer transfer({setIdxDst, dstMem + dstGhostBuff.mPitch(memPitch)},
    //                                                       {setIdxSrc, srcMem + srcBoundaryBuff.mPitch(memPitch)},
    //                                                       grid.getDimension().x * grid.getDimension().y * sizeof(T));
    //
    //
    //                    transfersVec.push_back(transfer);
    //                }
    //            }
    //        });
#endif
}

}  // namespace Neon::domain::details::bGrid
