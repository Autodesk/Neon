#pragma once

#include "Neon/domain/details/bGrid/bField.h"

namespace Neon::domain::details::bGrid {

template <typename T, int C, int8_t dataBlockSizeX, int8_t dataBlockSizeY, int8_t dataBlockSizeZ>
bField<T, C, dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ>::bField()
{
    mData = std::make_shared<Data>();
}

template <typename T, int C, int8_t dataBlockSizeX, int8_t dataBlockSizeY, int8_t dataBlockSizeZ>
bField<T, C, dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ>::bField(const std::string&         fieldUserName,
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

    // the allocation size is the number of blocks x block size x cardinality
    mData->memoryField = mData->grid->helpGetBlockViewGrid().template newField<T, 0>(
        "BitMask",
        [&] {
            int elPerBlock = dataBlockSize3D.rMul();
            elPerBlock = elPerBlock * cardinality;
            return elPerBlock;
        }(),
        0,
        dataUse,
        mData->grid->getBackend().getMemoryOptions(bSpan<dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ>::activeMaskMemoryLayout));


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

                partition = bPartition<T, C, dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ>(setIdx,
                                                                                             cardinality,
                                                                                             memoryFieldPartition.mem(),
                                                                                             blockConnectivity.mem(),
                                                                                             bitmask.mem(),
                                                                                             dataBlockOrigins.mem(),
                                                                                             mData->grid->helpGetStencilIdTo3dOffset().rawMem(execution, setIdx));
            });
    }

    initHaloUpdateTable();
}

template <typename T, int C, int8_t dataBlockSizeX, int8_t dataBlockSizeY, int8_t dataBlockSizeZ>
auto bField<T, C, dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ>::isInsideDomain(const Neon::index_3d& idx) const -> bool
{
    return mData->grid->isInsideDomain(idx);
}

template <typename T, int C, int8_t dataBlockSizeX, int8_t dataBlockSizeY, int8_t dataBlockSizeZ>
auto bField<T, C, dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ>::getReference(const Neon::index_3d& cartesianIdx,
                                                                                const int&            cardinality) -> T&
{
    auto& grid = this->getGrid();
    auto [setIdx, bIdx] = grid.helpGetSetIdxAndGridIdx(cartesianIdx);
    auto& partition = getPartition(Neon::Execution::host, setIdx, Neon::DataView::STANDARD);
    auto& result = partition(bIdx, cardinality);
    return result;
}

template <typename T, int C, int8_t dataBlockSizeX, int8_t dataBlockSizeY, int8_t dataBlockSizeZ>
auto bField<T, C, dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ>::operator()(const Neon::index_3d& cartesianIdx,
                                                                              const int&            cardinality) const -> T
{
    auto& grid = this->getGrid();
    auto [setIdx, bIdx] = grid.helpGetSetIdxAndGridIdx(cartesianIdx);
    if (setIdx.idx() == -1) {
        return this->getOutsideValue();
    }
    auto& partition = getPartition(Neon::Execution::host, setIdx, Neon::DataView::STANDARD);
    auto& result = partition(bIdx, cardinality);
    return result;
}

template <typename T, int C, int8_t dataBlockSizeX, int8_t dataBlockSizeY, int8_t dataBlockSizeZ>
auto bField<T, C, dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ>::updateHostData(int streamId) -> void
{
    mData->memoryField.updateHostData(streamId);
}

template <typename T, int C, int8_t dataBlockSizeX, int8_t dataBlockSizeY, int8_t dataBlockSizeZ>
auto bField<T, C, dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ>::updateDeviceData(int streamId) -> void
{
    mData->memoryField.updateDeviceData(streamId);
}

template <typename T, int C, int8_t dataBlockSizeX, int8_t dataBlockSizeY, int8_t dataBlockSizeZ>
auto bField<T, C, dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ>::getPartition(Neon::Execution       execution,
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

template <typename T, int C, int8_t dataBlockSizeX, int8_t dataBlockSizeY, int8_t dataBlockSizeZ>
auto bField<T, C, dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ>::getPartition(Neon::Execution       execution,
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

template <typename T, int C, int8_t dataBlockSizeX, int8_t dataBlockSizeY, int8_t dataBlockSizeZ>
auto bField<T, C, dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ>::newHaloUpdate(Neon::set::StencilSemantic stencilSemantic,
                                                                                 Neon::set::TransferMode    transferMode,
                                                                                 Neon::Execution            execution) const -> Neon::set::Container
{


    // We need to define a graph of Containers
    // One for the actual memory transfer
    // One for the synchronization
    // The order depends on the transfer mode: put or get
    Neon::set::Container dataTransferContainer;
    auto const&          bk = this->getGrid().getBackend();

    if (stencilSemantic == Neon::set::StencilSemantic::standard) {
        auto transfers = bk.template newDataSet<std::vector<Neon::set::MemoryTransfer>>();

        if (this->getMemoryOptions().getOrder() == Neon::MemoryLayout::structOfArrays) {
            for (auto byDirection : {tool::partitioning::ByDirection::up,
                                     tool::partitioning::ByDirection::down}) {

                auto const& tableEntryByDir = mData->soaHaloUpdateTable.get(transferMode,
                                                                            execution,
                                                                            byDirection);

                tableEntryByDir.forEachSeq([&](SetIdx setIdx, auto const& tableEntryByDirBySetIdx) {
                    transfers[setIdx].insert(std::end(transfers[setIdx]),
                                             std::begin(tableEntryByDirBySetIdx),
                                             std::end(tableEntryByDirBySetIdx));
                });
            }
            dataTransferContainer =
                Neon::set::Container::factoryDataTransfer(
                    *this,
                    transferMode,
                    stencilSemantic,
                    transfers,
                    execution);


        } else {
            NEON_THROW_UNSUPPORTED_OPERATION("");
        }
    } else {
        NEON_DEV_UNDER_CONSTRUCTION("");
    }
    Neon::set::Container SyncContainer =
        Neon::set::Container::factorySynchronization(
            *this,
            Neon::set::SynchronizationContainerType::hostOmpBarrier);

    Neon::set::container::Graph graph(this->getBackend());
    const auto&                 dataTransferNode = graph.addNode(dataTransferContainer);
    const auto&                 syncNode = graph.addNode(SyncContainer);

    switch (transferMode) {
        case Neon::set::TransferMode::put:
            graph.addDependency(dataTransferNode, syncNode, Neon::GraphDependencyType::data);
            break;
        case Neon::set::TransferMode::get:
            graph.addDependency(syncNode, dataTransferNode, Neon::GraphDependencyType::data);
            break;
        default:
            NEON_THROW_UNSUPPORTED_OPTION();
            break;
    }

    graph.removeRedundantDependencies();

    Neon::set::Container output =
        Neon::set::Container::factoryGraph("dGrid-Halo-Update",
                                           graph,
                                           [](Neon::SetIdx, Neon::set::Loader&) {});
    return output;
}

template <typename T, int C, int8_t dataBlockSizeX, int8_t dataBlockSizeY, int8_t dataBlockSizeZ>
auto bField<T, C, dataBlockSizeX, dataBlockSizeY, dataBlockSizeZ>::initHaloUpdateTable() -> void
{
    // NEON_THROW_UNSUPPORTED_OPERATION("");
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
                std::array<BlockViewPartition<T, C>*, Data::EndPointsUtils::nConfigs>                   blockViewPartitions;
                std::array<std::array<int, ByDirectionUtils::nConfigs>, Data::EndPointsUtils::nConfigs> ghostZBeginIdx;
                std::array<std::array<int, ByDirectionUtils::nConfigs>, Data::EndPointsUtils::nConfigs> boundaryZBeginIdx;
                std::array<Neon::size_4d, Data::EndPointsUtils::nConfigs>                               memPhyDim;

                partitions[Data::EndPoints::dst] = &this->getPartition(execution, setIdxDst, Neon::DataView::STANDARD);
                partitions[Data::EndPoints::src] = &this->getPartition(execution, setIdxSrc, Neon::DataView::STANDARD);
                blockViewPartitions[Data::EndPoints::dst] = &(mData->memoryField.getPartition(execution, setIdxDst, Neon::DataView::STANDARD));
                blockViewPartitions[Data::EndPoints::src] = &(mData->memoryField.getPartition(execution, setIdxSrc, Neon::DataView::STANDARD));

                for (auto endPoint : {Data::EndPoints::dst, Data::EndPoints::src}) {
                    for (auto direction : {ByDirection::down, ByDirection::up}) {
                        auto ghostFirst = mData->grid->mData->partitioner1D.getSpanLayout().getGhostBoundary(setIdxVec[endPoint], direction).first;
                        auto boundaryFirst = mData->grid->mData->partitioner1D.getSpanLayout().getBoundsBoundary(setIdxVec[endPoint], direction).first;
                        ghostZBeginIdx[endPoint][static_cast<int>(direction)] = ghostFirst;
                        boundaryZBeginIdx[endPoint][static_cast<int>(direction)] = boundaryFirst;
                    }

                    memPhyDim[endPoint] = Neon::size_4d(
                        dataBlockSize3D.rMul(),
                        1,
                        1,
                        size_t(blockViewPartitions[endPoint]->getCountAllocated()) * dataBlockSize3D.rMul());
                }

                if (ByDirection::up == byDirection && bk.isLastDevice(setIdxSrc)) {
                    return;
                }

                if (ByDirection::down == byDirection && bk.isFirstDevice(setIdxSrc)) {
                    return;
                }

                T* srcMem = blockViewPartitions[Data::EndPoints::src]->mem();
                T* dstMem = blockViewPartitions[Data::EndPoints::dst]->mem();

                Neon::size_4d srcBoundaryBuff(boundaryZBeginIdx[Data::EndPoints::src][static_cast<int>(byDirection)], 0, 0, 0);
                Neon::size_4d dstGhostBuff(ghostZBeginIdx[Data::EndPoints::dst][static_cast<int>(ByDirectionUtils::invert(byDirection))], 0, 0, 0);
                size_t        transferDataBlockCount = mData->grid->mData->partitioner1D.getSpanLayout().getBoundsBoundary(setIdxVec[Data::EndPoints::src], byDirection).count;

                //                std::cout << "To  " << dstGhostBuff << " prt " << blockViewPartitions[Data::EndPoints::dst]->prtID() << " From  " << srcBoundaryBuff << " prt " << blockViewPartitions[Data::EndPoints::src]->prtID() <<  std::endl;
                //                std::cout << "dst mem " << blockViewPartitions[Data::EndPoints::dst]->mem() << " " << std::endl;
                //                std::cout << "dst transferDataBlockCount " << transferDataBlockCount << " " << std::endl;
                //                std::cout << "dst pitch " << (dstGhostBuff * memPhyDim[Data::EndPoints::dst]).rSum() << " " << std::endl;
                //                std::cout << "dst dstGhostBuff " << dstGhostBuff << " " << std::endl;
                //                std::cout << "dst pitch all" << memPhyDim[Data::EndPoints::dst] << " " << std::endl;

                Neon::set::MemoryTransfer transfer({setIdxDst, dstMem + (dstGhostBuff * memPhyDim[Data::EndPoints::dst]).rSum(), dstGhostBuff},
                                                   {setIdxSrc, srcMem + (srcBoundaryBuff * memPhyDim[Data::EndPoints::src]).rSum(), srcBoundaryBuff},
                                                   sizeof(T) * dataBlockSize3D.rMul() * transferDataBlockCount);


                //                    std::cout << transfer.toString() << std::endl;
                transfersVec.push_back(transfer);
            }
        });
}


}  // namespace Neon::domain::details::bGrid