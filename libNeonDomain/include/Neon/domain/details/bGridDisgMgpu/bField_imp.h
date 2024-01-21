#pragma once

#include "Neon/domain/details/bGridDisgMgpu/bField.h"

namespace Neon::domain::details::bGridMgpu {

template <typename T, int C, typename SBlock>
bField<T, C, SBlock>::bField()
{
    mData = std::make_shared<Data>();
}

template <typename T, int C, typename SBlock>
bField<T, C, SBlock>::bField(const std::string&  fieldUserName,
                             Neon::DataUse       dataUse,
                             Neon::MemoryOptions memoryOptions,
                             const Grid&         grid,
                             int                 cardinality,
                             T                   inactiveValue)
    : Neon::domain::interface::FieldBaseTemplate<T, C, Grid, Partition, int>(&grid,
                                                                             fieldUserName,
                                                                             "bField",
                                                                             cardinality,
                                                                             inactiveValue,
                                                                             dataUse,
                                                                             memoryOptions,
                                                                             Neon::domain::haloStatus_et::e::ON)
{
    mData = std::make_shared<Data>(grid.getBackend());
    mData->grid = std::make_shared<Grid>(grid);

    if (memoryOptions.getOrder() == Neon::MemoryLayout::arrayOfStructs) {
        NEON_WARNING("bField does not support MemoryLayout::arrayOfStructs, enforcing MemoryLayout::structOfArrays");
        memoryOptions.setOrder(Neon::MemoryLayout::structOfArrays);
    }
    // the allocation size is the number of blocks x block size x cardinality
    mData->memoryField = mData->grid->getBlockViewGrid().template newField<T, 0>(
        "BitMask",
        [&] {
            int elPerBlock = SBlock::memBlockCountElements * cardinality;
            return elPerBlock;
        }(),
        inactiveValue,
        dataUse,
        mData->grid->getBackend().getMemoryOptions(bSpan<SBlock>::activeMaskMemoryLayout));


    {  // Setting up mPartitionTable
        // const int setCardinality = mData->grid->getBackend().getDeviceCount();
        mData->mPartitionTable.forEachConfiguration(
            [&](Neon::Execution execution,
                Neon::SetIdx    setIdx,
                Neon::DataView,
                Partition& partition) {
                auto& partitioner = mData->grid->helpGetPartitioner1D();
                auto  firstBup = partitioner.getSpanLayout().getBoundsBoundary(setIdx, Neon::domain::tool::partitioning::ByDirection::up).first;
                auto  firstBdw = partitioner.getSpanLayout().getBoundsBoundary(setIdx, Neon::domain::tool::partitioning::ByDirection::down).first;
                auto  firstGup = partitioner.getSpanLayout().getGhostBoundary(setIdx, Neon::domain::tool::partitioning::ByDirection::up).first;
                auto  firstGdw = partitioner.getSpanLayout().getGhostBoundary(setIdx, Neon::domain::tool::partitioning::ByDirection::down).first;
                auto  lastGdw = firstGdw + partitioner.getSpanLayout().getGhostBoundary(setIdx, Neon::domain::tool::partitioning::ByDirection::down).count;

                auto& memoryFieldPartition = mData->memoryField.getPartition(execution, setIdx, Neon::DataView::STANDARD);
                auto& blockConnectivity = mData->grid->helpGetBlockConnectivity().getPartition(execution, setIdx, Neon::DataView::STANDARD);
                auto& bitmask = mData->grid->getActiveBitMask().getPartition(execution, setIdx, Neon::DataView::STANDARD);
                auto& dataBlockOrigins = mData->grid->helpGetDataBlockOriginField().getPartition(execution, setIdx, Neon::DataView::STANDARD);

                partition = bPartition<T, C, SBlock>(setIdx,
                                                     cardinality,
                                                     memoryFieldPartition.mem(),
                                                     blockConnectivity.mem(),
                                                     bitmask.mem(),
                                                     dataBlockOrigins.mem(),
                                                     mData->grid->helpGetStencilIdTo3dOffset().rawMem(execution, setIdx),
                                                     mData->grid->getDimension(),
                                                     firstBup,
                                                     firstBdw,
                                                     firstGup,
                                                     firstGdw,
                                                     lastGdw);
            });
    }

    initHaloUpdateTable();
}

template <typename T, int C, typename SBlock>
auto bField<T, C, SBlock>::getMemoryField() -> BlockViewGrid::Field<T, C>&
{
    return mData->memoryField;
}

template <typename T, int C, typename SBlock>
auto bField<T, C, SBlock>::isInsideDomain(const Neon::index_3d& idx) const -> bool
{
    return mData->grid->isInsideDomain(idx);
}

template <typename T, int C, typename SBlock>
auto bField<T, C, SBlock>::getReference(const Neon::index_3d& cartesianIdx,
                                        const int&            cardinality) -> T&
{
    if constexpr (SBlock::isMultiResMode) {
        auto& grid = this->getGrid();
        auto  uniformCartesianIdx = cartesianIdx / grid.helGetMultiResDiscreteIdxSpacing();

        if (cartesianIdx.x % grid.helGetMultiResDiscreteIdxSpacing() != 0 ||
            cartesianIdx.y % grid.helGetMultiResDiscreteIdxSpacing() != 0 ||
            cartesianIdx.z % grid.helGetMultiResDiscreteIdxSpacing() != 0) {
            NeonException exp("bField::getReference");
            exp << "Input index is not multiple of the grid resolution";
            exp << "Index = " << cartesianIdx;
            NEON_THROW(exp);
        }
        auto [setIdx, bIdx] = grid.helpGetSetIdxAndGridIdx(uniformCartesianIdx);
        auto& partition = getPartition(Neon::Execution::host, setIdx, Neon::DataView::STANDARD);
        auto& result = partition(bIdx, cardinality);
        return result;
    } else {
        auto& grid = this->getGrid();
        auto [setIdx, bIdx] = grid.helpGetSetIdxAndGridIdx(cartesianIdx);
        auto& partition = getPartition(Neon::Execution::host, setIdx, Neon::DataView::STANDARD);
        auto& result = partition(bIdx, cardinality);
        return result;
    }
}

template <typename T, int C, typename SBlock>
auto bField<T, C, SBlock>::operator()(const Neon::index_3d& cartesianIdx,
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

template <typename T, int C, typename SBlock>
auto bField<T, C, SBlock>::updateHostData(int streamId) -> void
{
    mData->memoryField.updateHostData(streamId);
}

template <typename T, int C, typename SBlock>
auto bField<T, C, SBlock>::updateDeviceData(int streamId) -> void
{
    mData->memoryField.updateDeviceData(streamId);
}

template <typename T, int C, typename SBlock>
auto bField<T, C, SBlock>::getPartition(Neon::Execution       execution,
                                        Neon::SetIdx          setIdx,
                                        const Neon::DataView& dataView) const -> const Partition&
{
    const Neon::DataUse dataUse = this->getDataUse();
    bool                isOk = Neon::ExecutionUtils::checkCompatibility(dataUse, execution);
    if (isOk) {
        Partition const& result = mData->mPartitionTable.getPartition(execution, setIdx, dataView);
        return result;
    }
    std::stringstream message;
    message << "The requested execution mode ( " << execution << " ) is not compatible with the field DataUse (" << dataUse << ")";
    NEON_THROW_UNSUPPORTED_OPERATION(message.str());
}

template <typename T, int C, typename SBlock>
auto bField<T, C, SBlock>::getPartition(Neon::Execution       execution,
                                        Neon::SetIdx          setIdx,
                                        const Neon::DataView& dataView) -> Partition&
{
    const Neon::DataUse dataUse = this->getDataUse();
    bool                isOk = Neon::ExecutionUtils::checkCompatibility(dataUse, execution);
    if (isOk) {
        Partition& result = mData->mPartitionTable.getPartition(execution, setIdx, dataView);
        return result;
    }
    std::stringstream message;
    message << "The requested execution mode ( " << execution << " ) is not compatible with the field DataUse (" << dataUse << ")";
    NEON_THROW_UNSUPPORTED_OPERATION(message.str());
}

template <typename T, int C, typename SBlock>
auto bField<T, C, SBlock>::newHaloUpdate(Neon::set::StencilSemantic stencilSemantic,
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

                auto const& tableEntryByDir = mData->mStandardHaloUpdateTable.get(transferMode,
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

template <typename T, int C, typename SBlock>
auto bField<T, C, SBlock>::initHaloUpdateTable() -> void
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

    mData->mStandardHaloUpdateTable.forEachPutConfiguration(
        bk, [&](
                Neon::SetIdx                                  setIdxSend,
                Execution                                     execution,
                Neon::domain::tool::partitioning::ByDirection byDirection,
                std::vector<Neon::set::MemoryTransfer>&       transfersVec) {
            {
                using namespace Neon::domain::tool::partitioning;

                if (ByDirection::up == byDirection && bk.isLastDevice(setIdxSend)) {
                    return;
                }

                if (ByDirection::down == byDirection && bk.isFirstDevice(setIdxSend)) {
                    return;
                }


                Neon::SetIdx setIdxRecv = getNghSetIdx(setIdxSend, byDirection);

                Partition* partitionsRecv = &this->getPartition(execution, setIdxRecv, Neon::DataView::STANDARD);
                Partition* partitionsSend = &this->getPartition(execution, setIdxSend, Neon::DataView::STANDARD);

                auto const recvDirection = byDirection == ByDirection::up
                                               ? ByDirection::down
                                               : ByDirection::up;
                auto const sendDirection = byDirection;

                int const ghostSectorFirstBlockIdx =
                    recvDirection == ByDirection::up
                        ? partitionsRecv->helpGetSectorFirstBlock(Partition::Sectors::gUp)
                        : partitionsRecv->helpGetSectorFirstBlock(Partition::Sectors::gDw);

                int const boundarySectorFirstBlockIdx =
                    sendDirection == ByDirection::up
                        ? partitionsSend->helpGetSectorFirstBlock(Partition::Sectors::bUp)
                        : partitionsSend->helpGetSectorFirstBlock(Partition::Sectors::bDw);

                auto const msgLengthInBlocks = partitionsSend->helpGetSectorLength(sendDirection == ByDirection::up
                                                                                       ? Partition::Sectors::bUp
                                                                                       : Partition::Sectors::bDw);


                for (int c = 0; c < this->getCardinality(); c++) {

                    auto const recvPitch = [&] {
                        Idx                          idx;
                        typename Idx::InDataBlockIdx inDataBlockIdx;
                        if (recvDirection == ByDirection::up) {
                            inDataBlockIdx = typename Idx::InDataBlockIdx(0, 0, 0);
                        } else {
                            inDataBlockIdx = typename Idx::InDataBlockIdx(0, 0, SBlock::memBlockSizeZ - 1);
                        }
                        idx.setInDataBlockIdx(inDataBlockIdx);
                        idx.setDataBlockIdx(ghostSectorFirstBlockIdx);

                        auto pitch = partitionsRecv->helpGetPitch(idx, c);
                        return pitch;
                    }();

                    auto const sendPitch = [&] {
                        typename Idx::InDataBlockIdx inDataBlockIdx;
                        if (sendDirection == ByDirection::up) {
                            inDataBlockIdx = typename Idx::InDataBlockIdx(0, 0, SBlock::memBlockSizeZ - 1);
                        } else {
                            inDataBlockIdx = typename Idx::InDataBlockIdx(0, 0, 0);
                        }
                        Idx idx;
                        idx.setInDataBlockIdx(inDataBlockIdx);
                        idx.setDataBlockIdx(boundarySectorFirstBlockIdx);
                        auto pitch = partitionsSend->helpGetPitch(idx, c);
                        return pitch;
                    }();

                    auto const msgSizePerCardinality = [&] {
                        // All blocks are mapped into a 3D grid, where blocks are places one after the other in a 1D mapping
                        // for each block we send only the top or bottom slice
                        // Therefore the size of the message is equal to the number of blocks in the sector
                        // by the size of element in a slice of a block...
                        auto size = msgLengthInBlocks * SBlock::memBlockSizeX * SBlock::memBlockSizeY;
                        return size;
                    }();

                    T const* sendMem = partitionsSend->mem();
                    T const* recvMem = partitionsRecv->mem();


                    Neon::set::MemoryTransfer transfer({setIdxRecv, (void*)(recvMem + recvPitch)},
                                                       {setIdxSend, (void*)(sendMem + sendPitch)},
                                                       sizeof(T) * msgSizePerCardinality);

                    transfersVec.push_back(transfer);
                }
            }
        });
}


}  // namespace Neon::domain::details::bGridMgpu
