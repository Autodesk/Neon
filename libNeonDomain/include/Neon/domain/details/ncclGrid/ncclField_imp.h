#pragma once
#include "Neon/set/container/NcclTransferContainer.h"
#include "ncclField.h"

namespace Neon::domain::details::ncclGrid {

template <typename T, int C>
ncclField<T, C>::ncclField()
{
    mData = std::make_shared<Data>();
}

template <typename T, int C>
ncclField<T, C>::ncclField(const std::string&                         fieldUserName,
                           Neon::DataUse                              dataUse,
                           const Neon::MemoryOptions&                 memoryOptions,
                           const Grid&                                grid,
                           const Neon::set::RankData<Neon::index_3d>& dimsRank,
                           int                                        zHaloRadius,
                           Neon::domain::haloStatus_et::e             haloStatus,
                           int                                        cardinality,
                           Neon::set::MemSet<Neon::int8_3d>&          stencilIdTo3dOffset)
    : Neon::domain::interface::FieldBaseTemplate<T, C, Grid, Partition, int>(&grid,
                                                                             fieldUserName,
                                                                             "ncclField",
                                                                             cardinality,
                                                                             T(0),
                                                                             dataUse,
                                                                             memoryOptions,
                                                                             haloStatus)
{

    // only works if dims in x and y direction for all partitions match
    for (int i = 0; i < dimsRank.size() - 1; ++i) {
        for (int j = i + 1; j < dimsRank.size(); ++j) {
            if (dimsRank[i].x != dimsRank[j].x || dimsRank[i].y != dimsRank[j].y) {
                NeonException exc("ncclField_t");
                exc << "New ncclField only works on partitioning along z axis.";
                NEON_THROW(exc);
            }
        }
    }

    mData = std::make_shared<Data>(grid.getBackend());
    mData->dataUse = dataUse;
    mData->memoryOptions = memoryOptions;
    mData->cardinality = cardinality;
    mData->memoryOptions = memoryOptions;
    mData->grid = std::make_shared<Grid>(grid);
    mData->haloStatus = haloStatus_et::e::ON;
    const int haloRadius = mData->haloStatus == Neon::domain::haloStatus_et::ON ? zHaloRadius : 0;
    mData->zHaloDim = zHaloRadius;
    const auto myRank = mData->grid->getBackend().getNccl().getWorldRank();
    // const auto worldSize = mData->grid->getBackend().getNccl().getWorldSize();

    Neon::set::RankData<index_3d> originsRank = this->getGrid().getBackend().template newRankData<index_3d>({0, 0, 0});
    {  // Computing origins
        originsRank.forEachSeq(
            [&](int idxRank, Neon::index_3d& val) {
                if (idxRank == 0) {
                    val.z = 0;
                    return;
                }
                const auto proceedingRank = idxRank - 1;
                val.z = originsRank[proceedingRank].z + dimsRank[proceedingRank].z;
            });
    }

    {  // Computing Pitch
        mData->pitch.forEachSeq(
            [&]([[maybe_unused]] Neon::SetIdx setIdx, Neon::size_4d& pitch) {
                switch (mData->memoryOptions.getOrder()) {
                    case MemoryLayout::structOfArrays: {
                        pitch.x = 1;
                        pitch.y = pitch.x * dimsRank[myRank].x;
                        pitch.z = pitch.y * dimsRank[myRank].y;
                        pitch.w = pitch.z * (dimsRank[myRank].z + 2 * haloRadius);
                        break;
                    }
                    case MemoryLayout::arrayOfStructs: {
                        pitch.x = mData->cardinality;
                        pitch.y = pitch.x * dimsRank[myRank].x;
                        pitch.z = pitch.y * dimsRank[myRank].y;
                        pitch.w = 1;
                        break;
                    }
                }
            });
    }

    {  // Setting up partitions
        Neon::aGrid const& aGrid = mData->grid->helpFieldMemoryAllocator();
        mData->memoryField = aGrid.newField<T, C>(fieldUserName + "-storage", cardinality, T(), dataUse, memoryOptions);
        // const int setCardinality = mData->grid->getBackend().getDeviceCount();
        mData->partitionTable.forEachConfiguration(
            [&](Neon::Execution           execution,
                Neon::SetIdx              setIdx,
                Neon::DataView            dw,
                typename Self::Partition& partition) {
                auto memoryFielncclPartition = mData->memoryField.getPartition(execution, setIdx, Neon::DataView::STANDARD);

                partition = ncclPartition<T, C>(dw,
                                                memoryFielncclPartition.mem(),
                                                dimsRank[myRank],
                                                haloRadius,
                                                mData->zHaloDim,
                                                mData->pitch[setIdx],
                                                setIdx.idx(),
                                                originsRank[myRank],
                                                mData->cardinality,
                                                mData->grid->getDimension(),
                                                stencilIdTo3dOffset.rawMem(execution, setIdx));
            });
    }

    {  // Setting Reduction information
        mData->partitionTable.forEachConfigurationWithUserData(
            [&](Neon::Execution,
                [[maybe_unused]] Neon::SetIdx setIdx,
                Neon::DataView                dw,
                typename Self::Partition&,
                typename Data::ReductionInformation& reductionInfo) {
                switch (dw) {
                    case Neon::DataView::STANDARD: {
                        // old structure [dv_id][c][i]
                        if (grid.getBackend().devSet().numDevs() == 1) {
                            // As the number of devices is 1, we don't have halos.
                            reductionInfo.startIDByView.push_back(0);
                            reductionInfo.nElementsByView.push_back(int(dimsRank[myRank].rMul()));
                        } else {
                            switch (mData->memoryOptions.getOrder()) {
                                case MemoryLayout::structOfArrays: {
                                    for (int c = 0; c < mData->cardinality; ++c) {
                                        // To compute the start point we need to
                                        // jump the previous cardinalities -> c * dims[setIdx].x * dims[setIdx].y * (dims[setIdx].z + 2 * haloRadius)
                                        // jump one halo -> dims[setIdx].x * dims[setIdx].y * haloRadius
                                        int const startPoint = c * dimsRank[myRank].x * dimsRank[myRank].y * (dimsRank[myRank].z + 2 * haloRadius) +
                                                               dimsRank[myRank].x * dimsRank[myRank].y * haloRadius;
                                        int const nElements = dimsRank[myRank].rMul();

                                        reductionInfo.startIDByView.push_back(startPoint);
                                        reductionInfo.nElementsByView.push_back(nElements);
                                    }
                                    break;
                                }
                                case MemoryLayout::arrayOfStructs: {
                                    int const startPoint = dimsRank[myRank].x * dimsRank[myRank].y * haloRadius * mData->cardinality;
                                    int const nElements = dimsRank[myRank].x * dimsRank[myRank].y * dimsRank[myRank].z * mData->cardinality;

                                    reductionInfo.startIDByView.push_back(startPoint);
                                    reductionInfo.nElementsByView.push_back(nElements);

                                    break;
                                }
                            }
                        }
                        break;
                    }
                    case Neon::DataView::INTERNAL: {
                        if (grid.getBackend().devSet().numDevs() > 1) {
                            switch (mData->memoryOptions.getOrder()) {
                                case MemoryLayout::structOfArrays: {
                                    for (int c = 0; c < mData->cardinality; ++c) {

                                        auto const boundaryRadius = mData->zHaloDim;
                                        int const  startPoint = c * dimsRank[myRank].x * dimsRank[myRank].y * (dimsRank[myRank].z + 2 * haloRadius) +
                                                               dimsRank[myRank].x * dimsRank[myRank].y * (haloRadius + boundaryRadius);

                                        int const nElements = dimsRank[myRank].x * dimsRank[myRank].y * (dimsRank[myRank].z - 2 * haloRadius);

                                        reductionInfo.startIDByView.push_back(startPoint);
                                        reductionInfo.nElementsByView.push_back(nElements);
                                    }
                                    break;
                                }
                                case MemoryLayout::arrayOfStructs: {
                                    auto const boundaryRadius = mData->zHaloDim;
                                    int const  startPoint = dimsRank[myRank].x * dimsRank[myRank].y * (haloRadius + boundaryRadius) * mData->cardinality;
                                    int const  nElements = dimsRank[myRank].x * dimsRank[myRank].y * (dimsRank[myRank].z - 2 * haloRadius) * mData->cardinality;

                                    reductionInfo.startIDByView.push_back(startPoint);
                                    reductionInfo.nElementsByView.push_back(nElements);

                                    break;
                                }
                            }
                        }
                        break;
                    }
                    case Neon::DataView::BOUNDARY: {
                        if (grid.getBackend().devSet().numDevs() > 1) {
                            switch (mData->memoryOptions.getOrder()) {
                                case MemoryLayout::structOfArrays: {
                                    for (int c = 0; c < mData->cardinality; ++c) {
                                        {  // up
                                            auto const boundaryRadius = mData->zHaloDim;
                                            int const  startPoint = c * dimsRank[myRank].x * dimsRank[myRank].y * (dimsRank[myRank].z + 2 * haloRadius) +
                                                                   dimsRank[myRank].x * dimsRank[myRank].y * haloRadius;
                                            int const nElements = dimsRank[myRank].x * dimsRank[myRank].y * boundaryRadius;

                                            reductionInfo.startIDByView.push_back(startPoint);
                                            reductionInfo.nElementsByView.push_back(nElements);
                                        }

                                        {  // down
                                            auto const boundaryRadius = mData->zHaloDim;
                                            int const  startPoint = c * dimsRank[myRank].x * dimsRank[myRank].y * (dimsRank[myRank].z + 2 * haloRadius) +
                                                                   dimsRank[myRank].x * dimsRank[myRank].y * (dimsRank[myRank].z + haloRadius - boundaryRadius);
                                            int const nElements = dimsRank[myRank].x * dimsRank[myRank].y * boundaryRadius;

                                            reductionInfo.startIDByView.push_back(startPoint);
                                            reductionInfo.nElementsByView.push_back(nElements);
                                        }
                                    }
                                    break;
                                }
                                case MemoryLayout::arrayOfStructs: {
                                    {  // up
                                        auto const boundaryRadius = mData->zHaloDim;
                                        int const  startPoint = dimsRank[myRank].x * dimsRank[myRank].y * haloRadius * mData->cardinality;
                                        int const  nElements = dimsRank[myRank].x * dimsRank[myRank].y * boundaryRadius * mData->cardinality;

                                        reductionInfo.startIDByView.push_back(startPoint);
                                        reductionInfo.nElementsByView.push_back(nElements);
                                    }
                                    {  // down
                                        auto const boundaryRadius = mData->zHaloDim;
                                        int const  startPoint = dimsRank[myRank].x * dimsRank[myRank].y * (dimsRank[myRank].z + haloRadius - boundaryRadius) * mData->cardinality;
                                        int const  nElements = dimsRank[myRank].x * dimsRank[myRank].y * boundaryRadius * mData->cardinality;

                                        reductionInfo.startIDByView.push_back(startPoint);
                                        reductionInfo.nElementsByView.push_back(nElements);
                                    }
                                    break;
                                }
                            }
                        }
                        break;
                    }
                    default: {
                        NeonException exp("ncclFieldDev_t");
                        exp << " Invalid DataView";
                        NEON_THROW(exp);
                    }
                }
            });

        // this->initHaloUpdateTable();
    }
}


template <typename T, int C>
auto ncclField<T, C>::updateDeviceData(int streamSetId)
    -> void
{
    mData->memoryField.updateDeviceData(streamSetId);
}

template <typename T, int C>
auto ncclField<T, C>::updateHostData(int streamSetId)
    -> void
{
    mData->memoryField.updateHostData(streamSetId);
}

template <typename T, int C>
auto ncclField<T, C>::getPartition(Neon::Execution       execution,
                                   Neon::SetIdx          setIdx,
                                   const Neon::DataView& dataView)
    const
    -> const Partition&
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
auto ncclField<T, C>::getPartition(Neon::Execution       execution,
                                   Neon::SetIdx          setIdx,
                                   const Neon::DataView& dataView)
    -> Partition&
{
    const auto dataUse = this->getDataUse();
    bool       isOk = Neon::ExecutionUtils::checkCompatibility(dataUse, execution);
    if (isOk) {
        Partition& result = mData->partitionTable.getPartition(execution, setIdx, dataView);
        return result;
    }
    std::stringstream message;
    message << "The requested execution mode ( " << execution << " ) is not compatible with the field DataUse (" << dataUse << ")";
    NEON_THROW_UNSUPPORTED_OPERATION(message.str());
}

template <typename T, int C>
auto ncclField<T, C>::operator()(const Neon::index_3d& idxGlobal,
                                 const int&            cardinality) const
    -> Type
{
    auto [localIDx, partitionIdx] = helpGlobalIdxToPartitionIdx(idxGlobal);
    auto& partition = mData->partitionTable.getPartition(Neon::Execution::host,
                                                         partitionIdx,
                                                         Neon::DataView::STANDARD);
    auto& span = mData->grid->getSpan(Neon::Execution::host, partitionIdx, Neon::DataView::STANDARD);
    Idx   idx;
    bool  isOk = span.setAndValidate(idx, localIDx.x, localIDx.y, localIDx.z);
    if (!isOk) {
#pragma omp barrier
        NEON_THROW_UNSUPPORTED_OPERATION("");
    }
    auto& result = partition(idx, cardinality);
    return result;
}

template <typename T, int C>
auto ncclField<T, C>::getReference(const Neon::index_3d& idxGlobal,
                                   const int&            cardinality)
    -> Type&
{
    auto [localIDx, partitionIdx] = helpGlobalIdxToPartitionIdx(idxGlobal);
    auto& partition = mData->partitionTable.getPartition(Neon::Execution::host,
                                                         partitionIdx,
                                                         Neon::DataView::STANDARD);
    auto& span = mData->grid->getSpan(Neon::Execution::host, partitionIdx, Neon::DataView::STANDARD);
    Idx   idx;
    bool  isOk = span.setAndValidate(idx, localIDx.x, localIDx.y, localIDx.z);
    if (!isOk) {
#pragma omp barrier
        NEON_THROW_UNSUPPORTED_OPERATION("");
    }
    auto& result = partition(idx, cardinality);
    return result;
}


template <typename T, int C>
auto ncclField<T, C>::ioToVtiPartitions(std::string const& fname) const -> void
{
    auto bk = mData->grid->getBackend();
    bk.forEachDeviceSeq([&](Neon::SetIdx setIdx) {
        auto partition = this->getPartition(Neon::Execution::device, setIdx, Neon::DataView::STANDARD);
        partition.ioToVti(fname, "sdfsd");
    });
}

template <typename T, int C>
auto ncclField<T, C>::
    newHaloUpdate(Neon::set::StencilSemantic stencilSemantic,
                  Neon::set::TransferMode /*transferMode*/,
                  Neon::Execution execution)
        const -> Neon::set::Container
{


    // We need to define a graph of Containers
    // One for the actual memory transfer
    // One for the synchronization
    // The order depends on the transfer mode: put or get
    // Neon::set::Container dataTransferContainer;
    auto const&                      bk = this->getGrid().getBackend();
    int                              upRank = (bk.getNccl().getWorldRank() + 1) % bk.getNccl().getWorldSize();
    int                              downRank = (bk.getNccl().getWorldRank() - 1 + bk.getNccl().getWorldSize()) % bk.getNccl().getWorldSize();
    std::vector<Neon::set::NcclPtoP> nccSession;
    if (stencilSemantic == Neon::set::StencilSemantic::standard) {
        Neon::set::NcclPtoP upSend;
        Neon::set::NcclPtoP upRecv;
        Neon::set::NcclPtoP downSend;
        Neon::set::NcclPtoP downRecv;

        if (this->getMemoryOptions().getOrder() == Neon::MemoryLayout::structOfArrays) {
            Neon::SetIdx const setIdx = 0;
            auto               span = this->getGrid().getSpan(Neon::Execution::host, setIdx, Neon::DataView::STANDARD);
            auto               spanDim = span.helpGetDim();
            auto               haloRadius = span.helpGetZHaloRadius();

            ncclGrid::Idx dw_first_boundary = span.helpHalosetAndValidate(0, 0, 0);
            ncclGrid::Idx up_first_boundary = span.helpHalosetAndValidate(0, 0, spanDim.z - haloRadius);
            ncclGrid::Idx dw_first_halo_Idx = span.helpHalosetAndValidate(0, 0, -haloRadius);
            ncclGrid::Idx up_first_halo_Idx = span.helpHalosetAndValidate(0, 0, spanDim.z);

            size_t numElementForTransfer = size_t(spanDim.x) * spanDim.y * haloRadius;
            auto   partition = this->getPartition(Neon::Execution::device, setIdx, Neon::DataView::STANDARD);

            for (int i = 0; i < this->getCardinality(); i++) {
                upSend = Neon::set::NcclPtoP::init<Type>(bk, upRank, numElementForTransfer, &(partition(up_first_boundary, i)), Neon::set::NcclPtoP::send);
                upRecv = Neon::set::NcclPtoP::init<Type>(bk, upRank, numElementForTransfer, &(partition(up_first_halo_Idx, i)), Neon::set::NcclPtoP::receive);

                downSend = Neon::set::NcclPtoP::init<Type>(bk, downRank, numElementForTransfer, &(partition(dw_first_boundary, i)), Neon::set::NcclPtoP::send);
                downRecv = Neon::set::NcclPtoP::init<Type>(bk, downRank, numElementForTransfer, &(partition(dw_first_halo_Idx, i)), Neon::set::NcclPtoP::receive);

                if (upRank > bk.getNccl().getWorldRank()) {
                    nccSession.push_back(upSend);
                    nccSession.push_back(upRecv);
                }
                if (downRank < bk.getNccl().getWorldRank()) {
                    nccSession.push_back(downSend);
                    nccSession.push_back(downRecv);
                }
            }
            auto res = Neon::set::Container::factoryNcclTransfer(this->getGrid(),
                                                                 stencilSemantic,
                                                                 nccSession,
                                                                 execution);
            return res;
        } else {
            NEON_DEV_UNDER_CONSTRUCTION("");
        }
    } else {
        NEON_DEV_UNDER_CONSTRUCTION("");
    }


    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename T, int C>
auto ncclField<T, C>::self() -> ncclField::Self&
{
    return *this;
}

template <typename T, int C>
auto ncclField<T, C>::self() const -> const ncclField::Self&
{
    return *this;
}

template <typename T, int C>
auto ncclField<T, C>::constSelf() const -> const ncclField::Self&
{
    return *this;
}

template <typename T, int C>
auto ncclField<T, C>::swap(ncclField::Field& A, ncclField::Field& B) -> void
{
    Neon::domain::interface::FieldBaseTemplate<T, C, Grid, Partition, int>::swapUIDBeforeFullSwap(A, B);
    std::swap(A, B);
}

template <typename T, int C>
auto ncclField<T, C>::getData()
    -> Data&
{
    return *(mData.get());
}

template <typename T, int C>
auto ncclField<T, C>::helpGlobalIdxToPartitionIdx(Neon::index_3d const& index)
    const -> std::pair<Neon::index_3d, int>
{
    Neon::index_3d result = index;

    // since we partition along the z-axis, only the z-component of index will change
    const int32_t setCardinality = mData->grid->getBackend().devSet().numDevs();
    if (setCardinality == 1) {
        return {result, 0};
    }

    Neon::set::DataSet<int> firstZindex = mData->grid->helpGetFirstZindex();

    for (int i = 0; i < setCardinality - 1; i++) {
        if (index.z < firstZindex[i + 1]) {
            result.z -= firstZindex[i];
            return {result, i};
        }
    }
    if (index.z < this->getGrid().getDimension().z) {
        result.z -= firstZindex[setCardinality - 1];
        return {result, setCardinality - 1};
    }

    NeonException exc("ncclField");
    exc << "Data inconsistency was detected";
    NEON_THROW(exc);
}

}  // namespace Neon::domain::details::ncclGrid
