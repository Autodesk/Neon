#pragma once
#include "dField.h"

namespace Neon::domain::details::dGrid {

template <typename T, int C>
dField<T, C>::dField()
{
    mData = std::make_shared<Data>();
}

template <typename T, int C>
dField<T, C>::dField(const std::string&                        fieldUserName,
                     Neon::DataUse                             dataUse,
                     const Neon::MemoryOptions&                memoryOptions,
                     const Grid&                               grid,
                     const Neon::set::DataSet<Neon::index_3d>& dims,
                     int                                       zHaloDim,
                     Neon::domain::haloStatus_et::e            haloStatus,
                     int                                       cardinality,
                     Neon::set::MemSet<Neon::int8_3d>&         stencilIdTo3dOffset)
    : Neon::domain::interface::FieldBaseTemplate<T, C, Grid, Partition, int>(&grid,
                                                                             fieldUserName,
                                                                             "dField",
                                                                             cardinality,
                                                                             T(0),
                                                                             dataUse,
                                                                             memoryOptions,
                                                                             haloStatus) {

    // only works if dims in x and y direction for all partitions match
    for (int i = 0; i < dims.size() - 1; ++i) {
        for (int j = i + 1; j < dims.size(); ++j) {
            if (dims[i].x != dims[j].x || dims[i].y != dims[j].y) {
                NeonException exc("dField_t");
                exc << "New dField only works on partitioning along z axis.";
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
    mData->haloStatus = (mData->grid->getDevSet().setCardinality() == 1)
                            ? haloStatus_et::e::OFF
                            : haloStatus;
    const int haloRadius = mData->haloStatus == Neon::domain::haloStatus_et::ON ? zHaloDim : 0;
    mData->zHaloDim = zHaloDim;

    Neon::set::DataSet<index_3d> origins = this->getGrid().getBackend().template newDataSet<index_3d>({0, 0, 0});
    {  // Computing origins
        origins.forEachSeq(
            [&](Neon::SetIdx setIdx, Neon::index_3d& val) {
                if (setIdx == 0) {
                    val.z = 0;
                    return;
                }
                const Neon::SetIdx proceedingIdx = setIdx - 1;
                val.z = origins[proceedingIdx].z + dims[proceedingIdx].z;
            });
    }

    {  // Computing Pitch
        mData->pitch.forEachSeq(
            [&](Neon::SetIdx setIdx, Neon::size_4d& pitch) {
                switch (mData->memoryOptions.getOrder()) {
                    case MemoryLayout::structOfArrays: {
                        pitch.x = 1;
                        pitch.y = pitch.x * dims[setIdx.idx()].x;
                        pitch.z = pitch.y * dims[setIdx.idx()].y;
                        pitch.w = pitch.z * (dims[setIdx.idx()].z + 2 * haloRadius);
                        break;
                    }
                    case MemoryLayout::arrayOfStructs: {
                        pitch.x = mData->cardinality;
                        pitch.y = pitch.x * dims[setIdx.idx()].x;
                        pitch.z = pitch.y * dims[setIdx.idx()].y;
                        pitch.w = 1;
                        break;
                    }
                }
            });
    }

    {  // Setting up partitions
        Neon::domain::aGrid const& aGrid = mData->grid->helpFieldMemoryAllocator();
        mData->memoryField = aGrid.newField(fieldUserName + "-storage", cardinality, T(), dataUse, memoryOptions);
        // const int setCardinality = mData->grid->getBackend().getDeviceCount();
        mData->partitionTable.forEachConfiguration(
            [&](Neon::Execution           execution,
                Neon::SetIdx              setIdx,
                Neon::DataView            dw,
                typename Self::Partition& partition) {
                auto memoryFieldPartition = mData->memoryField.getPartition(execution, setIdx, Neon::DataView::STANDARD);

                partition = dPartition<T, C>(dw,
                                             memoryFieldPartition.mem(),
                                             dims[setIdx],
                                             haloRadius,
                                             mData->zHaloDim,
                                             mData->pitch[setIdx],
                                             setIdx.idx(),
                                             origins[setIdx],
                                             mData->cardinality,
                                             mData->grid->getDimension(),
                                             stencilIdTo3dOffset.rawMem(execution, setIdx));
            });
    }

    {  // Setting Reduction information
        mData->partitionTable.forEachConfigurationWithUserData(
            [&](Neon::Execution,
                Neon::SetIdx   setIdx,
                Neon::DataView dw,
                typename Self::Partition&,
                typename Data::ReductionInformation& reductionInfo) {
                switch (dw) {
                    case Neon::DataView::STANDARD: {
                        // old structure [dv_id][c][i]
                        if (grid.getBackend().devSet().setCardinality() == 1) {
                            // As the number of devices is 1, we don't have halos.
                            reductionInfo.startIDByView.push_back(0);
                            reductionInfo.nElementsByView.push_back(int(dims[setIdx.idx()].rMul()));
                        } else {
                            switch (mData->memoryOptions.getOrder()) {
                                case MemoryLayout::structOfArrays: {
                                    for (int c = 0; c < mData->cardinality; ++c) {
                                        // To compute the start point we need to
                                        // jump the previous cardinalities -> c * dims[setIdx].x * dims[setIdx].y * (dims[setIdx].z + 2 * haloRadius)
                                        // jump one halo -> dims[setIdx].x * dims[setIdx].y * haloRadius
                                        int const startPoint = c * dims[setIdx].x * dims[setIdx].y * (dims[setIdx].z + 2 * haloRadius) +
                                                               dims[setIdx].x * dims[setIdx].y * haloRadius;
                                        int const nElements = dims[setIdx].rMul();

                                        reductionInfo.startIDByView.push_back(startPoint);
                                        reductionInfo.nElementsByView.push_back(nElements);
                                    }
                                    break;
                                }
                                case MemoryLayout::arrayOfStructs: {
                                    int const startPoint = dims[setIdx].x * dims[setIdx].y * haloRadius * mData->cardinality;
                                    int const nElements = dims[setIdx].x * dims[setIdx].y * dims[setIdx].z * mData->cardinality;

                                    reductionInfo.startIDByView.push_back(startPoint);
                                    reductionInfo.nElementsByView.push_back(nElements);

                                    break;
                                }
                            }
                        }
                        break;
                    }
                    case Neon::DataView::INTERNAL: {
                        if (grid.getBackend().devSet().setCardinality() > 1) {
                            switch (mData->memoryOptions.getOrder()) {
                                case MemoryLayout::structOfArrays: {
                                    for (int c = 0; c < mData->cardinality; ++c) {

                                        auto const boundaryRadius = mData->zHaloDim;
                                        int const  startPoint = c * dims[setIdx].x * dims[setIdx].y * (dims[setIdx].z + 2 * haloRadius) +
                                                               dims[setIdx].x * dims[setIdx].y * (haloRadius + boundaryRadius);

                                        int const nElements = dims[setIdx].x * dims[setIdx].y * (dims[setIdx].z - 2 * haloRadius);

                                        reductionInfo.startIDByView.push_back(startPoint);
                                        reductionInfo.nElementsByView.push_back(nElements);
                                    }
                                    break;
                                }
                                case MemoryLayout::arrayOfStructs: {
                                    auto const boundaryRadius = mData->zHaloDim;
                                    int const  startPoint = dims[setIdx].x * dims[setIdx].y * (haloRadius + boundaryRadius) * mData->cardinality;
                                    int const  nElements = dims[setIdx].x * dims[setIdx].y * (dims[setIdx].z - 2 * haloRadius) * mData->cardinality;

                                    reductionInfo.startIDByView.push_back(startPoint);
                                    reductionInfo.nElementsByView.push_back(nElements);

                                    break;
                                }
                            }
                        }
                        break;
                    }
                    case Neon::DataView::BOUNDARY: {
                        if (grid.getBackend().devSet().setCardinality() > 1) {
                            switch (mData->memoryOptions.getOrder()) {
                                case MemoryLayout::structOfArrays: {
                                    for (int c = 0; c < mData->cardinality; ++c) {
                                        {  // up
                                            auto const boundaryRadius = mData->zHaloDim;
                                            int const  startPoint = c * dims[setIdx].x * dims[setIdx].y * (dims[setIdx].z + 2 * haloRadius) +
                                                                   dims[setIdx].x * dims[setIdx].y * haloRadius;
                                            int const nElements = dims[setIdx].x * dims[setIdx].y * boundaryRadius;

                                            reductionInfo.startIDByView.push_back(startPoint);
                                            reductionInfo.nElementsByView.push_back(nElements);
                                        }

                                        {  // down
                                            auto const boundaryRadius = mData->zHaloDim;
                                            int const  startPoint = c * dims[setIdx].x * dims[setIdx].y * (dims[setIdx].z + 2 * haloRadius) +
                                                                   dims[setIdx].x * dims[setIdx].y * (dims[setIdx].z + haloRadius - boundaryRadius);
                                            int const nElements = dims[setIdx].x * dims[setIdx].y * boundaryRadius;

                                            reductionInfo.startIDByView.push_back(startPoint);
                                            reductionInfo.nElementsByView.push_back(nElements);
                                        }
                                    }
                                    break;
                                }
                                case MemoryLayout::arrayOfStructs: {
                                    {  // up
                                        auto const boundaryRadius = mData->zHaloDim;
                                        int const  startPoint = dims[setIdx].x * dims[setIdx].y * haloRadius * mData->cardinality;
                                        int const  nElements = dims[setIdx].x * dims[setIdx].y * boundaryRadius * mData->cardinality;

                                        reductionInfo.startIDByView.push_back(startPoint);
                                        reductionInfo.nElementsByView.push_back(nElements);
                                    }
                                    {  // down
                                        auto const boundaryRadius = mData->zHaloDim;
                                        int const  startPoint = dims[setIdx].x * dims[setIdx].y * (dims[setIdx].z + haloRadius - boundaryRadius) * mData->cardinality;
                                        int const  nElements = dims[setIdx].x * dims[setIdx].y * boundaryRadius * mData->cardinality;

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
                        NeonException exp("dFieldDev_t");
                        exp << " Invalid DataView";
                        NEON_THROW(exp);
                    }
                }
            });
    }
}


template <typename T, int C>
auto dField<T, C>::updateDeviceData(int streamSetId)
    -> void
{
    mData->memoryField.updateCompute(streamSetId);
}

template <typename T, int C>
auto dField<T, C>::updateHostData(int streamSetId)
    -> void
{
    mData->memoryField.updateIO(streamSetId);
}

template <typename T, int C>
auto dField<T, C>::getPartition(Neon::Execution       execution,
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
auto dField<T, C>::getPartition(Neon::Execution       execution,
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
auto dField<T, C>::operator()(const Neon::index_3d& idxGlobal,
                              const int&            cardinality) const
    -> Type
{
    auto [localIDx, partitionIdx] = helpGlobalIdxToPartitionIdx(idxGlobal);
    auto& partition = mData->partitionTable.getPartition(Neon::Execution::host,
                                                         partitionIdx,
                                                         Neon::DataView::STANDARD);
    auto& span = mData->grid->getSpan(partitionIdx, Neon::DataView::STANDARD);
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
auto dField<T, C>::getReference(const Neon::index_3d& idxGlobal,
                                const int&            cardinality)
    -> Type&
{
    auto [localIDx, partitionIdx] = helpGlobalIdxToPartitionIdx(idxGlobal);
    auto& partition = mData->partitionTable.getPartition(Neon::Execution::host,
                                                         partitionIdx,
                                                         Neon::DataView::STANDARD);
    auto& span = mData->grid->getSpan(partitionIdx, Neon::DataView::STANDARD);
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
auto dField<T, C>::helpHaloUpdate(SetIdx                     setIdx,
                                  int                        streamIdx,
                                  Neon::set::StencilSemantic semantic,
                                  int const&                 cardIdx,
                                  Neon::set::TransferMode    transferMode,
                                  Neon::Execution            execution) const
    -> void
{
#if 0
    const int setCardinality = mData->grid->getBackend().devSet().setCardinality();

    switch (mData->memoryOptions.getOrder()) {
        case MemoryLayout::arrayOfStructs: {
            if (cardIdx != -1) {
                NEON_THROW_UNSUPPORTED_OPERATION("dField::haloUpdate() can not execute operation on a specific cardinality in a AoS memory layout");
            }
#pragma omp parallel for num_threads(setCardinality)
            for (int setId = 0; setId < setCardinality; setId++) {
                // Because we are in a arrayOfStructs layout mData->pitch[setId].z already includes the jump over cardinalities.
                const size_t transferBytes = sizeof(T) * mData->zHaloDim * mData->pitch[setId].z;

                auto messagingNorthRequired = [setCardinality](Neon::SetIdx setIdx) { return setIdx != setCardinality - 1; };

                if (messagingNorthRequired(setId)) {  // Addressing all partitions that needs to send data north

                    T* src = [&]() {
                        auto  southDevice = setId;
                        auto& partition = mData->partitionTable.getPartition(execution,
                                                                            southDevice,
                                                                            Neon::DataView::STANDARD);
                        dIndex firstBoundaryNorthCell(0, 0, partition.dim.z - mData->zHaloDim);
                        T*    result = partition.mem(firstBoundaryNorthCell, 0);
                        return result;
                    }();

                    T* dst = [&]() {
                        auto  northDevice = setId + 1;
                        auto& partition = mData->partitionTable.getPartition(execution,
                                                                            northDevice,
                                                                            Neon::DataView::STANDARD);
                        dIndex firstBoundarySouthCell(0, 0, 0);
                        T*    result = partition.mem(firstBoundarySouthCell, 0);
                        result = result - mData->zHaloDim * partition.getPitchData().z;
                        return result;
                    }();
                }
            }
            break;
        }
    }


#pragma omp parallel for num_threads(setCardinality)
    for (int setId = 0; setId < setCardinality; setId++) {
        switch (mData->memoryOptions.getOrder()) {
            case MemoryLayout::arrayOfStructs: {
                // send to the next partition (+z)
                const size_t transferBytes = sizeof(T) * mData->zHaloDim * mData->pitch[setId].z;

                if (setId != setCardinality - 1) {  // Addressing all partitions that needs to send data north
                    auto& partition = mData->partitionTable.getPartition(Neon::Execution::device,
                                                                        setId,
                                                                        Neon::DataView::STANDARD);

                    dIndex srcIdx(0, 0, partition.dim().z - 1);

                    T* src = partition(src_idx, 0);
                    field_compute[setId].mem() + field_compute[setId].elPitch(src_idx);

                    dCell dst_idx(0, 0, 0);
                    T*    dst = field_compute[setId + 1].mem() + +field_compute[setId + 1].elPitch(dst_idx);

                    if (mData->devType == Neon::DeviceType::CPU) {
                        std::memcpy(dst, src, transferBytes);
                    } else if (mData->devType == Neon::DeviceType::CUDA) {
                        bk.devSet().template peerTransfer<transferMode_ta>(
                            streamSet,
                            mData->grid->getDevSet().devId(setId + 1).idx(),  // dst
                            (char*)(dst),
                            mData->grid->getDevSet().devId(setId).idx(),  // src
                            (char*)(src),
                            transferBytes);
                    } else {
                        NEON_THROW_UNSUPPORTED_OPERATION("dFieldDev_t::haloUpdate() unsupported device.");
                    }
                }

                // send to the previous partition (-z)
                if (setId != 0) {
                    dCell src_idx(0, 0, mData->zHaloDim);
                    T*    src = field_compute[setId].mem() + field_compute[setId].elPitch(src_idx);

                    dCell dst_idx(0, 0, field_compute[setId - 1].dim().z + mData->zHaloDim);
                    T*    dst = field_compute[setId - 1].mem() + +field_compute[setId - 1].elPitch(dst_idx);

                    if (mData->devType == Neon::DeviceType::CPU) {
                        std::memcpy(dst, src, transferBytes);
                    } else if (mData->devType == Neon::DeviceType::CUDA) {
                        bk.devSet().template peerTransfer<transferMode_ta>(
                            streamSet,
                            mData->grid->getDevSet().devId(setId - 1).idx(),  // dst
                            (char*)(dst),
                            mData->grid->getDevSet().devId(setId).idx(),  // src
                            (char*)(src),
                            transferBytes);
                    } else {
                        NEON_THROW_UNSUPPORTED_OPERATION("dFieldDev_t::haloUpdate() unsupported device.");
                    }
                }
                break;
            }
            case Neon::memLayout_et::order_e::structOfArrays: {

                int card = 0;
                for (card = 0; card < mData->cardinality; card++) {

                    size_t transferBytes = sizeof(T) * mData->zHaloDim * mData->pitch[setId].z;

                    if (cardIdx != -1) {
                        card = cardIdx;
                        transferBytes /= mData->cardinality;
                    }

                    // send to the next partition (+z)
                    if (setId != ndevs - 1) {
                        dCell src_idx(0, 0, field_compute[setId].dim().z /*+mData->zHaloDim - mData->zHaloDim*/);
                        T*    src = field_compute[setId].mem() + field_compute[setId].elPitch(src_idx, card);

                        dCell dst_idx(0, 0, 0);
                        T*    dst = field_compute[setId + 1].mem() + +field_compute[setId + 1].elPitch(dst_idx, card);

                        if (mData->devType == Neon::DeviceType::CPU) {
                            std::memcpy(dst, src, transferBytes);
                        } else if (mData->devType == Neon::DeviceType::CUDA) {
                            bk.devSet().template peerTransfer<transferMode_ta>(
                                streamSet,
                                mData->grid->getDevSet().devId(setId + 1).idx(),  // dst
                                (char*)(dst),
                                mData->grid->getDevSet().devId(setId).idx(),  // src
                                (char*)(src),
                                transferBytes);
                        } else {
                            NEON_THROW_UNSUPPORTED_OPERATION("dFieldDev_t::haloUpdate() unsupported device.");
                        }
                    }

                    // send to the previous partition (-z)
                    if (setId != 0) {
                        dCell src_idx(0, 0, mData->zHaloDim);
                        T*    src = field_compute[setId].mem() + field_compute[setId].elPitch(src_idx, card);

                        dCell dst_idx(0, 0, field_compute[setId - 1].dim().z + mData->zHaloDim);
                        T*    dst = field_compute[setId - 1].mem() + +field_compute[setId - 1].elPitch(dst_idx, card);

                        if (mData->devType == Neon::DeviceType::CPU) {
                            std::memcpy(dst, src, transferBytes);
                        } else if (mData->devType == Neon::DeviceType::CUDA) {
                            bk.devSet().template peerTransfer<transferMode_ta>(
                                streamSet,
                                mData->grid->getDevSet().devId(setId - 1).idx(),  // dst
                                (char*)(dst),
                                mData->grid->getDevSet().devId(setId).idx(),  // src
                                (char*)(src),
                                transferBytes);
                        } else {
                            NEON_THROW_UNSUPPORTED_OPERATION("dFieldDev_t::haloUpdate() unsupported device.");
                        }
                    }

                    if (cardIdx != -1) {
                        break;
                    }
                }
                break;
            }
        }
    }
#endif
}


template <typename T, int C>
auto dField<T, C>::
    newHaloUpdate(Neon::set::StencilSemantic stencilSemantic,
                  Neon::set::TransferMode    transferMode,
                  Neon::Execution            execution)
        const -> Neon::set::Container
{
    // We need to define a graph of Containers
    // One for the actual memory transfer
    // One for the synchronization
    // The order depends on the transfer mode: put or get
    Neon::set::Container dataTransferContainer =
        Neon::set::Container::factoryDataTransfer(
            *this,
            transferMode,
            stencilSemantic);

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

template <typename T, int C>
auto dField<T, C>::self() -> dField::Self&
{
    return *this;
}

template <typename T, int C>
auto dField<T, C>::self() const -> const dField::Self&
{
    return *this;
}
template <typename T, int C>
auto dField<T, C>::swap(dField::Field& A, dField::Field& B) -> void
{
    Neon::domain::interface::FieldBaseTemplate<T, C, Grid, Partition, int>::swapUIDBeforeFullSwap(A, B);
    std::swap(A, B);
}

template <typename T, int C>
auto dField<T, C>::getData()
    -> Data&
{
    return *(mData.get());
}

template <typename T, int C>
auto dField<T, C>::helpGlobalIdxToPartitionIdx(Neon::index_3d const& index)
    const -> std::pair<Neon::index_3d, int>
{
    Neon::index_3d result = index;

    // since we partition along the z-axis, only the z-component of index will change
    const int32_t setCardinality = mData->grid->getBackend().devSet().setCardinality();
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

    NeonException exc("dField");
    exc << "Data inconsistency was detected";
    NEON_THROW(exc);
}

}  // namespace Neon::domain::details::dGrid
