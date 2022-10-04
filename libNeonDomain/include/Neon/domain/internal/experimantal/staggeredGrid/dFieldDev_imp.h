#pragma once
namespace Neon::domain::internal::dGrid {

template <typename T, int C>
dFieldDev<T, C>::dFieldDev(const grid_t&                             grid,
                           const Neon::set::DataSet<Neon::index_3d>& dims,
                           int                                       zHaloDim,
                           Neon::domain::haloStatus_et::e            haloStatus,
                           Neon::DeviceType                          deviceType,
                           Neon::memLayout_et::order_e               memOrder,
                           Neon::Allocator                           allocator,
                           int                                       cardinality)
{
    m_data = std::make_shared<data_t>();
    m_data->devType = deviceType;
    m_data->cardinality = cardinality;
    m_data->memOrder = memOrder;
    m_data->memAlloc = allocator;
    m_data->grid = std::make_shared<grid_t>(grid);
    m_data->zHaloDim = zHaloDim;
    m_data->haloStatus = (m_data->grid->getDevSet().setCardinality() == 1) ? haloStatus_et::e::OFF : haloStatus;

    if (m_data->memAlloc != Neon::Allocator::NULL_MEM) {
        h_init(dims, grid);
    } else {
        // init of dFieldComputeSet for each data view
        for (int i = 0; i < static_cast<int>(Neon::DataViewUtil::nConfig); i++) {
            m_data->dFieldComputeSetByView[i] = grid.getBackend().devSet().template newDataSet<local_t>();
            for (SetIdx setIdx = 0; setIdx < grid.getBackend().devSet().setCardinality(); setIdx++) {
                m_data->dFieldComputeSetByView[i][setIdx] = local_t();
            }
        }
        m_data->userPointersSet = grid.getBackend().devSet().template newDataSet<element_t*>();
        for (SetIdx setIdx = 0; setIdx < grid.getBackend().devSet().setCardinality(); setIdx++) {
            m_data->userPointersSet[setIdx] = nullptr;
        }
    }
}

template <typename T, int C>
dFieldDev<T, C>::dFieldDev()
{
    m_data = std::make_shared<data_t>();
    m_data->grid = std::shared_ptr<grid_t>();
}

template <typename T, int C>
dFieldDev<T, C>::dFieldDev(const dFieldDev& other)
{
    m_data = other.m_data;
}

template <typename T, int C>
dFieldDev<T, C>::dFieldDev(dFieldDev&& other)
{
    m_data = std::move(other.m_data);
    other.m_data = std::shared_ptr<data_t>();
}

template <typename T, int C>
auto dFieldDev<T, C>::operator=(const dFieldDev& other)
    -> dFieldDev&
{
    m_data = other.m_data;
    return *this;
}

template <typename T, int C>
auto dFieldDev<T, C>::operator=(dFieldDev&& other) -> dFieldDev&
{
    m_data = std::move(other.m_data);
    other.m_data = std::shared_ptr<data_t>();
    return *this;
}

template <typename T, int C>
auto dFieldDev<T, C>::uid() const -> Neon::set::MultiDeviceObjectUid
{
    void*                           addr = static_cast<void*>(m_data.get());
    Neon::set::MultiDeviceObjectUid uidRes = (size_t)addr;
    return uidRes;
}

template <typename T, int C>
auto dFieldDev<T, C>::grid() -> grid_t&
{
    return *(m_data->grid.get());
}

template <typename T, int C>
auto dFieldDev<T, C>::grid() const -> const grid_t&
{
    return *(m_data->grid.get());
}

template <typename T, int C>
auto dFieldDev<T, C>::cardinality() const -> int
{
    return m_data->cardinality;
}

template <typename T, int C>
auto dFieldDev<T, C>::devType() const -> Neon::DeviceType
{
    return m_data->devType;
}

template <typename T, int C>
auto dFieldDev<T, C>::dot(
    Neon::set::patterns::BlasSet<T>& blasSet,
    const dFieldDev<T>&              input,
    Neon::set::MemDevSet<T>&         output,
    const Neon::DataView&            dataView) -> T
{
    const int dataView_id = static_cast<int>(dataView);
    const int numSlices = int(m_data->startIDByView[dataView_id].size());
    T         ret = 0;
    for (int s = 0; s < numSlices; ++s) {
        ret += blasSet.dot(m_data->memory,
                           input.m_data->memory,
                           output,
                           m_data->startIDByView[dataView_id][s],
                           m_data->nElementsByView[dataView_id][s]);
    }
    return ret;
}

template <typename T, int C>
auto dFieldDev<T, C>::norm2(
    Neon::set::patterns::BlasSet<T>& blasSet,
    Neon::set::MemDevSet<T>&         output,
    const Neon::DataView&            dataView) -> T
{
    const int dataView_id = static_cast<int>(dataView);
    const int numSlices = int(m_data->startIDByView[dataView_id].size());
    T         ret = 0;
    for (int s = 0; s < numSlices; ++s) {
        T temp = blasSet.norm2(m_data->memory,
                               output,
                               m_data->startIDByView[dataView_id][s],
                               m_data->nElementsByView[dataView_id][s]);
        ret += temp * temp;
    }
    ret = std::sqrt(ret);
    return ret;
}

template <typename T, int C>
template <Neon::DataView Indexing_ta>
auto dFieldDev<T, C>::eRef(const Neon::index_3d& idx,
                           const int             cardinality) -> T&
{
    if (m_data->devType != Neon::DeviceType::CPU) {
        NeonException exc("dFieldDev");
        exc << "eRef operation can not be run on a GPU field.";
        NEON_THROW(exc);
    }
    dCell   local_idx(idx);
    int32_t part = convert_to_local(local_idx.set());

    return m_data->dFieldComputeSetByView[static_cast<int>(Indexing_ta)][part].operator()(local_idx, cardinality);
}

template <typename T, int C>
template <Neon::DataView Indexing_ta>
auto dFieldDev<T, C>::eRef(const Neon::index_3d& idx,
                           const int&            cardinality) const -> const T&
{
    if (m_data->devType != Neon::DeviceType::CPU) {
        NeonException exc("dFieldDev");
        exc << "eRef operation can not be run on a GPU field.";
        NEON_THROW(exc);
    }
    dCell                                                                      local_idx(idx);
    int32_t                                                                    part = convert_to_local(local_idx.set());
    return m_data->dFieldComputeSetByView[static_cast<int>(Indexing_ta)][part].operator()(local_idx, cardinality);
}

template <typename T, int C>
template <Neon::set::TransferMode transferMode_ta>
auto dFieldDev<T, C>::haloUpdate(const Neon::Backend& bk,
                                 bool                 startWithBarrier,
                                 int                  streamSetIdx)
    -> void
{
    haloUpdate<transferMode_ta>(bk, -1, startWithBarrier, streamSetIdx);
}

template <typename T, int C>
template <Neon::set::TransferMode transferMode_ta>
auto dFieldDev<T, C>::haloUpdate(Neon::SetIdx         setIdx,
                                 const Neon::Backend& bk,
                                 bool                 startWithBarrier,
                                 int                  streamSetIdx)
    -> void
{
    haloUpdate<transferMode_ta>(setIdx, bk, -1, startWithBarrier, streamSetIdx);
}

template <typename T, int C>
template <Neon::set::TransferMode transferMode_ta>
auto dFieldDev<T, C>::haloUpdate(const Neon::Backend& bk,
                                 const int            cardIdx,
                                 bool                 startWithBarrier,
                                 int                  streamSetIdx) -> void
{
    if (cardIdx != -1 && m_data->memOrder == Neon::memLayout_et::order_e::arrayOfStructs) {
        NEON_THROW_UNSUPPORTED_OPERATION("dFieldDev_t::haloUpdate() can not halo-update specific cardinality in a AoS memory layout");
    }

    if (startWithBarrier) {
        bk.syncAll();
    }

    const int ndevs = static_cast<int>(m_data->grid->partitions().size());
    auto&     streamSet = bk.streamSet(streamSetIdx);
    auto&     field_compute = m_data->dFieldComputeSetByView[static_cast<int>(DataView::STANDARD)];
#pragma omp parallel for num_threads(ndevs)
    for (int setId = 0; setId < ndevs; setId++) {
        switch (m_data->memOrder) {
            case Neon::memLayout_et::order_e::arrayOfStructs: {
                // send to the next partition (+z)
                const size_t transferBytes = sizeof(T) * m_data->zHaloDim * m_data->pitch[setId].z;


                if (setId != ndevs - 1) {

                    dCell src_idx(0, 0, field_compute[setId].dim().z /*+m_data->zHaloDim - m_data->zHaloDim*/);
                    T*    src = field_compute[setId].mem() + field_compute[setId].elPitch(src_idx);

                    dCell dst_idx(0, 0, 0);
                    T*    dst = field_compute[setId + 1].mem() + +field_compute[setId + 1].elPitch(dst_idx);

                    if (m_data->devType == Neon::DeviceType::CPU) {
                        std::memcpy(dst, src, transferBytes);
                    } else if (m_data->devType == Neon::DeviceType::CUDA) {
                        bk.devSet().template peerTransfer<transferMode_ta>(
                            streamSet,
                            m_data->grid->getDevSet().devId(setId + 1).idx(),  // dst
                            (char*)(dst),
                            m_data->grid->getDevSet().devId(setId).idx(),  // src
                            (char*)(src),
                            transferBytes);
                    } else {
                        NEON_THROW_UNSUPPORTED_OPERATION("dFieldDev_t::haloUpdate() unsupported device.");
                    }
                }

                // send to the previous partition (-z)
                if (setId != 0) {
                    dCell src_idx(0, 0, m_data->zHaloDim);
                    T*    src = field_compute[setId].mem() + field_compute[setId].elPitch(src_idx);

                    dCell dst_idx(0, 0, field_compute[setId - 1].dim().z + m_data->zHaloDim);
                    T*    dst = field_compute[setId - 1].mem() + +field_compute[setId - 1].elPitch(dst_idx);

                    if (m_data->devType == Neon::DeviceType::CPU) {
                        std::memcpy(dst, src, transferBytes);
                    } else if (m_data->devType == Neon::DeviceType::CUDA) {
                        bk.devSet().template peerTransfer<transferMode_ta>(
                            streamSet,
                            m_data->grid->getDevSet().devId(setId - 1).idx(),  // dst
                            (char*)(dst),
                            m_data->grid->getDevSet().devId(setId).idx(),  // src
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
                for (card = 0; card < m_data->cardinality; card++) {

                    size_t transferBytes = sizeof(T) * m_data->zHaloDim * m_data->pitch[setId].z;

                    if (cardIdx != -1) {
                        card = cardIdx;
                        transferBytes /= m_data->cardinality;
                    }

                    // send to the next partition (+z)
                    if (setId != ndevs - 1) {
                        dCell src_idx(0, 0, field_compute[setId].dim().z /*+m_data->zHaloDim - m_data->zHaloDim*/);
                        T*    src = field_compute[setId].mem() + field_compute[setId].elPitch(src_idx, card);

                        dCell dst_idx(0, 0, 0);
                        T*    dst = field_compute[setId + 1].mem() + +field_compute[setId + 1].elPitch(dst_idx, card);

                        if (m_data->devType == Neon::DeviceType::CPU) {
                            std::memcpy(dst, src, transferBytes);
                        } else if (m_data->devType == Neon::DeviceType::CUDA) {
                            bk.devSet().template peerTransfer<transferMode_ta>(
                                streamSet,
                                m_data->grid->getDevSet().devId(setId + 1).idx(),  // dst
                                (char*)(dst),
                                m_data->grid->getDevSet().devId(setId).idx(),  // src
                                (char*)(src),
                                transferBytes);
                        } else {
                            NEON_THROW_UNSUPPORTED_OPERATION("dFieldDev_t::haloUpdate() unsupported device.");
                        }
                    }

                    // send to the previous partition (-z)
                    if (setId != 0) {
                        dCell src_idx(0, 0, m_data->zHaloDim);
                        T*    src = field_compute[setId].mem() + field_compute[setId].elPitch(src_idx, card);

                        dCell dst_idx(0, 0, field_compute[setId - 1].dim().z + m_data->zHaloDim);
                        T*    dst = field_compute[setId - 1].mem() + +field_compute[setId - 1].elPitch(dst_idx, card);

                        if (m_data->devType == Neon::DeviceType::CPU) {
                            std::memcpy(dst, src, transferBytes);
                        } else if (m_data->devType == Neon::DeviceType::CUDA) {
                            bk.devSet().template peerTransfer<transferMode_ta>(
                                streamSet,
                                m_data->grid->getDevSet().devId(setId - 1).idx(),  // dst
                                (char*)(dst),
                                m_data->grid->getDevSet().devId(setId).idx(),  // src
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
}


template <typename T, int C>
template <Neon::set::TransferMode transferMode_ta>
auto dFieldDev<T, C>::haloUpdate(Neon::SetIdx         setIdx,
                                 const Neon::Backend& bk,
                                 const int            cardIdx,
                                 bool                 startWithBarrier,
                                 int                  streamSetIdx) -> void
{
    if (cardIdx != -1 && m_data->memOrder == Neon::memLayout_et::order_e::arrayOfStructs) {
        NEON_THROW_UNSUPPORTED_OPERATION("dFieldDev_t::haloUpdate() can not halo-update specific cardinality in a AoS memory layout");
    }

    if (startWithBarrier) {
        bk.syncAll();
    }
    int const setId = setIdx.idx();
    const int ndevs = static_cast<int>(m_data->grid->partitions().size());
    auto&     streamSet = bk.streamSet(streamSetIdx);
    auto&     field_compute = m_data->dFieldComputeSetByView[static_cast<int>(DataView::STANDARD)];
    switch (m_data->memOrder) {
        case Neon::memLayout_et::order_e::arrayOfStructs: {
            // send to the next partition (+z)
            const size_t transferBytes = sizeof(T) * m_data->zHaloDim * m_data->pitch[setId].z;


            if (setId != ndevs - 1) {

                dCell src_idx(0, 0, field_compute[setId].dim().z /*+m_data->zHaloDim - m_data->zHaloDim*/);
                T*    src = field_compute[setId].mem() + field_compute[setId].elPitch(src_idx);

                dCell dst_idx(0, 0, 0);
                T*    dst = field_compute[setId + 1].mem() + +field_compute[setId + 1].elPitch(dst_idx);

                if (m_data->devType == Neon::DeviceType::CPU) {
                    std::memcpy(dst, src, transferBytes);
                } else if (m_data->devType == Neon::DeviceType::CUDA) {
                    bk.devSet().template peerTransfer<transferMode_ta>(
                        streamSet,
                        m_data->grid->getDevSet().devId(setId + 1).idx(),  // dst
                        (char*)(dst),
                        m_data->grid->getDevSet().devId(setId).idx(),  // src
                        (char*)(src),
                        transferBytes);
                } else {
                    NEON_THROW_UNSUPPORTED_OPERATION("dFieldDev_t::haloUpdate() unsupported device.");
                }
            }

            // send to the previous partition (-z)
            if (setId != 0) {
                dCell src_idx(0, 0, m_data->zHaloDim);
                T*    src = field_compute[setId].mem() + field_compute[setId].elPitch(src_idx);

                dCell dst_idx(0, 0, field_compute[setId - 1].dim().z + m_data->zHaloDim);
                T*    dst = field_compute[setId - 1].mem() + +field_compute[setId - 1].elPitch(dst_idx);

                if (m_data->devType == Neon::DeviceType::CPU) {
                    std::memcpy(dst, src, transferBytes);
                } else if (m_data->devType == Neon::DeviceType::CUDA) {
                    bk.devSet().template peerTransfer<transferMode_ta>(
                        streamSet,
                        m_data->grid->getDevSet().devId(setId - 1).idx(),  // dst
                        (char*)(dst),
                        m_data->grid->getDevSet().devId(setId).idx(),  // src
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
            for (card = 0; card < m_data->cardinality; card++) {

                size_t transferBytes = sizeof(T) * m_data->zHaloDim * m_data->pitch[setId].z;

                if (cardIdx != -1) {
                    card = cardIdx;
                    transferBytes /= m_data->cardinality;
                }

                // send to the next partition (+z)
                if (setId != ndevs - 1) {
                    dCell src_idx(0, 0, field_compute[setId].dim().z /*+m_data->zHaloDim - m_data->zHaloDim*/);
                    T*    src = field_compute[setId].mem() + field_compute[setId].elPitch(src_idx, card);

                    dCell dst_idx(0, 0, 0);
                    T*    dst = field_compute[setId + 1].mem() + +field_compute[setId + 1].elPitch(dst_idx, card);

                    if (m_data->devType == Neon::DeviceType::CPU) {
                        std::memcpy(dst, src, transferBytes);
                    } else if (m_data->devType == Neon::DeviceType::CUDA) {
                        bk.devSet().template peerTransfer<transferMode_ta>(
                            streamSet,
                            m_data->grid->getDevSet().devId(setId + 1).idx(),  // dst
                            (char*)(dst),
                            m_data->grid->getDevSet().devId(setId).idx(),  // src
                            (char*)(src),
                            transferBytes);
                    } else {
                        NEON_THROW_UNSUPPORTED_OPERATION("dFieldDev_t::haloUpdate() unsupported device.");
                    }
                }

                // send to the previous partition (-z)
                if (setId != 0) {
                    dCell src_idx(0, 0, m_data->zHaloDim);
                    T*    src = field_compute[setId].mem() + field_compute[setId].elPitch(src_idx, card);

                    dCell dst_idx(0, 0, field_compute[setId - 1].dim().z + m_data->zHaloDim);
                    T*    dst = field_compute[setId - 1].mem() + +field_compute[setId - 1].elPitch(dst_idx, card);

                    if (m_data->devType == Neon::DeviceType::CPU) {
                        std::memcpy(dst, src, transferBytes);
                    } else if (m_data->devType == Neon::DeviceType::CUDA) {
                        bk.devSet().template peerTransfer<transferMode_ta>(
                            streamSet,
                            m_data->grid->getDevSet().devId(setId - 1).idx(),  // dst
                            (char*)(dst),
                            m_data->grid->getDevSet().devId(setId).idx(),  // src
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

template <typename T, int C>
auto dFieldDev<T, C>::forEach(std::function<void(bool,
                                                 const Neon::index_3d&,
                                                 const int32_t&,
                                                 T&)> fun) -> void
{
    if (m_data->devType != Neon::DeviceType::CPU) {
        NeonException exc("dFieldDev");
        exc << "forEach operation can not be run on a GPU field.";
        NEON_THROW(exc);
    }

    Neon::index_3d dim = m_data->grid->domainDim();
    for (int v = 0; v < m_data->cardinality; v++) {
        for (int z = 0; z < dim.z; z++) {
            for (int y = 0; y < dim.y; y++) {
                for (int x = 0; x < dim.x; x++) {
                    Neon::index_3d idx(x, y, z);
                    fun(true, idx, v, eRef(idx, v));
                }
            }
        }
    }
}

template <typename T, int C>
auto dFieldDev<T, C>::forEach(std::function<void(bool,
                                                 const Neon::index_3d&,
                                                 const int32_t&,
                                                 const T&)> fun) const -> void
{
    if (m_data->devType != Neon::DeviceType::CPU) {
        NeonException exc("dFieldDev");
        exc << "forEach operation can not be run on a GPU field.";
        NEON_THROW(exc);
    }

    Neon::index_3d dim = m_data->grid->domainDim();
    for (int v = 0; v < m_data->cardinality; v++) {
        for (int z = 0; z < dim.z; z++) {
            for (int y = 0; y < dim.y; y++) {
                for (int x = 0; x < dim.x; x++) {

                    Neon::index_3d idx(x, y, z);
                    fun(true, idx, v, eRef(idx, v));
                }
            }
        }
    }
}

template <typename T, int C>
auto dFieldDev<T, C>::forEachActive(std::function<void(const Neon::index_3d&,
                                                       const int&,
                                                       T&)> fun)
    -> void
{
    this->forEach([&](bool                  isActive,
                      const Neon::index_3d& idx,
                      const int&            cardinality,
                      T&                    val) {
        if (!isActive) {
            return;
        }
        fun(idx, cardinality, val);
    });
}

template <typename T, int C>
template <typename exportReal_ta>
auto dFieldDev<T, C>::ioToDense(Neon::memLayout_et::order_e order,
                                exportReal_ta*              dense) const
    -> void
{
    forEach([&](bool, const Neon::index_3d& idx, const int32_t& cardIdx, const T& val) {
        size_t         pitch;
        Neon::index_3d domain = m_data->grid->domainDim();
        switch (order) {
            case Neon::memLayout_et::order_e::structOfArrays: {
                pitch = idx.x +
                        idx.y * domain.x +
                        idx.z * domain.x * domain.y +
                        cardIdx * domain.x * domain.y * domain.z;
                break;
            }
            case Neon::memLayout_et::order_e::arrayOfStructs: {
                pitch = cardIdx +
                        idx.x * m_data->cardinality +
                        idx.y * m_data->cardinality * domain.x +
                        idx.z * m_data->cardinality * domain.x * domain.y;
                break;
            }
            default: {
                NEON_THROW_UNSUPPORTED_OPTION();
            }
        }
        dense[pitch] = exportReal_ta(val);
    });
}

template <typename T, int C>
template <typename exportReal_ta>
auto dFieldDev<T, C>::ioToDense(Neon::memLayout_et::order_e order) const
    -> std::shared_ptr<exportReal_ta>
{
    std::shared_ptr<exportReal_ta> dense(new exportReal_ta[m_data->grid->domainDim().template rMulTyped<size_t>() * m_data->cardinality],
                                         std::default_delete<exportReal_ta[]>());
    ioToDense(order, dense.get());
    return dense;
}

template <typename T, int C>
template <typename exportReal_ta>
auto dFieldDev<T, C>::ioFromDense(Neon::memLayout_et::order_e order,
                                  const exportReal_ta*        dense,
                                  exportReal_ta /* inactiveValue*/) -> void
{

    forEachActive([&](const Neon::index_3d& idx, const int& cardIdx, T& val) {
        size_t         pitch;
        Neon::index_3d domain = m_data->grid->domainDim();
        switch (order) {
            case Neon::memLayout_et::order_e::structOfArrays: {
                pitch = idx.x +
                        idx.y * domain.x +
                        idx.z * domain.x * domain.y +
                        cardIdx * domain.x * domain.y * domain.z;
                break;
            }
            case Neon::memLayout_et::order_e::arrayOfStructs: {
                pitch = cardIdx +
                        idx.x * m_data->cardinality +
                        idx.y * m_data->cardinality * domain.x +
                        idx.z * m_data->cardinality * domain.x * domain.y;
                break;
            }
            default: {
                NEON_THROW_UNSUPPORTED_OPTION();
            }
        }
        T denseVal = T(dense[pitch]);
        val = denseVal;
    });
}

template <typename T, int C>
auto dFieldDev<T, C>::getPartition(Neon::DeviceType,
                                   Neon::SetIdx   idx,
                                   Neon::DataView dataView)
    -> local_t&
{
    return m_data->dFieldComputeSetByView[static_cast<int>(dataView)][idx.idx()];
}

template <typename T, int C>
auto dFieldDev<T, C>::getPartition(Neon::DeviceType,
                                   Neon::SetIdx   idx,
                                   Neon::DataView dataView) const
    -> const local_t&
{
    return m_data->dFieldComputeSetByView[static_cast<int>(dataView)][idx.idx()];
}
template <typename T, int C>
auto dFieldDev<T, C>::haloStatus() const -> Neon::domain::haloStatus_et::e
{
    return m_data->haloStatus;
}

template <typename T, int C>
auto dFieldDev<T, C>::h_init(const Neon::set::DataSet<Neon::index_3d>& dims,
                             const grid_t&                             grid)
    -> void
{
    for (auto& dv : Neon::DataViewUtil::validOptions()) {
        int dv_id = static_cast<int>(dv);
        m_data->dFieldComputeSetByView[dv_id] = grid.getBackend().devSet().template newDataSet<local_t>();

        switch (dv) {
            case Neon::DataView::STANDARD: {
                if (grid.getBackend().devSet().setCardinality() == 1 ||
                    (grid.getBackend().devSet().setCardinality() > 1 && m_data->memOrder == Neon::memLayout_et::order_e::arrayOfStructs)) {
                    m_data->startIDByView[dv_id].resize(1);
                    m_data->startIDByView[dv_id][0] = grid.getBackend().devSet().template newDataSet<int>();

                    m_data->nElementsByView[dv_id].resize(1);
                    m_data->nElementsByView[dv_id][0] = grid.getBackend().devSet().template newDataSet<int>();
                } else {
                    m_data->startIDByView[dv_id].resize(m_data->cardinality);
                    m_data->nElementsByView[dv_id].resize(m_data->cardinality);
                    for (int i = 0; i < m_data->cardinality; ++i) {
                        m_data->startIDByView[dv_id][i] = grid.getBackend().devSet().template newDataSet<int>();
                        m_data->nElementsByView[dv_id][i] = grid.getBackend().devSet().template newDataSet<int>();
                    }
                }
                break;
            }
            case Neon::DataView::INTERNAL: {

                if (grid.getBackend().devSet().setCardinality() > 1) {
                    switch (m_data->memOrder) {
                        case Neon::memLayout_et::order_e::structOfArrays: {
                            m_data->startIDByView[dv_id].resize(m_data->cardinality);
                            m_data->nElementsByView[dv_id].resize(m_data->cardinality);
                            for (int i = 0; i < m_data->cardinality; ++i) {
                                m_data->startIDByView[dv_id][i] = grid.getBackend().devSet().template newDataSet<int>();
                                m_data->nElementsByView[dv_id][i] = grid.getBackend().devSet().template newDataSet<int>();
                            }
                            break;
                        }
                        case Neon::memLayout_et::order_e::arrayOfStructs: {
                            m_data->startIDByView[dv_id].resize(1);
                            m_data->startIDByView[dv_id][0] = grid.getBackend().devSet().template newDataSet<int>();
                            m_data->nElementsByView[dv_id].resize(1);
                            m_data->nElementsByView[dv_id][0] = grid.getBackend().devSet().template newDataSet<int>();
                            break;
                        }
                    }
                }
                break;
            }
            case Neon::DataView::BOUNDARY: {

                if (grid.getBackend().devSet().setCardinality() > 1) {
                    switch (m_data->memOrder) {
                        case Neon::memLayout_et::order_e::structOfArrays: {
                            m_data->startIDByView[dv_id].resize(2 * m_data->cardinality);
                            m_data->nElementsByView[dv_id].resize(2 * m_data->cardinality);
                            for (int i = 0; i < 2 * m_data->cardinality; ++i) {
                                m_data->startIDByView[dv_id][i] = grid.getBackend().devSet().template newDataSet<int>();
                                m_data->nElementsByView[dv_id][i] = grid.getBackend().devSet().template newDataSet<int>();
                            }
                            break;
                        }
                        case Neon::memLayout_et::order_e::arrayOfStructs: {
                            m_data->startIDByView[dv_id].resize(2);
                            m_data->startIDByView[dv_id][0] = grid.getBackend().devSet().template newDataSet<int>();
                            m_data->startIDByView[dv_id][1] = grid.getBackend().devSet().template newDataSet<int>();
                            m_data->nElementsByView[dv_id].resize(2);
                            m_data->nElementsByView[dv_id][0] = grid.getBackend().devSet().template newDataSet<int>();
                            m_data->nElementsByView[dv_id][1] = grid.getBackend().devSet().template newDataSet<int>();
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
    }

    const int haloRadius = haloStatus() == Neon::domain::haloStatus_et::ON ? m_data->zHaloDim : 0;

    Neon::set::DataSet<uint64_t> dims_flat = grid.getBackend().devSet().template newDataSet<uint64_t>();

    Neon::set::DataSet<uint64_t> stencil_dim = grid.getBackend().devSet().template newDataSet<uint64_t>();

    uint64_t stencil_num_ngh = uint64_t(grid.getStencil().neighbours().size());
    for (int64_t i = 0; i < dims.size(); ++i) {
        dims_flat[i] = dims[i].x * dims[i].y * (dims[i].z + 2 * haloRadius) * m_data->cardinality;
        stencil_dim[i] = stencil_num_ngh;
    }

    m_data->memory = grid.getBackend().devSet().template newMemDevSet<T>(m_data->devType,
                                                                         m_data->memAlloc,
                                                                         dims_flat);

    m_data->stencilNghIndex = grid.getBackend().devSet().template newMemDevSet<typename field_t::ngh_idx>(m_data->devType,
                                                                                                          m_data->memAlloc,
                                                                                                          stencil_dim);

    Neon::sys::MemDevice<typename field_t::ngh_idx> stencil_cpu(Neon::DeviceType::CPU, 0, Neon::Allocator::MALLOC, stencil_num_ngh);

    for (uint64_t s = 0; s < stencil_cpu.nElements(); ++s) {
        stencil_cpu.mem()[s].x = static_cast<typename field_t::ngh_idx::Integer>(grid.getStencil().neighbours()[s].x);
        stencil_cpu.mem()[s].y = static_cast<typename field_t::ngh_idx::Integer>(grid.getStencil().neighbours()[s].y);
        stencil_cpu.mem()[s].z = static_cast<typename field_t::ngh_idx::Integer>(grid.getStencil().neighbours()[s].z);
    }

    m_data->pitch = grid.getBackend().devSet().template newDataSet<Neon::size_4d>();

    index_3d origin_accumulator(0, 0, 0);
    for (uint64_t i = 0; i < uint64_t(grid.getBackend().devSet().setCardinality()); ++i) {

        // compute pitch
        if (m_data->cardinality == 1) {
            m_data->pitch[i].x = 1;
            m_data->pitch[i].y = m_data->pitch[i].x * dims[i].x;
            m_data->pitch[i].z = m_data->pitch[i].y * dims[i].y;
            m_data->pitch[i].w = 0;
        } else {
            switch (m_data->memOrder) {
                case Neon::memLayout_et::order_e::structOfArrays: {
                    m_data->pitch[i].x = 1;
                    m_data->pitch[i].y = m_data->pitch[i].x * dims[i].x;
                    m_data->pitch[i].z = m_data->pitch[i].y * dims[i].y;
                    m_data->pitch[i].w = m_data->pitch[i].z * (dims[i].z + 2 * haloRadius);
                    break;
                }
                case Neon::memLayout_et::order_e::arrayOfStructs: {
                    m_data->pitch[i].x = m_data->cardinality;
                    m_data->pitch[i].y = m_data->pitch[i].x * dims[i].x;
                    m_data->pitch[i].z = m_data->pitch[i].y * dims[i].y;
                    m_data->pitch[i].w = 1;
                    break;
                }
            }
        }


        T* mem = m_data->memory.mem(int(i));

        m_data->stencilNghIndex.getMemDev(int64_t(i)).copyFrom(stencil_cpu);

        typename field_t::ngh_idx* stencilNgh = m_data->stencilNghIndex.mem(int(i));

        const index_3d dim = dims[i];

        const int boundayRadius = m_data->zHaloDim;
        index_3d  origin = origin_accumulator;
        for (auto& dv : Neon::DataViewUtil::validOptions()) {
            int dv_id = static_cast<int>(dv);
            m_data->dFieldComputeSetByView[dv_id][i] =
                dPartition<T, C>(dv, mem, dim, haloRadius, boundayRadius,
                                 m_data->pitch[i], int(i),
                                 origin, m_data->cardinality,
                                 m_data->grid->getDimension(), stencilNgh);

            switch (dv) {
                case Neon::DataView::STANDARD: {
                    if (grid.getBackend().devSet().setCardinality() == 1) {
                        m_data->startIDByView[dv_id][0][i] = 0;
                        m_data->nElementsByView[dv_id][0][i] = int(dims_flat[i]);
                    } else {
                        switch (m_data->memOrder) {
                            case Neon::memLayout_et::order_e::structOfArrays: {
                                for (int c = 0; c < m_data->cardinality; ++c) {
                                    m_data->startIDByView[dv_id][c][i] =
                                        c * dims[i].x * dims[i].y * (dims[i].z + 2 * haloRadius) +
                                        dims[i].x * dims[i].y * haloRadius;
                                    m_data->nElementsByView[dv_id][c][i] = dims[i].x * dims[i].y * dims[i].z;
                                }
                                break;
                            }
                            case Neon::memLayout_et::order_e::arrayOfStructs: {
                                m_data->startIDByView[dv_id][0][i] = dims[i].x * dims[i].y * haloRadius * m_data->cardinality;
                                m_data->nElementsByView[dv_id][0][i] = dims[i].x * dims[i].y * dims[i].z * m_data->cardinality;
                                break;
                            }
                        }
                    }
                    break;
                }
                case Neon::DataView::INTERNAL: {
                    if (grid.getBackend().devSet().setCardinality() > 1) {
                        switch (m_data->memOrder) {
                            case Neon::memLayout_et::order_e::structOfArrays: {
                                for (int c = 0; c < m_data->cardinality; ++c) {
                                    m_data->startIDByView[dv_id][c][i] =
                                        c * dims[i].x * dims[i].y * (dims[i].z + 2 * haloRadius) +
                                        dims[i].x * dims[i].y * (haloRadius + boundayRadius);
                                    m_data->nElementsByView[dv_id][c][i] = dims[i].x * dims[i].y * (dims[i].z - 2 * haloRadius);
                                }
                                break;
                            }
                            case Neon::memLayout_et::order_e::arrayOfStructs: {
                                m_data->startIDByView[dv_id][0][i] = dims[i].x * dims[i].y * (haloRadius + boundayRadius) * m_data->cardinality;
                                m_data->nElementsByView[dv_id][0][i] = dims[i].x * dims[i].y * (dims[i].z - 2 * haloRadius) * m_data->cardinality;
                                break;
                            }
                        }
                    }
                    break;
                }
                case Neon::DataView::BOUNDARY: {
                    if (grid.getBackend().devSet().setCardinality() > 1) {
                        switch (m_data->memOrder) {
                            case Neon::memLayout_et::order_e::structOfArrays: {
                                for (int c = 0; c < m_data->cardinality; ++c) {
                                    // up
                                    m_data->startIDByView[dv_id][2 * c][i] =
                                        c * dims[i].x * dims[i].y * (dims[i].z + 2 * haloRadius) +
                                        dims[i].x * dims[i].y * haloRadius;
                                    m_data->nElementsByView[dv_id][2 * c][i] = dims[i].x * dims[i].y * boundayRadius;

                                    // down
                                    m_data->startIDByView[dv_id][2 * c + 1][i] =
                                        c * dims[i].x * dims[i].y * (dims[i].z + 2 * haloRadius) +
                                        dims[i].x * dims[i].y * (dims[i].z + haloRadius - boundayRadius);
                                    m_data->nElementsByView[dv_id][2 * c + 1][i] = dims[i].x * dims[i].y * boundayRadius;
                                }
                                break;
                            }
                            case Neon::memLayout_et::order_e::arrayOfStructs: {
                                // up
                                m_data->startIDByView[dv_id][0][i] = dims[i].x * dims[i].y * haloRadius * m_data->cardinality;
                                m_data->nElementsByView[dv_id][0][i] = dims[i].x * dims[i].y * boundayRadius * m_data->cardinality;

                                // down
                                m_data->startIDByView[dv_id][1][i] = dims[i].x * dims[i].y * (dims[i].z + haloRadius - boundayRadius) * m_data->cardinality;
                                m_data->nElementsByView[dv_id][1][i] = dims[i].x * dims[i].y * boundayRadius * m_data->cardinality;
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
        }
        origin_accumulator.z += dim.z;

        // Origin calculation only works with z partitions. We check this here
        if (i > 0) {
            if (dims[i].x != dims[i - 1].x || dims[i].y != dims[i - 1].y) {
                NeonException exp("dFieldDev_t");
                exp << " Origin computation only with z partitions.";
                NEON_THROW(exp);
            }
        }
    }
}


template <typename T, int C>
auto dFieldDev<T, C>::convert_to_local(Neon::index_3d& index,
                                       Neon::DataView  dataView) const
    -> int32_t
{
    // since we partition along the z-axis, only the z-component of index will change
    int32_t partition = 0;
    if (m_data->grid->partitions().size() > 1) {
        for (const auto& dim : m_data->grid->partitions()) {
            if (index.z >= dim.z) {
                index.z -= dim.z;
                partition++;
            } else {
                break;
            }
        }

        switch (dataView) {
            case Neon::DataView::STANDARD:
                index.z += m_data->zHaloDim;
                break;
            case Neon::DataView::INTERNAL:
                index.z += 2 * m_data->zHaloDim;
                break;
            case Neon::DataView::BOUNDARY:
                index.z += index.z < m_data->zHaloDim ? 0 : (m_data->dFieldComputeSetByView[static_cast<int>(dataView)][partition].dim().z - 1) + (-1 * m_data->zHaloDim);
                index.z += m_data->zHaloDim;
                break;
            default:
                break;
        }
    }
    return partition;
}

}  // namespace Neon::domain::internal::dGrid