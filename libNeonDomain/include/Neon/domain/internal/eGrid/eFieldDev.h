#pragma once

#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"
#include "Neon/domain/internal/eGrid/eInternals/builder/dsBuilderCommon.h"
#include "Neon/domain/internal/eGrid/eInternals/builder/dsFrame.h"
#include "Neon/set/DevSet.h"
#include "Neon/set/HuOptions.h"
#include "Neon/set/memory/memSet.h"
#include "Neon/sys/memory/memConf.h"
#include "ePartition.h"

namespace Neon::domain::internal::eGrid {

class eGrid;


template <typename T_ta,
          int cardinality_ta>
class eField;

template <typename T_ta,
          int cardinality_ta = 0>
class eFieldDevice_t
{
   public:
    // FRIENDS
    friend eGrid;
    friend eField<T_ta, cardinality_ta>;

   public:
    // Neon aliases
    using self_t = eFieldDevice_t<T_ta, cardinality_ta>;
    using element_t = T_ta; /**< Basic type of the field */
    using Cell = eCell;

    // GRID, FIELDS and LOCAL aliases
    using grid_t = eGrid;                                    /**< Type of the associated grid */
    using field_t = eField<T_ta, cardinality_ta>;            /**< Type of the associated field */
    using fieldDev_t = eFieldDevice_t<T_ta, cardinality_ta>; /**< Type of the associated fieldMirror */
    using local_t = ePartition<T_ta, cardinality_ta>;        /**< Type of the associated local_t */

   private:
    // ALIAS
    using LocalIndexingInfo_t = internals::LocalIndexingInfo_t;

    struct data_t
    {
        // INPUT
        Neon::DeviceType                      devType = {Neon::DeviceType::NONE};
        Neon::set::DevSet                     devSet;
        int                                   cardinality;
        element_t                             inactiveValue;
        std::shared_ptr<internals::dsFrame_t> frame_shp;
        Neon::domain::haloStatus_et::e        haloStatus;
        Neon::memLayout_et::order_e           memOrder;
        Neon::sys::MemAlignment               memAlignment;
        Neon::memLayout_et::padding_e         memPadding;

        // COMPUTED
        Neon::set::MemDevSet<element_t>                                      memoryStorage;
        Neon::set::DataSet<element_t*>                                       userPointersSet;
        std::array<Neon::set::DataSet<local_t>, Neon::DataViewUtil::nConfig> localSetByView;

        std::shared_ptr<grid_t> grid;

        std::array<
            std::array<
                std::vector<Neon::set::Transfer>,
                Neon::set::TransferSemanticUtils::nOptions>,
            Neon::set::TransferModeUtils::nOptions>
            m_haloUpdateInfo;
    };

   private:
    // MEMBERS
    std::shared_ptr<data_t> m_data;

   public:
    // CONSTRUCTORS AND ASSIGNMENT OPERATORS
    eFieldDevice_t(const eGrid&                          grid,
                   Neon::sys::memConf_t                  memConf,
                   const Neon::set::DevSet&              devSet,
                   int                                   cardinality,
                   T_ta                                  inactiveValue,
                   std::shared_ptr<internals::dsFrame_t> frame_shp,
                   Neon::domain::haloStatus_et::e        haloStatus)
    {
        m_data = std::make_shared<data_t>();
        m_data->grid = std::make_shared<grid_t>(grid);

        {  // INPUT
            m_data->devType = memConf.devEt();
            m_data->devSet = devSet;
            m_data->cardinality = cardinality;
            m_data->inactiveValue = inactiveValue;
            m_data->frame_shp = frame_shp;
            m_data->haloStatus = haloStatus;
            m_data->memOrder = memConf.order();
            m_data->memAlignment = memConf.alignment();
            m_data->memPadding = memConf.padding();
        }

        {  // COMPUTING
            if (memConf.allocEt() != Neon::Allocator::NULL_MEM) {
                h_initRawMem(memConf.allocEt());
                h_initFieldCompute();
            } else {
                for (int DataViewIdx = 0; DataViewIdx < Neon::DataViewUtil::nConfig; DataViewIdx++) {
                    m_data->localSetByView[DataViewIdx] = devSet.newDataSet<local_t>();
                }
                m_data->userPointersSet = devSet.newDataSet<element_t*>();
                for (SetIdx setIdx = 0; setIdx < devSet.setCardinality(); setIdx++) {
                    for (int DataViewIdx = 0; DataViewIdx < Neon::DataViewUtil::nConfig; DataViewIdx++) {
                        m_data->localSetByView[DataViewIdx][setIdx] = local_t();
                    }
                    m_data->userPointersSet[setIdx] = nullptr;
                }
            }
        }

        {  // COMPUTING HALO DATA
            if (memConf.allocEt() != Neon::Allocator::NULL_MEM && haloStatus == Neon::domain::haloStatus_et::ON) {
                {
                    for (auto mode : {Neon::set::TransferMode::get, Neon::set::TransferMode::put}) {

                        {  // (GET,PUT), FORWARD, GRID
                            const auto           structure = Neon::set::TransferSemantic::grid;
                            auto&                transfers = h_haloUpdateInfo(mode, structure);
                            Neon::set::HuOptions huOptions(mode, transfers, structure);
                            this->haloUpdate__(m_data->grid->getBackend(), huOptions);
                        }
                        {  // (GET,PUT), FORWARD, LATTICE
                           //                            const auto             structure = Neon::set::Transfer_t::Structure::lattice;
                           //                            auto&                  transfers = h_haloUpdateInfo(mode, structure, direction);
                           //                            Neon::set::HuOptions_t huOptions(mode, transfers, structure, direction);
                           //                            this->haloUpdateLattice(huOptions);
                        }
                    }
                }
            }
        }
    }

    auto h_haloUpdateInfo(Neon::set::TransferMode     mode,
                          Neon::set::TransferSemantic structure)
        -> std::vector<Neon::set::Transfer>&
    {
        return m_data->m_haloUpdateInfo[static_cast<int>(mode)]
                                       [static_cast<int>(structure)];
    }

   public:
    /**
     *
     */
    eFieldDevice_t()
    {
        m_data = std::make_shared<data_t>();
        m_data->grid = std::shared_ptr<grid_t>();
    }

    /**
     *
     */
    eFieldDevice_t(const eFieldDevice_t& other)
    {
        m_data = other.m_data;
    }

    /**
     *
     */
    eFieldDevice_t(eFieldDevice_t&& other)
    {
        m_data = std::move(other.m_data);
        other.m_data = std::shared_ptr<data_t>();
    }

    /**
     *
     */
    eFieldDevice_t& operator=(const eFieldDevice_t& other)
    {
        m_data = other.m_data;
        return *this;
    }

    /**
     *
     */
    eFieldDevice_t& operator=(eFieldDevice_t&& other)
    {
        m_data = std::move(other.m_data);
        other.m_data = std::shared_ptr<data_t>();
        return *this;
    }

    /**
     * Returns a unique identifier for this type of DataSet
     * @return
     */
    auto uid() const -> Neon::set::MultiDeviceObjectUid
    {
        void*                           addr = static_cast<void*>(m_data.get());
        Neon::set::MultiDeviceObjectUid uidRes = (size_t)addr;
        return uidRes;
    }

    auto grid() -> grid_t&
    {
        return *(m_data->grid.get());
    }

    auto grid() const -> const grid_t&
    {
        return *(m_data->grid.get());
    }
    /**
     * Returns device set
     */
    const Neon::set::DevSet& devSet() const
    {
        return m_data->devSet;
    }

    auto cardinality() const -> int
    {
        return m_data->cardinality;
    }

    /**
     * Return padding configuration for this object.
     */
    const Neon::memLayout_et::padding_e& padding() const
    {
        return m_data->memPadding;
    }

   public:
    /**
     * Returns the type of device
     */
    auto devType()
        const
        -> Neon::DeviceType
    {
        return m_data->devType;
    }

    /**
     * Get the value at given index
     * @param idx 3D index of the voxel
     * @param cardinality Cardinality of the value
     * @return Mutable reference to value
     */
    auto eRef(const Neon::index_3d& idx, const int cardinality)
        -> T_ta&
    {
        if (m_data->devType != Neon::DeviceType::CPU) {
            NeonException exc("eField");
            exc << "eRef operation cannot be run on a GPU field.";
            NEON_THROW(exc);
        }

        const auto& GtoL = m_data->frame_shp->globalToLocal();
        const auto& info = GtoL.elRef(idx);
        if (!(info.isActive())) {
            NeonException exc("eField");
            exc << "Cannot return reference to inactive value. Use eActive()"
                   "to check if the voxel is active before attempting to modify it";
            NEON_THROW(exc);
        }
        const auto& prtId = info.getPrtIdx();
        const auto& localIdx = info.getLocalIdx();

        auto  memStorage = m_data->memoryStorage.template get<Neon::Access::readWrite>(prtId);
        T_ta& el = memStorage.template elRef<Neon::Access::readWrite>(localIdx, cardinality);
        return el;
    }


    /**
     * Get the value at given index
     * @param idx 3D index of the voxel
     * @param cardinality Cardinality of the value
     * @return Const reference to value
     */
    auto eRef(const Neon::index_3d& idx, const int& cardinality)
        const
        -> const T_ta&
    {
        if (m_data->devType != Neon::DeviceType::CPU) {
            NeonException exc("eField");
            exc << "eRef operation cannot be run on a GPU field.";
            NEON_THROW(exc);
        }

        const auto& GtoL = m_data->frame_shp->globalToLocal();
        const auto& info = GtoL.elRef(idx);
        if (!(info.isActive())) {
            return m_data->inactiveValue;
        }
        const auto& prtId = info.getPrtIdx();
        const auto& localIdx = info.getLocalIdx();

        auto  memStorage = m_data->memoryStorage.template get<Neon::Access::readWrite>(prtId);
        T_ta& el = memStorage.template elRef<Neon::Access::readWrite>(localIdx, cardinality);
        return el;
    }

    /**
     * Returns whether the voxel is active
     * @param idx 3D index of the voxel
     * @return Whether active
     */
    auto eActive(const Neon::index_3d& idx) const
        -> bool
    {
        if (m_data->devType != Neon::DeviceType::CPU) {
            NeonException exc("eField");
            exc << "eActive cannot be called on a GPU field.";
            NEON_THROW(exc);
        }
        const auto& GtoL = m_data->frame_shp->globalToLocal();
        const auto& info = GtoL.elRef(idx);
        return info.isActive();
    }

    /**
     * Updating halo for all cardinality of the field at once.
     */
    template <Neon::set::TransferMode transferMode_ta>
    [[deprecated]] auto haloUpdate(const Neon::Backend& bk, bool startWithBarrier = true, int streamSetIdx = 0)
        -> void
    {
        /**
         * No halo helpUpdate operation can be performed on a filed where halo was not activated.
         */
        if (m_data->haloStatus != Neon::domain::haloStatus_et::ON) {
            NEON_THROW_UNSUPPORTED_OPERATION("Halo support was not activated for this field.");
        }

        /**
         * We don't need any helpUpdate if the number of devices is one.
         */
        if (m_data->devSet.setCardinality() == 1) {
            return;
        }

        switch (m_data->memOrder) {
            case Neon::memLayout_et::order_e::structOfArrays: {
                {
                    /*
                     * this sync goes before the following loop.
                     * This is a complete barrier over the stream and
                     * it can not be put inside the loop
                     */
                    if (startWithBarrier) {
                        bk.sync(streamSetIdx);
                    }
                }
                const int ndevs = m_data->devSet.setCardinality();
#pragma omp parallel for num_threads(ndevs) default(shared)
                for (int gpuIdx = 0; gpuIdx < ndevs; gpuIdx++) {
                    for (int cardIdx = 0; cardIdx < self().cardinality(); cardIdx++) {
                        constexpr bool latticeMode = false;
                        constexpr bool revertLattice = false;

                        h_huSoAByCardSingleDev<transferMode_ta>(gpuIdx, bk, cardIdx, streamSetIdx, latticeMode, revertLattice);
                    }
                }
                break;
            }

            case Neon::memLayout_et::order_e::arrayOfStructs: {
                {
                    /*
                     * this sync goes before the following loop.
                     * This is a complete barrier over the stream and
                     * it can not be put inside the loop
                     */
                    if (startWithBarrier) {
                        bk.sync(streamSetIdx);
                    }
                }
                {
                    const int nDevs = m_data->devSet.setCardinality();
#pragma omp parallel for num_threads(nDevs) default(shared)
                    for (int gpuIdx = 0; gpuIdx < nDevs; gpuIdx++) {
                        // ARRAYS OF STRUCTURES
                        // -> for each voxel the components are stored contiguously
                        // -> We follow the same configuration either for Lattice or Standard
                        // -> We use the Standard as reference

                        const int            dstIdx = gpuIdx;
                        LocalIndexingInfo_t& dst = m_data->frame_shp->template localIndexingInfo<Neon::Access::readWrite>(dstIdx);

                        const std::array<internals::partition_idx, ComDirection_e::COM_NUM>
                            srcIdx = {dst.nghIdx(ComDirection_e::COM_DW),
                                      dst.nghIdx(ComDirection_e::COM_UP)};

                        for (const auto& comDirection : {ComDirection_e::COM_DW, ComDirection_e::COM_UP}) {
                            /**
                             * In terms of elements we need to a number of values
                             * equivalent to the number of elements (voxels) by the number of values per element (=cardinality)
                             */
                            count_t      transferEl = dst.remoteBdrCount(comDirection) * self().cardinality();
                            Cell::Offset remoteOffset = dst.remoteBdrOff(comDirection);
                            Cell::Offset localOffset = dst.ghostOff(comDirection);

                            T_ta*       dstMem = m_data->memoryStorage.mem(dstIdx);
                            const T_ta* srcMem = m_data->memoryStorage.mem(srcIdx[comDirection]);

                            T_ta*       dstBuf = dstMem + localOffset;
                            const T_ta* srcBuf = srcMem + remoteOffset;

                            // For partition 0, communication are only in the UP direction
                            if (gpuIdx == 0 && comDirection == ComDirection_e::COM_DW)
                                continue;

                            // For the last, communication are only in the DW direction
                            if (gpuIdx == (nDevs - 1) && comDirection == ComDirection_e::COM_UP)
                                continue;

                            assert(transferEl > -1);
                            if (transferEl > 0) {
                                switch (m_data->devType) {
                                    case Neon::DeviceType::CPU: {
                                        std::memcpy(dstBuf, srcBuf, transferEl * sizeof(T_ta));
                                        break;
                                    }
                                    case Neon::DeviceType::CUDA: {
                                        // auto& devSet = m_data->devSet;
                                        auto& streamSet = bk.streamSet(streamSetIdx);
                                        m_data->devSet.template peerTransfer<transferMode_ta>(streamSet,
                                                                                              dstIdx, (char*)dstBuf,
                                                                                              srcIdx[comDirection], (char*)srcBuf,
                                                                                              transferEl * sizeof(T_ta));
                                        break;
                                    }
                                    default: {
                                        NEON_THROW_UNSUPPORTED_OPERATION("");
                                    }
                                }
                            }
                        }
                    }
                }
                break;
            }
        }
    }

    /**
     * Updating halo for all cardinality of the field at once.
     */
    auto haloUpdate__(const Neon::Backend&  bk,
                      Neon::set::HuOptions& opt) const
        -> void
    {

        // No halo update operation can be performed on a filed where halo was not activated.
        if (m_data->haloStatus != Neon::domain::haloStatus_et::ON) {
            NEON_THROW_UNSUPPORTED_OPERATION("Halo support was not activated for this field.");
        }

        // We don't need any update if the number of devices is one.
        if (m_data->devSet.setCardinality() == 1) {
            return;
        }

        // If we are in execution mode, than we use on omp thread per device
        // If we are in storeInfo mode, than we use only one thread, which will
        // insert information on the transfer vectors sequentially
        const bool isExecuteMode = opt.isExecuteMode();
        const int  ompNDevs = isExecuteMode ? m_data->devSet.setCardinality() : 1;
        const int  nDevs = m_data->devSet.setCardinality();

        // This sync goes before the following loop.
        // This is a complete barrier over the stream and
        // it can not be put inside the loop
        // The sync is done only if opt.isExecuteMode() == true
        if (opt.startWithBarrier() && opt.isExecuteMode()) {
            bk.sync(opt.streamSetIdx());
        }

        // Different behaviour base on the data layout
        switch (m_data->memOrder) {
            case Neon::memLayout_et::order_e::structOfArrays: {
#pragma omp parallel for num_threads(ompNDevs) default(shared)
                for (int setIdx = 0; setIdx < nDevs; setIdx++) {
                    for (int cardIdx = 0; cardIdx < self().cardinality(); cardIdx++) {
                        constexpr auto structure = Neon::set::TransferSemantic::grid;

                        auto& peerTransferOpt = opt.getPeerTransferOpt(bk);
                        h_huSoAByCardSingleDevFwd(peerTransferOpt,
                                                  setIdx,
                                                  cardIdx,
                                                  structure);
                    }
                }
                break;
            }
            //------------------------------------------------
            case Neon::memLayout_et::order_e::arrayOfStructs: {

#pragma omp parallel for num_threads(ompNDevs) default(shared)
                for (int gpuIdx = 0; gpuIdx < nDevs; gpuIdx++) {
                    // ARRAYS OF STRUCTURES
                    // -> for each voxel the components are stored contiguously
                    // -> We follow the same configuration either for Lattice or Standard
                    // -> We use the Standard as reference

                    const int            dstIdx = gpuIdx;
                    LocalIndexingInfo_t& dst = m_data->frame_shp->template localIndexingInfo<Neon::Access::readWrite>(dstIdx);

                    const std::array<internals::partition_idx, ComDirection_e::COM_NUM>
                        srcIdx = {dst.nghIdx(ComDirection_e::COM_DW),
                                  dst.nghIdx(ComDirection_e::COM_UP)};

                    for (const auto& comDirection : {ComDirection_e::COM_DW,
                                                     ComDirection_e::COM_UP}) {

                        // In terms of elements we need to a number of values
                        // equivalent to the number of elements (voxels)
                        // by the number of values per element (=cardinality)
                        count_t      transferEl = dst.remoteBdrCount(comDirection) * self().cardinality();
                        Cell::Offset remoteOffset = dst.remoteBdrOff(comDirection);
                        Cell::Offset localOffset = dst.ghostOff(comDirection);

                        T_ta*       dstMem = m_data->memoryStorage.mem(dstIdx);
                        const T_ta* srcMem = m_data->memoryStorage.mem(srcIdx[comDirection]);

                        T_ta*       dstBuf = dstMem + localOffset;
                        const T_ta* srcBuf = srcMem + remoteOffset;

                        // For partition 0, communication are only in the UP direction
                        if (gpuIdx == 0 && comDirection == ComDirection_e::COM_DW)
                            continue;

                        // For the last, communication are only in the DW direction
                        if (gpuIdx == (nDevs - 1) && comDirection == ComDirection_e::COM_UP)
                            continue;

                        assert(transferEl > -1);
                        if (transferEl > 0) {
                            // auto&                             streamSet = bk.streamSet(opt.streamSetIdx());
                            Neon::set::Transfer::Endpoint_t srcEndPoint(srcIdx[comDirection], (void*)srcBuf);
                            Neon::set::Transfer::Endpoint_t dstEndPoint(dstIdx, (void*)dstBuf);

                            Neon::set::Transfer transfer(opt.transferMode(),
                                                         dstEndPoint,
                                                         srcEndPoint,
                                                         transferEl * sizeof(T_ta));

                            m_data->devSet.peerTransfer(opt.getPeerTransferOpt(bk), transfer);
                        }
                    }
                }
            }
        }  //------------------------------------------------
    }

    /**
     * Do Update by cardinality
     * @param bk
     * @param cardIdx
     * @param startWithBarrier
     * @param streamSetIdx
     */
    template <Neon::set::TransferMode transferMode_ta>
    auto haloUpdateByCardinality(const Neon::Backend& bk,
                                 const int            cardIdx,
                                 bool                 startWithBarrier = true,
                                 int                  streamSetIdx = Neon::Backend::mainStreamIdx)
        -> void
    {
        if (m_data->haloStatus !=
            Neon::domain::haloStatus_et::ON) {
            NEON_THROW_UNSUPPORTED_OPERATION("Halo support was not activated for this field.");
        }

        if (m_data->devSet.setCardinality() == 1) {
            // With one single device we don't need to actually update any halos
            return;
        }

        if ((m_data->memOrder != Neon::memLayout_et::order_e::structOfArrays) && self().cardinality() != 1) {
            NEON_THROW_UNSUPPORTED_OPERATION("Halo helpUpdate by cardinality is supported only with a structOfArrays order.");
        }

        if (self().cardinality() <= cardIdx) {
            NEON_THROW_UNSUPPORTED_OPERATION("The specified cardinality for halo helpUpdate exceeds the cardinality of the field.");
        }

        {
            /*
             * this sync goes before the following loop.
             * This is a complete barrier over the stream and
             * it can not be put inside the loop
             */
            if (startWithBarrier) {
                bk.sync(streamSetIdx);
            }
        }
        {
            int ndevs = m_data->devSet.setCardinality();
#pragma omp parallel for num_threads(ndevs) default(shared)
            for (int devIdx = 0; devIdx < ndevs; devIdx++) {
                constexpr bool latticeMode = false;
                constexpr bool revertLattice = false;
                h_huSoAByCardSingleDev<transferMode_ta>(devIdx, bk, cardIdx, streamSetIdx, latticeMode, revertLattice);
            }
        }
    }

    /**
     * Updating halo for all cardinality of the field at once.
     */
    auto haloUpdateLattice(Neon::set::HuOptions& huOptions)
        -> void
    {
        const Neon::Backend& bk = m_data->grid->backend();
        /**
         * No halo helpUpdate operation can be performed on a filed where halo was not activated.
         */
        if (m_data->haloStatus != haloStatus_et::ON) {
            NEON_THROW_UNSUPPORTED_OPERATION("Halo support was not activated for this field.");
        }

        /**
         * We don't need any helpUpdate if the number of devices is one.
         */
        if (m_data->devSet.setCardinality() == 1) {
            return;
        }

        switch (m_data->memOrder) {
            case Neon::memLayout_et::order_e::structOfArrays: {
                {
                    /*
                     * this sync goes before the following loop.
                     * This is a complete barrier over the stream and
                     * it can not be put inside the loop
                     */
                    if (huOptions.startWithBarrier()) {
                        bk.sync(huOptions.streamSetIdx());
                    }
                }

                const int ndevs = m_data->devSet.setCardinality();
#pragma omp parallel for num_threads(ndevs) default(shared)
                for (int gpuIdx = 0; gpuIdx < ndevs; gpuIdx++) {
                    for (int cardIdx = 0; cardIdx < self().cardinality(); cardIdx++) {

                        Neon::set::PeerTransferOption& peerTransferOpt = huOptions.getPeerTransferOpt(bk);
                        // LATTICE + FORWARD
                        constexpr auto structure = Neon::set::TransferSemantic::lattice;

                        h_huSoAByCardSingleDevFwd(peerTransferOpt,
                                                  gpuIdx, cardIdx,
                                                  structure);
                    }
                }
                break;
            }

            case Neon::memLayout_et::order_e::arrayOfStructs: {
                NEON_THROW_UNSUPPORTED_OPERATION("");
                break;
            }
        }
    }

   private:
    /**
     *
     * @tparam transferMode_ta
     * @tparam haloUpdateMode_ta
     * @param devIdx
     * @param bk
     * @param cardIdx
     * @param streamSetIdx
     */
    template <Neon::set::TransferMode transferMode_ta>
    [[deprecated]] auto
    h_huSoAByCardSingleDev(int                  devIdx,
                           const Neon::Backend& bk,
                           const int            cardIdx,
                           int                  streamSetIdx,
                           bool                 latticeMode,
                           bool                 revertLattice)
        -> void
    {
        LocalIndexingInfo_t& dst = m_data->frame_shp->template localIndexingInfo<Neon::Access::readWrite>(devIdx);
        auto                 activeDependency = m_data->frame_shp->activeDependencyByDestination();

        const std::array<internals::partition_idx, ComDirection_e::COM_NUM> srcIdxByDir =
            {dst.nghIdx(ComDirection_e::COM_DW),
             dst.nghIdx(ComDirection_e::COM_UP)};

        for (const auto& comDirection : {ComDirection_e::COM_DW, ComDirection_e::COM_UP}) {

            const int dstIdx = devIdx;
            const int srcIdx = srcIdxByDir[comDirection];

            //            if (!activeDependency[devIdx][cardIdx].isActive[HaloUpdateMode_e::LATTICE][comDirection]) {
            //                continue;
            //            }

            // TODO@(Max)[Used >1 instead of >2 when the 2-GPU issue has been fixed]
            if (grid().getBackend().devSet().setCardinality() > 2) {
                switch (comDirection) {
                    case ComDirection_e::COM_DW: {
                        if (!(dstIdx > srcIdx)) {
                            continue;
                        }
                        break;
                    }
                    case ComDirection_e::COM_UP: {
                        if (!(dstIdx < srcIdx)) {
                            continue;
                        }
                        break;
                    }
                    default: {
                        NEON_THROW_UNSUPPORTED_OPTION("");
                    }
                }
            }

            if (latticeMode) {
                index_3d dir = grid().getStencil().neighbours().at(cardIdx);
                if (revertLattice) {
                    dir.x = -dir.x;
                    dir.y = -dir.y;
                    dir.z = -dir.z;
                }
                switch (comDirection) {
                    case ComDirection_e::COM_DW: {
                        if (dir.z <= 0 && dir.y <= 0 && dir.x <= 0) {
                            continue;
                        }
                        break;
                    }
                    case ComDirection_e::COM_UP: {
                        if (dir.z >= 0 && dir.y >= 0 && dir.x >= 0) {
                            // TODO@(Max)[Remove when new initiazation is done and the 2 gpu issue is fixed]
                            if (bk.devSet().setCardinality() != 2) {
                                continue;
                            }
                        }
                        break;
                    }
                    default: {
                        NEON_THROW_UNSUPPORTED_OPTION("");
                    }
                }
            }

            const count_t      transferEl = dst.remoteBdrCount(comDirection);
            const Cell::Offset remoteOffset = dst.remoteBdrOff(comDirection);
            const Cell::Offset localOffset = dst.ghostOff(comDirection);

            auto&       dstLocal = self().getPartition(bk.devType(), dstIdx);
            const auto& srcLocal = self().getPartition(bk.devType(), srcIdx);

            T_ta*       dstMem = dstLocal.ePtr(0, cardIdx);
            const T_ta* srcMem = srcLocal.ePtr(0, cardIdx);

            T_ta*       dstBuf = dstMem + localOffset;
            const T_ta* srcBuf = srcMem + remoteOffset;

            assert(transferEl > -1);
            if (transferEl > 0) {
                switch (m_data->devType) {
                    case Neon::DeviceType::CPU: {
                        std::memcpy(dstBuf, srcBuf, transferEl * sizeof(T_ta));
                        break;
                    }
                    case Neon::DeviceType::CUDA: {
                        // auto& devSet = m_data->devSet;
                        auto& streamSet = bk.streamSet(streamSetIdx);
                        m_data->devSet.template peerTransfer<transferMode_ta>(streamSet,
                                                                              dstIdx, (char*)dstBuf,
                                                                              srcIdx, (char*)srcBuf,
                                                                              transferEl * sizeof(T_ta));
                        break;
                    }
                    default: {
                        NEON_THROW_UNSUPPORTED_OPERATION("");
                    }
                }
            }
        }
        return;
    }

    /**
     * @tparam transferMode_ta
     * @tparam haloUpdateMode_ta
     * @param devIdx
     * @param bk
     * @param cardIdx
     * @param streamSetIdx
     */
    auto h_huSoAByCardSingleDevFwd(Neon::set::PeerTransferOption opt,
                                   int                           devIdx,
                                   const int                     cardIdx,
                                   Neon::set::TransferSemantic   structure) const
        -> void
    {
        const Neon::Backend& bk = m_data->grid->getBackend();

        LocalIndexingInfo_t& dst = m_data->frame_shp->template localIndexingInfo<Neon::Access::readWrite>(devIdx);
        auto                 activeDependency = m_data->frame_shp->activeDependencyByDestination();

        const std::array<internals::partition_idx, ComDirection_e::COM_NUM> srcIdxByDir =
            {dst.nghIdx(ComDirection_e::COM_DW),
             dst.nghIdx(ComDirection_e::COM_UP)};

        for (const auto& comDirection : {ComDirection_e::COM_DW, ComDirection_e::COM_UP}) {

            const int dstIdx = devIdx;
            const int srcIdx = srcIdxByDir[comDirection];

            // TODO@(Max)[Used >1 instead of >2 when the 2-GPU issue has been fixed]
            if (grid().getDevSet().setCardinality() > 2) {
                switch (comDirection) {
                    case ComDirection_e::COM_DW: {
                        if (!(dstIdx > srcIdx)) {
                            continue;
                        }
                        break;
                    }
                    case ComDirection_e::COM_UP: {
                        if (!(dstIdx < srcIdx)) {
                            continue;
                        }
                        break;
                    }
                    default: {
                        NEON_THROW_UNSUPPORTED_OPTION("");
                    }
                }
            }

            if (structure == Neon::set::TransferSemantic::lattice) {
                index_3d dir = grid().getStencil().neighbours().at(cardIdx);
                switch (comDirection) {
                    case ComDirection_e::COM_DW: {
                        if (dir.z <= 0 && dir.y <= 0 && dir.x <= 0) {
                            continue;
                        }
                        break;
                    }
                    case ComDirection_e::COM_UP: {
                        if (dir.z >= 0 && dir.y >= 0 && dir.x >= 0) {
                            // TODO@(Max)[Remove when new initiazation is done and the 2 gpu issue is fixed]
                            if (bk.devSet().setCardinality() != 2) {
                                continue;
                            }
                        }
                        break;
                    }
                    default: {
                        NEON_THROW_UNSUPPORTED_OPTION("");
                    }
                }
            }

            const count_t      transferEl = dst.remoteBdrCount(comDirection);
            const Cell::Offset remoteOffset = dst.remoteBdrOff(comDirection);
            const Cell::Offset localOffset = dst.ghostOff(comDirection);

            auto&       dstLocal = self().getPartition(bk.devType(), dstIdx);
            const auto& srcLocal = self().getPartition(bk.devType(), srcIdx);

            const T_ta* dstMem = dstLocal.pointer(Cell(0), cardIdx);
            const T_ta* srcMem = srcLocal.pointer(Cell(0), cardIdx);

            const T_ta* dstBuf = dstMem + localOffset;
            const T_ta* srcBuf = srcMem + remoteOffset;

            assert(transferEl > -1);
            if (transferEl > 0) {
                m_data->devSet.peerTransfer(opt, {opt.transferMode(),
                                                  {dstIdx, (char*)dstBuf},
                                                  {srcIdx, (char*)srcBuf},
                                                  transferEl * sizeof(T_ta)});
            }
        }
        return;
    }

   public:
    /**
     * Export stored values to VTI
     */
    template <typename Vti_Real_ta = double>
    auto ioToVti(bool                        includeGridInfo,
                 const std::string&          fileName,
                 const std::string&          fieldName,
                 const Neon::Vec_3d<double>& spacing = Neon::Vec_3d<double>(1.0),
                 const Neon::Vec_3d<double>& origin = Neon::Vec_3d<double>(0.0))
        const
        -> void
    {
        const index_3d nodeSpace = m_data->frame_shp->domain() + 1;
        const index_3d voxSpace = m_data->frame_shp->domain();

        if (m_data->devType != Neon::DeviceType::CPU) {
            NEON_DEV_UNDER_CONSTRUCTION("");
        }

        const auto gpuIds = [this](const index_3d& idx, int /* vIdx */) -> double {
            internals::elmLocalInfo_t info = m_data->frame_shp->globalToLocal().elRef(idx);
            int                       prtId = info.getPrtIdx();
            return static_cast<double>(prtId);
        };

        const auto activity = [this](const index_3d& idx, int /*vIdx*/) -> double {
            internals::elmLocalInfo_t info = m_data->frame_shp->globalToLocal().elRef(idx);
            if (info.isActive()) {
                return 1.0;
            }
            return -1.0;
        };

        const auto fieldValues = [&](const Neon::index_3d& idx, int cardIdx) -> double {
            const auto retVal = std::get<0>(eRef(idx, cardIdx));
            return static_cast<double>(retVal);
        };

        if (includeGridInfo) {
            Neon::ioToVTI<index_3d::Integer, Vti_Real_ta>({{fieldValues, m_data->cardinality, fieldName, false, Neon::IoFileType::ASCII},
                                                           {activity, 1, "activity", false, Neon::IoFileType::ASCII},
                                                           {gpuIds, 1, "gpuIdx", false, Neon::IoFileType::ASCII}},
                                                          fileName, nodeSpace, voxSpace, spacing, origin);

        } else {
            Neon::ioToVTI<index_3d::Integer, Vti_Real_ta>({{fieldValues, m_data->cardinality, fieldName, false, Neon::IoFileType::ASCII}},
                                                          fileName, nodeSpace, voxSpace, spacing, origin);
        }
    }

    /**
     * Export stored values to VTK
     */
    template <typename Vti_Real_ta = double>
    auto ioToVtk(bool                           includeGridInfo,
                 const std::string&             fileName,
                 const std::string&             fieldName,
                 Neon::ioToVTKns::VtiDataType_e nodeOrVox,
                 const Neon::Vec_3d<double>&    spacing = Neon::Vec_3d<double>(1.0),
                 const Neon::Vec_3d<double>&    origin = Neon::Vec_3d<double>(0.0),
                 Neon::IoFileType               ASCII_or_BINARY = Neon::IoFileType::ASCII)
        const
        -> void
    {
        index_3d nodeSpace;
        index_3d voxSpace;

        if (nodeOrVox == Neon::ioToVTKns::node) {
            nodeSpace = m_data->frame_shp->domain();
            voxSpace = m_data->frame_shp->domain() - 1;
        } else {
            voxSpace = m_data->frame_shp->domain();
            nodeSpace = m_data->frame_shp->domain() + 1;
        }

        if (m_data->devType != Neon::DeviceType::CPU) {
            NEON_DEV_UNDER_CONSTRUCTION("");
        }

        const auto gpuIds = [this](const index_3d& idx, [[maybe_unused]] int vIdx) -> double {
            internals::elmLocalInfo_t info = m_data->frame_shp->globalToLocal().elRef(idx);
            int                       prtId = info.getPrtIdx();
            return static_cast<double>(prtId);
        };

        const auto activity = [this](const index_3d& idx, [[maybe_unused]] int vIdx) -> double {
            internals::elmLocalInfo_t info = m_data->frame_shp->globalToLocal().elRef(idx);
            if (info.isActive()) {
                return 1.0;
            }
            return -1.0;
        };

        const auto fieldValues = [&](const Neon::index_3d& idx, int cardIdx) -> double {
            const auto retVal = std::get<0>(eRef(idx, cardIdx));
            return static_cast<double>(retVal);
        };

        if (includeGridInfo) {
            Neon::ioToVTKns::ioToVTK<index_3d::Integer, Vti_Real_ta>({{fieldValues, m_data->cardinality, fieldName, nodeOrVox},
                                                                      {activity, 1, "activity", nodeOrVox},
                                                                      {gpuIds, 1, "gpuIdx", nodeOrVox}},
                                                                     fileName, nodeSpace, spacing, origin, ASCII_or_BINARY);

        } else {
            Neon::ioToVTKns::ioToVTK<index_3d::Integer, Vti_Real_ta>({{fieldValues, m_data->cardinality, fieldName, nodeOrVox}},
                                                                     fileName, nodeSpace, spacing, origin, ASCII_or_BINARY);
        }
    }
    /**
     * Export stored values to getNewDenseField
     */
    template <typename exportReal_ta = T_ta>
    auto ioToDense(Neon::memLayout_et::order_e order, exportReal_ta* dense)
        const
        -> void
    {
        forEach([&](bool /*isActive*/, const Neon::index_3d& idx, const int& cardIdx, const T_ta& val) {
            size_t pitch;
            switch (order) {
                case Neon::memLayout_et::order_e::structOfArrays: {
                    auto domain = m_data->frame_shp->domain().template newType<size_t>();
                    pitch = idx.x +
                            idx.y * domain.x +
                            idx.z * domain.x * domain.y +
                            cardIdx * domain.x * domain.y * domain.z;
                    break;
                }
                case Neon::memLayout_et::order_e::arrayOfStructs: {
                    auto domain = m_data->frame_shp->domain().template newType<size_t>();
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

    /**
     *
     */
    template <typename exportReal_ta = T_ta>
    auto ioToDense(Neon::memLayout_et::order_e order) const -> std::shared_ptr<exportReal_ta>
    {
        std::shared_ptr<exportReal_ta> dense(new exportReal_ta[m_data->frame_shp->domain().template rMulTyped<size_t>() * m_data->cardinality],
                                             std::default_delete<exportReal_ta[]>());
        ioToDense(order, dense.get());
        return dense;
    }

    /**
     *
     */
    template <typename exportReal_ta = T_ta>
    auto ioFromDense(Neon::memLayout_et::order_e order,
                     const exportReal_ta*        dense,
                     exportReal_ta               inactiveValue)
        -> void
    {
        m_data->inactiveValue = (element_t)inactiveValue;
        forEachActive([&](const Neon::index_3d& idx, const int& cardIdx, T_ta& val) {
            size_t pitch;
            switch (order) {
                case Neon::memLayout_et::order_e::structOfArrays: {
                    auto domain = m_data->frame_shp->domain().template newType<size_t>();
                    pitch = idx.x +
                            idx.y * domain.x +
                            idx.z * domain.x * domain.y +
                            cardIdx * domain.x * domain.y * domain.z;
                    break;
                }
                case Neon::memLayout_et::order_e::arrayOfStructs: {
                    auto domain = m_data->frame_shp->domain().template newType<size_t>();
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
            T_ta denseVal = T_ta(dense[pitch]);
            val = denseVal;
        });
    }


    /**
     *
     * @param fun
     */
    auto forEach(std::function<void(bool isActive,
                                    const Neon::index_3d&,
                                    const int&,
                                    element_t&)> fun)
        -> void
    {
        index_3d cellSpace = m_data->frame_shp->domain();

        if (m_data->devType != Neon::DeviceType::CPU) {
            NEON_DEV_UNDER_CONSTRUCTION("");
        }

        const auto& GtoL = m_data->frame_shp->globalToLocal();

        for (int cardIdx = 0; cardIdx < m_data->cardinality; cardIdx++) {
            index_3d::forEach(cellSpace, [&](const Neon::index_3d& idx) {
                const auto& info = GtoL.elRef(idx);
                if (!info.isActive()) {
                    fun(false, idx, cardIdx, m_data->inactiveValue);
                    return;
                }
                const auto& prtId = info.getPrtIdx();
                const auto& localIdx = info.getLocalIdx();

                Neon::sys::MemDevice<T_ta>& mem = m_data->memoryStorage.template get<Neon::Access::readWrite>(prtId);
                T_ta&                       el = mem.template elRef<Neon::Access::readWrite>(localIdx, cardIdx);
                fun(true, idx, cardIdx, el);
            });
        }
    }

    /**
     *
     * @param fun
     */
    auto forEach(std::function<void(bool isActive,
                                    const Neon::index_3d&,
                                    const int&,
                                    const element_t&)> fun) const
        -> void
    {
        index_3d cellSpace = m_data->frame_shp->domain();

        if (m_data->devType != Neon::DeviceType::CPU) {
            NEON_DEV_UNDER_CONSTRUCTION("");
        }

        const auto& GtoL = m_data->frame_shp->globalToLocal();

        for (int cardIdx = 0; cardIdx < m_data->cardinality; cardIdx++) {
            index_3d::forEach(cellSpace, [&](const Neon::index_3d& idx) {
                const auto& info = GtoL.elRef(idx);
                if (!info.isActive()) {
                    fun(false, idx, cardIdx, m_data->inactiveValue);
                    return;
                }
                const auto& prtId = info.getPrtIdx();
                const auto& localIdx = info.getLocalIdx();

                Neon::sys::MemDevice<T_ta>& mem = m_data->memoryStorage.template get<Neon::Access::readWrite>(prtId);
                T_ta&                       el = mem.template elRef<Neon::Access::readWrite>(localIdx, cardIdx);
                fun(true, idx, cardIdx, el);
            });
        }
    }
    /**
     *
     * @param fun
     */
    auto forEachActive(std::function<void(const Neon::index_3d&, const int& cardinality, element_t&)> fun)
        -> void
    {
        this->forEach([&](bool                  isActive,
                          const Neon::index_3d& idx,
                          const int&            cardinality,
                          T_ta&                 val) {
            if (!isActive) {
                return;
            }
            fun(idx, cardinality, val);
        });
    }

   public:
    /**
     *
     */
    ~eFieldDevice_t() = default;


    /**
     * Return the local view for a specific device type and id
     */
    auto getPartition(Neon::DeviceType,
                      index_1d              idx,
                      const Neon::DataView& dataView = Neon::DataView::STANDARD) -> local_t&
    {
        return m_data->localSetByView[static_cast<int>(dataView)][idx];
    }

    /**
     * Return the local view for a specific device type and id
     */
    auto getPartition(Neon::DeviceType,
                      index_1d              idx,
                      const Neon::DataView& dataView = Neon::DataView::STANDARD)
        const
        -> const local_t&
    {
        return m_data->localSetByView[static_cast<int>(dataView)][idx];
    }

    /**
     * returns a reference to this object
     * @return
     */
    inline auto self()
        -> self_t&
    {
        return *this;
    }

    /**
     * returns a reference to this object
     * @return
     */
    inline auto self()
        const
        -> const self_t&
    {
        return *this;
    }

    /**
     * returns a const reference to this object
     * @return
     */
    inline auto cself()
        const
        -> const self_t&
    {
        return *this;
    }

    /**
     *
     * @return
     */
    auto haloStatus() const -> Neon::domain::haloStatus_et::e
    {
        return m_data->haloStatus;
    }

   private:
    // INTERNAL UTILITY FUNCTIONS

    /**
     *
     */
    void h_initRawMem(const Neon::Allocator& allocType)
    {
        Neon::set::DataSet<uint64_t> nElementVec = m_data->devSet.template newDataSet<uint64_t>();
        for (int gpuIdx = 0; gpuIdx < m_data->devSet.setCardinality(); gpuIdx++) {
            nElementVec[gpuIdx] = m_data->frame_shp->localIndexingInfo(gpuIdx).nElements(m_data->haloStatus == Neon::domain::haloStatus_et::ON);
        }
        m_data->memoryStorage = m_data->devSet.template newMemDevSet<element_t>(m_data->cardinality,
                                                                                m_data->devType,
                                                                                allocType,
                                                                                nElementVec,
                                                                                m_data->memOrder,
                                                                                m_data->memAlignment,
                                                                                m_data->memPadding);
        m_data->userPointersSet = m_data->memoryStorage.memSet();
    }

    /**
     *
     */
    void h_initFieldCompute()
    {
        for (int DataViewIdx = 0; DataViewIdx < Neon::DataViewUtil::nConfig; DataViewIdx++) {
            m_data->localSetByView[DataViewIdx] = m_data->devSet.template newDataSet<local_t>();
        }
        for (int gpuIdx = 0; gpuIdx < m_data->devSet.setCardinality(); gpuIdx++) {
            const LocalIndexingInfo_t& indexingInfo = m_data->frame_shp->localIndexingInfo(gpuIdx);

            ePitch_t ePitch = m_data->memoryStorage.get(gpuIdx).pitch();

            std::array<Cell::Offset, ComDirection_e::COM_NUM> bdrOff = {indexingInfo.bdrOff(ComDirection_e::COM_DW),
                                                                        indexingInfo.bdrOff(ComDirection_e::COM_UP)};
            std::array<Cell::Offset, ComDirection_e::COM_NUM> ghostOff = {indexingInfo.ghostOff(ComDirection_e::COM_DW),
                                                                          indexingInfo.ghostOff(ComDirection_e::COM_UP)};

            std::array<count_t, ComDirection_e::COM_NUM> bdrCount = {indexingInfo.bdrCount(ComDirection_e::COM_DW),
                                                                     indexingInfo.bdrCount(ComDirection_e::COM_UP)};
            std::array<count_t, ComDirection_e::COM_NUM> ghostCount = {indexingInfo.remoteBdrCount(ComDirection_e::COM_DW),
                                                                       indexingInfo.remoteBdrCount(ComDirection_e::COM_UP)};

            int32_t*   con = m_data->frame_shp->connectivity(m_data->devType).mem(gpuIdx);
            index64_2d conPitch = m_data->frame_shp->connectivity(m_data->devType).get(gpuIdx).pitch();
            index_t*   inverseMapping = m_data->frame_shp->inverseMapping(m_data->devType).mem(gpuIdx);

            for (int DataViewIdx = 0; DataViewIdx < Neon::DataViewUtil::nConfig; DataViewIdx++) {
                m_data->localSetByView[DataViewIdx][gpuIdx] = local_t(Neon::DataView(DataViewIdx), gpuIdx,
                                                                      m_data->userPointersSet[gpuIdx],
                                                                      m_data->cardinality,
                                                                      ePitch,
                                                                      bdrOff, ghostOff,
                                                                      bdrCount, ghostCount,
                                                                      con, conPitch,
                                                                      inverseMapping);
            }
        }
    }
};  // namespace eGrid

}  // namespace Neon::domain::internal::eGrid
