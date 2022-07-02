#pragma once

#include <functional>

#include "dsBuilderCommon.h"
#include "dsFrame.h"
#include "Neon/domain/internal/eGrid/eInternals/Partitioning.h"
#include "Neon/domain/interface/Stencil.h"
#include "Neon/sys/memory/mem3d.h"
#include "dsBuilderCommon.h"
#include "dsFrame.h"

namespace Neon::domain::internal::eGrid {

namespace internals {

struct flatPartitioning_t
{
   private:
    std::shared_ptr<dsFrame_t> m_frame{nullptr};

    DataSet<global_idx>       m_firstIds;
    DataSet<global_idx>       m_lastIds;
    DataSet<int64_t>          m_partitionSizes;
    ListDataSet<global_idx> m_tmpActiveToGlobal;

    Neon::sys::Mem3d_t<global_idx>
        m_globalToInternalAndBoundaryLocal;

   public:
    flatPartitioning_t(const Neon::set::DevSet&                   devSet,
                       const Neon::index_3d&                      domain,
                       std::function<bool(const Neon::index_3d&)> inOut,
                       int                                        nPartitions,
                       const Neon::domain::Stencil&               stencil)
        : m_firstIds(nPartitions),
          m_lastIds(nPartitions),
          m_partitionSizes(nPartitions),
          m_tmpActiveToGlobal(nPartitions),
          m_globalToInternalAndBoundaryLocal(1,
                                             Neon::DeviceType::CPU,
                                             Neon::sys::DeviceID(0),
                                             Neon::Allocator::MALLOC,
                                             domain,
                                             Neon::index_3d(0),
                                             Neon::memLayout_et::structOfArrays,
                                             Neon::sys::MemAlignment(),
                                             Neon::memLayout_et::OFF)
    {
        m_frame = std::make_shared<dsFrame_t>(devSet, domain, std::move(inOut), nPartitions, stencil);
        computeFirstLastSize();
        setFrame();
    }

    std::shared_ptr<dsFrame_t> getFrame()
    {
        return m_frame;
    }

   private:
    /**
     * For each partition computes
     * 1. first global id
     * 2. last global id
     * 3. number of active elements in the region
     */
    void computeFirstLastSize()
    {
        dsFrame_t&      frame = *m_frame;
        const int64_t   minimumNumber = frame.nActiveElements() / frame.nPartitions();
        elmLocalInfo_t* frameGlobalToLocal = frame.globalToLocal().mem();

        {  // Computing uniform partitioning of a 1D flat vector.
            m_partitionSizes = DataSet<int64_t>(frame.nPartitions(), minimumNumber);
            {
                const int32_t reminder = static_cast<int32_t>(frame.nActiveElements() - minimumNumber * frame.nPartitions());
                for (int i = 0; i < reminder; i++) {
                    m_partitionSizes.ref<Neon::Access::readWrite>(i) += 1;
                }
            }

            partition_idx targetPrtIdx = 0;

            auto       targetSize = m_partitionSizes.ref(targetPrtIdx);
            global_idx targetFirst = 0;
            global_idx targetLast = 0;
            int64_t    nDetectedActive = 0;
            int64_t    totalDetectedActive = 0;

            const auto& globalToLocal = frame.globalToLocal().mem();
            global_idx  globalIdx;

            /**
             * This loop is sequential.
             * We are creating the mapping between elements and partitions.
             * Because we don't know the distribution of the elements we do it sequentially.
             *
             * TODO Optimizations:
             * 1. start both from top and bottom
             * 2. do a binning to help clusterng the row of elements that belongs all to one partition
             *
             */
            for (globalIdx = 0; globalIdx < frame.nDomainElements(); globalIdx++) {
                auto globalInfo = globalToLocal[globalIdx];
                bool isActive = globalInfo.isActive();
                if (isActive) {
                    nDetectedActive++;
                    m_tmpActiveToGlobal.append(targetPrtIdx, globalIdx);
                    frameGlobalToLocal[globalIdx] = elmLocalInfo_t(targetPrtIdx);
                    if (nDetectedActive == targetSize) {
                        targetLast = globalIdx;
                        totalDetectedActive += nDetectedActive;
                        m_firstIds.ref<Neon::Access::readWrite>(targetPrtIdx) = targetFirst;
                        m_lastIds.ref<Neon::Access::readWrite>(targetPrtIdx) = targetLast;

                        // MOVING TO NEXT PARTITION
                        targetPrtIdx++;
                        if (targetPrtIdx == frame.nPartitions()) {
                            break;
                        }

                        // RESETTING
                        targetSize = m_partitionSizes.ref(targetPrtIdx);
                        targetFirst = targetLast + 1;
                        targetLast = 0;
                        nDetectedActive = 0;

                        continue;
                    }
                }
            }

            if (totalDetectedActive != frame.nActiveElements()) {
                NeonException exp("Flat Partitioning");
                NEON_THROW(exp);
            }
        }
    }

    partition_idx mapGlobalToPartitionId(const global_idx& id) const
    {
        if (id < m_firstIds.ref(0)) {
            return partition_et::invalid;
        }
        for (partition_idx i = 0; i < m_frame->nPartitions(); i++) {
            const bool bottomCheck = m_firstIds.ref(i) <= id;
            const bool topCheck = id <= m_lastIds.ref(i);
            if (bottomCheck && topCheck) {
                return i;
            }
        }
        return partition_et::invalid;
    }

    /**
     * Given a global index, it returns the global indexes of its neighbours.
     * If the neighbour is does not exists, the corresponding global id is set to null
     *
     * @param centerGIdx
     * @return
     */
    auto getNghGlobalIds(global_idx centerGIdx) const -> NghDataSet<global_idx>
    {
        const dsFrame_t& frame = *m_frame;

        /**
         * Returning global ids of neighbours
         */
        const auto&              stencilPoints = frame.stencil().neighbours();
        NghDataSet<global_idx> nhgGIdSet = frame.newNghDataSet<global_idx>();
        const Neon::index_3d     centerG3d = frame.domain().mapTo3dIdx(centerGIdx);

        // Neon::int64_3d pitch(1, frame.domain().x, frame.domain().x * size_t(frame.domain().y));

        for (neighbour_idx ngIdx = 0; ngIdx < frame.nNeighbours(); ngIdx++) {

            const Neon::index_3d nghG3d = centerG3d + stencilPoints[ngIdx];
            if (!nghG3d.isInsideBox(0, frame.domain() - 1)) {
                nhgGIdSet.ref(ngIdx) = neighbour_et::invalid;
                continue;
            }

            int64_t     neighbourID = nghG3d.mPitch(frame.domain());
            const auto& GtoL = frame.globalToLocal().mem();
            const auto  isAnActiveNgh = GtoL[neighbourID].isActive();
            neighbourID = isAnActiveNgh ? neighbourID : neighbour_et::invalid;
            nhgGIdSet.ref(ngIdx) = neighbourID;
        }

        return nhgGIdSet;
    };


    /**
     * For a set of neighbour global ids it returns the associated partition id
     *
     * @param neighboursIdG
     * @return
     */
    auto getNghPrtIds(const NghDataSet<global_idx>& neighboursIdG) const -> NghDataSet<partition_idx>
    {
        const dsFrame_t&            frame = *m_frame;
        NghDataSet<partition_idx> ngPrts = frame.newNghDataSet<partition_idx>();

        for (neighbour_idx ngIdx = 0; ngIdx < frame.nNeighbours(); ngIdx++) {
            const global_idx neighboutIdG = neighboursIdG.ref(ngIdx);
            if (neighboutIdG == neighbour_et::invalid) {
                /*
                 * If a neighbour node does not belong to the domain,
                 * we set its partition id to be prt_extertnal (which is negative)
                 */
                ngPrts.ref(ngIdx) = partition_et::invalid;
            }

            ngPrts.ref(ngIdx) = mapGlobalToPartitionId(neighboutIdG);
        }
        return ngPrts;
    }

    auto getNghType(global_idx                         idG,
                    const NghDataSet<partition_idx>& ngPrts) const -> NghDataSet<neighbour_et::e>
    {
        const dsFrame_t&              frame = *m_frame;
        NghDataSet<neighbour_et::e> nghEs = frame.newNghDataSet<neighbour_et::e>();

        const partition_idx targetPrt = mapGlobalToPartitionId(idG);

        for (neighbour_idx nghIdx = 0; nghIdx < frame.nNeighbours(); nghIdx++) {
            const auto& ngPrt = ngPrts.ref(nghIdx);
            auto&       nghE = nghEs.ref(nghIdx);

            if (ngPrt == partition_et::invalid) {
                nghE = neighbour_et::invalid;
                continue;
            }

            if (ngPrt == targetPrt) {
                nghE = neighbour_et::local;
                continue;
            }

            if (ngPrt != targetPrt) {
                nghE = neighbour_et::remote;
                continue;
            }
        }
        return nghEs;
    };

    /**
     * Returns true if at leas one of the neighbours is not active
     * @param idG
     * @param ngPrts
     * @return
     */
    auto isDomainBroundary(global_idx                         idG,
                           const NghDataSet<partition_idx>& ngPrts) -> bool
    {
        (void)idG;
        dsFrame_t&                    frame = *m_frame.get();
        NghDataSet<neighbour_et::e> nghEs = frame.newNghDataSet<neighbour_et::e>();

        // partition_idx targetPrt = mapGlobalToPartitionId(idG);

        for (neighbour_idx nghIdx = 0; nghIdx < frame.nNeighbours(); nghIdx++) {
            const auto& ngPrt = ngPrts.ref(nghIdx);
            // auto&       nghE = nghEs.ref(nghIdx);

            if (ngPrt == partition_et::invalid) {
                return true;
            }
        }
        return false;
    };

    /**
     * Search in all neighbor id on each stencil direction.
     * There should only one unique partition id as this is a constraint on the flat partitioning.
     * If more then one Id is found, an exception is fired.
     * @param idG
     * @param ngPrts
     * @return
     *
     */
    auto getUniquePrt(global_idx                         idG,
                      const NghDataSet<partition_idx>& ngPrts) const -> std::array<partition_idx, ComDirection_e::COM_NUM>
    {
        const dsFrame_t& frame = *m_frame;

        std::array<partition_idx, ComDirection_e::COM_NUM> resPart = {partition_et::invalid,
                                                                      partition_et::invalid};

        const partition_idx targetPrt = mapGlobalToPartitionId(idG);

        int writeLocaction = 0;

        for (neighbour_idx nghIdx = 0; nghIdx < frame.nNeighbours(); nghIdx++) {
            const auto& ngPrt = ngPrts.ref(nghIdx);

            if (ngPrt == partition_et::invalid || ngPrt == targetPrt || ngPrt == resPart[0] || ngPrt == resPart[1]) {
                continue;
            }
            resPart[writeLocaction] = ngPrt;
            writeLocaction++;
            if (writeLocaction > 1) {
                NeonException exp("eGrid::flatPartitioning::getUniqueNghPrt");
                exp << "A voxel is linked to more than two neighbour partition.";
                NEON_THROW(exp);
            }
        }
        return resPart;
    };


    auto getLocalType(const NghDataSet<neighbour_et::e>& nghEs) const -> element_et::e
    {
        const dsFrame_t& frame = *m_frame;

        bool          atLeastOneLocal = false;
        element_et::e res = element_et::internal;

        for (neighbour_idx nghIdx = 0; nghIdx < frame.nNeighbours(); nghIdx++) {
            const auto& nghE = nghEs.ref(nghIdx);
            if (nghE == neighbour_et::remote) {
                res = element_et::boundary;
                continue;
            }
            if (nghE == neighbour_et::local) {
                atLeastOneLocal = true;
                continue;
            }
        }

        if (!atLeastOneLocal) {
            NeonException exp("eGrid::localClassification");
            exp << "eGrid expect at least one element to be local, i.e. the domain is too small";
            NEON_THROW(exp);
        }

        return res;
    };

    void classificationOfAPartition_InternalAndBoundary(partition_idx targetPrtIdx)
    {
        dsFrame_t&                 frame = *m_frame;
        DataSet<BoundariesInfo_t>& frameBdrToG = frame.boundaryToGlobal();
        DataSet<InternalInfo_t>&   frameInternalToG = frame.internalToGlobal();

        std::array<partition_idx, ComDirection_e::COM_NUM> nghPartIds;
        nghPartIds[ComDirection_e::COM_DW] = static_cast<int32_t>((m_partitionSizes.nPartitions() + targetPrtIdx - 1) % m_partitionSizes.nPartitions());
        nghPartIds[ComDirection_e::COM_UP] = static_cast<int32_t>((m_partitionSizes.nPartitions() + targetPrtIdx + 1) % m_partitionSizes.nPartitions());

        frameBdrToG.ref<Neon::Access::readWrite>(targetPrtIdx).settingOp(nghPartIds);

        assert(int64_t(m_tmpActiveToGlobal.refList(targetPrtIdx).size()) == m_partitionSizes.ref(targetPrtIdx));

        // PASS 1: compute global ids for internal and boundaries
        size_t nInternal = 0;
        // LOOP OVER ACTIVE
        // FOR NOW - this loop can be parallelize as in its body we are appending to vectors.
        // If we add a critical section get worst
        //
        for (int64_t i = 0; i < m_partitionSizes.ref(targetPrtIdx); i++) {
            // During this loop we don't have the full local_id distribution
            // as we are visiting the local (internal and boundary) elements
            // for the first time
            global_idx targetGlbIdx = m_tmpActiveToGlobal.ref(targetPrtIdx, i);

            NghDataSet<global_idx>      globalNhgIds = getNghGlobalIds(targetGlbIdx);
            NghDataSet<partition_idx>   NhgPrts = getNghPrtIds(globalNhgIds);
            NghDataSet<neighbour_et::e> NhgClasses = getNghType(targetGlbIdx, NhgPrts);
            bool                          isDomainBoundary = this->isDomainBroundary(targetGlbIdx, NhgPrts);
            element_et::e                 localType = getLocalType(NhgClasses);

            switch (localType) {
                case element_et::internal: {
                    // NOTE: internal elements have the following property:
                    // internalIdx == localIdx
                    //
                    // Parsing internal nodes and saving globalToInternal map into the temporary data structure
                    // a. frameInternalToGlobal
                    // b. frameInternalCon (we temporary store global information that will be translated to local in classificationOfAPartitionConnectivity)
                    // c. frameGlobalToLocal
                    internal_idx internalIdx = frameInternalToG.ref<Neon::Access::readWrite>(targetPrtIdx).append(globalNhgIds) - 1;

                    elmLocalInfo_t* frameGlobalToLocal = frame.globalToLocal().mem();
                    assert(targetPrtIdx == frameGlobalToLocal[targetGlbIdx].getPrtIdx());
                    assert(!frameGlobalToLocal[targetGlbIdx].isFullySet());
                    frameGlobalToLocal[targetGlbIdx] = elmLocalInfo_t(targetPrtIdx, local_idx(internalIdx), isDomainBoundary, Neon::DataView::INTERNAL);

                    nInternal++;
                    continue;
                }
                case element_et::boundary: {
                    // Parsing internal nodes and saving globalToInternal map into the temporary data structure
                    // a. frameBoundaryToGlobal
                    // NOTE .. WE ARE NOT CREATING THE LOCAL ID FOR boundary AT THIS TIME
                    std::array<partition_idx, 2> nghPrt = getUniquePrt(targetGlbIdx, NhgPrts);
                    frameBdrToG.ref<Neon::Access::readWrite>(targetPrtIdx).append(nghPrt[0], nghPrt[1], targetGlbIdx, globalNhgIds, NhgPrts);
                    continue;
                }

                case element_et::ghost: {
                    NeonException exp("");
                    exp << "Unsupported case.";
                    NEON_THROW(exp);
                }
            }
        }

        {  // NOTE .. WE ARE NOW CREATING THE LOCAL ID FOR boundary
            BoundariesInfo_t& target_frameBdrToG = frameBdrToG.ref<Neon::Access::readWrite>(targetPrtIdx);
            target_frameBdrToG.setLoadingComplete(nInternal);

            // We do a pass for each boundary direction
            for (auto&& direction : std::vector<BdrDepClass_e::e>{BdrDepClass_e::DW,
                                                                  BdrDepClass_e::e::BOTH,
                                                                  BdrDepClass_e::UP}) {

                elmLocalInfo_t* frameGlobalToLocal = frame.globalToLocal().mem();

                const int64_t nBoundary = target_frameBdrToG.nGlobal(direction);

                // Going through the list of partition boundaries for the specific direction
                for (bdrRelative_idx boundaryRelativeIdx = 0; boundaryRelativeIdx < nBoundary; boundaryRelativeIdx++) {
                    local_idx  targetLocal = target_frameBdrToG.computeLocalIdx(direction, boundaryRelativeIdx);
                    global_idx targetGlobal = target_frameBdrToG.getGlobalIdx(direction, boundaryRelativeIdx);
                    bool       isDomainBoundary = target_frameBdrToG.isDomainBoundary(direction, boundaryRelativeIdx);
                    frameGlobalToLocal[targetGlobal] = elmLocalInfo_t(targetPrtIdx, targetLocal, isDomainBoundary, Neon::DataView::BOUNDARY);
                }
            }
        }

        {
            BoundariesInfo_t&    frameTargetBoundaryInfo = frameBdrToG.ref<Neon::Access::readWrite>(targetPrtIdx);
            LocalIndexingInfo_t& localIndexingInfo = frame.localIndexingInfo<Neon::Access::readWrite>(targetPrtIdx);
            localIndexingInfo = LocalIndexingInfo_t(targetPrtIdx, nInternal, frameTargetBoundaryInfo.getPrtIds(), frameTargetBoundaryInfo.getNumBoundaries());
        }
    }

    void initConnectivityVector(partition_idx targetPrtIdx)
    {
        dsFrame_t&                  frame = *m_frame;
        const BoundariesInfo_t&     boundaryToGlobal = frame.boundaryToGlobal().ref(targetPrtIdx);
        const InternalInfo_t&       internalToGlobal = frame.internalToGlobal().ref(targetPrtIdx);
        const elmLocalInfo_t* const frameGlobalToLocal = frame.globalToLocal().mem();

        {  // INTERNAL -> converting global Id to local
            InternalInfo_t& InternalInfo = frame.internalToGlobal().ref<Neon::Access::readWrite>(targetPrtIdx);
            InternalInfo.convertNeighboursToLocal(frameGlobalToLocal);
        }

        {  // BOuNDARIES -> converting global Id to local
            BoundariesInfo_t& frameTargetBoundaryInfo = frame.boundaryToGlobal().ref<Neon::Access::readWrite>(targetPrtIdx);
            auto&             localIndexingInfoDataSet = frame.localIndexingInfo();
            frameTargetBoundaryInfo.convertBoundaryConnectivityFromGlobalToLocal(targetPrtIdx, frameGlobalToLocal, localIndexingInfoDataSet);
        }

        const LocalIndexingInfo_t&       localIndexingInfo = frame.localIndexingInfo().ref(targetPrtIdx);
        Neon::set::MemDevSet<int32_t>& localConectivitySet = frame.connectivity(Neon::DeviceType::CPU);
        Neon::sys::MemDevice<int32_t>&   localConectivityStorage = localConectivitySet.get<Neon::Access::readWrite>(targetPrtIdx);

// COPY INTERNAL connectivity to final place
#pragma omp parallel for collapse(2) default(shared)
        for (internal_idx idx = 0; idx < localIndexingInfo.internalCount(); idx++) {
            for (neighbour_idx nghIdx = 0; nghIdx < frame.nNeighbours(); nghIdx++) {
                auto value = internalToGlobal.Ngh(idx, nghIdx);
                localConectivityStorage.elRef<Neon::Access::readWrite>(idx, nghIdx) = int32_t(value);
            }
        }

        // COPY BOUNDARY connectivity to final place
        size_t offset = localIndexingInfo.internalCount();
        for (const auto& bdrDepClass : {BdrDepClass_e::DW, BdrDepClass_e::BOTH, BdrDepClass_e::UP}) {
            const size_t nElements = boundaryToGlobal.getNumBoundaries(bdrDepClass);
            {
#pragma omp parallel for collapse(2) default(shared) num_threads(4) schedule(static)
                for (boundaries_idx idx = 0; idx < boundaries_idx(nElements); idx++) {
                    for (neighbour_idx nghIdx = 0; nghIdx < frame.nNeighbours(); nghIdx++) {
                        const int64_t       globalLocation = offset + idx;
                        const neighbour_idx neighbourIdx = static_cast<neighbour_idx>(boundaryToGlobal.getNghIdx(bdrDepClass, idx).ref(nghIdx));
                        localConectivityStorage.elRef<Neon::Access::readWrite>(globalLocation, nghIdx) = neighbourIdx;
                    }
                }
            }
            offset += nElements;
        }
    }

    auto inverseMapping() -> void
    {
        elmLocalInfo_t* frameGlobalToLocal = m_frame->globalToLocal().mem();
#pragma omp parallel for collapse(2) default(shared)
        for (int z = 0; z < m_frame->domain().z; z++) {
            for (int y = 0; y < m_frame->domain().y; y++) {
                for (int x = 0; x < m_frame->domain().x; x++) {
                    index_3d   xyz(x, y, z);
                    global_idx globalIdx = xyz.mPitch(m_frame->domain());
                    if (frameGlobalToLocal[globalIdx].isActive()) {
                        auto prtId = frameGlobalToLocal[globalIdx].getPrtIdx();
                        auto localId = frameGlobalToLocal[globalIdx].getLocalIdx();
                        m_frame->inverseMapping(Neon::DeviceType::CPU).elRef(prtId, localId, index_3d::x_axis) = x;
                        m_frame->inverseMapping(Neon::DeviceType::CPU).elRef(prtId, localId, index_3d::y_axis) = y;
                        m_frame->inverseMapping(Neon::DeviceType::CPU).elRef(prtId, localId, index_3d::z_axis) = z;
                    }
                }
            }
        }
    }

    void
    classification()
    {
        auto const& nPartitions = m_frame->nPartitions();
        for (partition_idx partitionIdx = 0; partitionIdx < nPartitions; partitionIdx++) {
            classificationOfAPartition_InternalAndBoundary(partitionIdx);
        }
    }


    void connectivity()
    {
        auto const& nPartitions = m_frame->nPartitions();

        for (partition_idx partitionIdx = 0; partitionIdx < nPartitions; partitionIdx++) {
            partition_idx                                                   UP = (partitionIdx + nPartitions + 1) % nPartitions;
            partition_idx                                                   DW = (partitionIdx + nPartitions - 1) % nPartitions;
            std::array<const LocalIndexingInfo_t*, ComDirection_e::COM_NUM> neighbourInfo;

            neighbourInfo[ComDirection_e::COM_DW] = &m_frame->localIndexingInfo(DW);
            neighbourInfo[ComDirection_e::COM_UP] = &m_frame->localIndexingInfo(UP);
            m_frame->localIndexingInfo<Neon::Access::readWrite>(partitionIdx).updateGhostInfo(neighbourInfo);
        }

        // Neon::set::MemDevSet<int64_t>& localConectivity = m_frame->localConectivity();
        for (partition_idx partitionIdx = 0; partitionIdx < nPartitions; partitionIdx++) {
            initConnectivityVector(partitionIdx);
        }
    }


    void setFrame()
    {
        classification();
        m_frame->setConnectivityAndIverseMappingStorage();
        connectivity();
        inverseMapping();
        m_frame->updateConnectivityAndInverseMapping();
    }
};  // namespace internals

}  // namespace internals
}  // namespace Neon::domain::internal::eGrid