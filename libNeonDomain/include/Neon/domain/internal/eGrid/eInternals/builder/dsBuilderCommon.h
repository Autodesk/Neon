#pragma once
#include <assert.h>

#include <functional>

#include "Neon/domain/interface/common.h"
#include "Neon/domain/interface/Stencil.h"
#include "Neon/domain/internal/eGrid/eCommon.h"
#include "Neon/domain/internal/eGrid/eInternals/Partitioning.h"
#include "Neon/domain/internal/eGrid/ePartitionIndexSpace.h"
#include "Neon/sys/memory/mem3d.h"

namespace Neon::domain::internal::eGrid {

namespace internals {
//
// using stencil = Neon::domain::stencil_t;
using count_t = Neon::domain::internal::eGrid::count_t; /** Local Id  */

using local_idx = int64_t;       /** Local Id */
using internal_idx = int64_t;    /** Local Id. Local ids includes internal boundaries and ghost */
using boundaries_idx = int64_t;  /** Id only relative to the supported boundary sections */
using bdrRelative_idx = int64_t; /** Id with respect to only one of the supported boundary*/
using ghost_idx = int64_t;       /** Id relative to ghost elements*/

using global_idx = int64_t;    /** Global Id  */
using partition_idx = int32_t; /** Partition Id */
using neighbour_idx = int32_t; /** Neighbour Id in the stencil defined by the user */

using ngh_idx = neighbour_idx;


struct partition_et
{
    enum e : int64_t
    {
        valid = 0,
        invalid = -1,
    };
};

struct global_et
{
    enum e : int64_t
    {
        active = -1,
        inactive = -2,
    };
};

struct neighbour_et
{
    enum e : int64_t
    {
        local = +1,   /** All active neighbours belong to the same partition */
        remote = -1,  /** At least one active neighbour belongs to a different partition */
        invalid = -3, /** All neighbours of a ghost element are classify as null */
    };
};

struct element_et
{
    /**
     * Classes of elements:
     * 1. Internal
     * 2.
     */
    enum e : int64_t
    {
        internal = 10,
        boundary = 20,
        ghost = 30,
    };
};

/**
 * Helper structure to store information on each partition
 */
template <typename T_ta>
struct DataSet
{
   private:
    std::vector<T_ta> m_data;

   public:
    DataSet() = default;
    DataSet(int nPartition)
        : m_data(nPartition) {}
    DataSet(int nPartition, T_ta defaultVal)
        : m_data(nPartition, defaultVal)
    {
    }

    template <Neon::Access accessType_ta = Neon::Access::read>
    std::enable_if_t<accessType_ta == Neon::Access::readWrite, T_ta&> ref(partition_idx partitionIdx)
    {
        return (m_data[partitionIdx]);
    }

    template <Neon::Access accessType_ta = Neon::Access::read>
    std::enable_if_t<accessType_ta == Neon::Access::read, const T_ta&> ref(partition_idx partitionIdx) const
    {
        return (m_data[partitionIdx]);
    }

    size_t nPartitions() const
    {
        return m_data.size();
    }
};


template <typename T_ta>
struct ListDataSet
{
   private:
    std::vector<std::vector<T_ta>> m_data;

   public:
    ListDataSet() = default;
    ListDataSet(int nPartition)
        : m_data(nPartition) {}
    ListDataSet(int nPartition, size_t defaultSize)
        : m_data(nPartition)
    {
        for (auto&& l : m_data) {
            l = std::vector<T_ta>(defaultSize);
        }
    }

    T_ta& ref(partition_idx partitionIdx, size_t elementId)
    {
        return (m_data[partitionIdx])[elementId];
    }

    std::vector<T_ta>& refList(partition_idx partitionIdx)
    {
        return (m_data[partitionIdx]);
    }

    template <typename... args>
    size_t append(partition_idx partitionIdx, args... value)
    {
        m_data[partitionIdx].emplace_back(value...);
        size_t idx = m_data.size() - 1;
        return idx;
    }

    size_t lengthList(partition_idx partitionIdx) const
    {
        return m_data[partitionIdx].size();
    }

    size_t nPartition() const
    {
        return m_data.size();
    }

    std::vector<T_ta>& list(partition_idx partitionIdx)
    {
        m_data[partitionIdx];
    }
    const std::vector<T_ta>& list(partition_idx partitionIdx) const
    {
        m_data[partitionIdx];
    }
    size_t listLenght(partition_idx partitionIdx) const
    {
        return m_data[partitionIdx].size();
    }
};

template <typename T_ta>
struct NghDataSet
{
   private:
    std::vector<T_ta> m_nData;

   public:
    NghDataSet() = default;
    NghDataSet(int nNeighbours)
        : m_nData(nNeighbours) {}

    T_ta& ref(neighbour_idx neighbourIdx)
    {
        return m_nData[neighbourIdx];
    }

    const T_ta& ref(neighbour_idx neighbourIdx) const
    {
        return m_nData[neighbourIdx];
    }

    bool isActive(neighbour_idx neighbourIdx) const
    {
        bool isActive = m_nData[neighbourIdx] >= 0;
        return isActive;
    }

    size_t nNeighbours() const
    {
        return m_nData.size();
    }

    std::vector<T_ta>& list()
    {
        return m_nData;
    }

    const std::vector<T_ta>& list() const
    {
        return m_nData;
    }
};

/**
 * A connectivityDataSet object stores the ids of neighbours for a set of points
 * @tparam T_ta
 */
template <typename T_ta>
struct connectivityListSet_t
{
   private:
    int                               m_nPartition{0};
    int                               m_nNeighbours{0};
    ListDataSet<NghDataSet<T_ta>> m_nData;

   public:
    connectivityListSet_t() = default;
    connectivityListSet_t(int nPartition, int nNeighbours)
        : m_nPartition(nPartition),
          m_nNeighbours(nNeighbours),
          m_nData(nPartition)
    {
        for (auto&& n : m_nData) {
            n = NghDataSet<T_ta>(nNeighbours);
        }
    }

    T_ta& ref(partition_idx partitionIdx, size_t elementId, neighbour_idx neighbourIdx)
    {
        return m_nData.ref(partitionIdx, elementId).ref(neighbourIdx);
    }

    NghDataSet<T_ta>& ref(partition_idx partitionIdx, size_t elementId)
    {
        return m_nData.ref(partitionIdx, elementId);
    }

    // Append a full neighbourDataSet<T_ta> for be partition.
    size_t append(partition_idx partitionIdx, const NghDataSet<T_ta>& data)
    {
        size_t appendIdx;
        appendIdx = m_nData.append(partitionIdx, data);
        return appendIdx;
    }

    size_t nElements(partition_idx partitionIdx) const
    {
        return m_nData.listLenght(partitionIdx);
    }
};

using BdrDepClass_e = Neon::domain::internal::eGrid::BdrDepClass_e;
using ComDirection_e = Neon::domain::internal::eGrid::ComDirection_e;

/**
 * Structure to store information on indexing for a single partition.
 * The structure defines length and position of the following type of elements:
 * 1. internal
 * 2. boundary
 * 3. ghost
 */
struct LocalIndexingInfo_t
{
   public:
   private:
    // CONST
    using Cell = eCell;
    static const Cell::Offset m_internalOff = 0; /** Index where internal data starts. It is always 0 */

    // INPUT OF CONSTRUCTOR
    count_t       m_internalCount = -1;                           /** Number of internal elements */
    count_t       m_bdrCount[ComDirection_e::COM_NUM] = {-1, -1}; /** Number of elements in the calsses up, dw, both */
    count_t       m_bdrOverlappedCount = {0};
    partition_idx m_partNghIdx[ComDirection_e::COM_NUM] = {-1, -1}; /** partition ID of the 2 neighbour partitions */
    partition_idx m_targetPrt{-1};                                  /** Id of the partition associated with this indexing */

    // COMPUTED BY CONSTRUCTOR
    Cell::Offset m_bdrOff[ComDirection_e::COM_NUM] = {-1, -1}; /** Offset for Up and Down elements that depends on HALO values */
    Cell::Offset m_haloOff = 0;                                /** Index where halo/ghost data starts. */

    // COMPUTED BY OTHER METHODS [as this needs global information]
    // -> (updateGhostInfo)
    Cell::Offset m_remoteBdrCount[ComDirection_e::COM_NUM] = {-1, -1}; /** Offset for where to store HALO data from other partitions */
    Cell::Offset m_remoteBdrOff[ComDirection_e::COM_NUM] = {-1, -1};   /** Offset for where to read data to store in HALOS */
    Cell::Offset m_ghostOff[ComDirection_e::COM_NUM] = {-1, -1};       /** Offset for where to read data to store in HALOS */


   public:
    LocalIndexingInfo_t() = default;

    LocalIndexingInfo_t(partition_idx                                             targetPrt,
                        size_t                                                    nInternal,
                        const std::array<partition_idx, ComDirection_e::COM_NUM>& nghPrt,
                        const std::array<local_idx, BdrDepClass_e::num>&          numBoundary)
    {
        // STORING SOME CONSTRUCTOR INPUT
        {
            m_internalCount = count_t(nInternal);

            m_bdrCount[ComDirection_e::COM_DW] = count_t(numBoundary[BdrDepClass_e::DW] + numBoundary[BdrDepClass_e::BOTH]);
            m_bdrCount[ComDirection_e::COM_UP] = count_t(numBoundary[BdrDepClass_e::UP] + numBoundary[BdrDepClass_e::BOTH]);

            m_partNghIdx[ComDirection_e::COM_DW] = nghPrt[ComDirection_e::COM_DW];
            m_partNghIdx[ComDirection_e::COM_UP] = nghPrt[ComDirection_e::COM_UP];

            assert(nghPrt[ComDirection_e::COM_DW] != partition_et::invalid || nghPrt[ComDirection_e::COM_UP] != partition_et::invalid);

            m_bdrOverlappedCount = count_t(numBoundary[BdrDepClass_e::BOTH]);
            m_targetPrt = targetPrt;
        }

        // COMPUTED BY CONSTRUCTOR
        {
            auto computeParas = [this] {
                // m_bdrOff
                m_bdrOff[ComDirection_e::COM_DW] = m_internalCount;
                m_bdrOff[ComDirection_e::COM_UP] = m_internalCount + m_bdrCount[BdrDepClass_e::DW] - m_bdrOverlappedCount;

                // m_haloOff
                m_haloOff = m_internalCount + m_bdrCount[ComDirection_e::COM_DW] + m_bdrCount[ComDirection_e::COM_UP] - m_bdrOverlappedCount;
            };
            computeParas();
        }
    }

    const partition_idx& prtIdx() const
    {
        return m_targetPrt;
    }

    auto nghIdx(ComDirection_e::e comDir) const -> const partition_idx&
    {
        return m_partNghIdx[comDir];
    }

    const Cell::Offset& internalOff() const
    {
        return m_internalOff;
    }

    const Cell::Offset& bdrOff(ComDirection_e::e prtDirection) const
    {
        return m_bdrOff[prtDirection];
    }

    const Cell::Offset& ghostOff(ComDirection_e::e prtDirection) const
    {
        return m_ghostOff[prtDirection];
    }

    const Cell::Offset& remoteBdrOff(ComDirection_e::e prtDirection) const
    {
        return m_remoteBdrOff[prtDirection];
    }

    const count_t& internalCount() const
    {
        return m_internalCount;
    }

    const count_t& bdrCount(ComDirection_e::e prtDirection) const
    {
        return m_bdrCount[prtDirection];
    }

    count_t bdrCount() const
    {
        return m_bdrCount[ComDirection_e::COM_DW] + m_bdrCount[ComDirection_e::COM_UP] - m_bdrOverlappedCount;
    }

    const count_t& remoteBdrCount(ComDirection_e::e prtDirection) const
    {
        return m_remoteBdrCount[prtDirection];
    }

    count_t nElements(bool doCountHalo) const
    {
        const size_t local = m_haloOff;
        const size_t ghost = (!doCountHalo) ? 0 : m_remoteBdrCount[ComDirection_e::COM_DW] + m_remoteBdrCount[ComDirection_e::COM_UP];
        const size_t totalCount = local + ghost;
        return count_t(totalCount);
    }

    //    BdrDepClass_e::e bdrDirection(neighbour_idx neighbourIdx) const
    //    {
    //        assert(neighbourIdx == m_partNghIdx[0] || neighbourIdx == m_partNghIdx[1]);
    //        BdrDepClass_e::e ret = BdrDepClass_e::num;
    //        ret = (neighbourIdx == m_partNghIdx[BdrDepClass_e::e::DW]) ? BdrDepClass_e::e::DW : ret;
    //        ret = (neighbourIdx == m_partNghIdx[BdrDepClass_e::e::UP]) ? BdrDepClass_e::e::UP : ret;
    //        return ret;
    //    }

    ComDirection_e::e bdrDirection(neighbour_idx neighbourIdx) const
    {
        assert(neighbourIdx == m_partNghIdx[ComDirection_e::COM_DW] || neighbourIdx == m_partNghIdx[ComDirection_e::COM_UP]);
        ComDirection_e::e ret = ComDirection_e::COM_NUM;
        ret = (neighbourIdx == m_partNghIdx[ComDirection_e::COM_DW]) ? ComDirection_e::COM_DW : ret;
        ret = (neighbourIdx == m_partNghIdx[ComDirection_e::COM_UP]) ? ComDirection_e::COM_UP : ret;
        return ret;
    }

    local_idx remoteToLocal(partition_idx neighbourIdx, local_idx remoteIdxForBdr) const
    {
        assert(neighbourIdx == m_partNghIdx[0] || neighbourIdx == m_partNghIdx[1]);
        const ComDirection_e::e direction = bdrDirection(neighbourIdx);
        const bdrRelative_idx   bdrRelativeIdx = remoteIdxForBdr - m_remoteBdrOff[direction];
        const local_idx         local = m_ghostOff[direction] + bdrRelativeIdx;
        return local;
    }

    void updateGhostInfo(std::array<const LocalIndexingInfo_t*, ComDirection_e::COM_NUM> neighbourInfo)
    {
        m_partNghIdx[ComDirection_e::COM_DW] = neighbourInfo[ComDirection_e::COM_DW]->prtIdx();
        m_partNghIdx[ComDirection_e::COM_UP] = neighbourInfo[ComDirection_e::COM_UP]->prtIdx();

        m_remoteBdrCount[ComDirection_e::COM_DW] = neighbourInfo[ComDirection_e::COM_DW]->bdrCount(ComDirection_e::COM_UP);
        m_remoteBdrCount[ComDirection_e::COM_UP] = neighbourInfo[ComDirection_e::COM_UP]->bdrCount(ComDirection_e::COM_DW);

        m_remoteBdrOff[ComDirection_e::COM_DW] = neighbourInfo[ComDirection_e::COM_DW]->bdrOff(ComDirection_e::COM_UP);
        m_remoteBdrOff[ComDirection_e::COM_UP] = neighbourInfo[ComDirection_e::COM_UP]->bdrOff(ComDirection_e::COM_DW);

        m_ghostOff[ComDirection_e::COM_DW] = m_haloOff;
        m_ghostOff[ComDirection_e::COM_UP] = m_haloOff + m_remoteBdrCount[ComDirection_e::COM_DW];
    }
};


/**
 * Information associated to each 3d global index.
 */
struct elmLocalInfo_t
{
   private:
    partition_idx  m_prtIdx{-1};
    local_idx      m_localIdx{-1};
    bool           m_isShapeBoundary{false};
    Neon::DataView m_dataView;

   public:
    const partition_idx& getPrtIdx() const
    {
        return m_prtIdx;
    }
    const local_idx& getLocalIdx() const
    {
        assert(isFullySet());
        return m_localIdx;
    }

    auto getDataView() const
        -> Neon::DataView
    {
        return m_dataView;
    }

   public:
    elmLocalInfo_t() = default;

    /**
     * Constructor for phases when we only know the partition Id
     * @param prtIdx_
     */
    elmLocalInfo_t(partition_idx prtIdx_)
    {
        m_prtIdx = prtIdx_;
        m_localIdx = global_et::inactive;
        m_dataView = Neon::DataView::STANDARD;
    }

    /**
     * Constructor for phases when we know the partition Id as well as the local Id
     * @param prtIdx_
     */
    elmLocalInfo_t(partition_idx prtIdx_, local_idx localIdx_, bool isShapeBoundary, DataView dw)
    {
        if (dw != Neon::DataView::INTERNAL && dw != Neon::DataView::BOUNDARY) {
            NEON_THROW_UNSUPPORTED_OPTION("");
        }
        m_prtIdx = prtIdx_;
        m_localIdx = localIdx_;
        m_isShapeBoundary = isShapeBoundary;
        m_dataView = dw;
    }

    /**
     * Constructor for phases when we only know about the status: active or not
     * @param status
     */
    elmLocalInfo_t(global_et::e status)
    {
        switch (status) {
            case global_et::active: {
                m_prtIdx = global_et::active;
                m_localIdx = global_et::active - 33;
                return;
            }
            case global_et::inactive: {
                m_prtIdx = global_et::inactive;
                m_localIdx = global_et::inactive;
                return;
            }
        }
    }

    /**
     * Returns true if the node is active
     * @return
     */
    bool isActive() const
    {
        return (m_prtIdx == global_et::active) || m_prtIdx >= 0;
    }

    /**
     * Returns true
     * @return
     */
    bool isInactive() const
    {
        return m_prtIdx == global_et::inactive;
    }

    bool isFullySet() const
    {
        return m_prtIdx >= 0 && m_localIdx >= 0;
    }

    auto isDomainBoundary()
        const
        -> bool
    {
        return m_isShapeBoundary;
    }
};


struct InternalInfo_t
{
    // TODO@Max: remove commented out code
   private:
    std::vector<NghDataSet<global_idx>> m_globalOrLocalNghList;
    bool                                  isLocal = {false};

   public:
    size_t append(const NghDataSet<global_idx>& Ngh)
    {
        m_globalOrLocalNghList.push_back(Ngh);
        return m_globalOrLocalNghList.size();
    }

    const std::vector<NghDataSet<global_idx>>& NghList() const
    {
        return m_globalOrLocalNghList;
    }

    size_t nInternal() const
    {
        return m_globalOrLocalNghList.size();
    }

    global_idx& NghRef(internal_idx internalIdx, neighbour_idx neighbourIdx)
    {
        return m_globalOrLocalNghList[internalIdx].ref(neighbourIdx);
    }

    const global_idx& Ngh(internal_idx internalIdx, neighbour_idx neighbourIdx) const
    {
        return m_globalOrLocalNghList[internalIdx].ref(neighbourIdx);
    }

    void convertNeighboursToLocal(const elmLocalInfo_t* const elmLocalInfo)
    {
        isLocal = true;
#pragma omp parallel for collapse(2) default(shared)
        for (int64_t idx = 0; idx < int64_t(m_globalOrLocalNghList.size()); idx++) {
            for (int32_t nghIdx = 0; nghIdx < int32_t(m_globalOrLocalNghList[0].nNeighbours()); nghIdx++) {
                global_idx globalIdx = m_globalOrLocalNghList[idx].list()[nghIdx];
                if (globalIdx >= 0) {
                    local_idx localIdx = elmLocalInfo[globalIdx].getLocalIdx();
                    m_globalOrLocalNghList[idx].list()[nghIdx] = localIdx;
                }
            }
        }
    }
};


struct BoundariesInfo_t
{

    enum nextOp_e
    {
        constructOP,
        settingOP,
        appendOP,
        queeringOP
    };

   private:
    // Set during "setting" operation
    std::array<partition_idx, ComDirection_e::COM_NUM>                       m_prtIds_{partition_et::invalid, partition_et::invalid};
    std::array<std::vector<global_idx>, BdrDepClass_e::num>                  m_globalIdxList;
    std::array<std::vector<NghDataSet<global_idx>>, BdrDepClass_e::num>    m_nghIdxList;
    std::array<std::vector<NghDataSet<partition_idx>>, BdrDepClass_e::num> m_nghPartList;

    nextOp_e m_supportedOp{nextOp_e::settingOP};
    size_t   m_nInternal{0};

   public:
    BoundariesInfo_t() = default;

    void settingOp(std::array<partition_idx, ComDirection_e::COM_NUM> prtIdxes)
    {
        assert(nextOp_e::settingOP == m_supportedOp);
        m_prtIds_[ComDirection_e::COM_DW] = prtIdxes[ComDirection_e::COM_DW];
        m_prtIds_[ComDirection_e::COM_UP] = prtIdxes[ComDirection_e::COM_UP];
        m_supportedOp = appendOP;
    }

    /**
     * Appends a new cell to one of the cell classes (BdrDir_e)..
     * Returns the relative local index for the
     *
     * @param neighbourIdA
     * @param neighbourIdB
     * @param newGlobal
     * @param nghGlobals
     * @param nghPartIds
     * @return
     */
    local_idx append(partition_idx                      neighbourIdA,
                     partition_idx                      neighbourIdB,
                     global_idx                         newGlobal,
                     const NghDataSet<global_idx>&    nghGlobals,
                     const NghDataSet<partition_idx>& nghPartIds)
    {

        assert(nextOp_e::appendOP == m_supportedOp);

        // OVERLAPPED BDR
        if ((neighbourIdA != partition_et::invalid) && (neighbourIdB != partition_et::invalid)) {
            const BdrDepClass_e::e target = BdrDepClass_e::BOTH;
            m_globalIdxList[target].push_back(newGlobal);
            m_nghIdxList[target].push_back(nghGlobals);
            m_nghPartList[target].push_back(nghPartIds);
            const local_idx localIdx = m_globalIdxList[BdrDepClass_e::DW].size() - 1;
            return localIdx;
        }

        assert(neighbourIdB == partition_et::invalid);

        const BdrDepClass_e::e target = (m_prtIds_[ComDirection_e::COM_DW] == neighbourIdA) ? BdrDepClass_e::DW : BdrDepClass_e::UP;
        m_globalIdxList[target].push_back(newGlobal);
        m_nghIdxList[target].push_back(nghGlobals);
        m_nghPartList[target].push_back(nghPartIds);
        local_idx localIdx = m_globalIdxList[BdrDepClass_e::DW].size() - 1;
        return localIdx;
    }

    void setLoadingComplete(size_t nInternal)
    {
        assert(nextOp_e::appendOP == m_supportedOp);
        m_nInternal = nInternal;
        m_supportedOp = queeringOP;
    }

    const std::vector<global_idx>& getGlobalIdxList(BdrDepClass_e::e direction) const
    {
        assert(nextOp_e::queeringOP == m_supportedOp);
        return m_globalIdxList[direction];
    }

    global_idx getGlobalIdx(BdrDepClass_e::e direction, bdrRelative_idx bdrRelativeIdx) const
    {
        assert(nextOp_e::queeringOP == m_supportedOp);
        return m_globalIdxList[direction][bdrRelativeIdx];
    }

    global_idx isDomainBoundary(BdrDepClass_e::e direction, bdrRelative_idx bdrRelativeIdx) const
    {
        assert(nextOp_e::queeringOP == m_supportedOp);
        auto nghPrtList = m_nghPartList[direction][bdrRelativeIdx];
        for (auto& nghPrt : nghPrtList.list()) {
            if (nghPrt == partition_et::invalid) {
                return true;
            }
        }
        return false;
    }

    const std::array<partition_idx, ComDirection_e::COM_NUM>& getPrtIds() const
    {
        return m_prtIds_;
    }

    std::array<local_idx, BdrDepClass_e::num> getNumBoundaries() const
    {
        std::array<local_idx, BdrDepClass_e::num> res;
        res[BdrDepClass_e::DW] = m_globalIdxList[BdrDepClass_e::DW].size();
        res[BdrDepClass_e::BOTH] = m_globalIdxList[BdrDepClass_e::BOTH].size();
        res[BdrDepClass_e::UP] = m_globalIdxList[BdrDepClass_e::UP].size();
        return res;
    }

    count_t getNumBoundaries(BdrDepClass_e::e bdrDepClass_e) const
    {
        return count_t(m_globalIdxList[bdrDepClass_e].size());
    }

    /**
     * Return full local id of the selected boundary element.
     * @param direction
     * @param offSet
     * @return
     */
    size_t computeLocalIdx(BdrDepClass_e::e direction, local_idx offSet) const
    {
        assert(nextOp_e::queeringOP == m_supportedOp);
        assert(direction != BdrDepClass_e::num);

        local_idx result = m_nInternal;

        switch (direction) {
            case BdrDepClass_e::DW: {
                result += 0;
                break;
            }
            case BdrDepClass_e::BOTH: {
                result += m_globalIdxList[BdrDepClass_e::DW].size();
                break;
            }
            case BdrDepClass_e::UP: {
                result += m_globalIdxList[BdrDepClass_e::DW].size();
                result += m_globalIdxList[BdrDepClass_e::BOTH].size();
                break;
            }
            case BdrDepClass_e::num: {
                assert(false);
            }
        }

        result += offSet;
        return result;
    }

    size_t nGlobal(BdrDepClass_e::e direction) const
    {
        assert(nextOp_e::queeringOP == m_supportedOp);
        assert(direction != BdrDepClass_e::num);
        return m_globalIdxList[direction].size();
    }

    auto getNghIdx(BdrDepClass_e::e direction, local_idx directionLocalId) -> NghDataSet<global_idx>&
    {
        assert(nextOp_e::queeringOP == m_supportedOp);
        assert(direction != BdrDepClass_e::num);

        return m_nghIdxList[direction][directionLocalId];
    }

    auto getNghIdx(BdrDepClass_e::e direction, local_idx directionLocalId) const -> const NghDataSet<global_idx>&
    {
        assert(nextOp_e::queeringOP == m_supportedOp);
        assert(direction != BdrDepClass_e::num);

        return m_nghIdxList[direction][directionLocalId];
    }

    auto getNghPrt(BdrDepClass_e::e direction, local_idx directionLocalId) -> NghDataSet<partition_idx>&
    {
        assert(nextOp_e::queeringOP == m_supportedOp);
        assert(direction != BdrDepClass_e::num);

        return m_nghPartList[direction][directionLocalId];
    };

    auto getNghPrt(BdrDepClass_e::e direction, local_idx directionLocalId) const -> const NghDataSet<partition_idx>&
    {
        assert(nextOp_e::queeringOP == m_supportedOp);
        assert(direction != BdrDepClass_e::num);

        return m_nghPartList[direction][directionLocalId];
    };


    void convertBoundaryConnectivityFromGlobalToLocal(partition_idx                         partIdx,
                                                      const elmLocalInfo_t*                 G2L,
                                                      const DataSet<LocalIndexingInfo_t>& localIndexingInfoDataSet)
    {
        for (auto&& direction : std::vector<BdrDepClass_e::e>{BdrDepClass_e::DW, BdrDepClass_e::BOTH, BdrDepClass_e::UP}) {

            int64_t nBoundary = m_nghIdxList[direction].size();
            int32_t nNeighbours = 0;
            if (!m_nghIdxList[direction].empty()) {
                nNeighbours = static_cast<neighbour_idx>(m_nghIdxList[direction][0].nNeighbours());
            }
#pragma omp parallel for collapse(2) default(shared)
            for (bdrRelative_idx bdrRelativeIdx = 0; bdrRelativeIdx < nBoundary; bdrRelativeIdx++) {
                for (neighbour_idx neighbourIdx = 0; neighbourIdx < nNeighbours; neighbourIdx++) {
                    if (!m_nghIdxList[direction][bdrRelativeIdx].isActive(neighbourIdx)) {
                        continue;
                    }
                    const global_idx    globalNghIdx = m_nghIdxList[direction][bdrRelativeIdx].ref(neighbourIdx);
                    const partition_idx partitionNghIdx = m_nghPartList[direction][bdrRelativeIdx].ref(neighbourIdx);
                    local_idx           localNghIdx = -1;

                    if (partitionNghIdx == partIdx) {
                        // Internal connectivity (either local or boundary)
                        localNghIdx = G2L[globalNghIdx].getLocalIdx();
                    } else {
                        // Ghost neighbour
                        const local_idx     remoteLocalIdx = G2L[globalNghIdx].getLocalIdx();
                        const partition_idx remoteNghIdx = G2L[globalNghIdx].getPrtIdx();

                        localNghIdx = localIndexingInfoDataSet.ref(partIdx).remoteToLocal(remoteNghIdx, remoteLocalIdx);
                    }
                    m_nghIdxList[direction][bdrRelativeIdx].ref(neighbourIdx) = localNghIdx;
                }
            }
        }
    }
};
}  // namespace internals
}  // namespace eGrid

