#pragma once
#include "Neon/set/DataSet.h"

#include <functional>

#include "Neon/core/tools/io/ioToVti.h"
#include "Neon/domain/internal/haloUpdateType.h"
#include "Neon/domain/interface/Stencil.h"
#include "dsBuilderCommon.h"
#include "Neon/set/DevSet.h"
#include "Neon/sys/memory/mem3d.h"
#include "dsBuilderCommon.h"

namespace Neon::domain::internal::eGrid {

namespace internals {

/**
 * Sparsity and connectivity information
 */
struct dsFrame_t
{


   private:
    Neon::set::DevSet     m_devSet;
    index_3d              m_cellDomain;
    int64_t               m_nDomainElements = {0};
    int64_t               m_nActiveElements = {0};
    Neon::domain::Stencil m_stencil;
    int                   m_nPartitions{0};
    int                   m_nNeighbours{0};

    Neon::sys::Mem3d_t<elmLocalInfo_t>
        m_globalToLocal;

    DataSet<LocalIndexingInfo_t> m_localIndexingInfo; /** all information about local indexing of a partition */
    DataSet<InternalInfo_t>      m_internalToGlobal;  /** data to map from partition internal Idx to global **/
    DataSet<BoundariesInfo_t>    m_boundaryToGlobal;  /** data to map from local boundary idx to global */

    // TODO@(Max){use mirror instead}
    Neon::set::MemDevSet<int32_t> m_localConectivityCPU;
    Neon::set::MemDevSet<int32_t> m_localConectivityGPU;

    // TODO@(Max){use mirror instead}
    Neon::set::MemDevSet<index_t> m_inverseMappingCPU;
    Neon::set::MemDevSet<index_t> m_inverseMappingGPU;

    struct dataDependencyFlag_t
    {
        bool isActive[Neon::domain::HaloUpdateMode_e::e::HALOUPDATEMODE_LEN][ComDirection_e::COM_NUM];

        dataDependencyFlag_t()
        {
            isActive[Neon::domain::HaloUpdateMode_e::e::STANDARD][ComDirection_e::COM_UP] = true;
            isActive[Neon::domain::HaloUpdateMode_e::e::STANDARD][ComDirection_e::COM_DW] = true;
            isActive[Neon::domain::HaloUpdateMode_e::e::STANDARD][ComDirection_e::COM_UP] = true;
            isActive[Neon::domain::HaloUpdateMode_e::e::STANDARD][ComDirection_e::COM_DW] = true;
        }
    };

    std::vector<std::vector<dataDependencyFlag_t>> m_DependencyFlagByDestination;

   public:
    auto globalToLocal() -> Neon::sys::Mem3d_t<elmLocalInfo_t>&
    {
        return m_globalToLocal;
    }

    auto globalToLocal() const -> const Neon::sys::Mem3d_t<elmLocalInfo_t>&
    {
        return m_globalToLocal;
    }

    Neon::set::MemDevSet<int32_t>& connectivity(const Neon::DeviceType& devEt)
    {
        switch (devEt) {
            case Neon::DeviceType::CPU:
            case Neon::DeviceType::OMP: {
                return m_localConectivityCPU;
            }
            case Neon::DeviceType::CUDA: {
                return m_localConectivityGPU;
            }
            default: {
                NeonException exception("Frame_t");
                exception << "Incompatible device type.";
                NEON_THROW(exception);
            }
        }
    }

    Neon::set::MemDevSet<index_t>& inverseMapping(const Neon::DeviceType& devEt)
    {
        switch (devEt) {
            case Neon::DeviceType::CPU:
            case Neon::DeviceType::OMP: {
                return m_inverseMappingCPU;
            }
            case Neon::DeviceType::CUDA: {
                return m_inverseMappingGPU;
            }
            default: {
                NeonException exception("Frame_t");
                exception << "Incompatible device type.";
                NEON_THROW(exception);
            }
        }
    }
    auto inverseMapping(const Neon::DeviceType& devEt) const -> const Neon::set::MemDevSet<index_t>&
    {
        switch (devEt) {
            case Neon::DeviceType::CPU:
            case Neon::DeviceType::OMP: {
                return m_inverseMappingCPU;
            }
            case Neon::DeviceType::CUDA: {
                return m_inverseMappingGPU;
            }
            default: {
                NeonException exception("Frame_t");
                exception << "Incompatible device type.";
                NEON_THROW(exception);
            }
        }
    }

    const Neon::set::MemDevSet<int32_t>& connectivity(const Neon::DeviceType& devEt) const
    {
        switch (devEt) {
            case Neon::DeviceType::CPU:
            case Neon::DeviceType::OMP: {
                return m_localConectivityCPU;
            }
            case Neon::DeviceType::CUDA: {
                return m_localConectivityGPU;
            }
            default: {
                NeonException exception("Frame_t");
                exception << "Incompatible device type.";
                NEON_THROW(exception);
            }
        }
    }

    auto updateConnectivityAndInverseMapping()
        -> void
    {
        if (m_devSet.type() == Neon::DeviceType::CUDA) {
            auto streamSet = m_devSet.newStreamSet();
            m_localConectivityGPU.updateFrom<Neon::run_et::sync>(streamSet, m_localConectivityCPU);
        }
    }

    DataSet<InternalInfo_t>& internalToGlobal()
    {
        return m_internalToGlobal;
    }

    DataSet<BoundariesInfo_t>& boundaryToGlobal()
    {
        return m_boundaryToGlobal;
    }

    DataSet<LocalIndexingInfo_t>& localIndexingInfo()
    {
        return m_localIndexingInfo;
    }

    template <Neon::Access accessType_ta = Neon::Access::read>
    auto
    localIndexingInfo(partition_idx partitionIdx) const
        -> std::enable_if_t<accessType_ta == Neon::Access::read, const LocalIndexingInfo_t&>
    {
        return m_localIndexingInfo.ref(partitionIdx);
    }

    template <Neon::Access accessType_ta = Neon::Access::read>
    std::enable_if_t<accessType_ta == Neon::Access::readWrite, LocalIndexingInfo_t&> localIndexingInfo(partition_idx partitionIdx)
    {
        return m_localIndexingInfo.ref<Neon::Access::readWrite>(partitionIdx);
    }


    const index_3d& domain() const
    {
        return m_cellDomain;
    }

    const int64_t& nDomainElements() const
    {
        return m_nDomainElements;
    }

    const int64_t& nActiveElements() const
    {
        return m_nActiveElements;
    }

    const int& nPartitions() const
    {
        return m_nPartitions;
    }

    const int& nNeighbours() const
    {
        return m_nNeighbours;
    }

    const Neon::domain::Stencil& stencil() const
    {
        return m_stencil;
    }

    auto activeDependencyByDestination() -> const std::vector<std::vector<dataDependencyFlag_t>>&
    {
        return m_DependencyFlagByDestination;
    }
    dsFrame_t() = default;
    dsFrame_t(const Neon::set::DevSet&                          devSet,
              const Neon::index_3d&                             domain,
              const std::function<bool(const Neon::index_3d&)>& inOut,
              int                                               nPartitions,
              const Neon::domain::Stencil&                      stencil)
        : m_devSet(devSet),
          m_cellDomain(domain),
          m_nDomainElements(domain.rMulTyped<size_t>()),
          m_nActiveElements(0),
          m_stencil(stencil),
          m_nPartitions(nPartitions),
          m_nNeighbours(stencil.nNeighbours()),
          m_globalToLocal(1, Neon::DeviceType::CPU, Neon::sys::DeviceID(0), Neon::Allocator::MALLOC, domain, Neon::index_3d(0), Neon::memLayout_et::structOfArrays, Neon::sys::MemAlignment(), Neon::memLayout_et::OFF),
          m_localIndexingInfo(nPartitions),
          m_internalToGlobal(m_nPartitions),
          m_boundaryToGlobal(m_nPartitions)
    {
        p_detectActiveElements(inOut);

        std::vector<dataDependencyFlag_t> dataDependencyFlagForInternalPartitions(m_stencil.nNeighbours());
        {
            // PARSE it fro both read update or write update...
            auto latticeMode = Neon::domain::HaloUpdateMode_e::LATTICE;

            for (int i = 0; i < m_stencil.nNeighbours(); i++) {

                // With respect to the partition needing the data
                {  // UP
                    bool putUp = true;
                    if (m_stencil.neighbours()[i].z >= 0 && m_stencil.neighbours()[i].y >= 0 && m_stencil.neighbours()[i].x >= 0) {
                        putUp = false;
                    }
                    dataDependencyFlagForInternalPartitions[i].isActive[latticeMode][ComDirection_e::COM_UP] = putUp;
                }
                {  // DOWN
                    bool putDown = true;
                    if (m_stencil.neighbours()[i].z <= 0 && m_stencil.neighbours()[i].y <= 0 && m_stencil.neighbours()[i].x <= 0) {
                        putDown = false;
                    }
                    dataDependencyFlagForInternalPartitions[i].isActive[latticeMode][ComDirection_e::COM_DW] = putDown;
                }
            }
        }

        m_DependencyFlagByDestination = std::vector<std::vector<dataDependencyFlag_t>>(m_nPartitions,
                                                                                       dataDependencyFlagForInternalPartitions);
    }  // namespace internals

    template <typename T_ta>
    Neon::set::DataSet<T_ta> newDataSet()
    {
        return Neon::set::DataSet<T_ta>(this->nPartitions());
    }

    template <typename T_ta>
    NghDataSet<T_ta> newNghDataSet() const
    {
        return NghDataSet<T_ta>(this->nNeighbours());
    }

   private:
    void p_detectActiveElements(const std::function<bool(const Neon::index_3d&)>& inOut)
    {
        size_t numActiveElem = 0;
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4849)  //  warning C4849: OpenMP 'collapse' clause ignored in 'parallel for' directive  (This is a bug in VS 2019 16.3 which will be hopefully fixed in the next version)
#endif
#pragma omp parallel for collapse(3) reduction(+ \
                                               : numActiveElem) default(shared)
        for (int z = 0; z < m_cellDomain.z; z++) {
            for (int y = 0; y < m_cellDomain.y; y++) {
                for (int x = 0; x < m_cellDomain.x; x++) {
                    const Neon::index_3d idx3d(x, y, z);
                    global_et::e         status = global_et::inactive;
                    if (inOut(idx3d)) {
                        numActiveElem += 1;
                        status = global_et::active;
                    }
                    m_globalToLocal.elRef(idx3d) = elmLocalInfo_t(status);
                }
            }
        }

        m_nActiveElements = numActiveElem;
        assert(numActiveElem != 0);
#ifdef _MSC_VER
#pragma warning(pop)
#endif
    }

   public:
    void exportTopology_vti(const std::string& fname)
    {
        auto gpuIds = [this](const index_3d& idx, int /*vIdx*/) -> double {
            const elmLocalInfo_t& info = m_globalToLocal.elRef(idx);
            const auto            prtId = info.getPrtIdx();
            return static_cast<double>(prtId);
        };

#if 0
        auto activity = [this](const index_3d& idx, int vIdx) -> double {
            const elmLocalInfo_t& info = m_globalToLocal.elRef(idx);
            if (info.isActive()) {
                return 1.0;
            }
            return -1.0;
        };
#endif
        auto activity2 = [this](const index_3d& idx, int /*vIdx*/) -> double {
            const elmLocalInfo_t& info = m_globalToLocal.elRef(idx);
            if (info.isActive()) {
                return static_cast<double>(info.getPrtIdx() * 100000 + 100000 + info.getLocalIdx());
            }
            return -1.0;
        };

        // Pradeep: this version of ioToVti is deprecated
        // Neon::ioToVTI({activity2, gpuIds}, {1, 1}, {"activity", "gpuIdx"}, {false, false}, fname, m_cellDomain + 1, m_cellDomain, 1.0, 0.0);
        Neon::ioToVTI({{activity2, 1, "activity", false, Neon::IoFileType::ASCII}, {gpuIds, 1, "gpuIdx", false, Neon::IoFileType::ASCII}}, fname, m_cellDomain + 1, m_cellDomain, 1.0, 0.0);
    }

    void setConnectivityAndIverseMappingStorage()
    {
        auto localLenDataSet = m_devSet.newDataSet<uint64_t>();
        for (int partIdx = 0; partIdx < nPartitions(); partIdx++) {
            localLenDataSet[partIdx] = m_localIndexingInfo.ref(partIdx).nElements(false);
            assert(localLenDataSet[partIdx] != 0);
        }
        m_localConectivityCPU = m_devSet.newMemDevSet<int32_t>(this->m_nNeighbours,
                                                               Neon::DeviceType::CPU,
                                                               Neon::Allocator::MALLOC,
                                                               localLenDataSet);

        m_inverseMappingCPU = m_devSet.newMemDevSet<index_t>(index_3d::num_axis,
                                                             Neon::DeviceType::CPU,
                                                             Neon::Allocator::MALLOC,
                                                             localLenDataSet);

        if (m_devSet.type() == Neon::DeviceType::CUDA) {
            int cardinalityNgh = this->m_nNeighbours;
            m_localConectivityGPU = m_devSet.newMemDevSet<int32_t>(cardinalityNgh,
                                                                   Neon::DeviceType::CUDA,
                                                                   Neon::Allocator::CUDA_MEM_DEVICE,
                                                                   localLenDataSet);

            m_inverseMappingGPU = m_devSet.newMemDevSet<index_t>(index_3d::num_axis,
                                                                 Neon::DeviceType::CUDA,
                                                                 Neon::Allocator::CUDA_MEM_DEVICE,
                                                                 localLenDataSet);
        }
    }
};  // namespace internals

}  // namespace internals
}  // namespace Neon::domain::internal::eGrid