#include "Neon/set/DevSet.h"

#include <array>

#include "Neon/core/types/Exceptions.h"
#include "Neon/sys/global/GpuSysGlobal.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace Neon {
namespace set {

// namespace internal {
///**
// * thread local storage to call cpuRun
// */
// thread_local launchGridInfo_t g_launchGridInfo_t;
//}
// #pragma omp threadprivate(a)

DevSet::DevSet(const std::vector<Neon::sys::ComputeID>& set)
{
    const Neon::DeviceType devType = Neon::DeviceType::CUDA;
    this->set(devType, set);
}

//
// DevSet::DevSet(const Neon::dev_et::enum_e& devType, const std::vector<Neon::sys::gpu_id>& set)
//{
//    this->set(devType, set);
//}

DevSet::DevSet(const Neon::DeviceType& devType, const std::vector<int>& set)
{
    std::vector<Neon::sys::ComputeID> gpuIds;
    for (auto& id : set) {
        gpuIds.push_back(Neon::sys::ComputeID(id));
    }
    this->set(devType, gpuIds);
}

auto DevSet::set(const Neon::DeviceType&                  devType,
                 const std::vector<Neon::sys::ComputeID>& set)
    -> void
{
    if (!m_devIds.empty()) {
        Neon::NeonException exp("DevSet");
        exp << "Error, device set was already initialized.";
        NEON_THROW(exp);
    }
    m_devIds = set;
    m_devType = devType;
    m_fullyConnected = false;

    if (set.empty()) {
        Neon::NeonException exp("DevSet");
        exp << "Error, initialization with no device ids.";
        NEON_THROW(exp);
    }

    {
        if (m_devType == Neon::DeviceType::CUDA) {
            auto invilidIdsVec = this->validateIds();
            if (!invilidIdsVec.empty()) {
                Neon::NeonException exp("GpuSet");
                exp << "Error, GpuSet::set does not accept invalid CUDA gpu ids.\n";
                exp << "The following Ids were detected as invalid: \n";
                for (auto&& invalidID : invilidIdsVec) {
                    exp << invalidID << " ";
                }
                NEON_THROW(exp);
            }
        }
    }
    m_fullyConnected = true;
    if (set.size() > 1 && m_devType == Neon::DeviceType::CUDA) {
        std::vector<std::vector<Neon::sys::ComputeID>> errors(m_devIds.size());

        this->forEachSetIdxPar(
            [&](const Neon::SetIdx& setIdx) {
                const Neon::sys::ComputeID gpuIdx = this->devId(setIdx.idx());
                for (auto&& peerId : this->m_devIds) {
                    if (gpuIdx == peerId) {
                        continue;
                    }
                    const Neon::sys::GpuDevice& gpuDev = Neon::sys::globalSpace::gpuSysObj().dev(gpuIdx);
                    try {
                        gpuDev.memory.enablePeerAccessWith(peerId);
                    } catch (...) {
                        errors[gpuIdx.idx()].push_back(peerId);
                        continue;
                    }
                }
            });
        for (auto&& errorList : errors) {
            if (!errorList.empty()) {
                m_fullyConnected = false;
            }
        }
    }
    h_init_defaultStreamSet();

    for (SetIdx setIdx = 0; setIdx < setCardinality(); setIdx++) {
        m_idxRange.push_back(setIdx);
    }
}


auto DevSet::isFullyConnected()
    const
    -> bool
{
    return m_fullyConnected;
}


auto DevSet::getRange()
    const
    -> const std::vector<SetIdx>&
{
    return m_idxRange;
}

auto DevSet::setCardinality()
    const
    -> int32_t
{
    return int(m_devIds.size());
}

auto DevSet::type() const
    -> const Neon::DeviceType&
{
    return m_devType;
}

auto DevSet::getInfo(Neon::sys::DeviceID gpuIdx)
    const
    -> void
{
    const Neon::sys::GpuDevice& gpuDev = Neon::sys::globalSpace::gpuSysObj().dev(m_devIds[gpuIdx.idx()]);
    auto                        info = gpuDev.tools.getDevInfo("   ");
    NEON_INFO("DevSet {}", info);
}


auto DevSet::getDevName(Neon::sys::ComputeID gpuIdx)
    const
    -> std::string
{
    if (m_devType == Neon::DeviceType::CUDA) {

        const Neon::sys::GpuDevice& gpuDev = Neon::sys::globalSpace::gpuSysObj().dev(m_devIds[gpuIdx.idx()]);
        auto                        name = gpuDev.tools.getDevName();
        return name;
    }
    return "CPU";
}


auto DevSet::validateIds()
    const
    -> std::vector<Neon::sys::ComputeID>
{
    std::vector<Neon::sys::ComputeID> invalidIds;
    if (m_devType == Neon::DeviceType::CUDA) {
        for (auto&& gpuId : this->m_devIds) {
            if (gpuId.idx() >= Neon::sys::globalSpace::gpuSysObj().numDevs()) {
                invalidIds.push_back(gpuId);
            }
        }
    }
    return invalidIds;
}

auto DevSet::h_init_defaultStreamSet() -> void
{
    if (m_devType == Neon::DeviceType::CUDA) {
        m_defaultStream = this->newStreamSet();
    }
}

auto DevSet::devId(SetIdx index)
    const
    -> Neon::sys::ComputeID
{
    if (index.idx() >= m_devIds.size()) {
        Neon::NeonException exp("GpuSet");
        exp << "Error, GpuSet::invalid index query.\n";
        exp << "The following relative index were detected as invalid: " << index;
        exp << "Supported Range is : 0 ->" << m_devIds.size() - 1;
        NEON_THROW(exp);
    }
    return m_devIds[index.idx()];
}


auto DevSet::devId()
    const
    -> const DataSet<Neon::sys::DeviceID>&
{
    return m_devIds;
}


auto DevSet::gpuDev(SetIdx index)
    const
    -> const Neon::sys::GpuDevice&
{
    if (m_devType == Neon::DeviceType::CUDA) {
        if (index.idx() >= m_devIds.size()) {
            Neon::NeonException exp("DevSet");
            exp << "Error, invalid index query.\n";
            exp << "The following relative index were detected as invalid: " << index;
            exp << "Supported Range is : 0 ->" << m_devIds.size() - 1;
            NEON_THROW(exp);
        }

        Neon::sys::ComputeID        devIdx = this->devId(SetIdx(index));
        const Neon::sys::GpuDevice& gpuDev = Neon::sys::globalSpace::gpuSysObj().dev(devIdx);

        return gpuDev;
    }
    Neon::NeonException exp("DevSet");
    exp << "Error, DevSet::invalid operation on a CPU type of device.\n";
    NEON_THROW(exp);
}


auto DevSet::setActiveDevContext(SetIdx index)
    const
    -> void
{
    if (m_devType != Neon::DeviceType::CUDA) {
        Neon::NeonException exp("DevSet");
        exp << "Error, DevSet::invalid operation on a CPU type of device.\n";
        NEON_THROW(exp);
    }
    if (index.idx() >= m_devIds.size()) {
        Neon::NeonException exp("DevSet");
        exp << "Error, invalid index query.\n";
        exp << "The following relative index were detected as invalid: " << index;
        exp << "Supported Range is : 0 ->" << m_devIds.size() - 1;
        NEON_THROW(exp);
    }
    const Neon::sys::GpuDevice& dev = Neon::sys::globalSpace::gpuSysObj().dev(m_devIds[index.idx()]);
    dev.tools.setActiveDevContext();
}


auto DevSet::toString()
    const
    -> std::string
{
    std::ostringstream msg;
    for (Neon::SetIdx idx = 0; idx < setCardinality(); idx++) {
        msg << "[" << idx.idx() << ": " << m_devIds[idx] << " -> " << getDevName(idx.idx()) << "] \n";
    }
    return msg.str();
}

auto DevSet::newStreamSet()
    const
    -> StreamSet
{
    if (m_devType == Neon::DeviceType::CUDA) {
        StreamSet streamSet(this->setCardinality());

        this->forEachSetIdxPar(
            [&](const Neon::SetIdx& setIdx) {
                const Neon::sys::ComputeID  gpuId = this->devId(setIdx.idx());
                const Neon::sys::GpuDevice& dev = Neon::sys::globalSpace::gpuSysObj().dev(gpuId);
                streamSet.set(setIdx.idx(), dev.tools.stream());
            });

        return streamSet;
    }
    Neon::NeonException exp("DevSet");
    exp << "Error, DevSet::invalid operation on a CPU type of device.\n";
    NEON_THROW(exp);
}

auto DevSet::defaultStreamSet() const
    -> const StreamSet&
{
    return m_defaultStream;
}

auto DevSet::sync() -> void
{
    m_defaultStream.sync();
}

auto DevSet::newEventSet(bool disableTiming)
    const
    -> GpuEventSet
{
    if (m_devType == Neon::DeviceType::CUDA) {

        GpuEventSet eventSet(this->setCardinality());

        this->forEachSetIdxPar([&](const Neon::SetIdx& setIdx) {
            const Neon::sys::ComputeID  gpuId = this->devId(setIdx.idx());
            const Neon::sys::GpuDevice& dev = Neon::sys::globalSpace::gpuSysObj().dev(gpuId);
            eventSet.event<Neon::Access::readWrite>(setIdx.idx()) = dev.tools.event(disableTiming);
        });

        return eventSet;
    }
    return GpuEventSet();
}


auto DevSet::newLaunchParameters()
    const
    -> LaunchParameters
{
    return LaunchParameters(this->setCardinality());
}

auto DevSet::devSync()
    const
    -> void
{
    if (m_devType == Neon::DeviceType::CUDA) {

        const int num_dev = static_cast<int>(m_devIds.size());
#pragma omp parallel for num_threads(num_dev)
        for (int id = 0; id < num_dev; ++id) {
            gpuDev(id).tools.sync();
        }
    }
}


auto DevSet::maxSet(const Neon::DeviceType devType)
    -> DevSet
{
    int32_t numDev;
    if (devType == Neon::DeviceType::CUDA) {

        numDev = Neon::sys::globalSpace::gpuSysObj().numDevs();
    } else {
#if defined(_OPENMP)
        numDev = omp_get_thread_num();
#else
        numDev = 1;
#endif
    }
    std::vector<int> gpus(numDev);
    for (int i = 0; i < numDev; i++) {
        gpus[i] = i;
    }
    DevSet newSet(devType, gpus);
    return newSet;
}


auto DevSet::memInUse(SetIdx setIdx)
    -> size_t
{
    if (m_devType == Neon::DeviceType::CUDA) {

        const Neon::sys::ComputeID id = m_devIds[setIdx.idx()];
        auto                       allocator = Neon::sys::globalSpace::gpuSysObj().allocator(id);
        return allocator.inUse();
    }
    NEON_DEV_UNDER_CONSTRUCTION("DevSet::memInUse");
}


auto DevSet::memMaxUse(SetIdx setIdx)
    -> size_t
{
    if (m_devType == Neon::DeviceType::CUDA) {
        const Neon::sys::ComputeID id = m_devIds[setIdx.idx()];
        auto                       allocator = Neon::sys::globalSpace::gpuSysObj().allocator(id);
        return allocator.maxUse();
    }
    NEON_DEV_UNDER_CONSTRUCTION("DevSet::memMaxUse");
}


const Neon::set::DataSet<Neon::sys::DeviceID>& DevSet::idSet() const
{
    return m_devIds;
}

const std::vector<int> DevSet::userIdVec() const
{
    std::vector<int> v;
    for (auto& e : m_devIds) {
        v.push_back(e.idx());
    }
    return v;
}


auto DevSet::transfer(TransferMode     transferMode,
                      const StreamSet& streamSet,
                      SetIdx           dstSetId,
                      char*            dstBuf,
                      SetIdx           srcSetIdx,
                      const char*      srcBuf,
                      size_t           numBytes)
    const
    -> void
{
    if (m_devType == Neon::DeviceType::CUDA) {
        Neon::sys::ComputeID dstGpuIdx = this->devId(dstSetId);
        Neon::sys::ComputeID srcGpuIdx = this->devId(srcSetIdx);

        switch (transferMode) {
            case Neon::set::TransferMode::put: {
                const Neon::sys::GpuDevice& srcDev = Neon::sys::globalSpace::gpuSysObj().dev(srcGpuIdx);
                srcDev.memory.peerTransfer(streamSet[srcSetIdx], dstGpuIdx, dstBuf, srcGpuIdx, srcBuf, numBytes);
                return;
            }
            case Neon::set::TransferMode::get: {
                const Neon::sys::GpuDevice& dstDev = Neon::sys::globalSpace::gpuSysObj().dev(dstGpuIdx);
                dstDev.memory.peerTransfer(streamSet[dstSetId], dstGpuIdx, dstBuf, srcGpuIdx, srcBuf, numBytes);
                return;
            }
        }
    }

    if (m_devType == Neon::DeviceType::CPU || m_devType == Neon::DeviceType::OMP) {
        std::memcpy(dstBuf, srcBuf, numBytes);
        return ;
    }
    Neon::NeonException exp("DevSet");
    exp << "Error, DevSet::invalid operation on a CPU type of device.\n";
    NEON_THROW(exp);
}

auto DevSet::peerTransfer(PeerTransferOption&        opt,
                          const Neon::set::Transfer& transfer)
    const
    -> void
{
    if (opt.operationMode() == PeerTransferOption::storeInfo) {
#pragma omp critical(peerTransfer)
        {
            // IF we are just storing the information,
            // we don't need to proceed further.
            opt.transfers().push_back(transfer);
        }
        return;
    }
    const Neon::SetIdx dstIdx = transfer.dst().devId;
    const Neon::SetIdx srcIdx = transfer.src().devId;
    char*              dstBuf = (char*)transfer.dst().mem;
    char*              srcBuf = (char*)transfer.src().mem;
    size_t             numBytes = transfer.size();

    switch (m_devType) {
        case Neon::DeviceType::CUDA: {
            Neon::sys::ComputeID dstGpuIdx = this->devId(dstIdx);
            Neon::sys::ComputeID srcGpuIdx = this->devId(srcIdx);

            switch (transfer.mode()) {
                case Neon::set::TransferMode::put: {
                    const Neon::sys::GpuDevice& srcDev = Neon::sys::globalSpace::gpuSysObj().dev(srcGpuIdx);
                    srcDev.memory.peerTransfer(opt.streamSet()[srcIdx], dstGpuIdx, dstBuf, srcGpuIdx, srcBuf, numBytes);
                    return;
                }
                case Neon::set::TransferMode::get: {
                    const Neon::sys::GpuDevice& dstDev = Neon::sys::globalSpace::gpuSysObj().dev(dstGpuIdx);
                    dstDev.memory.peerTransfer(opt.streamSet()[dstIdx], dstGpuIdx, dstBuf, srcGpuIdx, srcBuf, numBytes);
                    return;
                }
                default: {
                    NEON_THROW_UNSUPPORTED_OPTION("");
                }
            }
        }
        case (Neon::DeviceType::CPU): {
            std::memcpy(dstBuf, srcBuf, numBytes);
            return;
        }
        default: {
            Neon::NeonException exp("DevSet");
            exp << "Error, DevSet::invalid operation on a CPU type of device.\n";
            NEON_THROW(exp);
        }
    }
}  // namespace set


}  // namespace set
}  // namespace Neon
