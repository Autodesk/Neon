#include "Neon/set/Backend.h"
#include <cassert>
#include <functional>
#include <future>
#include <iostream>
#include <thread>
#include <tuple>
#include <vector>
#include "Neon/Neon.h"
#include "Neon/set/DevSet.h"

namespace Neon {


void Backend::Data_t::Nccl::checkNccl(ncclResult_t result, const char* func)
{
    if (result != ncclSuccess) {
        std::cerr << "NCCL error in " << func << ": " << ncclGetErrorString(result) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
}

void Backend::Data_t::Nccl::checkCuda(cudaError_t result, const char* func)
{
    if (result != cudaSuccess) {
        std::cerr << "CUDA error in " << func << ": " << cudaGetErrorString(result) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
}

auto Backend::Data_t::Nccl::initMPI() -> void
{
    auto& data = *this;

    MPI_Init(nullptr, nullptr);

    MPI_Comm_rank(MPI_COMM_WORLD, &data.worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &data.worldSize);

    {  // loca rank info
        MPI_Comm local_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
        MPI_Comm_rank(local_comm, &data.localRank);
        MPI_Comm_size(local_comm, &data.localSize);
        MPI_Comm_free(&local_comm);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    {  // Get the number of available GPUs
        checkCuda(cudaGetDeviceCount(&data.numLocalDevices), "cudaGetDeviceCount");
        std::cout << "MPI Rank " << data.worldRank
                  << " using GPU " << data.localRank
                  << " of " << data.numLocalDevices << std::endl;
    }


    if (data.sizeLocalRanks > data.numLocalDevices) {
        std::cout << "error..." << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        Neon::NeonException ex("distributed::Backend");
        ex << "Unsupported configuration. At the moment we require the number of devices (";
        ex << data.numLocalDevices;
        ex << ") to be the same as the MPI rank per node";
    } else {
        MPI_Barrier(MPI_COMM_WORLD);
    }
    std::cout << "MPI Rank " << data.worldRank
              << "MPI init completed" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
}

auto Backend::Data_t::Nccl::initNCCL() -> void
{
    auto& data = *this;
    if (data.worldRank == 0) {
        std::cout << "MPI rank 0 - ncclGetUniqueId " << std::endl;
        ncclGetUniqueId(&(data.nccl_id));
    }

    // Broadcast NCCL ID from rank 0 to all other ranks
    MPI_Bcast(&(data.nccl_id), sizeof(data.nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // Initialize NCCL communicator
    checkNccl(ncclCommInitRank(&(data.nccl_comm),
                               data.worldSize,
                               data.nccl_id,
                               data.worldRank),
              "initNCCL");
    std::cout << "NCCL initialized for rank " << data.worldRank << std::endl;
}

Backend::Data_t::Nccl::~Nccl()
{
    if (isDistributed()) {
        finiNCCL();
        finiMPI();
    }
    localRank = 0;
    localSize = 0;
    worldRank = 0;
    worldSize = 0;
    numLocalDevices = 0;
}

auto Backend::Data_t::Nccl::finiMPI() -> void
{
    std::cout << "MPI_Finalize" << std::endl;
    MPI_Finalize();
}

auto Backend::Data_t::Nccl::finiNCCL() -> void
{
    ncclCommDestroy(nccl_comm);
}

auto Backend::Data_t::Nccl::isDistributed() const -> bool
{
    return worldSize > 1;
}
auto Backend::Data_t::Nccl::getWorldRank() const -> int
{
    return worldRank;
}
auto Backend::Data_t::Nccl::getWorldSize() const -> int
{
    return worldSize;
}
auto Backend::Data_t::Nccl::getLocalRank() const -> int
{
    return localRank;
}
auto Backend::Data_t::Nccl::getLocalSize() const -> int
{
    return localSize;
}

auto Backend::selfData() -> Data_t&
{
    return *m_data;
}

auto Backend::selfData() const -> const Data_t&
{
    return *m_data;
}

Backend::Backend()
{
    m_data = std::make_shared<Data_t>();
    selfData().runtime = Neon::Runtime::none;
    selfData().devSet = std::make_shared<Neon::set::DevSet>();
    selfData().streamSetVec = std::vector<Neon::set::StreamSet>(0);
    selfData().eventSetVec = std::vector<Neon::set::GpuEventSet>(0);
}

Backend::Backend(Neon::Runtime runtime)
{
    m_data = std::make_shared<Data_t>();
    Neon::init();
    selfData().nccl.initMPI();

    if (!selfData().nccl.isDistributed()) {
        // We have only 1 process
        // we are going to set Neon backend with no MPI support with all the available devices
        int              numDevs = runtime == Neon::Runtime::openmp
                                       ? 1
                                       : Neon::sys::globalSpace::gpuSysObjStorage.numDevs();
        std::vector<int> devIds;
        for (int i = 0; i < numDevs; i++) {
            devIds.push_back(i);
        }
        selfData().runtime = runtime;
        selfData().devSet = std::make_shared<Neon::set::DevSet>(devType(), devIds);
        selfData().streamSetVec.push_back(selfData().devSet->defaultStreamSet());
        h_initFirstEvent();
        assert(selfData().eventSetVec.size() == selfData().streamSetVec.size());
        return;
    } else {
        std::vector<int> devIds;
        devIds.push_back(selfData().nccl.getLocalRank());
        selfData().runtime = runtime;
        selfData().devSet = std::make_shared<Neon::set::DevSet>(devType(), devIds);
        selfData().streamSetVec.push_back(selfData().devSet->defaultStreamSet());
        h_initFirstEvent();
        assert(selfData().eventSetVec.size() == selfData().streamSetVec.size());
        devSet().setActiveDevContext(0);
        selfData().nccl.initNCCL();
        return;
    }
}

Backend::Backend(int nGpus, Neon::Runtime runtime)
{
    m_data = std::make_shared<Data_t>();
    std::vector<int> devIds;
    for (int i = 0; i < nGpus; i++) {
        devIds.push_back(i);
    }
    selfData().runtime = runtime;
    selfData().devSet = std::make_shared<Neon::set::DevSet>(devType(), devIds);
    selfData().streamSetVec.push_back(selfData().devSet->defaultStreamSet());
    h_initFirstEvent();
    assert(selfData().eventSetVec.size() == selfData().streamSetVec.size());
}

Backend::Backend(const std::vector<int>& devIds,
                 Neon::Runtime           runtime)
{
    m_data = std::make_shared<Data_t>();
    selfData().runtime = runtime;
    selfData().devSet = std::make_shared<Neon::set::DevSet>(devType(), devIds);
    selfData().streamSetVec.push_back(selfData().devSet->defaultStreamSet());
    h_initFirstEvent();
    assert(selfData().eventSetVec.size() == selfData().streamSetVec.size());
}

Backend::Backend(const Neon::set::DevSet& devSet,
                 Neon::Runtime            runtime)
{

    m_data = std::make_shared<Data_t>();
    selfData().runtime = runtime;
    selfData().devSet = std::make_shared<Neon::set::DevSet>(devSet);
    selfData().streamSetVec.push_back(selfData().devSet->defaultStreamSet());
    h_initFirstEvent();
    assert(selfData().eventSetVec.size() == selfData().streamSetVec.size());
}

Backend::Backend(const Neon::set::DevSet&    devSet,
                 const Neon::set::StreamSet& streamSet)
{
    m_data = std::make_shared<Data_t>();
    selfData().runtime = Neon::Runtime::stream;
    selfData().devSet = std::make_shared<Neon::set::DevSet>(devSet);
    selfData().streamSetVec.push_back(streamSet);
    h_initFirstEvent();
    assert(selfData().eventSetVec.size() == selfData().streamSetVec.size());
}

Backend::Backend(const std::vector<int>&     devIds,
                 const Neon::set::StreamSet& streamSet)
{
    m_data = std::make_shared<Data_t>();
    selfData().runtime = Neon::Runtime::stream;
    selfData().devSet = std::make_shared<Neon::set::DevSet>(devType(), devIds);
    selfData().streamSetVec.push_back(streamSet);
    h_initFirstEvent();
    assert(selfData().eventSetVec.size() == selfData().streamSetVec.size());
}


auto Backend::clone(Neon::Runtime runtime) -> Backend
{
    Backend cloned;
    cloned.m_data = m_data;
    if (runtime != Neon::Runtime::system) {
        cloned.m_data->runtime = runtime;
    }
    return cloned;
}

auto Backend::h_initFirstEvent() -> void
{
    switch (selfData().runtime) {
        case Neon::Runtime::openmp: {
            selfData().eventSetVec = std::vector<Neon::set::GpuEventSet>(1);
            return;
        }
        case Neon::Runtime::stream: {
            const bool disableTimingOption = true;
            selfData().eventSetVec = std::vector<Neon::set::GpuEventSet>(1, selfData().devSet->newEventSet(disableTimingOption));
            return;
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("");
        }
    }
}

auto Backend::devType()
    const
    -> Neon::DeviceType
{
    switch (selfData().runtime) {
        case Neon::Runtime::openmp: {
            return Neon::DeviceType::CPU;
        }
        case Neon::Runtime::stream: {
            return Neon::DeviceType::CUDA;
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPERATION();
        }
    }
}

auto Backend::devSet()
    const
    -> const Neon::set::DevSet&
{
    return *selfData().devSet.get();
}

auto Backend::runtime()
    const
    -> const Neon::Runtime&
{
    return selfData().runtime;
}

auto Backend::streamSet(Neon::StreamIdx streamIdx) const -> const Neon::set::StreamSet&
{
    return selfData().streamSetVec.at(streamIdx);
}

auto Backend::streamSet(Neon::StreamIdx streamIdx) -> Neon::set::StreamSet&
{
    return selfData().streamSetVec.at(streamIdx);
}

auto Backend::eventSet(Neon::EventIdx eventdx) const -> const Neon::set::GpuEventSet&
{
    return selfData().userEventSetVec.at(eventdx);
}

auto Backend::eventSet(Neon::EventIdx eventdx) -> Neon::set::GpuEventSet&
{
    return selfData().userEventSetVec.at(eventdx);
}

// auto Backend::streamSetRotate(int&                    rotateIdx,
//                               const std::vector<int>& streamIdxVec)
//     const
//     -> const Neon::set::StreamSet&
//{
//     int streamIdx = Backend::streamSetIdxRotate(rotateIdx, streamIdxVec);
//     return selfData().streamSetVec.at(streamIdx);
// }
//
// auto Backend::streamSetIdxRotate(int&                    rotateIdx,
//                                  const std::vector<int>& streamIdxVec)
//     -> int
//{
//     int streamIdx = streamIdxVec.at(rotateIdx);
//     rotateIdx++;
//     rotateIdx = rotateIdx % streamIdxVec.size();
//     return streamIdx;
// }

auto Backend::pushEventOnStream(int eventId, int streamId)
    -> void
{
    switch (selfData().runtime) {
        case Neon::Runtime::openmp: {
            return;
        }
        case Neon::Runtime::stream: {
            break;
            ;
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("");
        }
    }

    try {
        auto& streamSet = selfData().streamSetVec.at(streamId);
        try {
            auto event = selfData().userEventSetVec.at(eventId);
            streamSet.enqueueEvent(event);
        } catch (...) {
            NeonException exp("pushEventOnStream");
            exp << "Error trying to enqueue an event";
            NEON_THROW(exp);
        }
    } catch (...) {
        NeonException exp("pushEventOnStream");
        exp << "Error trying to extract a stream";
        NEON_THROW(exp);
    }
}

auto Backend::pushEventOnStream(Neon::SetIdx setIdx,
                                int          eventId,
                                int          streamId)
    -> void
{
    switch (selfData().runtime) {
        case Neon::Runtime::openmp: {
            return;
        }
        case Neon::Runtime::stream: {
            break;
            ;
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("");
        }
    }

    try {
        auto& streamSet = selfData().streamSetVec.at(streamId);
        try {
            auto event = selfData().userEventSetVec.at(eventId);
            streamSet.enqueueEvent(setIdx, event);
        } catch (...) {
            NeonException exp("pushEventOnStream");
            exp << "Error trying to enqueue an event";
            NEON_THROW(exp);
        }
    } catch (...) {
        NeonException exp("pushEventOnStream");
        exp << "Error trying to extract a stream";
        NEON_THROW(exp);
    }
}

auto Backend::waitEventOnStream(int eventId, int streamId)
    -> void
{
    switch (selfData().runtime) {
        case Neon::Runtime::openmp: {
            return;
        }
        case Neon::Runtime::stream: {
            break;
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("");
        }
    }
    try {
        selfData().streamSetVec.at(streamId).waitForEvent(selfData().userEventSetVec.at(eventId));
    } catch (...) {
        NeonException exp("waitEventOnStream");
        exp << "Error trying to wait for stream";
        NEON_THROW(exp);
    }
}

auto Backend::waitEventOnStream(Neon::SetIdx setIdx, int eventId, int streamId)
    -> void
{
    switch (selfData().runtime) {
        case Neon::Runtime::openmp: {
            return;
        }
        case Neon::Runtime::stream: {
            break;
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("");
        }
    }
    try {
        selfData().streamSetVec.at(streamId).waitForEvent(setIdx, selfData().userEventSetVec.at(eventId));
    } catch (...) {
        NeonException exp("waitEventOnStream");
        exp << "Error trying to wait for stream";
        NEON_THROW(exp);
    }
}

// auto Backend::streamEventBarrier(const std::vector<int>& streamIdxVec) -> void
//{
//     switch (selfData().runtime) {
//         case Neon::Runtime::openmp: {
//             return;
//         }
//         case Neon::Runtime::stream: {
//             break;
//             ;
//         }
//         default: {
//             NEON_THROW_UNSUPPORTED_OPTION("");
//         }
//     }
//
//     for (int i = 0; i < int(streamIdxVec.size()); i++) {
//         selfData().streamSetVec.at(i).enqueueEvent(selfData().eventSetVec.at(i));
//     }
//
//
//     auto nextPowerOfTwo = [](unsigned int n) -> unsigned int {
//         // if n=4 v=4
//         // if n=5 v=8
//         int v = n;
//
//         v--;
//         v |= v >> 1;
//         v |= v >> 2;
//         v |= v >> 4;
//         v |= v >> 8;
//         v |= v >> 16;
//         v++;  // next power of 2
//
//         return v;
//     };
//     const unsigned int nstreams = (unsigned int)(streamIdxVec.size());
//     const unsigned int nstreamsPow2 = nextPowerOfTwo(nstreams);
//
//     for (int rLevel = 1; rLevel < int(nstreamsPow2); rLevel *= 2) {
//         const int waiterJump = rLevel * 2;
//         const int neighbourToWaitJump = rLevel;
//         for (int waiterId = 0; waiterId < int(nstreams); waiterId += waiterJump) {
//             const int neighbourId = waiterId + neighbourToWaitJump;
//             if (neighbourId < int(nstreams)) {
//                 int waiterStreamId = streamIdxVec[waiterId];
//                 int neighbourEventId = streamIdxVec[neighbourId];
//
//                 auto waiterStream = selfData().streamSetVec.at(waiterStreamId);
//                 auto neighborEvent = selfData().eventSetVec.at(neighbourEventId);
//                 waiterStream.waitForEvent(neighborEvent);
//                 // std::cout << " waiterStreamId " << waiterStreamId <<" neighbourEventId " <<neighbourEventId<<std::endl;
//             }
//         }
//     }
//
//     return;
// }

auto Backend::setAvailableStreamSet(int nStreamSets) -> void
{
    if (runtime() == Neon::Runtime::openmp) {
        selfData().streamSetVec = std::vector<Neon::set::StreamSet>(nStreamSets);
        selfData().eventSetVec = std::vector<Neon::set::GpuEventSet>(nStreamSets);
        return;
    }
    const int streamsToAdd = nStreamSets - int(selfData().streamSetVec.size());
    for (int i = 0; i < streamsToAdd; i++) {
        selfData().streamSetVec.push_back(selfData().devSet->newStreamSet());
        const bool disableTimingOption = true;
        selfData().eventSetVec.push_back(selfData().devSet->newEventSet(disableTimingOption));
    }
    assert(selfData().eventSetVec.size() == selfData().streamSetVec.size());
}

auto Backend::setAvailableUserEvents(int nUserEventSets) -> void
{
    if (runtime() == Neon::Runtime::openmp) {
        selfData().userEventSetVec = std::vector<Neon::set::GpuEventSet>(nUserEventSets);
        return;
    }
    const int eventToAdd = nUserEventSets - int(selfData().userEventSetVec.size());
    for (int i = 0; i < eventToAdd; i++) {
        const bool disableTimingOption = true;
        selfData().userEventSetVec.push_back(selfData().devSet->newEventSet(disableTimingOption));
    }
}


auto Backend::sync() const -> void
{
    if (runtime() == Neon::Runtime::openmp) {
        return;
    }
    if (runtime() == Neon::Runtime::stream) {
        return selfData().streamSetVec[0].sync();
    }
    NeonException exp("BackendConfig_t");
    exp << "Backend::sync() not permitted for a " << Neon::RuntimeUtils::toString(runtime()) << "backend";
    NEON_THROW(exp);
}

auto Backend::syncAll() const -> void
{
    if (runtime() == Neon::Runtime::openmp) {
        return;
    }
    if (runtime() == Neon::Runtime::stream) {
        int nStreamSetVec = int(selfData().streamSetVec.size());
        for (int i = 0; i < nStreamSetVec; i++) {
            selfData().streamSetVec[i].sync();
        }
        return;
    }
    NeonException exp("BackendConfig_t");
    exp << "Backend::syncAll() not permitted for a " << Neon::RuntimeUtils::toString(runtime()) << "backend";
    NEON_THROW(exp);
}

auto Backend::sync(int idx) const -> void
{
    if (runtime() == Neon::Runtime::openmp) {
        return;
    }
    if (runtime() == Neon::Runtime::stream) {
        selfData().streamSetVec[idx].sync();
        return;
    }
    NeonException exp("BackendConfig_t");
    exp << "Backend::sync with idx not permitted for a " << Neon::RuntimeUtils::toString(runtime()) << "backend";
    NEON_THROW(exp);
}

auto Backend::sync(Neon::SetIdx setIdx, int idx) const -> void
{
    if (runtime() == Neon::Runtime::openmp) {
        return;
    }
    if (runtime() == Neon::Runtime::stream) {
        selfData().streamSetVec[idx].sync(setIdx.idx());
        return;
    }
    NeonException exp("BackendConfig_t");
    exp << "Backend::sync with idx not permitted for a " << Neon::RuntimeUtils::toString(runtime()) << "backend";
    NEON_THROW(exp);
}

auto Backend::syncEvent(Neon::SetIdx setIdx, int eventIdx) const -> void
{
    if (runtime() == Neon::Runtime::openmp) {
        return;
    }
    if (runtime() == Neon::Runtime::stream) {
        const Neon::set::GpuEventSet& eventSet = selfData().userEventSetVec.at(eventIdx);
        eventSet.sync(setIdx);
        return;
    }
    NeonException exp("BackendConfig_t");
    exp << "Backend::sync with idx not permitted for a " << Neon::RuntimeUtils::toString(runtime()) << "backend";
    NEON_THROW(exp);
}

auto Backend::runMode() const -> const Neon::run_et ::et&
{
    return selfData().runMode;
}

auto Backend::runMode(Neon::run_et::et runMode) -> void
{
    selfData().runMode = runMode;
}


std::string Backend::toString(Neon::Runtime e)
{
    switch (e) {
        case Neon::Runtime::none: {
            return "none";
        }
        case Neon::Runtime::stream: {
            return "stream";
        }
        case Neon::Runtime::openmp: {
            return "openmp";
        }
        default: {
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

std::string Backend::toString() const
{
    std::ostringstream msg;
    msg << "Backend (" << this << ") - [runtime:" << toString(selfData().runtime) << "] [nDev:" << selfData().devSet->setCardinality() << "] ";
    switch (selfData().devSet->type()) {
        case Neon::DeviceType::OMP:
        case Neon::DeviceType::CPU: {
            for (int i = 0; i < selfData().devSet->setCardinality(); i++) {
                msg << "[dev" << i << ":" << selfData().devSet->devId(i) << "] ";
            }
            break;
        }
        case Neon::DeviceType::CUDA: {
            for (int i = 0; i < selfData().devSet->setCardinality(); i++) {
                msg << "[dev" << i << ":" << selfData().devSet->devId(i) << " " << selfData().devSet->getDevName(i) << "] ";
            }
            break;
        }
            //        case dev_et::NONE:
            //        case dev_et::MPI:
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("");
        }
    }
    return msg.str();
}

auto Backend::getMemoryOptions(Neon::MemoryLayout order) const
    -> Neon::MemoryOptions
{
    MemoryOptions memoryOptions(Neon::DeviceType::CPU,
                                devSet().type(),
                                order);
    return memoryOptions;
}

auto Backend::getMemoryOptions(Neon::Allocator    ioAllocator,
                               Neon::Allocator    computeAllocators[Neon::DeviceTypeUtil::nConfig],
                               Neon::MemoryLayout order) const
    -> Neon::MemoryOptions
{
    MemoryOptions memoryOptions(Neon::DeviceType::CPU,
                                ioAllocator,
                                devSet().type(),
                                computeAllocators,
                                order);
    return memoryOptions;
}

auto Backend::getMemoryOptions() const
    -> Neon::MemoryOptions
{
    const Neon::MemoryOptions defaultMemOption = getMemoryOptions(Neon::MemoryLayout::structOfArrays);
    return defaultMemOption;
}

auto Backend::toReport(Neon::Report& report, Report::SubBlock* subdocAPI) const -> void
{
    Report::SubBlock* targetSubDoc = subdocAPI;
    Report::SubBlock  tmp;
    if (nullptr == subdocAPI) {
        tmp = report.getSubdoc();
        targetSubDoc = &tmp;
    }

    report.addMember("Runtime", Neon::RuntimeUtils::toString(runtime()), targetSubDoc);
    report.addMember("DeviceType", Neon::DeviceTypeUtil::toString(devType()), targetSubDoc);
    report.addMember("NumberOfDevices", devSet().setCardinality(), targetSubDoc);
    report.addMember(
        "Devices", [&] {
            std::vector<int> idsList;
            for (auto const& id : devSet().devId()) {
                idsList.push_back(id.idx());
            }
            return idsList;
        }(),
        targetSubDoc);

    if (nullptr == subdocAPI) {
        report.addSubdoc("Backend", *targetSubDoc);
    }
}

auto Backend::getDeviceCount() const -> int
{
    return m_data->devSet->setCardinality();
}

auto Backend::getNccl() const -> Data_t::Nccl&
{
    return m_data->nccl;
}

auto Backend::helpDeviceToDeviceTransferByte(int                     streamId,
                                             size_t                  bytes,
                                             Neon::set::TransferMode transferMode,
                                             Neon::SetIdx            dstSet,
                                             char*                   dstAddr,
                                             Neon::SetIdx            srcSet,
                                             const char*             srcAddr) const -> void
{
    if (bytes == 0) {
        return;
    }
    auto& devSet = this->devSet();
    auto& targetStreamSet = streamSet(streamId);

    devSet.transfer(transferMode,
                    targetStreamSet,
                    dstSet,
                    dstAddr,
                    srcSet,
                    srcAddr,
                    bytes);
}

auto Backend::deviceCount() const -> int
{
    return devSet().setCardinality();
}

auto Backend::isFirstDevice(Neon::SetIdx id) const -> bool
{
    return id.idx() == 0;
}
auto Backend::isLastDevice(Neon::SetIdx id) const -> bool
{
    return id.idx() == (deviceCount() - 1);
}

auto Backend::isDistributed() const -> bool
{
    return selfData().nccl.isDistributed();
}

auto Backend::countAvailableGpus() -> int32_t
{
    return Neon::sys::globalSpace::gpuSysObjStorage.numDevs();
}

}  // namespace Neon
