#include "Neon/set/Backend.h"
#include <cassert>
#include <functional>
#include <future>
#include <iostream>
#include <thread>
#include <tuple>
#include <vector>
#include "Neon/set/DevSet.h"

namespace Neon {

auto Backend::getData() -> Data&
{
    return *m_data;
}

auto Backend::getData() const -> const Data&
{
    return *m_data;
}

Backend::Backend()
{
    m_data = std::make_shared<Data>();
    getData().nXpu = 0;
    getData().runtime = Neon::Runtime::none;
    getData().devSet = std::make_shared<Neon::set::DevSet>();
    getData().streamSetVec = std::vector<Neon::set::StreamSet>(0);
    getData().eventSetVec = std::vector<Neon::set::GpuEventSet>(0);
}

Backend::Backend(int nGpus, Neon::Runtime runtime)
{
    std::vector<int> devIds;
    for (int i = 0; i < nGpus; i++) {
        devIds.push_back(i);
    }
    getData().nXpu = static_cast<int>(devIds.size());
    m_data = std::make_shared<Data>();
    getData().runtime = runtime;
    getData().devSet = std::make_shared<Neon::set::DevSet>(devType(), devIds);
    getData().streamSetVec.push_back(getData().devSet->defaultStreamSet());
    h_initFirstEvent();
    assert(getData().eventSetVec.size() == getData().streamSetVec.size());
}

Backend::Backend(const std::vector<int>& devIds,
                 Neon::Runtime           runtime)
{
    m_data = std::make_shared<Data>();
    getData().runtime = runtime;
    getData().devSet = std::make_shared<Neon::set::DevSet>(devType(), devIds);
    getData().streamSetVec.push_back(getData().devSet->defaultStreamSet());
    h_initFirstEvent();
    assert(getData().eventSetVec.size() == getData().streamSetVec.size());
    getData().nXpu = static_cast<int>(devIds.size());
}

Backend::Backend(const Neon::set::DevSet& devSet,
                 Neon::Runtime            runtime)
{

    m_data = std::make_shared<Data>();
    getData().runtime = runtime;
    getData().devSet = std::make_shared<Neon::set::DevSet>(devSet);
    getData().streamSetVec.push_back(getData().devSet->defaultStreamSet());
    h_initFirstEvent();
    assert(getData().eventSetVec.size() == getData().streamSetVec.size());
    getData().nXpu = static_cast<int>(devSet.setCardinality());
}

Backend::Backend(const Neon::set::DevSet&    devSet,
                 const Neon::set::StreamSet& streamSet)
{
    m_data = std::make_shared<Data>();
    getData().runtime = Neon::Runtime::stream;
    getData().devSet = std::make_shared<Neon::set::DevSet>(devSet);
    getData().streamSetVec.push_back(streamSet);
    h_initFirstEvent();
    assert(getData().eventSetVec.size() == getData().streamSetVec.size());
    getData().nXpu = static_cast<int>(devSet.setCardinality());
}

Backend::Backend(const std::vector<int>&     devIds,
                 const Neon::set::StreamSet& streamSet)
{
    m_data = std::make_shared<Data>();
    getData().runtime = Neon::Runtime::stream;
    getData().devSet = std::make_shared<Neon::set::DevSet>(devType(), devIds);
    getData().streamSetVec.push_back(streamSet);
    h_initFirstEvent();
    assert(getData().eventSetVec.size() == getData().streamSetVec.size());
    getData().nXpu = static_cast<int>(devIds.size());
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
    switch (getData().runtime) {
        case Neon::Runtime::openmp: {
            getData().eventSetVec = std::vector<Neon::set::GpuEventSet>(1);
            return;
        }
        case Neon::Runtime::stream: {
            const bool disableTimingOption = true;
            getData().eventSetVec = std::vector<Neon::set::GpuEventSet>(1, getData().devSet->newEventSet(disableTimingOption));
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
    switch (getData().runtime) {
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
    return *getData().devSet.get();
}
auto Backend::runtime()
    const
    -> const Neon::Runtime&
{
    return getData().runtime;
}

auto Backend::streamSet(Neon::StreamIdx streamIdx) const -> const Neon::set::StreamSet&
{
    return getData().streamSetVec.at(streamIdx);
}

auto Backend::streamSet(Neon::StreamIdx streamIdx) -> Neon::set::StreamSet&
{
    return getData().streamSetVec.at(streamIdx);
}

auto Backend::eventSet(Neon::EventIdx eventdx) const -> const Neon::set::GpuEventSet&
{
    return getData().userEventSetVec.at(eventdx);
}

auto Backend::eventSet(Neon::EventIdx eventdx) -> Neon::set::GpuEventSet&
{
    return getData().userEventSetVec.at(eventdx);
}

// auto Backend::streamSetRotate(int&                    rotateIdx,
//                               const std::vector<int>& streamIdxVec)
//     const
//     -> const Neon::set::StreamSet&
//{
//     int streamIdx = Backend::streamSetIdxRotate(rotateIdx, streamIdxVec);
//     return getData().streamSetVec.at(streamIdx);
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
    switch (getData().runtime) {
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
        auto& streamSet = getData().streamSetVec.at(streamId);
        try {
            auto event = getData().userEventSetVec.at(eventId);
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
    switch (getData().runtime) {
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
        auto& streamSet = getData().streamSetVec.at(streamId);
        try {
            auto event = getData().userEventSetVec.at(eventId);
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
    switch (getData().runtime) {
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
        getData().streamSetVec.at(streamId).waitForEvent(getData().userEventSetVec.at(eventId));
    } catch (...) {
        NeonException exp("waitEventOnStream");
        exp << "Error trying to wait for stream";
        NEON_THROW(exp);
    }
}

auto Backend::waitEventOnStream(Neon::SetIdx setIdx, int eventId, int streamId)
    -> void
{
    switch (getData().runtime) {
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
        getData().streamSetVec.at(streamId).waitForEvent(setIdx, getData().userEventSetVec.at(eventId));
    } catch (...) {
        NeonException exp("waitEventOnStream");
        exp << "Error trying to wait for stream";
        NEON_THROW(exp);
    }
}

// auto Backend::streamEventBarrier(const std::vector<int>& streamIdxVec) -> void
//{
//     switch (getData().runtime) {
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
//         getData().streamSetVec.at(i).enqueueEvent(getData().eventSetVec.at(i));
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
//                 auto waiterStream = getData().streamSetVec.at(waiterStreamId);
//                 auto neighborEvent = getData().eventSetVec.at(neighbourEventId);
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
        getData().streamSetVec = std::vector<Neon::set::StreamSet>(nStreamSets);
        getData().eventSetVec = std::vector<Neon::set::GpuEventSet>(nStreamSets);
        return;
    }
    const int streamsToAdd = nStreamSets - int(getData().streamSetVec.size());
    for (int i = 0; i < streamsToAdd; i++) {
        getData().streamSetVec.push_back(getData().devSet->newStreamSet());
        const bool disableTimingOption = true;
        getData().eventSetVec.push_back(getData().devSet->newEventSet(disableTimingOption));
    }
    assert(getData().eventSetVec.size() == getData().streamSetVec.size());
}

auto Backend::setAvailableUserEvents(int nUserEventSets) -> void
{
    if (runtime() == Neon::Runtime::openmp) {
        getData().userEventSetVec = std::vector<Neon::set::GpuEventSet>(nUserEventSets);
        return;
    }
    const int eventToAdd = nUserEventSets - int(getData().userEventSetVec.size());
    for (int i = 0; i < eventToAdd; i++) {
        const bool disableTimingOption = true;
        getData().userEventSetVec.push_back(getData().devSet->newEventSet(disableTimingOption));
    }
}


auto Backend::sync() const -> void
{
    if (runtime() == Neon::Runtime::openmp) {
        return;
    }
    if (runtime() == Neon::Runtime::stream) {
        return getData().streamSetVec[0].sync();
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
        int nStreamSetVec = int(getData().streamSetVec.size());
        for (int i = 0; i < nStreamSetVec; i++) {
            getData().streamSetVec[i].sync();
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
        getData().streamSetVec[idx].sync();
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
        getData().streamSetVec[idx].sync(setIdx.idx());
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
        const Neon::set::GpuEventSet& eventSet = getData().userEventSetVec.at(eventIdx);
        eventSet.sync(setIdx);
        return;
    }
    NeonException exp("BackendConfig_t");
    exp << "Backend::sync with idx not permitted for a " << Neon::RuntimeUtils::toString(runtime()) << "backend";
    NEON_THROW(exp);
}

auto Backend::runMode() const -> const Neon::run_et ::et&
{
    return getData().runMode;
}

auto Backend::runMode(Neon::run_et::et runMode) -> void
{
    getData().runMode = runMode;
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
    msg << "Backend (" << this << ") - [runtime:" << toString(getData().runtime) << "] [nDev:" << getData().devSet->setCardinality() << "] ";
    switch (getData().devSet->type()) {
        case Neon::DeviceType::OMP:
        case Neon::DeviceType::CPU: {
            for (int i = 0; i < getData().devSet->setCardinality(); i++) {
                msg << "[dev" << i << ":" << getData().devSet->devId(i) << "] ";
            }
            break;
        }
        case Neon::DeviceType::CUDA: {
            for (int i = 0; i < getData().devSet->setCardinality(); i++) {
                msg << "[dev" << i << ":" << getData().devSet->devId(i) << " " << getData().devSet->getDevName(i) << "] ";
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

auto Backend::getXpuCount() const -> int
{
    return getData().nXpu;
}
}  // namespace Neon
