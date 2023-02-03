#pragma once

#include <functional>
#include <future>
#include <iostream>
#include <thread>
#include <tuple>
#include <vector>

#include "Neon/Report.h"
#include "Neon/core/core.h"
#include "Neon/set/Execution.h"
#include "Neon/set/MemoryOptions.h"
#include "Neon/set/Runtime.h"

// #include "Neon/core/types/mode.h"
// #include "Neon/core/types/devType.h"

namespace Neon {
using StreamIdx = int;
using EventIdx = int;
namespace set {
class DevSet;
class StreamSet;
class GpuEventSet;
}  // namespace set


class Backend
{
   public:
    static constexpr int mainStreamIdx{0};

    /**
     * Empty constructor
     */
    Backend();

    /**
     * Creating a Backend object with the first nGpus devices.
     */
    Backend(int           nGpus /*!   Number of devices. The devices are selected in the order specifies by CUDA */,
            Neon::Runtime runtime /*! Type of runtime to use */);

    /**
     *
     */
    Backend(const std::vector<int>& devIds /*!  Vectors of device ids. There are CUDA device ids */,
            Neon::Runtime           runtime /*! Type of runtime to use */);

    /**
     *
     */
    Backend(const Neon::set::DevSet& devSet,
            Neon::Runtime);
    /**
     *
     * @param streamSet
     */
    Backend(const std::vector<int>&     devIds,
            const Neon::set::StreamSet& streamSet);

    /**
     *
     * @param streamSet
     */
    Backend(const Neon::set::DevSet&    devSet,
            const Neon::set::StreamSet& streamSet);

    auto clone(Neon::Runtime runtime = Neon::Runtime::system) -> Backend;

    auto getXpuCount()
        const
        -> int;

    auto h_initFirstEvent() -> void;
    /**
     * Returns target device for computation
     * @return
     */
    auto devType()
        const
        -> Neon::DeviceType;

    auto devSet()
        const
        -> const Neon::set::DevSet&;

    /**
     * Returns the mode for the kernel lauch
     * @return
     */
    auto runtime()
        const
        -> const Neon::Runtime&;

    /**
     * Run mode: sync/async
     */
    auto runMode()
        const
        -> const Neon::run_et ::et&;

    /**
     * Set the run mode
     */
    auto runMode(Neon::run_et ::et)
        -> void;

    /**
     *
     */
    auto streamSet(int streamIdx)
        const
        -> const Neon::set::StreamSet&;

    auto streamSet(int streamIdx)
        -> Neon::set::StreamSet&;


    auto eventSet(Neon::EventIdx eventdx)
        const
        -> const Neon::set::GpuEventSet&;

    auto eventSet(Neon::EventIdx eventdx)
        -> Neon::set::GpuEventSet&;

    /**
     *
     * @param nStreamSets
     */
    auto setAvailableStreamSet(int nStreamSets)
        -> void;

    auto setAvailableUserEvents(int nUserEventSets)
        -> void;

    /**
     *
     */
    auto sync()
        const
        -> void;

    /**
     *
     */
    auto syncAll()
        const
        -> void;

    /**
     *
     */
    auto sync(int idx)
        const
        -> void;

    auto sync(Neon::SetIdx setIdx,
              int          idx)
        const
        -> void;

    auto pushEventOnStream(int eventId,
                           int streamId)
        -> void;

    auto waitEventOnStream(int eventId,
                           int streamId)
        -> void;

    auto pushEventOnStream(Neon::SetIdx setIdx,
                           int          eventId,
                           int          streamId)
        -> void;

    auto waitEventOnStream(Neon::SetIdx setIdx, int eventId, int streamId)
        -> void;
    //    /**
    //     * Create a set of cuda events to create an exit barrier.
    //     * I.e. one streams sync with all the others
    //     * The stream holding the barrier is the first in the streamIdxVec vector.
    //     *
    //     * @param streamIdxVec
    //     */
    //    auto streamEventBarrier(const std::vector<int>& streamIdxVec) -> void;

    auto getMemoryOptions(Neon::MemoryLayout order)
        const
        -> Neon::MemoryOptions;

    auto getMemoryOptions(Neon::Allocator    ioAllocator,
                          Neon::Allocator    computeAllocators[Neon::DeviceTypeUtil::nConfig],
                          Neon::MemoryLayout order)
        const
        -> Neon::MemoryOptions;

    auto getMemoryOptions() const
        -> Neon::MemoryOptions;

    static std::string toString(Neon::Runtime e);

    /**
     * Log the information of this backend object to a string
     */
    auto toString()
        const
        -> std::string;

    /**
     *
     */
    auto toReport(Neon::Report&     report,
                  Report::SubBlock* subdocAPI = nullptr)
        const
        -> void;

    void syncEvent(SetIdx setIdx, int eventIdx)
        const;

    template <Neon::Execution, typename UserFunction>
    auto forEachXpu(UserFunction function)
        const
        -> void;

   private:
    struct Data
    {
        int              nXpu{0};
        Neon::Runtime    runtime{Neon::Runtime::none};
        Neon::run_et::et runMode{Neon::run_et::async};

        std::vector<Neon::set::StreamSet>   streamSetVec;
        std::vector<Neon::set::GpuEventSet> eventSetVec;
        std::vector<Neon::set::GpuEventSet> userEventSetVec;

        std::shared_ptr<Neon::set::DevSet> devSet;
    };

    auto getData() -> Data&;

    auto getData()
        const
        -> const Data&;

    std::shared_ptr<Data> m_data;
};

}  // namespace Neon

#include "Neon/set/Backend_imp.h"