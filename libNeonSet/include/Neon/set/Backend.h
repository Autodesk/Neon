#pragma once
#include <mpi.h>
#include <nccl.h>
#include <functional>
#include <future>
#include <iostream>
#include <thread>
#include <tuple>
#include <vector>

#include "Neon/Report.h"
#include "Neon/core/core.h"
#include "Neon/set/MemoryOptions.h"
#include "Neon/set/Runtime.h"
// #include "Neon/core/types/mode.h"
// #include "Neon/core/types/devType.h"
#include "Neon/set/DataSet.h"

#include <Neon/sys/devices/gpu/GpuSys.h>

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

   private:
    struct Data_t
    {
        Neon::Runtime    runtime{Neon::Runtime::none};
        Neon::run_et::et runMode{Neon::run_et::async};

        std::vector<Neon::set::StreamSet>   streamSetVec;
        std::vector<Neon::set::GpuEventSet> eventSetVec;
        std::vector<Neon::set::GpuEventSet> userEventSetVec;

        std::shared_ptr<Neon::set::DevSet> devSet;

        struct Nccl
        {
            ~Nccl();
            auto initMPI() -> void;
            auto initNCCL() -> void;

            auto isDistributed() const -> bool;
            auto getWorldRank() const -> int;
            auto getWorldSize() const -> int;
            auto getLocalRank() const -> int;
            auto getLocalSize() const -> int;

           private:
            auto finiMPI() -> void;
            auto finiNCCL() -> void;
            void checkNccl(ncclResult_t result, const char* func);
            void checkCuda(cudaError_t result, const char* func);

            int          worldRank;
            int          worldSize;
            ncclUniqueId nccl_id;
            ncclComm_t   nccl_comm;
            int          numLocalDevices;
            int          sizeLocalRanks;
            int          localRank = 0;
            int          localSize = 0;
        } nccl;
    };

    auto selfData() -> Data_t&;

    auto selfData() const -> const Data_t&;

    std::shared_ptr<Data_t> m_data;

   public:
    //--------------------------------------------------------------------------
    // INITIALIZATION
    //--------------------------------------------------------------------------

    /**
     * Empty constructor
     */
    Backend();

    /**
     * Initialization with just the type of runtime
     * For streaming it will use all available devices
     * This is the only constructor that works with NCCL/MPI
     */
    explicit Backend(Neon::Runtime runtime /*! Type of runtime to use */);

    /**
     * Creating a Backend object with the first nGpus devices.
     */
    explicit Backend(int           nGpus /*!   Number of devices. The devices are selected in the order specifies by CUDA */,
                     Neon::Runtime runtime /*! Type of runtime to use */);

    /**
     *
     */
    explicit Backend(const std::vector<int>& devIds /*!  Vectors of device ids. There are CUDA device ids */,
                     Neon::Runtime           runtime /*! Type of runtime to use */);

    /**
     *
     */
    explicit Backend(const Neon::set::DevSet& devSet,
                     Neon::Runtime);
    /**
     *
     * @param streamSet
     */
    explicit Backend(const std::vector<int>&     devIds,
                     const Neon::set::StreamSet& streamSet);

    /**
     *
     * @param streamSet
     */
    explicit Backend(const Neon::set::DevSet&    devSet,
                     const Neon::set::StreamSet& streamSet);


    template <typename T>
    auto newDataSet()
        const -> Neon::set::DataSet<T>;

    template <typename T>
    auto newRankData()
        const -> Neon::set::RankData<T>;

    template <typename T>
    auto newDataSet(T const& val)
        const -> Neon::set::DataSet<T>;

    template <typename T>
    auto newRankData(T const& val)
        const -> Neon::set::RankData<T>;

    template <typename T, typename Lambda>
    auto newDataSet(Lambda lambda)
        const -> Neon::set::DataSet<T>;

    template <typename Lambda>
    auto forEachDeviceSeq(const Lambda& lambda)
        const -> void;

    template <typename Lambda>
    auto forEachDevicePar(const Lambda& lambda)
        const -> void;

    template <typename Lambda>
    auto forEachMPIRank(const Lambda& lambda)
        const -> void;

    auto getDeviceCount()
        const -> int;

    auto getNccl()
        const -> Data_t::Nccl&;

    auto clone(Neon::Runtime runtime = Neon::Runtime::system) -> Backend;

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

    auto deviceCount()
        const
        -> int;

    auto isFirstDevice(Neon::SetIdx)
        const
        -> bool;

    auto isLastDevice(Neon::SetIdx)
        const
        -> bool;

    auto isDistributed()
        const
        -> bool;
    /**
     * Returns the mode for the kernel lauch
     * @return
     */
    auto runtime()
        const
        -> const Neon::Runtime&;

    template <typename T>
    auto deviceToDeviceTransfer(int                     streamId,
                                size_t                  nItems,
                                Neon::set::TransferMode transferMode,
                                Neon::SetIdx            dstSet,
                                T*                      dstAddr,
                                Neon::SetIdx            srcSet,
                                T const*                srcAddr)
        const -> void;
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
     * @param streamIdx
     * @return
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
    //    /**
    //     * Extract a stream base in the provided streamIds.
    //     * The method uses a circular policy to select the stream.
    //     * The parameter rotateIdx is the index used to store the
    //     * state of the circular polity in between calls.
    //     *
    //     * @param rotateIdx
    //     * @param streamIdxVec
    //     * @return
    //     */
    //    auto streamSetRotate(int&                    rotateIdx,
    //                         const std::vector<int>& streamIdxVec)
    //        const
    //        -> const Neon::set::StreamSet&;
    //
    //    /**
    //     * Extract a stream base in the provided streamIds.
    //     * The method uses a circular policy to select the stream.
    //     * The parameter rotateIdx is the index used to store the
    //     * state of the circular polity in between calls.
    //     *
    //     * @param rotateIdx
    //     * @param streamIdxVec
    //     * @return
    //     */
    //    static auto streamSetIdxRotate(int&                    rotateIdx,
    //                                   const std::vector<int>& streamIdxVec)
    //        -> int;

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
     * @param idx
     */
    auto sync(int idx) const
        -> void;

    auto sync(Neon::SetIdx setIdx, int idx) const
        -> void;

    auto pushEventOnStream(int eventId, int streamId)
        -> void;

    auto waitEventOnStream(int eventId, int streamId)
        -> void;

    auto pushEventOnStream(Neon::SetIdx setIdx, int eventId, int streamId)
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

    auto getMemoryOptions(Neon::MemoryLayout order) const
        -> Neon::MemoryOptions;

    auto getMemoryOptions(Neon::Allocator    ioAllocator,
                          Neon::Allocator    computeAllocators[Neon::DeviceTypeUtil::nConfig],
                          Neon::MemoryLayout order) const
        -> Neon::MemoryOptions;

    auto getMemoryOptions() const
        -> Neon::MemoryOptions;

    static std::string toString(Neon::Runtime e);

    static auto countAvailableGpus() -> int32_t;

    /**
     *
     * @return
     */
    std::string toString() const;

    auto toReport(Neon::Report& report, Report::SubBlock* subdocAPI = nullptr) const -> void;
    void syncEvent(SetIdx setIdx, int eventIdx) const;

   private:
    auto helpDeviceToDeviceTransferByte(int                     streamId,
                                        size_t                  bytes,
                                        Neon::set::TransferMode transferMode,
                                        Neon::SetIdx            dstSet,
                                        char*                   dstAddr,
                                        Neon::SetIdx            srcSet,
                                        const char*             srcAddr)
        const -> void;
};


}  // namespace Neon

#include "Neon/set/Backend_imp.h"