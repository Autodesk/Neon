#pragma once
#include "Neon/core/tools/metaprogramming.h"
#include "Neon/core/tools/metaprogramming/applyTuple.h"

#include "Neon/set/Capture.h"
#include "Neon/set/MemoryOptions.h"

#include <omp.h>
#include <functional>
#include <future>
#include <iostream>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

// #include "Neon/set/backend.h"
#include "Neon/set/DataSet.h"
#include "Neon/set/GpuEventSet.h"
#include "Neon/set/GpuStreamSet.h"
#include "Neon/set/KernelConfig.h"
#include "Neon/set/LambdaExecutor.h"
#include "Neon/set/LaunchParameters.h"
#include "Neon/set/SingletonSet.h"
#include "Neon/set/Transfer.h"
#include "Neon/set/memory/memDevSet.h"
#include "Neon/set/memory/memSet.h"
#include "Neon/sys/global/GpuSysGlobal.h"
#include "Neon/sys/memory/memConf.h"
#include "Neon/set/KernelConfig.h"

namespace Neon {
namespace set {

/**
 * Abstraction for a set of devices.
 */
class DevSet
{
   private:
    DataSet<Neon::sys::DeviceID> m_devIds;                /** DataSet of device ids */
    Neon::DeviceType             m_devType;               /** Types of the device */
    bool                         m_fullyConnected{false}; /** true if any device can communicate with the others in the set */
    Neon::set::StreamSet         m_defaultStream;         /** a default stream used by the device set */
    std::vector<Neon::SetIdx>    m_idxRange;

   public:
    //--------------------------------------------------------------------------
    // INITIALIZATION
    //--------------------------------------------------------------------------

    /**
     * Empty constructor
     */
    DevSet() = default;

    /**
     * Constructing a DevSet object for a vector of GPU ids.
     * The ids are considered as for a CUDA set of devices
     */
    DevSet(const std::vector<Neon::sys::ComputeID>& set);

    /**
     * Constructing a DevSet object for a vector of GPU ids.
     */
    explicit DevSet(const Neon::DeviceType& devType, const std::vector<int>& set);

    /**
     * Set the ids of a DevSet object.
     * The method can be called only on DevSet object that have not been
     * associated to any GPU. If this is the case, an exception if thrown.
     *
     * @param set[in]: id of gpu devices
     */
    [[deprecated]] auto set(const Neon::DeviceType&                  devType,
                            const std::vector<Neon::sys::ComputeID>& set)
        -> void;

    /**
     * Returns the gpu set covering all gpu in the system
     * The set may be not fully connected.
     */
    static auto maxSet(Neon::DeviceType devType = Neon::DeviceType::CUDA)
        -> DevSet;

    //--------------------------------------------------------------------------
    // INSPECTION
    //--------------------------------------------------------------------------

    /**
     * Retrieve the device index for the ith gpu managed by this DevSet
     */
    auto devId(SetIdx index) const
        -> Neon::sys::DeviceID;

    /**
     * Retrieve all device indexes
     */
    auto devId() const
        -> const DataSet<Neon::sys::DeviceID>&;

    /**
     * Retrieve the gpu system index for the ith gpu managed by this DevSet
     */
    [[deprecated]] auto gpuDev(SetIdx index) const
        -> const Neon::sys::GpuDevice&;

    /**
     * Provides the vector of GPU ids that is managed by the GpuSet
     * @return an ordered vector of Ids.
     */
    auto idSet() const
        -> const DataSet<Neon::sys::DeviceID>&;

    /**
     * returns true if all the GPUs are fully connected.
     * Fully connected means that any GPU in the set can read and write buffers
     * that are located in any of the other GPUs in the set.
     */
    auto isFullyConnected() const
        -> bool;

    /**
     * Returns the number of devices managed by this object
     **/
    auto setCardinality() const
        -> int32_t;

    auto getRange()
        const
        -> const std::vector<SetIdx>&;

    /**
     * Returns the type of this device
     * @return
     */
    auto type()
        const
        -> const Neon::DeviceType&;

    /**
     * Prints on standard output information on the specified GPU
     * @param gpuIdx[in]: id of the selected gpu devices
     **/
    auto getInfo(Neon::sys::ComputeID gpuIdx)
        const
        -> void;

    /**
     * the method returns a string with the name of the selected GPU model
     * @param gpuIdx[in]: id of the selected gpu devices
     **/
    auto getDevName(Neon::sys::ComputeID gpuIdx)
        const
        -> std::string;

    [[deprecated]] auto userIdVec()
        const
        -> const std::vector<int>;

    template <typename ta_Lambda>
    auto forEachSetIdxPar(const ta_Lambda& lambda) const
        -> void
    {
        const int setCard = setCardinality();
#pragma omp parallel for num_threads(setCard)
        for (int index = 0; index < setCard; index++) {
            Neon::SetIdx setIdx(index);
            lambda(setIdx);
        }
    }

    template <typename ta_Lambda>
    auto forEachSetIdxSeq(const ta_Lambda& lambda) const
        -> void
    {
        const int setCard = setCardinality();
        for (int index = 0; index < setCard; index++) {
            Neon::SetIdx setIdx(index);
            lambda(setIdx);
        }
    }

    //--------------------------------------------------------------------------
    // SYNCRONIZATIONS
    //--------------------------------------------------------------------------

    /**
     * Creates a new streamSet for this DevSet instance.
     * The resources acquired by the return object must be explicitly released.
     */
    auto newStreamSet() const
        -> StreamSet;

    /**
     * Returns the default streamSet that is managed by this object.
     * The stremSet vector is of length zero for non CUDA devices
     * @return
     */
    auto defaultStreamSet()
        const
        -> const StreamSet&;

    /**
     * Synchronizes each GPU w.r.t the default streamSet managed.
     */
    auto sync() -> void;

    /**
     * Creates a new cuda event for each GPU
     * @param disableTiming
     * @return
     */
    auto newEventSet(bool disableTiming) const
        -> GpuEventSet;

    /**
     * Calls cudaDeviceSynchronize on all device in the set effectively syncing
     * all devices on the set
     */
    auto devSync() const
        -> void;
    //--------------------------------------------------------------------------
    // KERNELS
    //--------------------------------------------------------------------------

    auto newLaunchParameters() const
        -> LaunchParameters;

    template <typename DataSetContainer_ta, typename Lambda_ta>
    inline auto kernelLambdaWithIterator(
        Neon::Execution                          execution,
        const Neon::set::KernelConfig&           kernelConfig,
        DataSetContainer_ta&                     dataSetContainer,
        std::function<Lambda_ta(SetIdx,
                                Neon::DataView)> lambdaHolder) const -> void
    {
        Neon::Runtime mode = kernelConfig.backend().runtime();
        // ORDER is IMPORTANT
        // KEEP OPENMP for last
        switch (mode) {
            case Neon::Runtime::stream: {
                if (execution == Neon::Execution::device) {
                    this->template h_kLambdaWithIterator_cudaStreams<DataSetContainer_ta, Lambda_ta>(kernelConfig,
                                                                                                     dataSetContainer,
                                                                                                     lambdaHolder);
                    return;
                }
                [[fallthrough]];
            };
            case Neon::Runtime::openmp: {
                this->template h_kLambdaWithIterator_openmp<DataSetContainer_ta, Lambda_ta>(execution,
                                                                                            kernelConfig,
                                                                                            dataSetContainer,
                                                                                            lambdaHolder);
                return;
            };
            default: {
                NeonException exception;
                exception << "Unsupported configuration";
                NEON_THROW(exception);
            }
        }
    }

    template <typename DataSetContainer_ta, typename Lambda_ta>
    inline auto kernelLambdaWithIterator(
        Neon::Execution                          execution,
        Neon::SetIdx                             setIdx,
        const Neon::set::KernelConfig&           kernelConfig,
        DataSetContainer_ta&                     dataSetContainer,
        std::function<Lambda_ta(SetIdx,
                                Neon::DataView)> lambdaHolder) const -> void
    {
        Neon::Runtime mode = kernelConfig.backend().runtime();
        // ORDER is IMPORTANT
        // KEEP OPENMP for last
        switch (mode) {
            case Neon::Runtime::stream: {
                if (execution == Neon::Execution::device) {

                    this->template h_kLambdaWithIterator_cudaStreams<DataSetContainer_ta, Lambda_ta>(setIdx,
                                                                                                     kernelConfig,
                                                                                                     dataSetContainer,
                                                                                                     lambdaHolder);
                    return;
                }
                [[fallthrough]];
            };
            case Neon::Runtime::openmp: {
                this->template h_kLambdaWithIterator_openmp<DataSetContainer_ta, Lambda_ta>(execution,
                                                                                            setIdx,
                                                                                            kernelConfig,
                                                                                            dataSetContainer,
                                                                                            lambdaHolder);
                return;
            };
            default: {
                NeonException exception;
                exception << "Unsupported configuration";
                NEON_THROW(exception);
            }
        }
    }

    template <typename DataSetContainer_ta, typename Lambda_ta>
    inline auto kernelHostLambdaWithIterator(
        const Neon::set::KernelConfig&           kernelConfig,
        DataSetContainer_ta&                     dataSetContainer,
        std::function<Lambda_ta(Neon::Execution,
                                SetIdx,
                                Neon::DataView)> lambdaHolder) const -> void
    {
        Neon::Runtime mode = Neon::Runtime::openmp;
        // ORDER is IMPORTANT
        // KEEP OPENMP for last
        switch (mode) {
            case Neon::Runtime::openmp: {
                this->template h_kLambdaWithIterator_openmp<DataSetContainer_ta, Lambda_ta>(kernelConfig,
                                                                                            dataSetContainer,
                                                                                            lambdaHolder);
                return;
            };
            default: {
                NeonException exception;
                exception << "Unsupported configuration";
                NEON_THROW(exception);
            }
        }
    }

    template <typename DataSetContainer_ta, typename Lambda_ta>
    inline auto kernelHostLambdaWithIterator(Neon::SetIdx                             setIdx,
                                             const Neon::set::KernelConfig&           kernelConfig,
                                             DataSetContainer_ta&                     dataSetContainer,
                                             std::function<Lambda_ta(Neon::Execution,
                                                                     SetIdx,
                                                                     Neon::DataView)> lambdaHolder) const -> void
    {
        Neon::Runtime mode = Neon::Runtime::openmp;
        switch (mode) {
            case Neon::Runtime::openmp: {
                this->template h_kLambdaWithIterator_openmp<DataSetContainer_ta, Lambda_ta>(setIdx,
                                                                                            kernelConfig,
                                                                                            dataSetContainer,
                                                                                            lambdaHolder);
                return;
            };
            default: {
                NeonException exception;
                exception << "Unsupported configuration";
                NEON_THROW(exception);
            }
        }
    }

    template <typename DataSetContainer_ta, typename Lambda_ta>
    inline auto h_kLambdaWithIterator_cudaStreams([[maybe_unused]] const Neon::set::KernelConfig&           kernelConfig,
                                                  [[maybe_unused]] DataSetContainer_ta&                     dataSetContainer,
                                                  [[maybe_unused]] std::function<Lambda_ta(SetIdx,
                                                                                           Neon::DataView)> lambdaHolder)
        const -> void
    {
        if (m_devType != Neon::DeviceType::CUDA) {
            Neon::NeonException exp("DevSet");
            exp << "Error, DevSet::invalid operation on a non GPU type of device.\n";
            NEON_THROW(exp);
        }
#ifdef NEON_COMPILER_CUDA

        const StreamSet&        gpuStreamSet = kernelConfig.streamSet();
        const LaunchParameters& launchInfoSet = kernelConfig.launchInfoSet();
        const int               nGpus = int(m_devIds.size());
        {
#pragma omp parallel num_threads(nGpus) default(shared) firstprivate(lambdaHolder)
            {
                int                         idx = omp_get_thread_num();
                const Neon::sys::GpuDevice& dev = Neon::sys::globalSpace::gpuSysObj().dev(m_devIds[idx]);
                // std::tuple<funParametersType_ta& ...>argsForIthGpuFunction(parametersVec.at(i) ...);

                auto      iterator = dataSetContainer.getSpan(idx, kernelConfig.dataView());
                Lambda_ta lambda = lambdaHolder(idx, kernelConfig.dataView());
                void*     untypedParams[2] = {&iterator, &lambda};
                void*     executor = (void*)Neon::set::internal::execLambdaWithIterator_cuda<DataSetContainer_ta, Lambda_ta>;

                dev.kernel.template cudaLaunchKernel<Neon::run_et::async>(gpuStreamSet[idx],
                                                                          launchInfoSet[idx],
                                                                          executor,
                                                                          untypedParams);
            }
            if (kernelConfig.runMode() == Neon::run_et::sync) {
                gpuStreamSet.sync();
            }
        }
#else
        NeonException exp("DevSet");
        exp << "A lambda with CUDA device code must be compiled within a .cu file.";
        NEON_THROW(exp);
#endif

        return;
    }

    template <typename DataSetContainer_ta, typename Lambda_ta>
    inline auto h_kLambdaWithIterator_cudaStreams([[maybe_unused]] Neon::SetIdx                             setIdx,
                                                  [[maybe_unused]] const Neon::set::KernelConfig&           kernelConfig,
                                                  [[maybe_unused]] DataSetContainer_ta&                     dataSetContainer,
                                                  [[maybe_unused]] std::function<Lambda_ta(SetIdx,
                                                                                           Neon::DataView)> lambdaHolder)
        const -> void
    {
        if (m_devType != Neon::DeviceType::CUDA) {
            Neon::NeonException exp("DevSet");
            exp << "Error, DevSet::invalid operation on a non GPU type of device.\n";
            NEON_THROW(exp);
        }
#ifdef NEON_COMPILER_CUDA

        const StreamSet&        gpuStreamSet = kernelConfig.streamSet();
        const LaunchParameters& launchInfoSet = kernelConfig.launchInfoSet();
        const int               nGpus = int(m_devIds.size());
        {

            const Neon::sys::GpuDevice& dev = Neon::sys::globalSpace::gpuSysObj().dev(m_devIds[setIdx.idx()]);
            // std::tuple<funParametersType_ta& ...>argsForIthGpuFunction(parametersVec.at(i) ...);

            auto      iterator = dataSetContainer.getSpan(setIdx.idx(), kernelConfig.dataView());
            Lambda_ta lambda = lambdaHolder(setIdx.idx(), kernelConfig.dataView());
            void*     untypedParams[2] = {&iterator, &lambda};
            void*     executor = (void*)Neon::set::internal::execLambdaWithIterator_cuda<DataSetContainer_ta, Lambda_ta>;

            dev.kernel.template cudaLaunchKernel<Neon::run_et::async>(gpuStreamSet[setIdx.idx()],
                                                                      launchInfoSet[setIdx.idx()],
                                                                      executor,
                                                                      untypedParams);
        }
        if (kernelConfig.runMode() == Neon::run_et::sync) {
            gpuStreamSet.sync();
        }

#else
        NeonException exp("DevSet");
        exp << "A lambda with CUDA device code must be compiled within a .cu file.";
        NEON_THROW(exp);
#endif

        return;
    }

    template <typename DataSetContainer_ta, typename Lambda_ta>
    inline auto h_kLambdaWithIterator_openmp(
        Neon::Execution                                           ,
        [[maybe_unused]] const Neon::set::KernelConfig&           kernelConfig,
        [[maybe_unused]] DataSetContainer_ta&                     dataSetContainer,
        [[maybe_unused]] std::function<Lambda_ta(SetIdx,
                                                 Neon::DataView)> lambdaHolder)
        const -> void
    {
        const LaunchParameters& launchInfoSet = kernelConfig.launchInfoSet();
        const int               nGpus = static_cast<int>(m_devIds.size());
        {
            for (int idx = 0; idx < nGpus; idx++) {
                auto      iterator = dataSetContainer.getSpan(idx,
                                                              kernelConfig.dataView());
                Lambda_ta lambda = lambdaHolder(idx, kernelConfig.dataView());
                Neon::set::internal::execLambdaWithIterator_omp<DataSetContainer_ta, Lambda_ta>(launchInfoSet[idx].domainGrid(), iterator, lambda);
            }
        }
        return;
    }

    template <typename DataSetContainer_ta, typename Lambda_ta>
    inline auto h_kLambdaWithIterator_openmp(Neon::Execution                                           ,
                                             Neon::SetIdx                                              setIdx,
                                             [[maybe_unused]] const Neon::set::KernelConfig&           kernelConfig,
                                             [[maybe_unused]] DataSetContainer_ta&                     dataSetContainer,
                                             [[maybe_unused]] std::function<Lambda_ta(SetIdx,
                                                                                      Neon::DataView)> lambdaHolder)
        const -> void
    {
        if (m_devType != Neon::DeviceType::CPU) {
            Neon::NeonException exp("DevSet");
            exp << "Error, DevSet::invalid operation on a non GPU type of device.\n";
            NEON_THROW(exp);
        }
        const LaunchParameters& launchInfoSet = kernelConfig.launchInfoSet();
        {
            auto      iterator = dataSetContainer.getSpan(setIdx,
                                                          kernelConfig.dataView());
            Lambda_ta lambda = lambdaHolder(setIdx, kernelConfig.dataView());
            Neon::set::internal::execLambdaWithIterator_omp<DataSetContainer_ta, Lambda_ta>(launchInfoSet[setIdx.idx()].domainGrid(), iterator, lambda);
        }
        return;
    }

   public:
    //--------------------------------------------------------------------------
    // MEMORY MANAGEMENT
    //--------------------------------------------------------------------------


    auto getMemoryOptions(Neon::MemoryLayout order) const -> Neon::MemoryOptions
    {
        MemoryOptions memoryOptions(Neon::DeviceType::CPU,
                                    type(),
                                    order);
        return memoryOptions;
    }

    auto getMemoryOptions(Neon::Allocator    ioAllocator,
                          Neon::Allocator    computeAllocators[Neon::DeviceTypeUtil::nConfig],
                          Neon::MemoryLayout order) const
        -> Neon::MemoryOptions
    {
        MemoryOptions memoryOptions(Neon::DeviceType::CPU,
                                    ioAllocator,
                                    type(),
                                    computeAllocators,
                                    order);
        return memoryOptions;
    }

    auto sanitizeMemoryOption(const Neon::MemoryOptions& memOpt) const
        -> Neon::MemoryOptions
    {
        if (!memOpt.helpWasInitCompleted()) {
            const Neon::MemoryOptions defaultMemOption = getMemoryOptions(Neon::MemoryLayout::structOfArrays);
            return defaultMemOption;
        }
        return memOpt;
    }
    /**
     * Returns a memSet for this GPU set
     * @param allocType: CPU or CUDA
     * @param allocType: type of the allocation
     * @param nElement: number of elements to be allocated
     * @param alignment: memory alignment, by default is 1 i.e. the alignment of the used allocator.
     * @return
     */
    template <typename T_ta>
    auto newMemDevSet(Neon::DeviceType        devType,
                      const Neon::Allocator&  allocType,
                      uint64_t                nElement,
                      Neon::sys::MemAlignment alignment = Neon::sys::MemAlignment()) const
        -> MemDevSet<T_ta>
    {
        const auto nElementDataSet = this->newDataSet(nElement);
        auto       out = newMemDevSet<T_ta>(devType, allocType, nElementDataSet, alignment);
        return out;
    }

    /**
     * Returns a memSet for this GPU set
     * @param allocType: CPU or CUDA
     * @param allocType: type of the allocation
     * @param nElementVec: vector of number of elements to be allocated on each device
     * @param alignment: memory alignment, by default is 1 i.e. the alignment of the used allocator.
     * @return
     */

    template <typename T_ta>
    auto newMemDevSet(Neon::DeviceType                    devType,
                      const Neon::Allocator&              allocType,
                      const Neon::set::DataSet<uint64_t>& nElementVec,
                      Neon::sys::MemAlignment             alignment = Neon::sys::MemAlignment()) const
        -> MemDevSet<T_ta>
    {
        switch (devType) {
            case Neon::DeviceType::CUDA: {
                return MemDevSet<T_ta>(devType,
                                       m_devIds,
                                       std::forward<const Neon::Allocator>(allocType),
                                       nElementVec,
                                       alignment);
            }
            case Neon::DeviceType::CPU: {
                std::vector<Neon::sys::DeviceID> idVec(this->setCardinality(), 0);
                return MemDevSet<T_ta>(devType, idVec, allocType, nElementVec, alignment);
            }
            default: {
                Neon::NeonException exp("GpuSet");
                exp << "GpuSet::newMemDevSet, a non supported devType was detected. DevType is: " << devType;
                NEON_THROW(exp);
            }
        }
    }

    template <typename T_ta>
    auto newMemDevSet(int                                 cardinality,
                      Neon::sys::memConf_t                memConfig,
                      const Neon::set::DataSet<uint64_t>& nElementVec) const
    {

        return newMemDevSet<T_ta>(cardinality,
                                  memConfig.devEt(),
                                  memConfig.allocEt(),
                                  nElementVec,
                                  memConfig.order(),
                                  memConfig.alignment(),
                                  memConfig.padding());
    }

    template <typename T_ta>
    auto newMemDevSet(int                                 cardinality,
                      Neon::DeviceType                    devType,
                      const Neon::Allocator&              allocType,
                      const Neon::set::DataSet<uint64_t>& nElementVec,
                      Neon::memLayout_et::order_e         order = Neon::memLayout_et::order_e::structOfArrays,
                      Neon::sys::MemAlignment             alignment = Neon::sys::MemAlignment(),
                      Neon::memLayout_et::padding_e       padding = Neon::memLayout_et::padding_e::OFF) const
        -> MemDevSet<T_ta>
    {
        if (m_devType != Neon::DeviceType::CUDA &&
            devType == Neon::DeviceType::CUDA &&
            allocType != Neon::Allocator::NULL_MEM) {
            Neon::NeonException exp("DevSet");
            exp << "Error, DevSet::invalid operation on a non GPU type of device.\n";
            NEON_THROW(exp);
        }

        switch (devType) {
            case Neon::DeviceType::CUDA: {
                return MemDevSet<T_ta>(cardinality,
                                       order,
                                       padding,
                                       devType,
                                       m_devIds,
                                       std::forward<const Neon::Allocator>(allocType),
                                       nElementVec,
                                       alignment);
            }
            case Neon::DeviceType::CPU: {
                std::vector<Neon::sys::DeviceID> idVec(this->setCardinality(), 0);
                return MemDevSet<T_ta>(cardinality,
                                       order,
                                       padding,
                                       devType,
                                       idVec,
                                       std::forward<const Neon::Allocator>(allocType),
                                       nElementVec,
                                       alignment);
            }
            default: {
                Neon::NeonException exp("GpuSet");
                exp << "GpuSet::newMemDevSet, a non supported devType was detected. DevType is: " << devType;
                NEON_THROW(exp);
            }
        }
    }

    template <typename T_ta>
    auto newMemSet(int                                 cardinality,
                   Neon::sys::memConf_t                cpuConfig,
                   Neon::sys::memConf_t                gpuConfig,
                   const Neon::set::DataSet<uint64_t>& nElementVec) const
        -> Neon::set::MemSet<T_ta>
    {
        if (cpuConfig.devEt() != Neon::DeviceType::CPU || gpuConfig.devEt() != Neon::DeviceType::CUDA) {
            Neon::NeonException exp("DevSet");
            exp << "Error, DevSet::invalid configuration for memory.\n";
            NEON_THROW(exp);
        }
        Neon::set::MemSet<T_ta> mirror(this->setCardinality());

        MemDevSet<T_ta> memCpu = newMemDevSet<T_ta>(cardinality, cpuConfig, nElementVec);
        MemDevSet<T_ta> memGpu = newMemDevSet<T_ta>(cardinality, gpuConfig, nElementVec);

        mirror.link(memCpu);
        mirror.link(memGpu);

        return mirror;
    }

    template <typename T_ta>
    auto newMemSet(Neon::DataUse                       dataUse,
                   int                                 cardinality,
                   Neon::MemoryOptions                 memoryOptions,
                   const Neon::set::DataSet<uint64_t>& nElementVec) const
        -> Neon::set::MemSet<T_ta>
    {
        memoryOptions = sanitizeMemoryOption(memoryOptions);
        Neon::sys::memConf_t cpuConfig(Neon::DeviceType::CPU,
                                       memoryOptions.getIOAllocator(dataUse),
                                       memoryOptions.getOrder() == Neon::MemoryLayout::structOfArrays ? Neon::memLayout_et::order_e::structOfArrays : Neon::memLayout_et::order_e::arrayOfStructs);

        Neon::sys::memConf_t gpuConfig(Neon::DeviceType::CUDA,
                                       memoryOptions.getComputeAllocator(dataUse),
                                       memoryOptions.getOrder() == Neon::MemoryLayout::structOfArrays ? Neon::memLayout_et::order_e::structOfArrays : Neon::memLayout_et::order_e::arrayOfStructs);

        return this->template newMemSet<T_ta>(cardinality, cpuConfig, gpuConfig, nElementVec);
    }

    template <typename T_ta>
    auto newDataSet() const
        -> DataSet<T_ta>
    {
        DataSet<T_ta> newData(setCardinality());
        return newData;
    }

    template <typename T_ta>
    auto newDataSet(const T_ta& dataReplicated) const
        -> DataSet<T_ta>
    {
        DataSet<T_ta> newData(setCardinality(), dataReplicated);
        return newData;
    }

    template <typename T_ta>
    auto newDataSet(std::function<void(Neon::SetIdx idx, T_ta&)> f) const
        -> DataSet<T_ta>
    {
        DataSet<T_ta> newData(setCardinality());

        for (int i = 0; i < setCardinality(); i++) {
            f(i, newData[i]);
        }
        return newData;
    }

    /**
     * Returns the amount of memory used the ith GPU of this DevSet
     */
    auto memInUse(SetIdx)
        -> size_t;

    /**
     * Returns the amount of memory used the ith GPU of this DevSet
     */
    auto memMaxUse(SetIdx)
        -> size_t;

    auto memForUse(SetIdx)
        -> size_t;


    //--------------------------------------------------------------------------
    // MEMORY TRANSFERS
    //--------------------------------------------------------------------------


    template <TransferMode transferMode_ta>
    [[deprecated]] auto peerTransfer(const StreamSet& streamSet,
                                     SetIdx           dstSetId,
                                     char*            dstBuf,
                                     SetIdx           srcSetIdx,
                                     const char*      srcBuf,
                                     size_t           numBytes)
        const
        -> void;


    auto peerTransfer(PeerTransferOption&        opt,
                      const Neon::set::Transfer& transfer)
        const
        -> void;


    //--------------------------------------------------------------------------
    // TOOLS
    //--------------------------------------------------------------------------

    /**
     * Method use to execute a cudaSetDevice for a specific GPU in the set
     * @param index
     */
    auto setActiveDevContext(SetIdx index)
        const
        -> void;

    auto toString()
        const
        -> std::string;

    void sanitize(Neon::sys::memConf_t& mem)
    {
        switch (mem.devEt()) {
            case Neon::DeviceType::CPU: {
                break;
            }
            case Neon::DeviceType::CUDA: {
                if (m_devType != Neon::DeviceType::CUDA) {
                    mem.m_allocEt = {Neon::Allocator::NULL_MEM};
                }
                break;
            }
            default: {
                NEON_THROW_UNSUPPORTED_OPTION("");
            }
        }
    }

   private:
    /**
     * Validates the set of gpu ids.
     * Return a empty vector if all ids are valid.
     * If an id is not valid it is returned within the output vector.
     */
    auto validateIds()
        const
        -> std::vector<Neon::sys::ComputeID>;

    auto h_init_defaultStreamSet() -> void;

};  // namespace set

extern template auto Neon::set::DevSet::peerTransfer<Neon::set::TransferMode::put>(const StreamSet& streamSet,
                                                                                   SetIdx           dstSetId,
                                                                                   char*            dstBuf,
                                                                                   SetIdx           srcSetIdx,
                                                                                   const char*      srcBuf,
                                                                                   size_t           numBytes)
    const
    -> void;

extern template auto Neon::set::DevSet::peerTransfer<Neon::set::TransferMode::get>(const StreamSet& streamSet,
                                                                                   SetIdx           dstSetId,
                                                                                   char*            dstBuf,
                                                                                   SetIdx           srcSetIdx,
                                                                                   const char*      srcBuf,
                                                                                   size_t           numBytes)
    const
    -> void;


}  // namespace set
}  // namespace Neon
