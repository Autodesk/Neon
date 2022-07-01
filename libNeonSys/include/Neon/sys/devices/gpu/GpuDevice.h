#pragma once

#include <cublas_v2.h>
#include <utility>

#include "Neon/core/core.h"

#include "Neon/sys/devices/DevInterface.h"
#include "Neon/sys/devices/gpu/ComputeID.h"
#include "Neon/sys/devices/gpu/GpuEvent.h"
#include "Neon/sys/devices/gpu/GpuKernelInfo.h"
#include "Neon/sys/devices/gpu/GpuStream.h"
#include "Neon/sys/devices/memType.h"

#include <cuda_runtime.h>

#include <atomic>
#include <functional>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace Neon::sys {


/**
 * Abstracting a GPU device
 */
class GpuDevice : public DeviceInterface
{

   public:
    /**
     * Initialization with gpu device ID
     * @param devIdx
     */
    explicit GpuDevice(const DeviceID& devIdx);

    /**
     * Move constructor
     */
    GpuDevice(GpuDevice&&);

    /**
     * Empty constructor
     */
    GpuDevice();

    /**
     * We don't want a copy constructor. One a GpuDevicehas been created it can only be moved.
     * We should delete this constructor but if we do it, VS complains (clang, gcc, intel are ok).
     * So We leave the constructor but we fire an exception if called.
     */
    GpuDevice(const GpuDevice&) = delete;

    /**
     * We don't want a copy operator. One a GpuDevicehas been created it can only be moved.
     * We should delete this operator but if we do it, VS complains (clang, gcc, intel are ok).
     * So we leave the copy but we fire an exception if called.
     */
    GpuDevice& operator=(const GpuDevice&)
    {
        NeonException excp("GpuDev");
        excp << "Assignment operator has been called. This operation is not permitted on a GpuDeviceobject.";
        NEON_THROW(excp);
    };

    /**
     * Destructor
     */
    virtual ~GpuDevice() = default;

    /**
     * Returns ratio of used memory to total memory
     * @return ratio of used memory to total memory
     */
    double usage() const override;

    /**
     * Returns the size of available virtual memory
     */
    int64_t virtMemory() const override
    {
        return -1;
    }

    /**
     * Returns the size of used virtual memory
     */
    int64_t usedVirtMemory() const override
    {
        return -1;
    }

    /**
     * Returns the size of available physical memory
     */
    int64_t physMemory() const override
    {
        return -1;
    }

    /**
     * Returns the size of used physical memory
     */
    int64_t usedPhysMemory() const override
    {
        return -1;
    }

    /**
     *
     * @return
     */
    int64_t processUsedPhysMemory() const override
    {
        return -1;
    }

   public:
    /**
     *
     */
    struct kernel_t
    {
       public:
       private:
        GpuDevice& gpuDev;

        const GpuDevice& getGpuDev() const
        {
            return gpuDev;
        }

       public:
        kernel_t(GpuDevice& gpuDev)
            : gpuDev(gpuDev){};

       public:
        /**
         * cudaLaunchKernel mode I
         * Launch a cuda kernel. Parameters for the kernels are passed as vector of untyped pointers.
         * @tparam runMode: sync or async
         * @param gpuStream: stream for the operation
         * @param kernelInfo: sizing of the cuda grid and the shared memory for the kernel
         * @param fun: function implementing the cuda kernel. It must be a __global__ function
         * @param argumentPtrs: input parameters for the kernel.
         */
        template <run_et::et runMode>
        void cudaLaunchKernel(
            const GpuStream&     gpuStream, /** Define on which stream the kernel will be queue on */
            const GpuLaunchInfo& kernelInfo,
            void*                fun,
            void*                argumentPtrs[]) const
        {

            gpuDev.tools.setActiveDevContext();

            const auto         cudaBlock = kernelInfo.cudaBlock();
            const auto         cudaGrid = kernelInfo.cudaGrid();
            const auto         shrMemSize = kernelInfo.shrMemSize();
            cudaFuncAttributes func_attr = cudaFuncAttributes();
            auto               error = cudaFuncGetAttributes(&func_attr, fun);
            if (error != cudaSuccess) {
                NeonException exc;
                exc << "\n Error: " << cudaGetErrorString(error);
                NEON_THROW(exc);
            }

            error = ::cudaLaunchKernel((void*)fun,
                                       cudaGrid,
                                       cudaBlock,
                                       argumentPtrs,
                                       shrMemSize,
                                       gpuStream.stream());

            if (error != cudaSuccess) {
                NeonException exc;
                exc << "\n Error: " << cudaGetErrorString(error);
                exc << "\n Kernel requires " << func_attr.numRegs << " registers per thread";
                exc << "\n Kernel requires " << func_attr.sharedSizeBytes << " bytes of static shared memory";
                exc << "\n Kernel requires " << func_attr.constSizeBytes << " bytes of user-allocated constant memory";
                exc << "\n Kernel requires " << func_attr.localSizeBytes << " bytes of local memory per thread";
                exc << "\n Kernel maximum thread/block is " << func_attr.maxThreadsPerBlock << " while launched block is " << cudaBlock.x * cudaBlock.y * cudaBlock.z;
                exc << "\n Kernel maximum dynamic shared memory is " << func_attr.maxDynamicSharedSizeBytes << " bytes while launched dynamic shared memory is " << shrMemSize << "bytes\n";
                NEON_THROW(exc);
            }

            gpuStream.sync<runMode>();
        }
    };

    /**
     * Structure to group methods related to memory management.
     */
    struct memory_t
    {
       private:
        GpuDevice& gpuDev;

       public:
        memory_t(GpuDevice& gpuDev)
            : gpuDev(gpuDev) {}

        /**
         * Wrapper around cudaMalloc.
         */
        template <typename Type_ta>
        Type_ta* malloc(int64_t nElements) const;
        void*    mallocByte(size_t size) const;


        /**
         * Free.
         */
        template <typename Type_ta>
        void free(Type_ta*& mem) const;
        /**
         * Free.
         */
        template <typename Type_ta>
        void free(const Type_ta*& mem) const;

        /**
         * Wrapper around cudaMemcpy.
         */
        template <typename Type_ta, mem_et::enum_e dest_ta, mem_et::enum_e src_ta, run_et::et runMode>
        void transfer(const GpuStream& gpuStream, Type_ta* dest, const Type_ta* src, const int64_t nElemensts) const;

        void enablePeerAccsessWith(ComputeID gpuId) const;

        void peerTransfer(const GpuStream& gpuStream, ComputeID dstDevId, char* dest, ComputeID srcDevId, const char* src, size_t numBytes) const;

        void memSet(void* mem, uint8_t val, size_t size) const;
    };  // End of memory section


    /**
     * Internal structure to collect some tools
     */
    struct tools_t
    {
       private:
        GpuDevice& gpuDev;

       public:
        enum arch_e
        {
            OldGen = 0,
            Kepler = 1,
            Maxwell = 2,
            Pascal = 3,
            Volta = 4,
            Turing = 5,
            Ampere = 6,
            NextGen = 7
        };

        explicit tools_t(GpuDevice& gpuDev);

        /**
         * Set the cuda context to the device
         */
        void setActiveDevContext() const;

        arch_e arch() const;

        /**
         * Returns the name of the device
         * @return
         */
        std::string getDevName() const;

        /**
         * Returns major compute capability of the device
         * @return
         */
        int majorComputeCapability() const;

        /**
         * Returns minor compute capability of the device
         * @return
         */
        int minorComputeCapability() const;

        /**
         * Synchronize with any operation on kernel;
         */
        void sync() const;

        /**
         * Create a GpuStream_t object associated to this gpu device
         */
        GpuStream stream() const;

        /**
         *
         * @param stream
         */
        void streamDestroy(GpuStream& stream) const;

        /**
         * Create a GpuEvent_t object associated to this gpu device
         */
        GpuEvent event(bool disableTiming) const;

        /**
         * Create and initialize a new cublas handle          
        */
        cublasHandle_t cublasHandle(bool device_pointer_mode = true) const;

        /**
         * return device properties          
        */
        cudaDeviceProp getDeviceProp() const;
        /**
         *
         */
        void eventDestroy(GpuEvent& event) const;

        /**
         * Returns a string with all the information of the device.
         * @param prefix
         * @return
         */
        std::string getDevInfo(const std::string& prefix = std::string("")) const;
    };


    tools_t  tools;
    memory_t memory;
    kernel_t kernel;
};

}  // namespace Neon::sys

#include "Neon/sys/devices/gpu/gpuDevice/gpuDeviceMemory.h"
