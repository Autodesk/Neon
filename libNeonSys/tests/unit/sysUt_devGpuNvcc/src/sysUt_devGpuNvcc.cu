
#include "gtest/gtest.h"

#include <cuda_runtime.h>
#include <cstring>
#include <iostream>

#include "Neon/Neon.h"

#include "Neon/core/types/vec.h"
#include "Neon/sys/devices/gpu/GpuDevice.h"
#include "Neon/sys/devices/gpu/GpuTools.h"
#include "Neon/sys/devices/memType.h"
#include "Neon/sys/global/GpuSysGlobal.h"


namespace kernelRun {
//int a, int* memhelp_recursive_t
__global__ void kernel(int a, int* mem)
{
    if (threadIdx.x < 1 && threadIdx.y < 1 && threadIdx.z < 1) {
        printf("a\n");
        printf(
            "b %d\n", a);
    }

    if (threadIdx.x < 1 && threadIdx.y < 1 && threadIdx.z < 1) {
        printf("a\n");
    }
}

}  // end of namespace kernelRun

/**
 * Launching a kernel passing parameters as vector of pointers (type unsafe mode).
 */
TEST(gpuDev, cudaLaunchKernelII)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        using namespace Neon;
        Vec_3d<int> domainGrid(33, 1, 1);
        int32_3d    cudaBlock(1024, 1, 1);
        size_t      sharedMem = 0;

        ASSERT_NO_THROW(Neon::sys::globalSpace::gpuSysObj().dev(0));
        const Neon::sys::GpuDevice& gpuDev = Neon::sys::globalSpace::gpuSysObj().dev(0);

        int* gpuMem;
        ASSERT_NO_THROW(gpuMem = gpuDev.memory.malloc<int>(domainGrid.rMul()));

        Neon::sys::GpuStream gpuStream;
        ASSERT_NO_THROW(gpuStream = gpuDev.tools.stream());

        Neon::sys::GpuLaunchInfo kernelInfo(Neon::sys::GpuLaunchInfo::mode_e::domainGridMode, domainGrid, cudaBlock, sharedMem);
        int                      val = 33;

        auto tuple = std::make_tuple(gpuStream, kernelInfo, kernelRun::kernel, val, gpuMem);


        void* params[2] = {&val, &gpuMem};
        ASSERT_NO_THROW(gpuDev.kernel.cudaLaunchKernel<run_et::sync>(gpuStream, kernelInfo, (void*)kernelRun::kernel, params));
    }
}

TEST(gpuDev, cudaLaunchKernelIII)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        using namespace Neon;
        Vec_3d<int> domainGrid(33, 1, 1);
        int32_3d    cudaBlock(1024, 1, 1);
        size_t      sharedMem = 0;

        ASSERT_NO_THROW(Neon::sys::globalSpace::gpuSysObj().dev(0));
        const Neon::sys::GpuDevice& gpuDev = Neon::sys::globalSpace::gpuSysObj().dev(0);

        int* gpuMem;
        ASSERT_NO_THROW(gpuMem = gpuDev.memory.malloc<int>(domainGrid.rMul()));

        Neon::sys::GpuStream gpuStream;
        ASSERT_NO_THROW(gpuStream = gpuDev.tools.stream());

        Neon::sys::GpuLaunchInfo kernelInfo(Neon::sys::GpuLaunchInfo::mode_e::domainGridMode, domainGrid, cudaBlock, sharedMem);
        int                      val = 33;

        auto tuple = std::make_tuple(gpuStream, kernelInfo, kernelRun::kernel, val, gpuMem);


        void* params[2] = {&val, &gpuMem};
        ASSERT_NO_THROW(gpuDev.kernel.cudaLaunchKernel<run_et::sync>(gpuStream, {Neon::sys::GpuLaunchInfo::domainGridMode, domainGrid, cudaBlock, sharedMem}, (void*)kernelRun::kernel, params));

        {
            Vec_3d<int> domainGrid(33, 1, 1);
            int32_3d    cudaBlock(5024, 1, 1);
            size_t      sharedMem = 0;

            ASSERT_ANY_THROW(gpuDev.kernel.cudaLaunchKernel<run_et::sync>(gpuStream, {Neon::sys::GpuLaunchInfo::domainGridMode, domainGrid, cudaBlock, sharedMem}, (void*)kernelRun::kernel, params));
            NEON_INFO("UnitTest: An exception was expected before this message, don't panic everything is fine so far...");
        }
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
