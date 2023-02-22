
#include <cstring>
#include <iostream>

#include "Neon/Neon.h"

#include "Neon/core/tools/metaprogramming.h"
#include "Neon/set/DevSet.h"
#include "Neon/set/memory/memSet.h"
#include "gtest/gtest.h"

#define TEST_M_VAL 5

template <typename T_ta>
struct testDataRedundancy_t
{
    T_ta* mem;
    T_ta  val;
    int   nEl;
};

namespace TestKernels {


template <typename T_ta>
__global__ void resetWithExtraParam(T_ta* mem, const T_ta setVal, const int nEl, testDataRedundancy_t<T_ta> redundancy, int m)
{
    int myIdx = threadIdx.x + blockIdx.x * blockDim.x;

    if (myIdx < nEl) {
        if (mem != redundancy.mem || setVal != redundancy.val || nEl != redundancy.nEl || m != TEST_M_VAL) {
            mem[myIdx] = -setVal - 1;
        } else {
            mem[myIdx] = setVal;
        }
    }
}

template <typename T_ta>
__global__ void reset(T_ta* mem, const T_ta setVal, const int nEl, testDataRedundancy_t<T_ta> redundancy)
{
    int myIdx = threadIdx.x + blockIdx.x * blockDim.x;

    if (myIdx < nEl) {
        if (mem != redundancy.mem || setVal != redundancy.val || nEl != redundancy.nEl) {
            mem[myIdx] = -setVal - 1;
        } else {
            mem[myIdx] = setVal;
        }
    }
}

template <typename T_ta>
__global__ void add(T_ta* mem, const T_ta add, const int nEl, testDataRedundancy_t<T_ta> redundancy)
{
    int myIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (myIdx < nEl) {
        if (mem != redundancy.mem || add != redundancy.val || nEl != redundancy.nEl) {
            mem[myIdx] += (myIdx + add);
            mem[myIdx] *= -1;
        } else {
            mem[myIdx] += (myIdx + add);
        }
    }
}

template <typename T_ta>
__global__ void add2(T_ta* mem, const T_ta add, const int nEl, testDataRedundancy_t<T_ta> redundancy)
{
    int myIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (myIdx < nEl) {
        if (mem != redundancy.mem || add != redundancy.val || nEl != redundancy.nEl) {
            mem[myIdx] += (myIdx + add);
            mem[myIdx] *= -1;
        } else {
            mem[myIdx] += (myIdx + add);
        }
    }
}

template <typename T_ta>
__global__ void addWithExtraParam(T_ta* mem, const T_ta add, const int nEl, testDataRedundancy_t<T_ta> redundancy, int m)
{
    int myIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (myIdx < nEl) {
        if (mem != redundancy.mem || add != redundancy.val || nEl != redundancy.nEl || m != TEST_M_VAL) {
            mem[myIdx] += (myIdx + add);
            mem[myIdx] *= -1;
        } else {
            mem[myIdx] += (myIdx + add);
        }
    }
}

}  // namespace TestKernels


using namespace Neon;
class cudaLaunchKernel_test
{
   public:
    Neon::set::DevSet          m_devSet;
    Neon::set::StreamSet       m_gpuStreamSet;
    std::vector<int32_3d>      m_domainGridVec;
    Neon::set::LaunchParameters   m_kernelInfoSet;
    Neon::set::MemSet<int>   m_mirror;

    Neon::set::DataSet<testDataRedundancy_t<int>> m_testDataRedundancyVec;

    int valToBeAdded(int gpuId)
    {
        return 33 * (gpuId + 1);
    }

    void init()
    {
        using namespace Neon;

        // Allocating a gpu set
        m_devSet = Neon::set::DevSet::maxSet();
        {
            // If there is only one GPU we oversubscribe the same GPU
            if (m_devSet.setCardinality() == 1) {
                Neon::SetIdx gpuId = 0;
                m_devSet = Neon::set::DevSet(Neon::DeviceType::CUDA, {gpuId, gpuId, gpuId});
            }
        }
        m_gpuStreamSet = m_devSet.newStreamSet();
        m_testDataRedundancyVec = std::vector<testDataRedundancy_t<int>>(m_devSet.setCardinality());


        // Defining the vector size for each GPU
        m_domainGridVec = std::vector<int32_3d>(m_devSet.setCardinality());
        for (int i = 0; i < m_devSet.setCardinality(); i++) {
            m_domainGridVec[i].set(1000 * (i + 1), 1, 1);
        }

        // Kernel Info
        m_kernelInfoSet = m_devSet.newLaunchParameters();
        for (int i = 0; i < m_devSet.setCardinality(); i++) {
            int32_3d cudaBlock(1024, 1, 1);
            size_t   sharedMem = 0;
            m_kernelInfoSet[i] = Neon::sys::GpuLaunchInfo(Neon::sys::GpuLaunchInfo::domainGridMode, m_domainGridVec[i], cudaBlock, sharedMem);
        }

        // Creating some memory
        {
            std::vector<uint64_t> eachGpuMemSize(m_devSet.setCardinality());
            for (int i = 0; i < m_devSet.setCardinality(); i++) {
                eachGpuMemSize[i] = m_domainGridVec[i].rMulTyped<size_t>();
            }
            m_mirror = m_devSet.newMemSet<int>(int(1), {}, {}, eachGpuMemSize);
        }

        // Set
        for (int i = 0; i < m_devSet.setCardinality(); i++) {
            int* cpuRawMem = (int*)m_mirror.rawMem(1, Neon::DeviceType::CPU);
            for (int j = 0; j < m_domainGridVec[i].rMulTyped<size_t>(); j++) {
                cpuRawMem[j] = j;
            }
        }

        // Transfer
        m_mirror.update<Neon::run_et::async>(m_devSet.defaultStreamSet(), Neon::DeviceType::CPU);


        for (int i = 0; i < m_devSet.setCardinality(); i++) {
            m_testDataRedundancyVec[i].nEl = m_domainGridVec[i].rMulTyped<int>();
            m_testDataRedundancyVec[i].val = valToBeAdded(i);
            m_testDataRedundancyVec[i].mem = (int*)m_mirror.rawMem(1, Neon::DeviceType::CUDA);
        }
    }

    void fini()
    {
        // Transfer back
        m_mirror.update<Neon::run_et::async>(m_gpuStreamSet, Neon::DeviceType::CPU);

        // Wait for all the operation on the stream to be completed.
        m_gpuStreamSet.sync();

        // Check results
        for (int i = 0; i < m_devSet.setCardinality(); i++) {
            for (int j = 0; j < m_domainGridVec[i].rMulTyped<size_t>(); j++) {
                int* cpuRawMem = (int*)m_mirror.rawMem(i, Neon::DeviceType::CPU);
                ASSERT_EQ(cpuRawMem[j], j + j + valToBeAdded(i)) << cpuRawMem[j] + 5 << " .. " << i;
            }
        }
    }
};

///**
// * Run kernel with parameters
// */
//TEST(sysUt_gpuSetNvcc, run_i)
//{
//    cudaLaunchKernel_test test;
//    test.init();
//
//    for (int i = 0; i < test.m_devSet.setCardinality(); i++) {
//        // Run Kernels
//        int  nEl = test.m_domainGridVec[i].rMul();
//        int* gpuRawMem = (int*)test.m_mirror.devRawMem(i);
//        int  valToBeAdded = test.valToBeAdded(i);
//
//        ASSERT_NO_THROW(test.m_devSet.kernel<run_et::async>(i,
//                                                          test.m_gpuStreamSet,
//                                                          test.m_kernelInfoSet,
//                                                          &TestKernels::add<int>,
//                                                          gpuRawMem,
//                                                          valToBeAdded,
//                                                          nEl,
//                                                          test.m_testDataRedundancyVec[i]));
//    }
//
//    test.fini();
//}

///**
// * Run kernel with parameters
// */
//TEST(sysUt_gpuSetNvcc, run)
//{
//    cudaLaunchKernel_test test;
//    test.init();
//
//    Neon::set::DataSet<int*> gpuRawMemVec(test.m_devSet.setCardinality());
//    Neon::set::DataSet<int>  valToBeAddedVec(test.m_devSet.setCardinality());
//    Neon::set::DataSet<int>  nElVec(test.m_devSet.setCardinality());
//
//    for (int i = 0; i < test.m_devSet.setCardinality(); i++) {
//        int  nEl = test.m_domainGridVec[i].rMul();
//        int* gpuRawMem = (int*)test.m_mirror.devRawMem(i);
//        int  valToBeAdded = test.valToBeAdded(i);
//
//        gpuRawMemVec[i] = gpuRawMem;
//        valToBeAddedVec[i] = valToBeAdded;
//        nElVec[i] = nEl;
//    }
//    auto fun = &TestKernels::add<int>;
//    ASSERT_NO_THROW(test.m_devSet.kRun<run_et::async>(test.m_gpuStreamSet,
//                                                      test.m_kernelInfoSet,
//                                                      fun,
//                                                      gpuRawMemVec,
//                                                      valToBeAddedVec,
//                                                      nElVec,
//                                                      test.m_testDataRedundancyVec));
//
//    test.fini();
//}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
