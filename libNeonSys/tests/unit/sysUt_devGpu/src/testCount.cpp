
#include "gtest/gtest.h"

#include "Neon/Neon.h"

#include "Neon/core/core.h"
#include "Neon/sys/devices/gpu/GpuDevice.h"
#include "Neon/sys/devices/memType.h"


#include <cstring>
#include <iostream>

TEST(gpuDev, info)
{
    Neon::sys::GpuDevice gpuDev;
    auto                 res = gpuDev.info();
    NEON_INFO("GoogleTest::gpuDev {}", res);
}

TEST(gpuDev, transferWithError)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        Neon::sys::GpuDevice gpuDev(0);
        Neon::sys::GpuStream gpuStream = gpuDev.tools.stream();
        ASSERT_ANY_THROW((gpuDev.memory.transfer<double, Neon::sys::mem_et::cpu, Neon::sys::mem_et::cpu, Neon::run_et::sync>(gpuStream, nullptr, nullptr, 33)));
    }
}

namespace transferData {
template <typename T_ta>
void trasferTest(int nElements)
{
    Neon::sys::GpuDevice gpuDev(0);


    T_ta* cpuMemA = new T_ta[nElements];
    T_ta* gpuMemB = gpuDev.memory.malloc<T_ta>(nElements);
    T_ta* gpuMemC = gpuDev.memory.malloc<T_ta>(nElements);

    auto val = [](int value) -> int { return (value * 17 + 13) % 11; };

    for (int i = 0; i < nElements; i++) {
        cpuMemA[i] = val(i);
    }
    Neon::sys::GpuStream gpuStream = gpuDev.tools.stream();

    gpuDev.memory.transfer<T_ta, Neon::sys::mem_et::gpu, Neon::sys::mem_et::cpu, Neon::run_et::sync>(gpuStream, gpuMemB, cpuMemA, nElements);

    for (int i = 0; i < nElements; i++) {
        cpuMemA[i] = 0;
    }

    gpuDev.memory.transfer<T_ta, Neon::sys::mem_et::gpu, Neon::sys::mem_et::gpu, Neon::run_et::sync>(gpuStream, gpuMemC, gpuMemB, nElements);
    // Transferring back only half of the vector
    gpuDev.memory.transfer<T_ta, Neon::sys::mem_et::cpu, Neon::sys::mem_et::gpu, Neon::run_et::sync>(gpuStream, cpuMemA, gpuMemC, nElements / 2);

    // Checking the first half of the vector
    for (int i = 0; i < nElements / 2; i++) {
        ASSERT_TRUE(cpuMemA[i] == val(i));
    }

    // Checking the second half of the vector
    for (int i = nElements / 2; i < nElements; i++) {
        ASSERT_TRUE(cpuMemA[i] == 0);
    }

    gpuDev.memory.free(gpuMemC);
    gpuDev.memory.free(gpuMemB);
    delete[] cpuMemA;
}
};  // namespace transferData

TEST(gpuDev, transferData)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        transferData::trasferTest<int>(100);
        transferData::trasferTest<int>(1000);
    }
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
