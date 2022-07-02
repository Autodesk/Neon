
#include "gtest/gtest.h"

#include "Neon/Neon.h"

#include "Neon/sys/devices/cpu/CpuDevice.h"
#include <cstring>
#include <iostream>

TEST(cpuDev, memory)
{
    Neon::sys::CpuDev   cpuDev;
    auto                res = cpuDev.info();
    NEON_INFO("GoogleTest::cpuDev {}", res);
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);    
    Neon::init();
    return RUN_ALL_TESTS();
}
