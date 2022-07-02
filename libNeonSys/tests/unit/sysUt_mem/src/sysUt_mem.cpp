
#include "gtest/gtest.h"

#include "Neon/Neon.h"

#include "Neon/core/types/vec.h"

#include "Neon/sys/global/CpuSysGlobal.h"
#include "Neon/sys/global/GpuSysGlobal.h"

#include "Neon/sys/memory/MemDevice.h"

#include <cstring>
#include <iostream>

namespace global {
size_t allocationSize = 1024 * sizeof(int);
size_t allocationMultiplier = 10;
}  // namespace global

TEST(Allocator, CUDA_DEVICE)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        using namespace Neon;
        Neon::sys::DeviceID targetDev = 0;

        size_t allocationSize = global::allocationSize;
        size_t allocationMultiplier = global::allocationMultiplier;
        size_t iterations = 0;

        bool done = false;

        while (!done) {
            try {
                size_t                     newSize = allocationSize * allocationMultiplier;
                Neon::sys::MemDevice<char> buffer(Neon::DeviceType::CUDA, targetDev, Neon::Allocator::CUDA_MEM_DEVICE, newSize);
                size_t                     inUse = Neon::sys::globalSpace::gpuSysObj().allocator(targetDev).inUse();
                ASSERT_TRUE(inUse == newSize) << "Recorded used Memory " << inUse << ", it should be " << newSize;

            } catch (...) {
                done = true;
                break;
            }
            iterations++;
            if (allocationSize > 1024 * 1024) {
                break;
            }
            allocationMultiplier *= allocationMultiplier;
        }

        allocationMultiplier = global::allocationMultiplier;

        ASSERT_TRUE(Neon::sys::globalSpace::gpuSysObj().allocator(targetDev).inUse() == 0);

        for (size_t i = 0; i < iterations; i++) {

            size_t                     newSize = allocationSize * allocationMultiplier;
            Neon::sys::MemDevice<char> buffer(Neon::DeviceType::CUDA, targetDev, Neon::Allocator::CUDA_MEM_DEVICE, newSize);

            size_t inUse = Neon::sys::globalSpace::gpuSysObj().allocator(targetDev).inUse();
            ASSERT_TRUE(inUse == newSize) << "Recorded used Memory " << inUse << ", it should be " << newSize;
            allocationMultiplier *= allocationMultiplier;
        }

        ASSERT_TRUE(Neon::sys::globalSpace::gpuSysObj().allocator(targetDev).inUse() == 0);
    }
}

TEST(Allocator, MALLOC)
{
    using namespace Neon;
    size_t allocationSize = global::allocationSize;
    size_t allocationMultiplier = global::allocationMultiplier;
    size_t iterations = 0;


    bool done = false;

    while (!done) {
        try {
            size_t                     newSize = allocationSize * allocationMultiplier;
            Neon::sys::MemDevice<char> buffer(Neon::DeviceType::CPU, 0, Neon::Allocator::MALLOC, newSize);
            size_t                     inUse = Neon::sys::globalSpace::cpuSysObj().allocator().inUsedMemPageable();
            ASSERT_TRUE(inUse == newSize);
        } catch (...) {
            done = true;
            break;
        }
        iterations++;
        if (allocationSize > 1024 * 1024) {
            break;
        }
        allocationMultiplier *= allocationMultiplier;
    }

    allocationMultiplier = global::allocationMultiplier;


    for (size_t i = 0; i < iterations; i++) {
        size_t newSize = allocationSize * allocationMultiplier;

        Neon::sys::MemDevice<char> buffer(
            Neon::DeviceType::CPU, 0, Neon::Allocator::MALLOC, newSize);
        ASSERT_TRUE(Neon::sys::globalSpace::cpuSysObj().allocator().inUsedMemPageable() == newSize);
        ASSERT_TRUE(Neon::sys::globalSpace::cpuSysObj().allocator().inUsedMemPinned() == 0);
        allocationMultiplier *= allocationMultiplier;
    }
}

TEST(Allocator, CUDA_UNIFIED)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        using namespace Neon;
        Neon::sys::DeviceID targetDev = 0;


        size_t allocationSize = global::allocationSize;
        size_t allocationMultiplier = global::allocationMultiplier;
        size_t iterations = 0;


        bool done = false;

        while (!done) {
            try {
                size_t                     newSize = allocationSize * allocationMultiplier;
                Neon::sys::MemDevice<char> buffer(Neon::DeviceType::CPU, targetDev, Neon::Allocator::CUDA_MEM_UNIFIED, newSize);
                size_t                     inUse = Neon::sys::globalSpace::cpuSysObj().allocator().inUsedMemPinned();
                ASSERT_TRUE(inUse == newSize);

            } catch (...) {
                done = true;
                break;
            }
            break;
        }


        allocationMultiplier = global::allocationMultiplier;


        for (size_t i = 0; i < iterations; i++) {
            size_t newSize = allocationSize * allocationMultiplier;

            Neon::sys::MemDevice<char> buffer(Neon::DeviceType::CPU, 0, Neon::Allocator::CUDA_MEM_UNIFIED, newSize);
            size_t                     inUsePinned = Neon::sys::globalSpace::cpuSysObj().allocator().inUsedMemPinned();
            size_t                     inUsePagable = Neon::sys::globalSpace::cpuSysObj().allocator().inUsedMemPageable();
            ASSERT_TRUE(inUsePinned == newSize) << "Recorded used Pinned Memory " << inUsePinned << ", it should be " << newSize;
            ASSERT_TRUE(inUsePagable == 0) << "Recorded used Pageable Memory " << inUsePagable << ", it should be " << newSize;
            allocationMultiplier *= allocationMultiplier;
        }
    }
}

TEST(Allocator, CUDA_HOST)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        using namespace Neon;
        Neon::sys::DeviceID targetDev = 0;


        size_t allocationSize = global::allocationSize;
        size_t allocationMultiplier = global::allocationMultiplier;
        size_t iterations = 0;


        bool done = false;

        while (!done) {
            try {
                size_t                     newSize = allocationSize * allocationMultiplier;
                Neon::sys::MemDevice<char> buffer(Neon::DeviceType::CPU, targetDev, Neon::Allocator::CUDA_MEM_HOST, newSize);
                size_t                     inUse = Neon::sys::globalSpace::cpuSysObj().allocator().inUsedMemPinned();
                ASSERT_TRUE(inUse == newSize);
            } catch (...) {
                done = true;
                break;
            }
            break;
        }


        allocationMultiplier = global::allocationMultiplier;


        for (size_t i = 0; i < iterations; i++) {
            size_t newSize = allocationSize * allocationMultiplier;

            Neon::sys::MemDevice<char> buffer(Neon::DeviceType::CPU, 0, Neon::Allocator::CUDA_MEM_HOST, newSize);
            size_t                     inUsePinned = Neon::sys::globalSpace::cpuSysObj().allocator().inUsedMemPinned();
            size_t                     inUsePagable = Neon::sys::globalSpace::cpuSysObj().allocator().inUsedMemPageable();
            ASSERT_TRUE(inUsePinned == newSize) << "Recorded used Pinned Memory " << inUsePinned << ", it should be " << newSize;
            ;
            ASSERT_TRUE(inUsePagable == 0) << "Recorded used Pageable Memory " << inUsePagable << ", it should be " << newSize;
            ;
            allocationMultiplier *= allocationMultiplier;
        }
    }
}
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
