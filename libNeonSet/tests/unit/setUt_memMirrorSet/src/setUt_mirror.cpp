
#include <cstring>
#include <iostream>

#include "Neon/Neon.h"

#include "Neon/core/core.h"
#include "Neon/set/DevSet.h"
#include "gtest/gtest.h"

#include "Neon/set/memory/memory.h"

namespace tools {
using namespace Neon::sys;
using namespace Neon;

template <typename T_ta>
void setVal_Phase_1(SetIdx setIdx, T_ta* addr, size_t nElements)
{
    for (size_t i = 0; i < nElements; i++) {
        addr[i] = T_ta(setIdx.idx() + 2 * i + 17);
    }
}

template <typename T_ta>
bool checkVal_Phase_1(SetIdx setIdx, T_ta* addr, size_t nElements)
{
    for (size_t i = 0; i < nElements; i++) {
        auto lVal = addr[i];
        auto rVal = T_ta(setIdx.idx() + 2 * i + 17);
        if (std::abs(lVal - rVal) > 0.000001) {
            return false;
        }
    }
    return true;
};


template <typename T_ta>
void setVal_Phase_2(SetIdx setIdx, T_ta* addr, size_t nElements)
{
    for (size_t i = 0; i < nElements; i++) {
        addr[i] = T_ta(setIdx.idx() + 7 * i + 33);
    }
}

template <typename T_ta>
bool checkVal_Phase_2(SetIdx setIdx, T_ta* addr, size_t nElements)
{
    for (size_t i = 0; i < nElements; i++) {
        if (addr[i] != T_ta(setIdx.idx() + 7 * i + 33)) {
            return false;
        }
    }
    return true;
};

}  // namespace tools

template <typename T_ta>
void mirrorTest(int nElements)
{
    using namespace Neon::sys;
    using namespace Neon::set;

    using namespace Neon;

    //Test on just two gpus
    Neon::set::DevSet devSet = DevSet::maxSet(Neon::DeviceType::CUDA);
    if (devSet.setCardinality() > 2) {
        devSet = Neon::set::DevSet(Neon::DeviceType::CUDA, {0, 1});
    }

    Neon::set::DataSet<size_t> mirrorSize(devSet.setCardinality(), nElements);
    auto                       mirrorSetA = devSet.template newMemSet<T_ta>(1, Neon::DeviceType::CPU, Neon::DeviceType::CUDA, mirrorSize);
    auto                       mirrorSetB = devSet.template newMemSet<T_ta>(1, Neon::DeviceType::CPU, {Neon::DeviceType::CUDA, Neon::Allocator::NULL_MEM}, mirrorSize);

    for (int i = 0; i < devSet.setCardinality(); i++) {
        Neon::sys::MemMirror cpuMem = mirrorSetA.get(i);
        T_ta*                rawCpu = cpuMem.rawMem(Neon::DeviceType::CPU);
        ::tools::setVal_Phase_1(i, rawCpu, nElements);
    }

    Neon::set::StreamSet stream = devSet.newStreamSet();
    mirrorSetA.template update<Neon::run_et::sync>(stream, Neon::DeviceType::CUDA);

    Neon::Backend bk(devSet, Neon::Runtime::stream);
    Neon::set::Memory::MemSet<int>(bk, 1, 4, Neon::DataUse::HOST_DEVICE);
}


TEST(symmetric, global)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        mirrorTest<int>(100);
        mirrorTest<int>(1000);
        mirrorTest<double>(100);
        mirrorTest<double>(1000);
    }
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
