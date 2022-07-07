
#include "gtest/gtest.h"

#include "Neon/Neon.h"

#include "Neon/core/tools/metaprogramming.h"
#include "Neon/set/DevSet.h"

#include <cstring>
#include <iostream>


TEST(gpuDev, maxSet)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        ASSERT_NO_THROW(Neon::set::DevSet devSet = Neon::set::DevSet::maxSet());
    }
}

TEST(gpuDev, multiRun)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        int nEl = 100;

        Neon::set::DevSet    gpuSet = Neon::set::DevSet::maxSet();
        Neon::set::MemDevSet memSet = gpuSet.newMemDevSet<char>(Neon::DeviceType::CUDA,
                                                                Neon::Allocator::CUDA_MEM_DEVICE, sizeof(int) * nEl);

        //Neon::sys::Mem_t
        // 1. Add MemDevSet memset = gpuSet.getMemSet
        // 1. Add mirror MemSet.
        //    Mirror set contains also the smartpointers...
    }
}

TEST(tupleTools, extractType)
{
    using namespace Neon;
    using namespace Neon::sys;

    using vectorTupleType_t = std::tuple<std::vector<int>, std::vector<double>>;
    using linearizedType_t = Neon::meta::TupleOfVecInnertType_t<vectorTupleType_t>;

    ASSERT_TRUE((std::is_same<linearizedType_t, std::tuple<int, double>>::value == true));
    ASSERT_FALSE((std::is_same<linearizedType_t, std::tuple<int, int>>::value == true));
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
