
#include "Neon/Neon.h"
#include "cuda_fp16.h"
#include "gtest/gtest.h"
#include "map.h"
#include "runHelper.h"

TEST(domain, map)
{
    int nGpus = 3;
    runAllTestConfiguration(std::function(map::run<Neon::domain::eGrid, int64_t, 0, double>),
                            nGpus,
                            1);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
