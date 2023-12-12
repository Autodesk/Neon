
#include "Neon/Neon.h"
#include "gtest/gtest.h"
#include "globalIdx.h"
#include "runHelper.h"
#include "Neon/domain/details/dGridDisg/dGrid.h"

TEST(domain_globalIdx, dGrid)
{
    int nGpus = 3;
    using Type = int64_t;
    runAllTestConfiguration(std::function(globalIdx::run<Neon::dGrid, Type, 0>),
                            nGpus,
                            1);
}

TEST(domain_globalIdx, eGrid)
{
    int nGpus = 3;
    using Type = int64_t;
    runAllTestConfiguration(std::function(globalIdx::run<Neon::eGrid, Type, 0>),
                            nGpus,
                            1);
}

TEST(domain_globalIdx, bGrid)
{
    int nGpus = 3;
    using Type = int64_t;
    runAllTestConfiguration(std::function(globalIdx::run<Neon::bGrid, Type, 0>),
                            nGpus,
                            1);
}

TEST(domain_globalIdx, dGridDisg)
{
    int nGpus = 3;
    using Type = int64_t;
    runAllTestConfiguration(std::function(globalIdx::run<Neon::domain::details::disaggregated::dGrid::dGrid , Type, 0>),
                            nGpus,
                            1);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
