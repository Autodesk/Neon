
#include "./testsAndContainers.h"
#include "Neon/Neon.h"
#include "gtest/gtest.h"
#include "runHelper.h"

TEST(domain_neighbour_globalIdx, dGrid)
{
    int nGpus = 5;
    using Type = int64_t;
    runAllTestConfiguration(std::function(globalIdx::run<Neon::dGrid, Type, 0>),
                            nGpus,
                            1);
}

TEST(domain_neighbour_globalIdx, eGrid)
{
    int nGpus = 5;
    using Type = int64_t;
    runAllTestConfiguration(std::function(globalIdx::run<Neon::eGrid, Type, 0>),
                            nGpus,
                            1);
}

TEST(domain_neighbour_globalIdx, bGrid)
{
    int nGpus = 5;
    using Type = int64_t;
    runAllTestConfiguration(std::function(globalIdx::run<Neon::bGrid, Type, 0>),
                            nGpus,
                            1);
}

TEST(domain_neighbour_globalIdx, dGridSoA)
{
    int nGpus = 5;
    using Type = int64_t;
    runAllTestConfiguration(std::function(globalIdx::run<Neon::dGridSoA, Type, 0>),
                            nGpus,
                            1);
}

///////////////////////////////////////////

TEST(domain_neighbour_globalIdx, dGrid_template)
{
    int nGpus = 5;
    using Type = int64_t;
    runAllTestConfiguration(std::function(globalIdx::runTemplate<Neon::dGrid, Type, 0>),
                            nGpus,
                            1);
}

TEST(domain_neighbour_globalIdx, eGrid_template)
{
    int nGpus = 5;
    using Type = int64_t;
    runAllTestConfiguration(std::function(globalIdx::runTemplate<Neon::eGrid, Type, 0>),
                            nGpus,
                            1);
}

TEST(domain_neighbour_globalIdx, bGrid_template)
{
    int nGpus = 5;
    using Type = int64_t;
    runAllTestConfiguration(std::function(globalIdx::runTemplate<Neon::bGrid, Type, 0>),
                            nGpus,
                            1);
}

TEST(domain_neighbour_globalIdx, dGridSoA_template)
{
    int nGpus = 5;
    using Type = int64_t;
    runAllTestConfiguration(std::function(globalIdx::runTemplate<Neon::dGridSoA, Type, 0>),
                            nGpus,
                            1);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
