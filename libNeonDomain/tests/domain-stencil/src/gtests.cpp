
#include "Neon/Neon.h"
#include "gtest/gtest.h"
#include "runHelper.h"
#include "stencil.h"

TEST(domain_stencil, dGrid_NoTemplate)
{
    int nGpus = 3;
    using Type = int64_t;
    runAllTestConfiguration(std::function(map::runNoTemplate<Neon::dGrid, Type, 0>),
                            nGpus,
                            1);
}

TEST(domain_stencil, eGrid_NoTemplate)
{
    int nGpus = 3;
    using Type = int64_t;
    runAllTestConfiguration(std::function(map::runNoTemplate<Neon::eGrid, Type, 0>),
                            nGpus,
                            1);
}

TEST(domain_stencil, bGri_NoTemplate)
{
    int nGpus = 5;
    using Type = int64_t;
    runAllTestConfiguration(std::function(map::runNoTemplate<Neon::bGrid, Type, 0>),
                            nGpus,
                            1);
}

TEST(domain_stencil, dGridSoA_NoTemplate)
{
    int nGpus = 5;
    using Type = int64_t;
    runAllTestConfiguration(std::function(map::runNoTemplate<Neon::dGridSoA, Type, 0>),
                            nGpus,
                            1);
}

TEST(domain_stencil, dGrid_Template)
{
    int nGpus = 3;
    using Type = int64_t;
    runAllTestConfiguration(std::function(map::runTemplate<Neon::dGrid, Type, 0>),
                            nGpus,
                            1);
}

TEST(domain_stencil, eGrid_Template)
{
    int nGpus = 3;
    using Type = int64_t;
    runAllTestConfiguration(std::function(map::runTemplate<Neon::eGrid, Type, 0>),
                            nGpus,
                            1);
}

TEST(domain_stencil, bGri_Template)
{
    int nGpus = 5;
    using Type = int64_t;
    runAllTestConfiguration(std::function(map::runTemplate<Neon::bGrid, Type, 0>),
                            nGpus,
                            1);
}

TEST(domain_stencil, dGridSoA_Template)
{
    int nGpus = 5;
    using Type = int64_t;
    runAllTestConfiguration(std::function(map::runTemplate<Neon::dGridSoA, Type, 0>),
                            nGpus,
                            1);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
