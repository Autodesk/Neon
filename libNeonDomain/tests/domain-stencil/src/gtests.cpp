
#include "Neon/Neon.h"
#include "gtest/gtest.h"
#include "runHelper.h"
#include "stencil.h"

TEST(domain_stencil, dGrid)
{
    int nGpus = 3;
    using Type = int64_t;
    runAllTestConfiguration(std::function(map::run<Neon::dGrid, Type, 0>),
                            nGpus,
                            1);
}

TEST(domain_stencil, eGrid)
{
    int nGpus = 3;
    using Type = int64_t;
    runAllTestConfiguration(std::function(map::run<Neon::eGrid, Type, 0>),
                            nGpus,
                            1);
}

TEST(domain_stencil, bGri )
{
    int nGpus = 5;
    using Type = int64_t;
    runAllTestConfiguration(std::function(map::run<Neon::bGrid, Type, 0>),
                            nGpus,
                            1);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
