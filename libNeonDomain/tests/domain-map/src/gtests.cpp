
#include "Neon/Neon.h"
#include "gtest/gtest.h"
#include "map.h"
#include "runHelper.h"

TEST(domain_map, dGrid)
{
    int nGpus = 3;
    using Type = int64_t;
    runAllTestConfiguration(std::function(map::run<Neon::dGrid, Type, 0>),
                            nGpus,
                            1);
}

TEST(domain_map, eGrid)
{
    int nGpus = 3;
    using Type = int64_t;
    runAllTestConfiguration(std::function(map::run<Neon::eGrid, Type, 0>),
                            nGpus,
                            1);
}

TEST(domain_map, bGrid)
{
    int nGpus = 1;
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
