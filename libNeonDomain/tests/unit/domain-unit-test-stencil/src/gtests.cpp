
#include "Neon/Neon.h"
#include "gtest/gtest.h"
#include "runHelper.h"
#include "stencil.h"

TEST(domain_unit_test_stencil, dGrid)
{
    int nGpus = 3;
    using Type = int64_t;
    runAllTestConfiguration(std::function(map::run<Neon::domain::details::dGrid::dGrid, Type, 0>),
                            nGpus,
                            1);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
