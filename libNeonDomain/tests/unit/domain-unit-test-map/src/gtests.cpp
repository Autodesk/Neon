
#include "Neon/Neon.h"
#include "gtest/gtest.h"
#include "map.h"
#include "runHelper.h"

TEST(domain, map)
{
    int nGpus = 3;
    using Type = int64_t;
    runAllTestConfiguration(std::function(map::run<Neon::domain::eGrid, Type, 0>),
                            nGpus,
                            1);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
