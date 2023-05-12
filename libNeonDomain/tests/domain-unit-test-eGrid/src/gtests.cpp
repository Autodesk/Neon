
#include "Neon/Neon.h"
#include "eGridTest.h"
#include "gtest/gtest.h"
#include "runHelper.h"


TEST(domain_unit_test_eGrid, eGrid)
{
    int nGpus = 3;
    using Type = int64_t;
    runAllTestConfiguration(std::function(eGridTesting::run<Neon::eGrid, Type, 0>),
                            nGpus,
                            1);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
