
#include "Neon/Neon.h"
#include "gtest/gtest.h"
#include "runHelper.h"
#include "testSignature.h"

TEST(domain, hostContainer)
{
    int nGpus = 3;
    using Type = int64_t;
    runAllTestConfiguration(std::function(host::runHost<Neon::domain::dGrid, Type, 0>),
                            nGpus,
                            1);
}

TEST(domain, deviceContainer)
{
    int nGpus = 3;
    using Type = int64_t;
    runAllTestConfiguration(std::function(device::runDevice<Neon::domain::dGrid, Type, 0>),
                            nGpus,
                            1);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}