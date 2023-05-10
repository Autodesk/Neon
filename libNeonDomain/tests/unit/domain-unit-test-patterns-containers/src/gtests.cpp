
#include "Neon/Neon.h"
#include "containerRun.h"
#include "gtest/gtest.h"
#include "runHelper.h"

#include "Neon/domain/details/dGrid/dGrid.h"


TEST(domain_unit_test_patterns_containers, DISABLED_dGrid)
{
    int nGpus = 3;
    using Type = int64_t;

    runAllTestConfiguration(
        std::function(runContainer<Neon::domain::details::dGrid::dGrid, Type, 0>),
       {Neon::sys::patterns::Engine::CUB,
                                          Neon::sys::patterns::Engine::cuBlas} ,
                            nGpus,
                            1);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
