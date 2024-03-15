
#include "Neon/Neon.h"
#include "Neon/domain/Grids.h"

#include "gtest/gtest.h"
#include "map.h"
#include "runHelper.h"

TEST(domain_map_disg, bGridDisg)
{
    int nGpus = 1;
    using Type = int64_t;
    runAllTestConfiguration(std::function(map::run<Neon::bGridDisg, Type, 0>),
                            nGpus,
                            1);
}

TEST(domain_map_disg, bGridMask)
{
    int nGpus = 1;
    using Type = int64_t;
    runAllTestConfiguration(std::function(map::run<Neon::bGridMask, Type, 0>),
                            nGpus,
                            1);
}

TEST(domain_map_disg_dataView, bGridDisg)
{
    int nGpus = 1;
    using Type = int64_t;
    runAllTestConfiguration(std::function(map::dataView::run<Neon::bGridDisg, Type, 0>),
                            nGpus,
                            1);
}

TEST(domain_map_disg_dataView, bGridMask)
{
    int nGpus = 1;
    using Type = int64_t;
    runAllTestConfiguration(std::function(map::dataView::run<Neon::bGridMask, Type, 0>),
                            nGpus,
                            1);
}
//
//TEST(domain_map_dataView, bGrid)
//{
//    int nGpus = 3;
//    using Type = int64_t;
//    runAllTestConfiguration(std::function(map::dataView::run<Neon::bGrid, Type, 0>),
//                            nGpus,
//                            2);
//}
//
//TEST(domain_map_dataView, dGrid)
//{
//    int nGpus = 3;
//    using Type = int64_t;
//    runAllTestConfiguration(std::function(map::dataView::run<Neon::dGrid, Type, 0>),
//                            nGpus,
//                            2);
//}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
