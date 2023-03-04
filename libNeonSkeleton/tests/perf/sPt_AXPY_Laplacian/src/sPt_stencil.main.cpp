#include "gtest/gtest.h"

#include "Neon/Neon.h"

#include "Neon/core/tools/clipp.h"
#include "sPt_common.h"
#include "sPt_stencil.h"

#include <map>

TestConfigurations testConfigurations;

TEST(Skeleton, DISABLED_AXPY_then_Laplacian)
{
    ASSERT_TRUE(testConfigurations.m_dim > Neon::index64_3d(0, 0, 0)) << testConfigurations.m_dim;
    ASSERT_TRUE(testConfigurations.m_nIterations > 0);
    ASSERT_TRUE(testConfigurations.m_nRepetitions > 0);
    ASSERT_TRUE(testConfigurations.m_nGPUs > 0);

    if (testConfigurations.m_dataType == DataType::DOUBLE_TYPE) {
        runAllConfig(filterAverage<Neon::domain::details::eGrid::eGrid, double>, testConfigurations);
    }

    if (testConfigurations.m_dataType == DataType::INT64_TYPE) {
        runAllConfig(filterAverage<Neon::domain::details::eGrid::eGrid, int64_t>, testConfigurations);
    }
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    // Occ::standard, Occ::extended, Occ::twoWayExtended, Occ::none
    std::string occ = std::string("none");
    std::string dataType("DOUBLE");
    std::string transfer("PUT");

    auto cli = (clipp::option("-dimx") & clipp::opt_values("dimx", testConfigurations.m_dim.x),
                clipp::option("-dimy") & clipp::opt_values("dimy", testConfigurations.m_dim.y),
                clipp::option("-dimz") & clipp::opt_values("dimz", testConfigurations.m_dim.z),
                clipp::option("-iterations") & clipp::opt_values("Number of iterations", testConfigurations.m_nIterations),
                clipp::option("-repetitions") & clipp::opt_values("Number of repetitions", testConfigurations.m_nRepetitions),
                clipp::option("-gpus") & clipp::opt_values("Number of gpus", testConfigurations.m_nGPUs),
                clipp::option("-occ") & clipp::opt_values("data_type: none, standard, extended, twoWayExtended", occ),
                clipp::option("-transfer") & clipp::opt_values("data_type: PUT, GET", transfer),
                clipp::option("-data_type") & clipp::opt_values("data_type: double or int64", dataType),
                clipp::option("-correctness").set(testConfigurations.m_compare),
                clipp::option("-o") & clipp::opt_values("o", testConfigurations.m_fnamePrefix));


    if (!clipp::parse(argc, argv, cli)) {
        auto fmt = clipp::doc_formatting{}.doc_column(31);
        std::cout << "Invalid input arguments!\n";
        std::cout << make_man_page(cli, argv[0], fmt) << '\n';
        exit(EXIT_FAILURE);
    }

    testConfigurations.m_optSkelOCC = Neon::skeleton::OccUtils::fromString(occ);
    testConfigurations.m_optSkelTransfer = Neon::set::TransferModeUtils::fromString(transfer);
    testConfigurations.m_dataType = DataTypeStr2Val(dataType);

    NEON_INFO(testConfigurations.toString());
    return RUN_ALL_TESTS();
}
