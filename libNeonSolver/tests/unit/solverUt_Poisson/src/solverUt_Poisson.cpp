
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>

#include "Neon/Neon.h"

#include "Neon/core/core.h"
#include "Neon/core/tools/clipp.h"
#include "Neon/set/DevSet.h"
#include "gtest/gtest.h"
#include "Poisson.h"

// Unit test params
constexpr int    DOMAIN_SIZE = 100;
constexpr size_t MAX_ITERATIONS = 1000;
constexpr double TOLERANCE = 1e-10;

using namespace Neon::set;
using namespace Neon::solver;
using namespace Neon::domain;

// Gets two GPU ids based on availability. If there's only one GPU,
// then it is oversubscribed.
const std::vector<int> getDevices()
{
    std::vector<int>    gpuIds{0};
    Neon::set::DevSet   maxDeviceSet = Neon::set::DevSet::maxSet();
    if (maxDeviceSet.setCardinality() >= 2) {
        gpuIds.push_back(1);
    } else {
        gpuIds.push_back(0);
    }
    return gpuIds;
}

TEST(PoissonTest, DISABLED_CG_Scalar_eGrid_GPU)
{
    Neon::Backend         backend = Neon::Backend(getDevices(), Neon::Runtime::stream);
    std::array<double, 1> bdZMin{-20.0};
    std::array<double, 1> bdZMax{20.0};

    Neon::skeleton::Occ occE;
    Neon::set::TransferMode transferE;

    {
        occE = Neon::skeleton::Occ::none;
        transferE = Neon::set::TransferMode::get;
        auto [result, status] = testPoissonContainers<eGrid, double, 1>(backend, "CG", DOMAIN_SIZE, bdZMin, bdZMax, MAX_ITERATIONS, TOLERANCE, occE, transferE);
        ASSERT_TRUE(status != SolverStatus::Error);
        ASSERT_TRUE(result.residualEnd <= TOLERANCE);
    }
    {
        occE = Neon::skeleton::Occ::standard;
        transferE = Neon::set::TransferMode::get;
        auto [result, status] = testPoissonContainers<eGrid, double, 1>(backend, "CG", DOMAIN_SIZE, bdZMin, bdZMax, MAX_ITERATIONS, TOLERANCE, occE, transferE);
        ASSERT_TRUE(status != SolverStatus::Error);
        ASSERT_TRUE(result.residualEnd <= TOLERANCE);
    }
    {
        occE = Neon::skeleton::Occ::standard;
        transferE = Neon::set::TransferMode::put;
        auto [result, status] = testPoissonContainers<eGrid, double, 1>(backend, "CG", DOMAIN_SIZE, bdZMin, bdZMax, MAX_ITERATIONS, TOLERANCE, occE, transferE);
        ASSERT_TRUE(status != SolverStatus::Error);
        ASSERT_TRUE(result.residualEnd <= TOLERANCE);
    }
}

TEST(PoissonTest, DISABLED_CG_Vector_eGrid_GPU)
{
    Neon::Backend         backend = Neon::Backend(getDevices(), Neon::Runtime::stream);
    std::array<double, 3> bdZMin{0, -20.0, 0};
    std::array<double, 3> bdZMax{0, 0, 20.0};

    Neon::skeleton::Occ occE;
    Neon::set::TransferMode transferE;
    occE = Neon::skeleton::Occ::standard;
    transferE = Neon::set::TransferMode::get;

    auto [result, status] = testPoissonContainers<eGrid, double, 3>(backend, "CG", DOMAIN_SIZE, bdZMin, bdZMax, MAX_ITERATIONS, TOLERANCE, occE, transferE);
    ASSERT_TRUE(status != SolverStatus::Error);
    ASSERT_TRUE(result.residualEnd <= TOLERANCE);
}


TEST(PoissonTest, DISABLED_CG_Vector_dGrid_GPU)
{
    Neon::Backend         backend = Neon::Backend(getDevices(), Neon::Runtime::stream);
    std::array<double, 3> bdZMin{0, -20.0, 0};
    std::array<double, 3> bdZMax{0, 0, 20.0};

    Neon::skeleton::Occ occE;
    Neon::set::TransferMode transferE;
    occE = Neon::skeleton::Occ::standard;
    transferE = Neon::set::TransferMode::get;

    auto [result, status] = testPoissonContainers<dGrid, double, 3>(backend, "CG", DOMAIN_SIZE, bdZMin, bdZMax, MAX_ITERATIONS, TOLERANCE, occE, transferE);
    ASSERT_TRUE(status != SolverStatus::Error);
    ASSERT_TRUE(result.residualEnd <= TOLERANCE);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
