#include "Neon/domain/aGrid.h"
#include "Neon/domain/dGrid.h"
#include "Neon/domain/eGrid.h"

#include "Neon/skeleton/Skeleton.h"

#include "./containers.h"
#include "./runHelper.h"
#include "Neon/domain/tools/TestData.h"
#include "gtest/gtest.h"


template <typename G, typename T, int C>
auto oneStageXPYPipe(Neon::domain::tool::testing::TestData<G, T, C>& data) -> void
{
    data.resetValuesToLinear(1, 100);
    using namespace Neon::domain::tool::testing;

    auto& grid = data.getGrid();
    auto  bk = grid.getBackend();

    {  // Neon

        auto& X = data.getField(FieldNames::X);
        auto& Y = data.getField(FieldNames::Y);

        Neon::skeleton::Skeleton          skl(bk);
        Neon::skeleton::Options           opt;
        std::vector<Neon::set::Container> sVec;
        sVec.push_back(UserTools::xpy(X, Y));
        skl.sequence(sVec, "single-XPY", opt);

        for (int i = 0; i < 10; i++) {
            skl.run();
        }
    }
    {  // Reference
        for (int i = 0; i < 10; i++) {
            using namespace Neon::domain::tool::testing;

            auto& X = data.getIODomain(FieldNames::X);
            auto& Y = data.getIODomain(FieldNames::Y);
            data.sum(X, Y);
        }
    }
    bk.syncAll();
    bool isOk = data.compare(FieldNames::X);
    isOk = isOk && data.compare(FieldNames::Y);
    ASSERT_TRUE(isOk);
}

template <typename G, typename T, int C>
auto twoStageXPYPipe(Neon::domain::tool::testing::TestData<G, T, C>& data) -> void
{
    data.resetValuesToLinear(1, 100);
    using namespace Neon::domain::tool::testing;

    auto& grid = data.getGrid();
    auto  bk = grid.getBackend();

    int const nIterations = 10;

    {  // Neon

        auto& X = data.getField(FieldNames::X);
        auto& Y = data.getField(FieldNames::Y);
        auto& Z = data.getField(FieldNames::Z);

        Neon::skeleton::Skeleton          skl(bk);
        Neon::skeleton::Options           opt;
        std::vector<Neon::set::Container> sVec;

        sVec.push_back(UserTools::xpy(X, Y));
        sVec.push_back(UserTools::xpy(Y, Z));
        skl.sequence(sVec, "twoStageXPYPipe", opt);

        for (int i = 0; i < nIterations; i++) {
            skl.run();
        }
    }

    {  // Reference
        for (int i = 0; i < nIterations; i++) {
            using namespace Neon::domain::tool::testing;

            auto& X = data.getIODomain(FieldNames::X);
            auto& Y = data.getIODomain(FieldNames::Y);
            auto& Z = data.getIODomain(FieldNames::Z);

            data.sum(X, Y);
            data.sum(Y, Z);
        }
    }

    bk.syncAll();
    bool isOk = data.compare(FieldNames::X);
    isOk = isOk && data.compare(FieldNames::Y);
    isOk = isOk && data.compare(FieldNames::Z);
    ASSERT_TRUE(isOk);
}

template <typename G, typename T, int C>
auto threeLevelXPYTree(Neon::domain::tool::testing::TestData<G, T, C>& data) -> void
{
    data.resetValuesToLinear(1, 100);
    using namespace Neon::domain::tool::testing;

    auto& grid = data.getGrid();
    auto  bk = grid.getBackend();

    int const nIterations = 10;

    {  // Neon

        auto& X = data.getField(FieldNames::X);
        auto& Y = data.getField(FieldNames::Y);
        auto& Z = data.getField(FieldNames::Z);

        Neon::skeleton::Skeleton          skl(bk);
        Neon::skeleton::Options           opt;
        std::vector<Neon::set::Container> sVec;

        sVec.push_back(UserTools::xpy(X, X));
        sVec.push_back(UserTools::xpy(Y, Y));
        sVec.push_back(UserTools::xpy(Z, Z));
        sVec.push_back(UserTools::xpy(Z, X));
        sVec.push_back(UserTools::xpy(Z, Y));
        sVec.push_back(UserTools::xpy(X, Y));

        skl.sequence(sVec, "threeLevelXPYTree", opt);

        for (int i = 0; i < nIterations; i++) {
            skl.run();
        }
    }

    {  // Reference
        for (int i = 0; i < nIterations; i++) {
            using namespace Neon::domain::tool::testing;

            auto& X = data.getIODomain(FieldNames::X);
            auto& Y = data.getIODomain(FieldNames::Y);
            auto& Z = data.getIODomain(FieldNames::Z);

            data.sum(X, X);
            data.sum(Y, Y);
            data.sum(Z, Z);
            data.sum(Z, X);
            data.sum(Z, Y);
            data.sum(X, Y);
        }
    }

    bk.syncAll();
    bool isOk = data.compare(FieldNames::X);
    isOk = isOk && data.compare(FieldNames::Y);
    isOk = isOk && data.compare(FieldNames::Z);
    ASSERT_TRUE(isOk);
}


TEST(OneStageXPYPipe, eGrid)
{
    int           nGpus = 3;
    constexpr int C = 0;
    using Grid = Neon::eGrid;
    using T = int64_t;
    // runAllTestConfiguration(AXPY_struct<eGrid_t, int64_t>, nGpus);
    runAllTestConfiguration(std::function(oneStageXPYPipe<Grid, T, C>), nGpus, 1);
}

TEST(OneStageXPYPipe, dGrid)
{
    int           nGpus = 3;
    constexpr int C = 0;
    using Grid = Neon::dGrid;
    using T = int64_t;
    // runAllTestConfiguration(AXPY_struct<eGrid_t, int64_t>, nGpus);
    runAllTestConfiguration(std::function(oneStageXPYPipe<Grid, T, C>), nGpus, 1);
}

TEST(OneStageXPYPipe, bGrid)
{
    int           nGpus = 3;
    constexpr int C = 0;
    using Grid = Neon::dGrid;
    using T = int64_t;
    // runAllTestConfiguration(AXPY_struct<eGrid_t, int64_t>, nGpus);
    runAllTestConfiguration(std::function(oneStageXPYPipe<Grid, T, C>), nGpus, 1);
}

// ===============================================================================
TEST(twoStageXPYPipe, eGrid)
{
    int           nGpus = 3;
    constexpr int C = 0;
    using Grid = Neon::eGrid;
    using T = int64_t;
    // runAllTestConfiguration(AXPY_struct<eGrid_t, int64_t>, nGpus);
    runAllTestConfiguration(std::function(twoStageXPYPipe<Grid, T, C>), nGpus, 1);
}

TEST(twoStageXPYPipe, dGrid)
{
    int           nGpus = 3;
    constexpr int C = 0;
    using Grid = Neon::dGrid;
    using T = int64_t;
    // runAllTestConfiguration(AXPY_struct<eGrid_t, int64_t>, nGpus);
    runAllTestConfiguration(std::function(twoStageXPYPipe<Grid, T, C>), nGpus, 1);
}

TEST(twoStageXPYPipe, bGrid)
{
    int           nGpus = 3;
    constexpr int C = 0;
    using Grid = Neon::dGrid;
    using T = int64_t;
    // runAllTestConfiguration(AXPY_struct<eGrid_t, int64_t>, nGpus);
    runAllTestConfiguration(std::function(twoStageXPYPipe<Grid, T, C>), nGpus, 1);
}


// ===============================================================================
TEST(threeLevelXPYTree, eGrid)
{
    int           nGpus = 3;
    constexpr int C = 0;
    using Grid = Neon::eGrid;
    using T = int64_t;
    // runAllTestConfiguration(AXPY_struct<eGrid_t, int64_t>, nGpus);
    runAllTestConfiguration(std::function(threeLevelXPYTree<Grid, T, C>), nGpus, 1);
}

TEST(threeLevelXPYTree, dGrid)
{
    int           nGpus = 3;
    constexpr int C = 0;
    using Grid = Neon::dGrid;
    using T = int64_t;
    // runAllTestConfiguration(AXPY_struct<eGrid_t, int64_t>, nGpus);
    runAllTestConfiguration(std::function(threeLevelXPYTree<Grid, T, C>), nGpus, 1);
}

TEST(threeLevelXPYTree, bGrid)
{
    int           nGpus = 3;
    constexpr int C = 0;
    using Grid = Neon::dGrid;
    using T = int64_t;
    // runAllTestConfiguration(AXPY_struct<eGrid_t, int64_t>, nGpus);
    runAllTestConfiguration(std::function(threeLevelXPYTree<Grid, T, C>), nGpus, 1);
}