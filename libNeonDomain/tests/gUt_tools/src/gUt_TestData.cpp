#include "gtest/gtest.h"

#include "Neon/core/core.h"

#include "Neon/domain/eGrid.h"
#include "Neon/domain/tools/TestData.h"

template <typename G>
auto TestDataTest(Neon::Backend&  backend,
                  Neon::index_3d& dimension) -> void
{
    using Grid = Neon::domain::details::eGrid::eGrid;
    using TestData = Neon::domain::tool::testing::TestData<Grid, int, 0>;
    using namespace Neon::domain::tool::testing;

    constexpr auto geometryName = Neon::domain::tool::Geometry::FullDomain;
    constexpr int  cardinality = 1;

    Neon::MemoryOptions memoryOptions = backend.getMemoryOptions();
    TestData            data(backend,
                             dimension,
                             cardinality,
                             memoryOptions,
                             geometryName);

    data.resetValuesToRandom(0, 100);

    data.getField(FieldNames::X).getReference({0, 0, 0}, 0) += 1;
    data.getField(FieldNames::X).getReference(dimension - 1, 0) += 2;
    data.getField(FieldNames::X).updateDeviceData(0);

    data.compare(FieldNames::X, [&](const Neon::index_3d& idx,
                                    int,
                                    const int& valGolden,
                                    const int& valField) {
        if (valField != valGolden) {
            ASSERT_TRUE(idx == Neon::index_3d(0, 0, 0) || idx == (dimension - 1));
            if (idx == Neon::index_3d(0, 0, 0)) {
                ASSERT_TRUE(valField - valGolden == 1) << " valField " << valField << " " << valGolden;
            }
            if (idx == (dimension - 1)) {
                ASSERT_TRUE(valField - valGolden == 2) << " valField " << valField << " " << valGolden;
            }
            return;
        }
        ASSERT_FALSE(idx == Neon::index_3d(0, 0, 0)) << idx.to_string();
        ASSERT_FALSE(idx == (dimension - 1)) << idx.to_string();
    });
}

TEST(gUt_tools_TestData, HollowSphere)
{
    Neon::index_3d   dimension(10, 20, 10);
    std::vector<int> devIds = {0};
    Neon::Backend    backend(devIds, Neon::Runtime::openmp);
    TestDataTest<Neon::domain::details::eGrid::eGrid>(backend, dimension);
}
