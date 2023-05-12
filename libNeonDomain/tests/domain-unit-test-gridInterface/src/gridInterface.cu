#include <functional>
#include "Neon/domain/Grids.h"

#include "Neon/domain/tools/TestData.h"
#include "TestInformation.h"
#include "gtest/gtest.h"


namespace globalIdx {

using namespace Neon::domain::tool::testing;

template <typename G, typename T, int C>
auto run(TestData<G, T, C>& data) -> void
{

    using Type = typename TestData<G, T, C>::Type;

    {  // NEON
        auto& X = data.getField(FieldNames::X);
        data.resetValuesToConst(1, 1);
        X.ioToVtk(data.name(), "f", true);

        auto& grid = X.getGrid();
        {
            auto dim = grid.getDimension();
            ASSERT_EQ(dim.x, data.getDimention().x) << "Computed "<<dim.x << " golden reference " << data.getDimention().x;
            ASSERT_EQ(dim.y, data.getDimention().y);
            ASSERT_EQ(dim.z, data.getDimention().z);
        }

        {
            auto dim = grid.getDimension();
            auto numAllCells = grid.getNumAllCells();
            ASSERT_EQ(numAllCells, dim.rMul());
        }

        {
            auto  numActiveCells = grid.getNumActiveCells();
            auto& X = data.getIODomain(FieldNames::X);
            int   count = 0;
            data.template forEachActiveIODomain([&](const Neon::index_3d&  idx,
                                                    [[maybe_unused]] int   cardinality,
                                                    [[maybe_unused]] Type& a) {
#pragma omp critical
                {
                    count++;
                }
            },
                                                X);
            ASSERT_EQ(numActiveCells, count);
        }

        {
            auto                       numActiveCells = grid.getNumActiveCells();
            Neon::set::DataSet<size_t> numActiveCellsPerPartition = grid.getNumActiveCellsPerPartition();
            int                        count = 0;
            numActiveCellsPerPartition.forEachSeq([&](Neon::SetIdx, auto val) {
                count += val;
            });
            ASSERT_EQ(numActiveCells, count);
        }
    }
}

template auto run<Neon::dGrid, int64_t, 0>(TestData<Neon::dGrid, int64_t, 0>&) -> void;
template auto run<Neon::eGrid, int64_t, 0>(TestData<Neon::eGrid, int64_t, 0>&) -> void;


}  // namespace globalIdx