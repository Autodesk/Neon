#include <functional>
#include "Neon/domain/eGrid.h"
#include "Neon/domain/details/dGrid/dGrid.h"

#include "Neon/domain/tools/TestData.h"
#include "TestInformation.h"
#include "containerRun.h"
#include "gtest/gtest.h"


using namespace Neon::domain::tool::testing;

template <typename G, typename T, int C>
auto runContainer(TestData<G, T, C>&                data,
                  const Neon::sys::patterns::Engine eng) -> void
{
    using Type = typename TestData<G, T, C>::Type;
    auto&             grid = data.getGrid();
    const std::string appName = TestInformation::fullName(grid.getImplementationName());

    data.resetValuesToLinear(1, 100);

    Type neonRes = 0;
    Type goldenRes = 0;
    {  // NEON
        const Neon::index_3d        dim = grid.getDimension();
        std::vector<Neon::index_3d> elements;

        data.getGrid().setReduceEngine(eng);

        data.resetValuesToConst(1);

        auto& X = data.getField(FieldNames::X);
        auto  scalar = data.getGrid().template newPatternScalar<Type>();
        auto  norm2_container = data.getGrid().norm2("GridNorm2", X, scalar, Neon::Execution::device);
        norm2_container.run(Neon::Backend::mainStreamIdx);
        data.getBackend().sync(0);
    }

    {  // Golden data
        auto& X = data.getIODomain(FieldNames::X);

        data.forEachActiveIODomain([&](const Neon::index_3d& ,
                                       int                   ,
                                       Type&                 a) {
            auto tmp = a * a;
#pragma omp single
            {
                goldenRes += tmp;
            }
        },
                                   X);
    }

    ASSERT_TRUE(goldenRes == neonRes) << "goldenRes " << goldenRes << " neonRes " << neonRes;
}

template auto runContainer<Neon::domain::details::dGrid::dGrid, int64_t, 0>(TestData<Neon::domain::details::dGrid::dGrid, int64_t, 0>&,
                                                                                  const Neon::sys::patterns::Engine eng) -> void;
