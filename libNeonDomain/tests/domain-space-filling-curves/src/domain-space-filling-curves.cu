#include <cmath>
#include <functional>
#include <iostream>
#include "Neon/domain/Grids.h"
#include "Neon/domain/details/dGridSoA/dGridSoA.h"
#include "Neon/domain/tools/SpaceCurves.h"
#include "Neon/domain/tools/TestData.h"
#include "TestInformation.h"
#include "gtest/gtest.h"

#include <cmath>
#include <iostream>

namespace space_filling_curves {

template <typename Field>
auto defHostContainer(Field& filedSweep,
                      Field& filedMorton,
                      Field& filedHilbert)
    -> Neon::set::Container
{
    const auto& grid = filedSweep.getGrid();
    return grid.template newContainer<Neon::Execution::host>(
        "defContainer",
        [&](Neon::set::Loader& loader) {
            auto sweep = loader.load(filedSweep);
            auto morton = loader.load(filedMorton);
            auto hilbert = loader.load(filedHilbert);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& gidx) mutable {
                Neon::index_3d p = sweep.getGlobalIndex(gidx);
                Neon::index_3d dim = sweep.getDomainSize();
                using namespace Neon::domain::tool::spaceCurves;
                sweep(gidx, 0) = Encoder::encode(EncoderType::sweep, dim, p);
                morton(gidx, 0) = Encoder::encode(EncoderType::morton, dim, p);
                hilbert(gidx, 0) = Encoder::encode(EncoderType::hilbert, dim, p);
            };
        });
}


using namespace Neon::domain::tool::testing;

template <typename G, typename T, int C>
auto run(TestData<G, T, C>& data) -> void
{

    using Type = typename TestData<G, T, C>::Type;
    auto&             grid = data.getGrid();
    const std::string appName = TestInformation::fullName(grid.getImplementationName());

    data.resetValuesToLinear(1, 100);

    {  // NEON
        const Neon::index_3d        dim = grid.getDimension();
        std::vector<Neon::index_3d> elements;

        auto& X = data.getField(FieldNames::X);
        auto& Y = data.getField(FieldNames::Y);
        auto& Z = data.getField(FieldNames::Z);

        defHostContainer(X, Y, Z).run(0);
        data.getBackend().sync(0);

        data.getField(FieldNames::X).ioToVtk("spaceCurveSweep", "code", false);
        data.getField(FieldNames::Y).ioToVtk("spaceCurveMorton", "code", false);
        data.getField(FieldNames::Z).ioToVtk("spaceCurveHilbert", "code", false);
    }
}

template auto run<Neon::dGrid, int64_t, 0>(TestData<Neon::dGrid, int64_t, 0>&) -> void;


}  // namespace space_filling_curves