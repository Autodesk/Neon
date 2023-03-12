#include <functional>
#include "Neon/domain/Grids.h"
#include "Neon/domain/interface/GridConcept.h"
#include "Neon/domain/tools/TestData.h"
#include "TestInformation.h"
#include "gtest/gtest.h"


namespace map {

template <typename Field>
auto stencilContainer_laplace(const Field& filedA,
                              Field&       fieldB)
    -> Neon::set::Container
{
    const auto& grid = filedA.getGrid();
    return grid.newContainer(
        "mapContainer_axpy",
        [&](Neon::set::Loader& loader) {
            const auto a = loader.load(filedA, Neon::Compute::STENCIL);
            auto       b = loader.load(fieldB);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& idx) mutable {
                for (int i = 0; i < a.cardinality(); i++) {
                    // printf("GPU %ld <- %ld + %ld\n", lc(e, i) , la(e, i) , val);
                    typename Field::Type partial = 0;
                    int                  count = 0;
                    using Ngh3DIdx = Neon::int8_3d;
                    std::array< Ngh3DIdx, 6> stencil{
                        Ngh3DIdx(1, 0, 0),
                        Ngh3DIdx(-1, 0, 0),
                        Ngh3DIdx(0, 1, 0),
                        Ngh3DIdx(0, -1, 0),
                        Ngh3DIdx(0, 0, 1),
                        Ngh3DIdx(0, 0, -1)};
                    for (auto const& direction : stencil) {
                        typename Field::NghData nghData = a.getNghData(idx, direction, i, 0);
                        if (nghData.isValid()) {
                            partial += nghData.getData();
                            count++;
                        }
                    };
                    b(idx, i) = a(idx, i);  // - count * partial;
                }
            };
        },
        Neon::Execution::device);
}

using namespace Neon::domain::tool::testing;

template <typename G, typename T, int C>
auto run(TestData<G, T, C>& data) -> void
{

    using Type = typename TestData<G, T, C>::Type;
    auto&             grid = data.getGrid();
    const std::string appName = TestInformation::fullName(grid.getImplementationName());

    NEON_INFO(grid.toString());

    data.resetValuesToLinear(1, 100);

    {  // NEON
        const Neon::index_3d        dim = grid.getDimension();
        std::vector<Neon::index_3d> elements;

        auto& X = data.getField(FieldNames::X);
        auto& Y = data.getField(FieldNames::Y);

        for (int iter = 4; iter > 0; iter--) {
            X.newHaloUpdate(Neon::set::StencilSemantic::standard,
                            Neon::set::TransferMode::get,
                            Neon::Execution::device)
                .run(Neon::Backend::mainStreamIdx);

            stencilContainer_laplace(X, Y).run(Neon::Backend::mainStreamIdx);

            Y.newHaloUpdate(Neon::set::StencilSemantic::standard,
                            Neon::set::TransferMode::get,
                            Neon::Execution::device)
                .run(Neon::Backend::mainStreamIdx);

            stencilContainer_laplace(Y, X).run(Neon::Backend::mainStreamIdx);
        }
        data.getBackend().sync(0);
    }

    {  // Golden data
        auto& X = data.getIODomain(FieldNames::X);
        auto& Y = data.getIODomain(FieldNames::Y);
        for (int iter = 4; iter > 0; iter--) {

            data.laplace(X, Y);
            data.laplace(Y, X);
        }
    }

    data.updateHostData();
    bool isOk = data.compare(FieldNames::X);
    ASSERT_TRUE(isOk);
}

template auto run<Neon::dGrid, int64_t, 0>(TestData<Neon::dGrid, int64_t, 0>&) -> void;
template auto run<Neon::eGrid, int64_t, 0>(TestData<Neon::eGrid, int64_t, 0>&) -> void;

}  // namespace map