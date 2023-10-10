#include <functional>
#include "Neon/domain/Grids.h"

#include "Neon/domain/details/dGridSoA/dGridSoA.h"
#include "Neon/domain/tools/TestData.h"
#include "TestInformation.h"
#include "gtest/gtest.h"


namespace map {

template <typename Field>
auto mapContainer_axpy(int                   streamIdx,
                       typename Field::Type& val,
                       const Field&          filedA,
                       Field&                fieldB)
    -> Neon::set::Container
{
    const auto& grid = filedA.getGrid();
    return grid.newContainer(
        "mapContainer_axpy",
        [&, val](Neon::set::Loader& loader) {
            const auto a = loader.load(filedA);
            auto       b = loader.load(fieldB);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& e) mutable {
                for (int i = 0; i < a.cardinality(); i++) {
                    // printf("GPU %ld <- %ld + %ld\n", lc(e, i) , la(e, i) , val);
                    b(e, i) += a(e, i) * val;
                }
            };
        });
}

template <typename Field>
auto mapContainer_add(int                   streamIdx,
                       typename Field::Type& val,
                       Field&                fieldB)
    -> Neon::set::Container
{
    const auto& grid = fieldB.getGrid();
    return grid.newContainer(
        "mapContainer_axpy",
        [&, val](Neon::set::Loader& loader) {
            auto       b = loader.load(fieldB);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& e) mutable {
                for (int i = 0; i < b.cardinality(); i++) {
                    // printf("GPU %ld <- %ld + %ld\n", lc(e, i) , la(e, i) , val);
                    b(e, i) += val;
                }
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
    T val = T(33);

    {  // NEON
        const Neon::index_3d        dim = grid.getDimension();
        std::vector<Neon::index_3d> elements;

        auto& X = data.getField(FieldNames::X);
        auto& Y = data.getField(FieldNames::Y);


        mapContainer_axpy(Neon::Backend::mainStreamIdx,
                          val, X, Y)
            .run(0);

        X.updateHostData(0);
        Y.updateHostData(0);

        data.getBackend().sync(0);
    }

    {  // Golden data
        auto& X = data.getIODomain(FieldNames::X);
        auto& Y = data.getIODomain(FieldNames::Y);
        data.axpy(&val, X, Y);
    }

    bool isOk = data.compare(FieldNames::Y);
    ASSERT_TRUE(isOk);
}

template auto run<Neon::dGrid, int64_t, 0>(TestData<Neon::dGrid, int64_t, 0>&) -> void;
template auto run<Neon::eGrid, int64_t, 0>(TestData<Neon::eGrid, int64_t, 0>&) -> void;
template auto run<Neon::bGrid, int64_t, 0>(TestData<Neon::bGrid, int64_t, 0>&) -> void;
template auto run<Neon::domain::details::dGridSoA::dGridSoA, int64_t, 0>(TestData<Neon::domain::details::dGridSoA::dGridSoA, int64_t, 0>&) -> void;

namespace dataView {
template <typename G, typename T, int C>
auto run(TestData<G, T, C>& data) -> void
{

    using Type = typename TestData<G, T, C>::Type;
    auto&             grid = data.getGrid();
    const std::string appName = TestInformation::fullName(grid.getImplementationName());

    data.resetValuesToLinear(1, 100);
    T val = T(33);

    {  // NEON
        const Neon::index_3d        dim = grid.getDimension();
        std::vector<Neon::index_3d> elements;

        auto& X = data.getField(FieldNames::X);
        auto& Y = data.getField(FieldNames::Y);


        mapContainer_axpy(Neon::Backend::mainStreamIdx,
                          val, X, Y)
            .run(0, Neon::DataView::BOUNDARY);

        mapContainer_axpy(Neon::Backend::mainStreamIdx,
                          val, X, Y)
            .run(0, Neon::DataView::INTERNAL);

        X.updateHostData(0);
        Y.updateHostData(0);

        data.getBackend().sync(0);
    }

    {  // Golden data
        auto& X = data.getIODomain(FieldNames::X);
        auto& Y = data.getIODomain(FieldNames::Y);
        data.axpy(&val, X, Y);
    }

    bool isOk = data.compare(FieldNames::Y);
    ASSERT_TRUE(isOk);
}
template auto run<Neon::dGrid, int64_t, 0>(TestData<Neon::dGrid, int64_t, 0>&) -> void;
template auto run<Neon::eGrid, int64_t, 0>(TestData<Neon::eGrid, int64_t, 0>&) -> void;
template auto run<Neon::bGrid, int64_t, 0>(TestData<Neon::bGrid, int64_t, 0>&) -> void;
template auto run<Neon::domain::details::dGridSoA::dGridSoA, int64_t, 0>(TestData<Neon::domain::details::dGridSoA::dGridSoA, int64_t, 0>&) -> void;

}  // namespace dataView
}  // namespace map