#include <functional>
#include "Neon/domain/Grids.h"
#include "Neon/domain/details/bGridDisg/bGrid.h"
#include "Neon/domain/details/dGridDisg/dGrid.h"

#include "Neon/domain/details/dGridSoA/dGridSoA.h"
#include "Neon/domain/tools/TestData.h"
#include "TestInformation.h"
#include "gtest/gtest.h"


namespace map {

template <typename Field>
auto set(const Field& filedA)
    -> Neon::set::Container
{
    const auto& grid = filedA.getGrid();
    return grid.newContainer(
        "set",
        [&](Neon::set::Loader& loader) {
            auto a = loader.load(filedA);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& e) mutable {
                for (int i = 0; i < a.cardinality(); i++) {
                    // printf("GPU %ld <- %ld + %ld\n", lc(e, i) , la(e, i) , val);
                    a(e, i) = 0;
                }
            };
        });
}

template <typename Field>
auto addAlpha(typename Field::Type val,
              Field&               filedA)
    -> Neon::set::Container
{
    const auto& grid = filedA.getGrid();
    return grid.newAlphaContainer(
        "AlphaSet",
        [&](Neon::set::Loader& loader) {
            auto a = loader.load(filedA);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& e) mutable {
                for (int i = 0; i < a.cardinality(); i++) {
                    // Neon::index_3d globalPoint = a.getGlobalIndex(e);
                    // printf("AlphaSet %d %d %d\n", globalPoint.x, globalPoint.y, globalPoint.z);
                    a(e, i) += val;
                }
            };
        });
}

template <typename Field>
auto addBeta(typename Field::Type val,
             Field&               filedA)
    -> Neon::set::Container
{
    const auto& grid = filedA.getGrid();
    return grid.newBetaContainer(
        "AlphaSet",
        [&](Neon::set::Loader& loader) {
            auto a = loader.load(filedA);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& e) mutable {
                for (int i = 0; i < a.cardinality(); i++) {
                    // Neon::index_3d globalPoint = a.getGlobalIndex(e);
                    // printf("AlphaSet %d %d %d\n", globalPoint.x, globalPoint.y, globalPoint.z);
                    a(e, i) += val;
                }
            };
        });
}

template <typename Field>
auto axpyAlphaBeta(typename Field::Type& val,
                   const Field&          filedA,
                   Field&                fieldB)
    -> Neon::set::Container
{
    const auto& grid = filedA.getGrid();
    return grid.newAlphaBetaContainer(
        "BetaSet",
        [&](Neon::set::Loader& loader) {
            auto const a = loader.load(filedA);
            auto       b = loader.load(fieldB);
            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& e) mutable {
                for (int i = 0; i < a.cardinality(); i++) {
                    b(e, i) += a(e, i) * val;
                }
            };
        },
        [&](Neon::set::Loader& loader) {
            auto const a = loader.load(filedA);
            auto       b = loader.load(fieldB);
            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& e) mutable {
                for (int i = 0; i < a.cardinality(); i++) {
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
            auto b = loader.load(fieldB);

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
    auto& grid = data.getGrid();
    G     tmp = G(
        grid.getBackend(),
        grid.getDimension(),
        [&](Neon::index_3d idx) {
            bool isInside = grid.isInsideDomain(idx);
            if (!isInside) {
                return Neon::domain::details::disaggregated::bGrid::details::cGrid::ClassSelector::outside;
            }
            if (idx.x == 0 || idx.y == 0 || idx.z == 0 || idx.x == grid.getDimension().x - 1 || idx.y == grid.getDimension().y - 1 || idx.z == grid.getDimension().z - 1) {
                return Neon::domain::details::disaggregated::bGrid::details::cGrid::ClassSelector::beta;
            }
            return Neon::domain::details::disaggregated::bGrid::details::cGrid::ClassSelector::alpha;
        },
        grid.getStencil(),
        1,
        grid.getSpacing(),
        grid.getOrigin(),
        grid.getSpaceCurve());


    grid = tmp;
    grid.helpGetClassField().template ioToVtk<int>("classField", "classField");
    const std::string appName = TestInformation::fullName(grid.getImplementationName());
    data.resetValuesToLinear(1, 100);

    set(data.getField(FieldNames::X)).run(0);
    addAlpha(11, data.getField(FieldNames::X)).run(0);
    addBeta(17, data.getField(FieldNames::X)).run(0);
    data.getField(FieldNames::X).ioToVtk("XAlphaBeta", "XAlphaBeta");

    data.resetValuesToLinear(1, 100);

    T val = T(33);

    {  // NEON
        const Neon::index_3d        dim = grid.getDimension();
        std::vector<Neon::index_3d> elements;

        auto& X = data.getField(FieldNames::X);
        auto& Y = data.getField(FieldNames::Y);


        axpyAlphaBeta(val, X, Y).run(0);

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
        data.getField(FieldNames::X).ioToVtk("X_t0", "X_t0");
        data.getField(FieldNames::Y).ioToVtk("Y_t0", "Y_t0");

        axpyAlphaBeta(val, X, Y).run(0, Neon::DataView::BOUNDARY);
        axpyAlphaBeta(val, X, Y).run(0, Neon::DataView::INTERNAL);

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
    if (!isOk) {
        auto flagField = data.compareAndGetField(FieldNames::X);

        data.getField(FieldNames::X).ioToVtk("X", "X");
        data.getField(FieldNames::Y).ioToVtk("Y", "Y");

        flagField.ioToVti("X_diffFlag", "X_diffFlag");
        flagField = data.compareAndGetField(FieldNames::Y);
        flagField.ioToVti("Y_diffFlag", "Y_diffFlag");
    }
    ASSERT_TRUE(isOk);
}

}  // namespace dataView

template auto run<Neon::bGridDisg, int64_t, 0>(TestData<Neon::bGridDisg, int64_t, 0>&) -> void;
// template auto run<Neon::bGrid, int64_t, 0>(TestData<Neon::bGrid, int64_t, 0>&) -> void;

namespace dataView {
template auto run<Neon::bGridDisg, int64_t, 0>(TestData<Neon::bGridDisg, int64_t, 0>&) -> void;
// template auto run<Neon::bGrid, int64_t, 0>(TestData<Neon::bGrid, int64_t, 0>&) -> void;
// template auto run<Neon::dGrid, int64_t, 0>(TestData<Neon::dGrid, int64_t, 0>&) -> void;

}  // namespace dataView
}  // namespace map