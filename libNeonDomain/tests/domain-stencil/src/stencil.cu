#include <functional>
#include "Neon/domain/Grids.h"
#include "Neon/domain/interface/GridConcept.h"
#include "Neon/domain/tools/TestData.h"
#include "TestInformation.h"
#include "gtest/gtest.h"
#include "Neon/domain/details/dGridDisg/dGrid.h"


namespace map {

template <typename Field>
auto laplaceNoTemplate(const Field& filedA,
                       Field&       fieldB)
    -> Neon::set::Container
{
    const auto& grid = filedA.getGrid();
    return grid.newContainer(
        "stencilFun",
        [&](Neon::set::Loader& loader) {
            const auto a = loader.load(filedA, Neon::Pattern::STENCIL);
            auto       b = loader.load(fieldB);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& idx) mutable {
                int maxCard = a.cardinality();
                for (int i = 0; i < maxCard; i++) {
                    // printf("GPU %ld <- %ld + %ld\n", lc(e, i) , la(e, i) , val);
                    typename Field::Type partial = 0;
                    int                  count = 0;
                    using Ngh3DIdx = Neon::int8_3d;

                    constexpr std::array<const Ngh3DIdx, 6> stencil{
                        Ngh3DIdx(1, 0, 0),
                        Ngh3DIdx(-1, 0, 0),
                        Ngh3DIdx(0, 1, 0),
                        Ngh3DIdx(0, -1, 0),
                        Ngh3DIdx(0, 0, 1),
                        Ngh3DIdx(0, 0, -1)};

                    for (auto const& direction : stencil) {
                        typename Field::NghData nghData = a.getNghData(idx, direction, i);
                        if (nghData.isValid()) {
                            partial += nghData.getData();
                            count++;
                        }
                    }

                    b(idx, i) = a(idx, i) - count * partial;
                }
            };
        });
}

using Ngh3DIdx = Neon::int8_3d;
static constexpr std::array<const Ngh3DIdx, 6> stencil{
    Ngh3DIdx(1, 0, 0),
    Ngh3DIdx(-1, 0, 0),
    Ngh3DIdx(0, 1, 0),
    Ngh3DIdx(0, -1, 0),
    Ngh3DIdx(0, 0, 1),
    Ngh3DIdx(0, 0, -1)};

template <int sIdx, typename IDX, typename Partition, typename Partial>
NEON_CUDA_HOST_DEVICE inline auto viaTemplate(const IDX& idx, int i, const Partition& a, Partial& partial, int& count)
{
    //    Neon::index_3d direction(X, Y, Z);
    //    auto           nghData = a.getNghData(idx, direction.newType<int8_t>(), i);
    //    if (nghData.isValid()) {
    //        partial += nghData.getData();
    //        count++;
    //    }
    a.template getNghData<stencil[sIdx].x,
                          stencil[sIdx].y,
                          stencil[sIdx].z>(idx, i,
                                           [&](typename Partition::Type const& val) {
                                               partial += val;
                                               count++;
                                           });
};


//template <auto Start, auto End, auto Inc, class F>
//constexpr void constexpr_for(F&& f)
//{
//    if constexpr (Start < End) {
//        f(std::integral_constant<decltype(Start), Start>());
//        constexpr_for<Start + Inc, End, Inc>(f);
//    }
//}

template <typename Field>
auto laplaceTemplate(const Field& filedA,
                     Field&       fieldB)
    -> Neon::set::Container
{
    const auto& grid = filedA.getGrid();
    return grid.newContainer(
        "stencilFun",
        [&](Neon::set::Loader& loader) {
            const auto a = loader.load(filedA, Neon::Pattern::STENCIL);
            auto       b = loader.load(fieldB);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& idx) mutable {
                int maxCard = a.cardinality();
                for (int i = 0; i < maxCard; i++) {
                    // printf("GPU %ld <- %ld + %ld\n", lc(e, i) , la(e, i) , val);
                    typename Field::Type partial = 0;
                    int                  count = 0;
                    using Ngh3DIdx = Neon::int8_3d;

                    Neon::ConstexprFor<0, 6, 1>([&](auto sIdx) {
                        a.template getNghData<stencil[sIdx].x,
                                              stencil[sIdx].y,
                                              stencil[sIdx].z>(idx, i,
                                                               [&](auto const& val) {
                                                                   partial += val;
                                                                   count++;
                                                               });
                    });
                    
                    b(idx, i) = a(idx, i) - count * partial;
                }
            };
        });
}

using namespace Neon::domain::tool::testing;

template <typename G, typename T, int C>
auto runNoTemplate(TestData<G, T, C>& data) -> void
{

    using Type = typename TestData<G, T, C>::Type;
    auto&             grid = data.getGrid();
    const std::string appName = TestInformation::fullName(grid.getImplementationName());
    const int         maxIters = 1;

    NEON_INFO(grid.toString());

    // data.resetValuesToLinear(1, 100);
    data.resetValuesToMasked(1);

    {  // NEON
        const Neon::index_3d        dim = grid.getDimension();
        std::vector<Neon::index_3d> elements;
        auto                        bk = grid.getBackend();
        auto&                       X = data.getField(FieldNames::X);
        auto&                       Y = data.getField(FieldNames::Y);
        for (int iter = maxIters; iter > 0; iter--) {
            bk.sync(Neon::Backend::mainStreamIdx);
            X.newHaloUpdate(Neon::set::StencilSemantic::standard,
                            Neon::set::TransferMode::put,
                            Neon::Execution::device)
                .run(Neon::Backend::mainStreamIdx);

            bk.sync(Neon::Backend::mainStreamIdx);
            laplaceNoTemplate(X, Y).run(Neon::Backend::mainStreamIdx);

            bk.sync(Neon::Backend::mainStreamIdx);
            Y.newHaloUpdate(Neon::set::StencilSemantic::standard,
                            Neon::set::TransferMode::get,
                            Neon::Execution::device)
                .run(Neon::Backend::mainStreamIdx);

            bk.sync(Neon::Backend::mainStreamIdx);
            laplaceNoTemplate(Y, X).run(Neon::Backend::mainStreamIdx);
        }
        data.getBackend().sync(0);
    }

    {  // Golden data
        auto& X = data.getIODomain(FieldNames::X);
        auto& Y = data.getIODomain(FieldNames::Y);
        for (int iter = maxIters; iter > 0; iter--) {
            data.laplace(X, Y);
            data.laplace(Y, X);
        }
    }

    data.updateHostData();

    data.getField(FieldNames::X).ioToVtk("X", "X", true);
    //    data.getField(FieldNames::Y).ioToVtk("Y", "Y", false);
    //    data.getField(FieldNames::Z).ioToVtk("Z", "Z", false);
    //
    data.getIODomain(FieldNames::X).ioToVti("X_", "X_");
    //    data.getField(FieldNames::Y).ioVtiAllocator("Y_");
    //    data.getField(FieldNames::Z).ioVtiAllocator("Z_");

    bool isOk = data.compare(FieldNames::X);
    isOk = data.compare(FieldNames::Y);
    if (!isOk) {
        auto flagField = data.compareAndGetField(FieldNames::X);
        flagField.ioToVti("X_diffFlag", "X_diffFlag");
        flagField = data.compareAndGetField(FieldNames::Y);
        flagField.ioToVti("Y_diffFlag", "Y_diffFlag");
    }
    ASSERT_TRUE(isOk);
    if (!isOk) {
        exit(99);
    }
}

template <typename G, typename T, int C>
auto runTemplate(TestData<G, T, C>& data) -> void
{

    using Type = typename TestData<G, T, C>::Type;
    auto&             grid = data.getGrid();
    const std::string appName = TestInformation::fullName(grid.getImplementationName());
    const int         maxIters = 1;

    NEON_INFO(grid.toString());

    // data.resetValuesToLinear(1, 100);
    data.resetValuesToMasked(1);

    {  // NEON
        const Neon::index_3d        dim = grid.getDimension();
        std::vector<Neon::index_3d> elements;
        auto                        bk = grid.getBackend();
        auto&                       X = data.getField(FieldNames::X);
        auto&                       Y = data.getField(FieldNames::Y);
        for (int iter = maxIters; iter > 0; iter--) {
            bk.sync(Neon::Backend::mainStreamIdx);
            X.newHaloUpdate(Neon::set::StencilSemantic::standard,
                            Neon::set::TransferMode::put,
                            Neon::Execution::device)
                .run(Neon::Backend::mainStreamIdx);

            bk.sync(Neon::Backend::mainStreamIdx);
            laplaceTemplate(X, Y).run(Neon::Backend::mainStreamIdx);

            bk.sync(Neon::Backend::mainStreamIdx);
            Y.newHaloUpdate(Neon::set::StencilSemantic::standard,
                            Neon::set::TransferMode::get,
                            Neon::Execution::device)
                .run(Neon::Backend::mainStreamIdx);

            bk.sync(Neon::Backend::mainStreamIdx);
            laplaceTemplate(Y, X).run(Neon::Backend::mainStreamIdx);
        }
        data.getBackend().sync(0);
    }

    {  // Golden data
        auto& X = data.getIODomain(FieldNames::X);
        auto& Y = data.getIODomain(FieldNames::Y);
        for (int iter = maxIters; iter > 0; iter--) {
            data.laplace(X, Y);
            data.laplace(Y, X);
        }
    }

    data.updateHostData();

    data.getField(FieldNames::X).ioToVtk("X", "X", true);
    //    data.getField(FieldNames::Y).ioToVtk("Y", "Y", false);
    //    data.getField(FieldNames::Z).ioToVtk("Z", "Z", false);
    //
    data.getIODomain(FieldNames::X).ioToVti("X_", "X_");
    //    data.getField(FieldNames::Y).ioVtiAllocator("Y_");
    //    data.getField(FieldNames::Z).ioVtiAllocator("Z_");

    bool isOk = data.compare(FieldNames::X);
    isOk = data.compare(FieldNames::Y);
    if (!isOk) {
        auto flagField = data.compareAndGetField(FieldNames::X);
        flagField.ioToVti("X_diffFlag", "X_diffFlag");
        flagField = data.compareAndGetField(FieldNames::Y);
        flagField.ioToVti("Y_diffFlag", "Y_diffFlag");
    }
    ASSERT_TRUE(isOk);
    if (!isOk) {
        exit(99);
    }
}

template auto runNoTemplate<Neon::dGrid, int64_t, 0>(TestData<Neon::dGrid, int64_t, 0>&) -> void;
template auto runNoTemplate<Neon::eGrid, int64_t, 0>(TestData<Neon::eGrid, int64_t, 0>&) -> void;
template auto runNoTemplate<Neon::bGrid, int64_t, 0>(TestData<Neon::bGrid, int64_t, 0>&) -> void;
template auto runNoTemplate<Neon::dGridSoA, int64_t, 0>(TestData<Neon::dGridSoA, int64_t, 0>&) -> void;
template auto runNoTemplate<Neon::domain::details::disaggregated::dGrid::dGrid, int64_t, 0>(TestData<Neon::domain::details::disaggregated::dGrid::dGrid, int64_t, 0>&) -> void;

template auto runTemplate<Neon::dGrid, int64_t, 0>(TestData<Neon::dGrid, int64_t, 0>&) -> void;
template auto runTemplate<Neon::eGrid, int64_t, 0>(TestData<Neon::eGrid, int64_t, 0>&) -> void;
template auto runTemplate<Neon::bGrid, int64_t, 0>(TestData<Neon::bGrid, int64_t, 0>&) -> void;
template auto runTemplate<Neon::dGridSoA, int64_t, 0>(TestData<Neon::dGridSoA, int64_t, 0>&) -> void;
template auto runTemplate<Neon::domain::details::disaggregated::dGrid::dGrid, int64_t, 0>(TestData<Neon::domain::details::disaggregated::dGrid::dGrid, int64_t, 0>&) -> void;

}  // namespace map