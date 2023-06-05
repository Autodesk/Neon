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

template<int stencilIdx, typename IDX, typename Field>
inline auto viaTemplate (const IDX& idx, int i, const Field& a, int& partial, int& count){
        a.template getNghData<stencil[stencilIdx].x,
                              stencil[stencilIdx].y,
                              stencil[stencilIdx].z>(idx, i,
                                                     [&](typename Field::Type const& val) {
                                                         partial += val;
                                                         count++;
                                                     });
};

template <typename Field>
auto stencilContainerLaplaceTemplate(const Field& filedA,
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

                    constexpr std::array<const Ngh3DIdx, 6> stencil{
                        Ngh3DIdx(1, 0, 0),
                        Ngh3DIdx(-1, 0, 0),
                        Ngh3DIdx(0, 1, 0),
                        Ngh3DIdx(0, -1, 0),
                        Ngh3DIdx(0, 0, 1),
                        Ngh3DIdx(0, 0, -1)};

#if 0
                    auto viaTemplate = [&]<int stencilIdx>() {
                        if constexpr (std::is_same_v<typename Field::Grid, Neon::dGrid>) {
                            a.template getNghData<stencil[stencilIdx].x,
                                                  stencil[stencilIdx].y,
                                                  stencil[stencilIdx].z>(idx, i,
                                                                         [&](Field::Type const& val) {
                                                                             partial += val;
                                                                             count++;
                                                                         });
                        }
                    };
#endif
                    viaTemplate<0>(idx, i, a, partial, count);
                    viaTemplate<1>(idx, i, a, partial, count);
                    viaTemplate<2>(idx, i, a, partial, count);
                    viaTemplate<3>(idx, i, a, partial, count);
                    viaTemplate<4>(idx, i, a, partial, count);
                    viaTemplate<5>(idx, i, a, partial, count);

                    b(idx, i) = a(idx, i) - count * partial;
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
            stencilContainer_laplace(X, Y).run(Neon::Backend::mainStreamIdx);

            bk.sync(Neon::Backend::mainStreamIdx);
            Y.newHaloUpdate(Neon::set::StencilSemantic::standard,
                            Neon::set::TransferMode::get,
                            Neon::Execution::device)
                .run(Neon::Backend::mainStreamIdx);

            bk.sync(Neon::Backend::mainStreamIdx);
            stencilContainer_laplace(Y, X).run(Neon::Backend::mainStreamIdx);
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

template auto run<Neon::dGrid, int64_t, 0>(TestData<Neon::dGrid, int64_t, 0>&) -> void;
template auto run<Neon::eGrid, int64_t, 0>(TestData<Neon::eGrid, int64_t, 0>&) -> void;
template auto run<Neon::bGrid, int64_t, 0>(TestData<Neon::bGrid, int64_t, 0>&) -> void;


}  // namespace map