#include "gtest/gtest.h"

#include "Neon/Neon.h"

#include "Neon/core/types/chrono.h"

#include "Neon/domain/dGrid.h"
#include "Neon/domain/tools/Geometries.h"
#include "Neon/domain/tools/TestData.h"

#include "Neon/skeleton/Options.h"
#include "Neon/skeleton/Skeleton.h"

#include "runHelper.h"

using namespace Neon::domain::tool::testing;

template <typename Field>
auto laplaceOnIntegers(const Field& filedA,
                       Field&       fieldB)
    -> Neon::set::Container
{
    const auto& grid = filedA.getGrid();
    return grid.newContainer(
        "laplaceOnIntegers",
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
                    //        b = a - count * res;
                    b(idx, i) = a(idx, i) - count * partial;
                }
            };
        });
}


template <typename G, typename T, int C>
void singleStencil(TestData<G, T, C>&  data,
                   Neon::skeleton::Occ occ)
{
    using Type = typename TestData<G, T, C>::Type;

    const int nIterations = 5;

    const T val = 89;

    data.getBackend().syncAll();

    data.resetValuesToRandom(1, 50);


    {
        auto& X = data.getField(FieldNames::X);
        auto& Y = data.getField(FieldNames::Y);

        std::vector<Neon::set::Container> ops;

        ops.push_back(laplaceOnIntegers(X, Y));
        ops.push_back(laplaceOnIntegers(Y, X));

        Neon::skeleton::Skeleton skl(data.getBackend());
        Neon::skeleton::Options  opt(occ, Neon::set::TransferMode::get);
        skl.sequence(ops, "sUt_dGridStencil", opt);

        for (int j = 0; j < nIterations; j++) {
            skl.run();
        }
    }


    {  // Golden data
        auto& X = data.getIODomain(FieldNames::X);
        auto& Y = data.getIODomain(FieldNames::Y);

        for (int i = 0; i < nIterations; i++) {
            data.laplace(X, Y);
            data.laplace(Y, X);
        }
    }
    data.getBackend().syncAll();


    bool isOk = data.compare(FieldNames::X);
    isOk = isOk && data.compare(FieldNames::Y);

    ASSERT_TRUE(isOk);
}

TEST(singleStencil, dGrid)
{
    int nGpus = 1;
    using Grid = Neon::dGrid;
    using Type = int32_t;
    constexpr int C = 0;
    runAllTestConfiguration<Grid, Type, 0>("dGrid", singleStencil<Grid, Type, C>, Neon::skeleton::Occ::none, nGpus, 1);
}

TEST(singleStencil, bGridSingleGpu)
{
    int nGpus = 1;
    using Grid = Neon::bGrid;
    using Type = int32_t;
    constexpr int C = 0;
    runAllTestConfiguration<Grid, Type, 0>("bGrid", singleStencil<Grid, Type, C>, Neon::skeleton::Occ::none, nGpus, 1);
}