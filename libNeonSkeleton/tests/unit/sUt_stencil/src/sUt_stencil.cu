#include "gtest/gtest.h"

#include "Neon/Neon.h"

#include "Neon/core/types/chrono.h"

#include "Neon/domain/dGrid.h"
#include "Neon/domain/tools/Geometries.h"
#include "Neon/domain/tools/TestData.h"

#include "Neon/skeleton/Options.h"
#include "Neon/skeleton/Skeleton.h"

#include "sUt.runHelper.h"

using namespace Neon::domain::tool::testing;

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}


template <typename FieldT>
auto laplace(const FieldT& x, FieldT& y, bool use_relative_ids) -> Neon::set::Container
{
    return x.getGrid().getContainer(
        "Laplace",
        [&, use_relative_ids ](Neon::set::Loader & L) -> auto {
            auto& xLocal = L.load(x, Neon::Compute::STENCIL);
            auto& yLocal = L.load(y);

            return [=] NEON_CUDA_HOST_DEVICE(const typename FieldT::Cell& cell) mutable {
                using Type = typename FieldT::Type;
                for (int card = 0; card < xLocal.cardinality(); card++) {
                    typename FieldT::Type res = 0;


                    auto checkNeighbor = [&res](Neon::domain::NghInfo<Type>& neighbor) {
                        if (neighbor.isValid) {
                            res += neighbor.value;
                        }
                    };

                    if (use_relative_ids) {
                        for (int8_t nghIdx = 0; nghIdx < 6; ++nghIdx) {
                            auto neighbor = xLocal.nghVal(cell, nghIdx, card, Type(0));
                            checkNeighbor(neighbor);
                        }
                    } else {

                        typename FieldT::Partition::nghIdx_t ngh(0, 0, 0);

                        //+x
                        ngh.x = 1;
                        ngh.y = 0;
                        ngh.z = 0;
                        auto neighbor = xLocal.nghVal(cell, ngh, card, Type(0));
                        checkNeighbor(neighbor);

                        //-x
                        ngh.x = -1;
                        ngh.y = 0;
                        ngh.z = 0;
                        neighbor = xLocal.nghVal(cell, ngh, card, Type(0));
                        checkNeighbor(neighbor);

                        //+y
                        ngh.x = 0;
                        ngh.y = 1;
                        ngh.z = 0;
                        neighbor = xLocal.nghVal(cell, ngh, card, Type(0));
                        checkNeighbor(neighbor);

                        //-y
                        ngh.x = 0;
                        ngh.y = -1;
                        ngh.z = 0;
                        neighbor = xLocal.nghVal(cell, ngh, card, Type(0));
                        checkNeighbor(neighbor);

                        //+z
                        ngh.x = 0;
                        ngh.y = 0;
                        ngh.z = 1;
                        neighbor = xLocal.nghVal(cell, ngh, card, Type(0));
                        checkNeighbor(neighbor);

                        //-z
                        ngh.x = 0;
                        ngh.y = 0;
                        ngh.z = -1;
                        neighbor = xLocal.nghVal(cell, ngh, card, Type(0));
                        checkNeighbor(neighbor);
                    }

                    yLocal(cell, card) = -6 * res;
                }
            };
        });
}


template <typename G, typename T, int C>
void SingleStencil(TestData<G, T, C>& data)
{
    using Type = typename TestData<G, T, C>::Type;

    const int nIterations = 5;

    const T val = 89;

    data.getBackend().syncAll();

    data.resetValuesToRandom(1, 50);

    {  // Golden data
        auto& X = data.getIODomain(FieldNames::X);
        auto& Y = data.getIODomain(FieldNames::Y);

        for (int i = 0; i < nIterations; i++) {
            data.laplace(X, Y);
        }
    }

    //run the test twice; one with relative stencil index and another with direct stencil index
    for (int i = 0; i < 2; ++i) {

        auto& X = data.getField(FieldNames::X);
        auto& Y = data.getField(FieldNames::Y);

        std::vector<Neon::set::Container> ops;

        ops.push_back(laplace(X, Y, i == 0));

        Neon::skeleton::Skeleton skl(data.getBackend());
        skl.sequence(ops, "sUt_dGridStencil");

        for (int j = 0; j < nIterations; j++) {
            skl.run();
        }
        data.getBackend().syncAll();


        bool isOk = data.compare(FieldNames::X);
        isOk = isOk && data.compare(FieldNames::Y);

        ASSERT_TRUE(isOk);
    }
}


TEST(Stencil_NoOCC, dGrid)
{
    int nGpus = 1;
    using Grid = Neon::domain::dGrid;
    using Type = int32_t;
    runAllTestConfiguration<Grid, Type, 0>("dGrid_t", SingleStencil<Grid, Type, 0>, nGpus, 1);
}

TEST(DISABLED_Stencil_NoOCC, DISABLED_bGrid)
{
    int nGpus = 1;
    using Grid = Neon::domain::bGrid;
    using Type = int32_t;
    runAllTestConfiguration<Grid, Type, 0>("bGrid_t", SingleStencil<Grid, Type, 0>, nGpus, 1);
}