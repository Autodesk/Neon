#include "gtest/gtest.h"

#include "Neon/Neon.h"

#include "Neon/domain/bGrid.h"

#include "Neon/skeleton/Options.h"
#include "Neon/skeleton/Skeleton.h"

#include "sUt_runHelper.h"

using namespace Neon::domain::tool::testing;

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}


template <typename FieldT>
auto laplace(const FieldT& x, FieldT& y) -> Neon::set::Container
{
    return x.getGrid().getContainer(
        "Laplace",
        [&](Neon::set::Loader & L) -> auto {
            auto& xLocal = L.load(x, Neon::Compute::STENCIL);
            auto& yLocal = L.load(y);

            return [=] NEON_CUDA_HOST_DEVICE(const typename FieldT::Cell& cell) mutable {
                using Type = typename FieldT::Type;
                for (int card = 0; card < xLocal.cardinality(); card++) {
                    Type res = 0;

                    auto checkNeighbor = [&res](Neon::domain::NghInfo<Type>& neighbor) {
                        if (neighbor.isValid) {
                            res += neighbor.value;
                        }
                    };

                    for (int8_t nghIdx = 0; nghIdx < 6; ++nghIdx) {
                        auto neighbor = xLocal.nghVal(cell, nghIdx, card, Type(0));
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


    for (int i = 0; i < nIterations; i++) {
        data.laplace(data.getIODomain(FieldNames::X), data.getIODomain(FieldNames::Y));
    }

    std::vector<Neon::set::Container> ops;

    ops.push_back(laplace(data.getField(FieldNames::X), data.getField(FieldNames::Y)));

    Neon::skeleton::Skeleton skl(data.getBackend());
    skl.sequence(ops, "sUt_dGridStencil");

    for (int i = 0; i < nIterations; i++) {
        skl.run();
    }
    data.getBackend().syncAll();


    ASSERT_TRUE(data.compare(FieldNames::Y));
}


template <typename FieldT>
auto axpy(const int level, const typename FieldT::Type a, const FieldT& x, FieldT& y) -> Neon::set::Container
{
    return x.getGrid().getContainer(
        "AXPY",
        [&, a, level ](Neon::set::Loader & loader) -> auto {
            auto& xLocal = loader.load(x);
            auto& yLocal = loader.load(y);

            return [=] NEON_CUDA_HOST_DEVICE(const typename FieldT::Cell& cell) mutable {
                for (int card = 0; card < xLocal.cardinality(); card++) {
                    yLocal(cell, card) = a * xLocal(cell, card) + yLocal(cell, card);
                }
            };
        });
}

void SingleMap()
{
    using Type = int32_t;
    const int              nGPUs = 1;
    const Neon::int32_3d   dim(24, 24, 24);
    const std::vector<int> gpusIds(nGPUs, 0);
    auto                   bk = Neon::Backend(gpusIds, Neon::Runtime::openmp);

    const Type a = 10.0;
    const Type XLevelVal[3] = {2, 5, 10};
    const Type YInitVal = 1;

    const Neon::domain::internal::bGrid::bGridDescriptor descriptor({1, 1, 1});

    Neon::domain::bGrid BGrid(
        bk,
        dim,
        {[&](const Neon::index_3d id) -> bool {
             return id.x < 8;
         },
         [&](const Neon::index_3d& id) -> bool {
             return id.x >= 8 && id.x < 16;
         },
         [&](const Neon::index_3d& id) -> bool {
             return id.x >= 16;
         }},
        Neon::domain::Stencil::s7_Laplace_t(),
        descriptor);
    //BGrid.topologyToVTK("bGrid111.vtk", false);

    auto XField = BGrid.newField<Type>("myField", 1, -1);
    auto YField = BGrid.newField<Type>("myField", 1, -1);

    //Init fields
    for (int l = 0; l < descriptor.getDepth(); ++l) {
        XField.forEachActiveCell(
            l,
            [&](const Neon::int32_3d /*idx*/, const int /*card*/, Type& val) {
                val = XLevelVal[l];
            });

        YField.forEachActiveCell(
            l,
            [&](const Neon::int32_3d /*idx*/, const int /*card*/, Type& val) {
                val = YInitVal;
            });
    }
    //XField.ioToVtk("f", "f");


    for (int level = 0; level < descriptor.getDepth(); ++level) {
        XField.setCurrentLevel(level);
        YField.setCurrentLevel(level);
        BGrid.setCurrentLevel(level);

        auto container = BGrid.getContainer(
            "AXPY", [&, a, level](Neon::set::Loader& loader) {
                auto& xLocal = loader.load(XField);
                auto& yLocal = loader.load(YField);

                return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::bGrid::Cell& cell) mutable {
                    for (int card = 0; card < xLocal.cardinality(); card++) {
                        yLocal(cell, card) = a * xLocal(cell, card) + yLocal(cell, card);
                    }
                };
            });

        container.run(0);
        BGrid.getBackend().syncAll();
    }

    if (bk.runtime() == Neon::Runtime::stream) {
        YField.updateIO();
    }


    //verify
    for (int l = 0; l < descriptor.getDepth(); ++l) {
        YField.forEachActiveCell(
            l,
            [&](const Neon::int32_3d /*idx*/, const int /*card*/, Type& val) {
                EXPECT_EQ(val, a * XLevelVal[l] + YInitVal);
            });
    }
}


TEST(MultiRes, Map)
{
    SingleMap();
}

//TEST(MultiRes, Stencil)
//{
//    int nGpus = 1;
//    using Grid = Neon::domain::bGrid;
//    using Type = int32_t;
//    runAllTestConfiguration<Grid, Type, 0>("bGrid_t", SingleStencil<Grid, Type, 0>, nGpus, 1);
//}