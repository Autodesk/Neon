#include "gtest/gtest.h"

#include "Neon/Neon.h"

#include "Neon/domain/bGrid.h"

#include "Neon/skeleton/Options.h"
#include "Neon/skeleton/Skeleton.h"

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}

void MultiResSingleMap()
{
    using Type = int32_t;
    const int              nGPUs = 1;
    const Neon::int32_3d   dim(24, 24, 24);
    const std::vector<int> gpusIds(nGPUs, 0);
    const Type             a = 10.0;
    const Type             XLevelVal[3] = {2, 5, 10};
    const Type             YInitVal = 1;

    const Neon::domain::internal::bGrid::bGridDescriptor descriptor({1, 1, 1});

    for (auto runtime : {Neon::Runtime::stream, Neon::Runtime::openmp}) {

        auto bk = Neon::Backend(gpusIds, runtime);

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

        auto XField = BGrid.newField<Type>("XField", 1, -1);
        auto YField = BGrid.newField<Type>("YField", 1, -1);

        //Init fields
        for (int l = 0; l < descriptor.getDepth(); ++l) {
            XField.forEachActiveCell(
                l,
                [&](const Neon::int32_3d, const int, Type& val) {
                    val = XLevelVal[l];
                });

            YField.forEachActiveCell(
                l,
                [&](const Neon::int32_3d, const int, Type& val) {
                    val = YInitVal;
                });
        }

        if (bk.runtime() == Neon::Runtime::stream) {
            XField.updateCompute();
            YField.updateCompute();
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
                [&](const Neon::int32_3d, const int, Type& val) {
                    EXPECT_EQ(val, a * XLevelVal[l] + YInitVal);
                });
        }
    }
}
TEST(MultiRes, Map)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        MultiResSingleMap();
    }
}


void MultiResSameLevelStencil()
{
    using Type = int32_t;
    const int              nGPUs = 1;
    const Neon::int32_3d   dim(24, 24, 24);
    const std::vector<int> gpusIds(nGPUs, 0);
    const Type             XVal = 42;
    const Type             YVal = 55;

    const Neon::domain::internal::bGrid::bGridDescriptor descriptor({1, 1, 1});

    for (auto runtime : {Neon::Runtime::openmp, Neon::Runtime::stream}) {

        auto bk = Neon::Backend(gpusIds, runtime);

        Neon::domain::bGrid BGrid(
            bk,
            dim,
            {[&](const Neon::index_3d) -> bool {
                 return true;
             },
             [&](const Neon::index_3d) -> bool {
                 return true;
             },
             [&](const Neon::index_3d) -> bool {
                 return true;
             }},
            Neon::domain::Stencil::s7_Laplace_t(),
            descriptor);
        //BGrid.topologyToVTK("bGrid111.vtk", false);

        auto XField = BGrid.newField<Type>("XField", 1, -1);
        auto YField = BGrid.newField<Type>("YField", 1, -1);

        //Init fields
        for (int l = 0; l < descriptor.getDepth(); ++l) {
            XField.forEachActiveCell(
                l,
                [&](const Neon::int32_3d, const int, Type& val) {
                    val = XVal;
                });

            YField.forEachActiveCell(
                l,
                [&](const Neon::int32_3d, const int, Type& val) {
                    val = YVal;
                });
        }

        if (bk.runtime() == Neon::Runtime::stream) {
            XField.updateCompute();
            YField.updateCompute();
        }
        //XField.ioToVtk("f", "f");


        for (int level = 0; level < descriptor.getDepth(); ++level) {
            XField.setCurrentLevel(level);
            YField.setCurrentLevel(level);
            BGrid.setCurrentLevel(level);

            auto container = BGrid.getContainer(
                "SameLevelStencil", [&, level](Neon::set::Loader& loader) {
                    auto& xLocal = loader.load(XField);
                    auto& yLocal = loader.load(YField);

                    return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::bGrid::Cell& cell) mutable {
                        for (int card = 0; card < xLocal.cardinality(); card++) {
                            Type res = 0;

                            for (int8_t nghIdx = 0; nghIdx < 6; ++nghIdx) {
                                auto neighbor = xLocal.nghVal(cell, nghIdx, card, Type(0));
                                res += neighbor.value;
                            }
                            yLocal(cell, card) = res;
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
                [&](const Neon::int32_3d idx, const int, Type& val) {
                    EXPECT_NE(val, YVal);
                    Type TrueVal = 6 * XVal;
                    for (int i = 0; i < 3; ++i) {
                        if (idx.v[i] + BGrid.getDescriptor().getSpacing(l - 1) >= dim.v[i]) {
                            TrueVal -= XVal;
                        }

                        if (idx.v[i] - BGrid.getDescriptor().getSpacing(l - 1) < 0) {
                            TrueVal -= XVal;
                        }
                    }

                    EXPECT_EQ(val, TrueVal);
                });
        }
    }
}
TEST(MultiRes, Stencil)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        MultiResSameLevelStencil();
    }
}

void MultiResParent()
{
    using Type = int32_t;
    const int              nGPUs = 1;
    const Neon::int32_3d   dim(24, 24, 24);
    const std::vector<int> gpusIds(nGPUs, 0);

    const Neon::domain::internal::bGrid::bGridDescriptor descriptor({1, 1, 1});

    for (auto runtime : {Neon::Runtime::stream, Neon::Runtime::openmp}) {

        auto bk = Neon::Backend(gpusIds, runtime);

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

        auto XField = BGrid.newField<Type>("XField", 1, -1);
        auto hasParentField = BGrid.newField<Type>("hasParent", 1, -1);

        //Init fields
        for (int l = 0; l < descriptor.getDepth(); ++l) {
            XField.forEachActiveCell(
                l,
                [&](const Neon::int32_3d, const int, Type& val) {
                    val = l;
                });
            hasParentField.forEachActiveCell(
                l,
                [&](const Neon::int32_3d, const int, Type& val) {
                    val = -1;
                });
        }

        if (bk.runtime() == Neon::Runtime::stream) {
            XField.updateCompute();
            hasParentField.updateCompute();
        }
        //XField.ioToVtk("f", "f");


        for (int level = 0; level < descriptor.getDepth(); ++level) {
            XField.setCurrentLevel(level);
            hasParentField.setCurrentLevel(level);
            BGrid.setCurrentLevel(level);

            auto container = BGrid.getContainer(
                "Parent", [&, level](Neon::set::Loader& loader) {
                    auto& xLocal = loader.load(XField);
                    auto& hasParentLocal = loader.load(hasParentField);

                    return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::bGrid::Cell& cell) mutable {
                        if (xLocal.hasParent(cell)) {
                            hasParentLocal(cell, 0) = 1;
                            xLocal(cell, 0) = xLocal.parent(cell, 0);
                        }
                    };
                });

            container.run(0);
            BGrid.getBackend().syncAll();
        }

        if (bk.runtime() == Neon::Runtime::stream) {
            XField.updateIO();
            hasParentField.updateIO();
        }


        //verify
        for (int l = 0; l < descriptor.getDepth(); ++l) {
            hasParentField.forEachActiveCell(
                l,
                [&](const Neon::int32_3d, const int, Type& val) {
                    if (l != descriptor.getDepth() - 1) {
                        EXPECT_EQ(val, 1);
                    } else {
                        EXPECT_EQ(val, -1);
                    }
                });


            XField.forEachActiveCell(
                l,
                [&](const Neon::int32_3d, const int, Type& val) {
                    if (l != descriptor.getDepth() - 1) {
                        EXPECT_EQ(val, l + 1);
                    } else {
                        EXPECT_EQ(val, l);
                    }
                });
        }
    }
}
TEST(MultiRes, Parent)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        MultiResParent();
    }
}