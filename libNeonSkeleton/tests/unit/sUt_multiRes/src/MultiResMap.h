#pragma once

void MultiResSingleMap()
{
    using Type = int32_t;
    const int              nGPUs = 1;
    const Neon::int32_3d   dim(24, 24, 24);
    const std::vector<int> gpusIds(nGPUs, 0);
    const Type             a = 10.0;
    const Type             XLevelVal[3] = {2, 5, 10};
    const Type             YInitVal = 1;

    const Neon::domain::mGridDescriptor descriptor({1, 1, 1});

    for (auto runtime : {Neon::Runtime::stream, Neon::Runtime::openmp}) {

        auto bk = Neon::Backend(gpusIds, runtime);

        Neon::domain::mGrid grid(
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
        //grid.topologyToVTK("grid111.vtk", false);

        auto XField = grid.newField<Type>("XField", 1, -1);
        auto YField = grid.newField<Type>("YField", 1, -1);

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

            auto container = grid.getContainer(
                "AXPY", level, [&, a, level](Neon::set::Loader& loader) {
                    auto& xLocal = XField.load(loader, level, Neon::MultiResCompute::MAP);
                    auto& yLocal = YField.load(loader, level, Neon::MultiResCompute::MAP);

                    return [=] NEON_CUDA_HOST_DEVICE(const Neon::domain::mGrid::Cell& cell) mutable {
                        for (int card = 0; card < xLocal.cardinality(); card++) {
                            yLocal(cell, card) = a * xLocal(cell, card) + yLocal(cell, card);
                        }
                    };
                });

            container.run(0);
            grid.getBackend().syncAll();
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