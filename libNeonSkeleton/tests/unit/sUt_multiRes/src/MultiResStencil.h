#pragma once

void MultiResSameLevelStencil()
{
    using Type = int32_t;
    const int              nGPUs = 1;
    const Neon::int32_3d   dim(24, 24, 24);
    const std::vector<int> gpusIds(nGPUs, 0);
    const Type             XVal = 42;
    const Type             YVal = 55;

    Neon::mGridDescriptor<1> descriptor(3);

    for (auto runtime : {Neon::Runtime::openmp, Neon::Runtime::stream}) {

        auto bk = Neon::Backend(gpusIds, runtime);

        Neon::domain::mGrid grid(
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

        auto XField = grid.newField<Type>("XField", 1, -1);
        auto YField = grid.newField<Type>("YField", 1, -1);

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
            XField.updateDeviceData();
            YField.updateDeviceData();
        }
        //XField.ioToVtk("f");


        for (int level = 0; level < descriptor.getDepth(); ++level) {

            auto container = grid.newContainer(
                "SameLevelStencil", level, [&, level](Neon::set::Loader& loader) {
                    const auto& xLocal = static_cast<const typename Neon::domain::mGrid::Field<Type>>(XField).load(loader, level, Neon::MultiResCompute::STENCIL);
                    auto&       yLocal = YField.load(loader, level, Neon::MultiResCompute::MAP);


                    return [=] NEON_CUDA_HOST_DEVICE(const Neon::domain::mGrid::Idx& cell) mutable {
                        for (int card = 0; card < xLocal.cardinality(); card++) {
                            Type res = 0;

                            for (int8_t nghIdx = 0; nghIdx < 6; ++nghIdx) {
                                auto neighbor = xLocal.getNghData(cell, nghIdx, card);
                                if (neighbor.mIsValid) {
                                    res += neighbor.mData;
                                }
                            }
                            yLocal(cell, card) = res;
                        }
                    };
                });

            container.run(0);
            grid.getBackend().syncAll();
        }

        if (bk.runtime() == Neon::Runtime::stream) {
            YField.updateHostData();
        }


        //verify
        for (int l = 0; l < descriptor.getDepth(); ++l) {
            YField.forEachActiveCell(
                l,
                [&](const Neon::int32_3d idx, const int, Type& val) {
                    EXPECT_NE(val, YVal);
                    Type TrueVal = 6 * XVal;
                    for (int i = 0; i < 3; ++i) {
                        if (idx.v[i] + grid.getDescriptor().getSpacing(l - 1) >= dim.v[i]) {
                            TrueVal -= XVal;
                        }

                        if (idx.v[i] - grid.getDescriptor().getSpacing(l - 1) < 0) {
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