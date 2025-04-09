#pragma once

void MultiResUncle()
{
    using Type = int32_t;
    const int              nGPUs = 1;
    const Neon::index_3d   dim(32, 24, 24);
    const std::vector<int> gpuIds(nGPUs, 0);

    Neon::mGridDescriptor<1> descriptor(3);

    for (auto runtime : {Neon::Runtime::openmp /*, Neon::Runtime::stream*/}) {

        auto bk = Neon::Backend(gpuIds, runtime);

        int SectionX[3];
        SectionX[0] = 16;
        SectionX[1] = 24;
        SectionX[2] = 32;

        Neon::domain::mGrid grid(
            bk,
            dim,
            {[&](const Neon::index_3d id) -> bool {
                 return id.x < SectionX[0];
             },
             [&](const Neon::index_3d& id) -> bool {
                 return id.x < SectionX[1];
             },
             [&](const Neon::index_3d& id) -> bool {
                 return true;
             }},
            Neon::domain::Stencil::s7_Laplace_t(),
            descriptor, true, true);

        auto XField = grid.newField<Type>("XField", 1, -1);
        auto hasUncleField = grid.newField<Type>("hasUncle", 6, -1);
        auto hasParentField = grid.newField<Type>("hasParent", 1, -1);


        // Init fields
        for (int l = 0; l < descriptor.getDepth(); ++l) {
            XField.forEachActiveCell(
                l,
                [&](const Neon::int32_3d, const int, Type& val) {
                    val = l + 100;
                },
                false);
            XField.forEachActiveCell(
                l,
                [&](const Neon::int32_3d, const int, Type& val) {
                    val = l;
                },
                true);
            hasUncleField.forEachActiveCell(
                l,
                [&](const Neon::int32_3d, const int, Type& val) {
                    val = -1;
                });
            hasParentField.forEachActiveCell(
                l,
                [&](const Neon::int32_3d, const int, Type& val) {
                    val = -33;
                });
        }

        if (bk.runtime() == Neon::Runtime::stream) {
            XField.updateDeviceData();
            hasUncleField.updateDeviceData();
            hasParentField.updateDeviceData();
        }

        for (int level = 0; level < descriptor.getDepth(); ++level) {
            if (level != 0) {
                continue;
            }
            auto container = grid.newContainer(

                "hasUncle", level, [=, &hasUncleField, &hasParentField](Neon::set::Loader& loader) {
                    auto& hasUncleLocal = hasUncleField.load(loader, level, Neon::MultiResCompute::STENCIL_UP);

                    auto& hasParentLocal = hasParentField.load(loader, level, Neon::MultiResCompute::MAP);

                    return [=] NEON_CUDA_HOST_DEVICE(const Neon::domain::mGrid::Idx& cell) mutable {
                        if (hasUncleLocal.hasParent(cell)) {
                            hasParentLocal(cell, 0) = 1;
                        }else {
                            hasParentLocal(cell, 0) = 33;
                        }
                    };
                });

            container.run(0);
            grid.getBackend().syncAll();
        }

        if (bk.runtime() == Neon::Runtime::stream) {
            XField.updateHostData();
            hasParentField.updateHostData();
            hasUncleField.updateHostData();
        }
        XField.ioToVtk("XField", true, true, true, false);
        hasParentField.ioToVtk("hasParentField", true, true, true, false);
        hasUncleField.ioToVtk("hasUncleField", true, true, true, false);

        // // verify
        // for (int l = 0; l < descriptor.getDepth(); ++l) {
        //     isRefinedField.forEachActiveCell(
        //         l,
        //         [&](const Neon::int32_3d id, const int, Type& val) {
        //             if (l == 0) {
        //                 EXPECT_EQ(val, -1);
        //             } else {
        //                 if (id.x < SectionX[l - 1]) {
        //                     EXPECT_EQ(val, 1);
        //                 } else {
        //                     EXPECT_EQ(val, -1);
        //                 }
        //             }
        //         });
        //
        //
        //     XField.forEachActiveCell(
        //         l,
        //         [&](const Neon::int32_3d id, const int, Type& val) {
        //             if (l < descriptor.getDepth() - 1) {
        //                 const int      refFactor = descriptor.getRefFactor(l);
        //                 Neon::index_3d blockOrigin = descriptor.toBaseIndexSpace(descriptor.childToParent(id, l), l + 1);
        //                 EXPECT_EQ(val, blockOrigin.mPitch(refFactor, refFactor));
        //             }
        //         });
        // }
    }
}

TEST(MultiRes, Uncle)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        MultiResUncle();
    }
}
