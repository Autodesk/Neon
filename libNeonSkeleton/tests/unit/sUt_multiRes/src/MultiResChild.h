#pragma once

void MultiResChild()
{
    using Type = int32_t;
    const int              nGPUs = 1;
    const Neon::index_3d   dim(24, 24, 24);
    const std::vector<int> gpuIds(nGPUs, 0);

    const Neon::domain::mGridDescriptor descriptor({1, 1, 1});

    for (auto runtime : {Neon::Runtime::openmp, Neon::Runtime::stream}) {

        auto bk = Neon::Backend(gpuIds, runtime);

        int SectionX[3];
        SectionX[0] = 8;
        SectionX[1] = 16;
        SectionX[2] = 24;

        Neon::domain::mGrid grid(
            bk,
            dim,
            {[&](const Neon::index_3d id) -> bool {
                 return id.x < SectionX[0];
             },
             [&](const Neon::index_3d& id) -> bool {
                 return id.x >= SectionX[0] && id.x < SectionX[1];
             },
             [&](const Neon::index_3d& id) -> bool {
                 return id.x >= SectionX[1] && id.x < SectionX[2];
             }},
            Neon::domain::Stencil::s7_Laplace_t(),
            descriptor);
        //grid.topologyToVTK("grid112.vtk", false);

        auto XField = grid.newField<Type>("XField", 1, -1);
        auto isRefinedField = grid.newField<Type>("isRefined", 1, -1);


        //Init fields
        for (int l = 0; l < descriptor.getDepth(); ++l) {
            XField.forEachActiveCell(
                l,
                [&](const Neon::int32_3d, const int, Type& val) {
                    val = l;
                });
            isRefinedField.forEachActiveCell(
                l,
                [&](const Neon::int32_3d, const int, Type& val) {
                    val = -1;
                });
        }

        if (bk.runtime() == Neon::Runtime::stream) {
            XField.updateCompute();
            isRefinedField.updateCompute();
        }
        //XField.ioToVtk("f", "f");

        for (int level = 0; level < descriptor.getDepth(); ++level) {

            auto container = grid.getContainer(

                "hasChildren", level, [&, level, descriptor](Neon::set::Loader& loader) {
                    auto& xLocal = XField.load(loader, level, Neon::MultiResCompute::MAP);
                    auto& isRefinedLocal = isRefinedField.load(loader, level, Neon::MultiResCompute::MAP);


                    return [=] NEON_CUDA_HOST_DEVICE(const Neon::domain::mGrid::Cell& cell) mutable {
                        if (xLocal.hasChildren(cell)) {
                            isRefinedLocal(cell, 0) = 1;

                            Neon::index_3d cellOrigin = xLocal.mapToGlobal(cell);

                            const int refFactor = xLocal.getRefFactor(level - 1);

                            for (int8_t z = 0; z < refFactor; ++z) {
                                for (int8_t y = 0; y < refFactor; ++y) {
                                    for (int8_t x = 0; x < refFactor; ++x) {

                                        Neon::int8_3d child_dir(x, y, z);

                                        auto child = xLocal.getChild(cell, child_dir);

                                        if (child.isActive()) {
                                            xLocal.childVal(child) = cellOrigin.mPitch(refFactor, refFactor);
                                        }
                                    }
                                }
                            }
                        }
                    };
                });

            container.run(0);
            grid.getBackend().syncAll();
        }

        if (bk.runtime() == Neon::Runtime::stream) {
            XField.updateIO();
            isRefinedField.updateIO();
        }

        //verify
        for (int l = 0; l < descriptor.getDepth(); ++l) {
            isRefinedField.forEachActiveCell(
                l,
                [&](const Neon::int32_3d id, const int, Type& val) {
                    if (l == 0) {
                        EXPECT_EQ(val, -1);
                    } else {
                        if (id.x < SectionX[l - 1]) {
                            EXPECT_EQ(val, 1);
                        } else {
                            EXPECT_EQ(val, -1);
                        }
                    }
                });


            XField.forEachActiveCell(
                l,
                [&](const Neon::int32_3d id, const int, Type& val) {
                    if (l < descriptor.getDepth() - 1) {
                        const int      refFactor = descriptor.getRefFactor(l);
                        Neon::index_3d blockOrigin = descriptor.toBaseIndexSpace(descriptor.childToParent(id, l), l + 1);
                        EXPECT_EQ(val, blockOrigin.mPitch(refFactor, refFactor));
                    }
                });
        }
    }
}

TEST(MultiRes, Child)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        MultiResChild();
    }
}