#pragma once

void MultiResParent()
{
    using Type = int32_t;
    const int              nGPUs = 1;
    const Neon::int32_3d   dim(24, 24, 24);
    const std::vector<int> gpusIds(nGPUs, 0);

    int SectionX[3];
    SectionX[0] = 8;
    SectionX[1] = 16;
    SectionX[2] = 24;

    const Neon::domain::mGridDescriptor descriptor({1, 1, 1});

    for (auto runtime : {Neon::Runtime::openmp, Neon::Runtime::stream}) {

        auto bk = Neon::Backend(gpusIds, runtime);


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
        //grid.topologyToVTK("grid111.vtk", false);

        auto XField = grid.newField<Type>("XField", 1, -1);
        auto hasParentField = grid.newField<Type>("hasParent", 1, -1);


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

            auto container = grid.getContainer(
                "Parent", level, [&, level](Neon::set::Loader& loader) {
                    auto& xLocal = XField.load(loader, level, Neon::MultiResCompute::MAP);
                    auto& hasParentLocal = hasParentField.load(loader, level, Neon::MultiResCompute::MAP);

                    return [=] NEON_CUDA_HOST_DEVICE(const Neon::domain::mGrid::Cell& cell) mutable {
                        if (xLocal.hasParent(cell)) {
                            hasParentLocal(cell, 0) = 1;
                            xLocal(cell, 0) = xLocal.parent(cell, 0);
                        } else {
                            hasParentLocal(cell, 0) = -1;
                        }
                    };
                });

            container.run(0);
            grid.getBackend().syncAll();
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
                [&](const Neon::int32_3d id, const int, Type& val) {
                    if (l != descriptor.getDepth() - 1) {
                        EXPECT_EQ(val, l + 1) << "l = " << l << " id = " << id;
                    } else {
                        EXPECT_EQ(val, l) << "l = " << l << " id = " << id;
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


void MultiResAtomicAddParent()
{
    using Type = int32_t;
    const int              nGPUs = 1;
    const Neon::int32_3d   dim(24, 24, 24);
    const std::vector<int> gpusIds(nGPUs, 0);

    const Neon::domain::mGridDescriptor descriptor({1, 1, 1});

    for (auto runtime : {
             Neon::Runtime::openmp,
             Neon::Runtime::stream}) {

        auto bk = Neon::Backend(gpusIds, runtime);

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
        //grid.topologyToVTK("grid111.vtk", false);

        auto XField = grid.newField<Type>("XField", 1, -1);


        //Init fields
        for (int l = 0; l < descriptor.getDepth(); ++l) {
            XField.forEachActiveCell(
                l,
                [&](const Neon::int32_3d, const int, Type& val) {
                    val = l + 1;
                });
        }

        if (bk.runtime() == Neon::Runtime::stream) {
            XField.updateCompute();
        }
        //XField.ioToVtk("f", "f");


        for (int level = 0; level < descriptor.getDepth(); ++level) {

            auto container = grid.getContainer(
                "Parent", level, [&, level](Neon::set::Loader& loader) {
                    auto& xLocal = XField.load(loader, level, Neon::MultiResCompute::MAP);

                    return [=] NEON_CUDA_HOST_DEVICE(const Neon::domain::mGrid::Cell& cell) mutable {
                        if (xLocal.hasParent(cell)) {

#ifdef NEON_PLACE_CUDA_DEVICE
                            atomicAdd(&xLocal.parent(cell, 0), xLocal(cell, 0));
#else

#pragma omp atomic
                            xLocal.parent(cell, 0) += xLocal(cell, 0);
#endif
                        }
                    };
                });

            container.run(0);
            grid.getBackend().syncAll();
        }

        if (bk.runtime() == Neon::Runtime::stream) {
            XField.updateIO();
        }


        //verify
        for (int l = 0; l < descriptor.getDepth(); ++l) {

            XField.forEachActiveCell(
                l,
                [&](const Neon::int32_3d id, const int, Type& val) {
                    if (l == 0) {
                        //the lowest/most refined level won't chance since it does not
                        //have children to write into it
                        EXPECT_EQ(val, l + 1) << "level= " << l << " id= " << id;
                    } else {
                        //only the part of this level that overlaps with Level 0 will change
                        //otherwise, it will stay the same since there is no children to write to it
                        if (id.x < SectionX[l - 1]) {
                            Type init_val = l + 1;
                            Type val_added_by_child = (l)*descriptor.getRefFactor(l) * descriptor.getRefFactor(l) * descriptor.getRefFactor(l);
                            Type val_added_by_self = 0;
                            if (l > 1) {
                                if (id.x < SectionX[l - 2]) {
                                    val_added_by_self = (l - 1) * descriptor.getRefFactor(l - 1) * descriptor.getRefFactor(l - 1) * descriptor.getRefFactor(l - 1) *
                                                        descriptor.getRefFactor(l) * descriptor.getRefFactor(l) * descriptor.getRefFactor(l);
                                }
                            }
                            EXPECT_EQ(val, init_val +                //init value on this level
                                               val_added_by_child +  //value added by all children
                                               val_added_by_self)    //value added by the level itself from previous run
                                << "level= " << l << " id= " << id << " val= " << val;
                        } else {
                            EXPECT_EQ(val, l + 1) << "level= " << l << " id= " << id << " val= " << val;
                        }
                    }
                });
        }
    }
}
TEST(MultiRes, AtomicAddParent)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        MultiResAtomicAddParent();
    }
}