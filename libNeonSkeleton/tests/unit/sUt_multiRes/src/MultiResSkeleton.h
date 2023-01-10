#pragma once
#include <assert.h>

void MultiResSkeleton()
{
    using Type = int32_t;
    const int              nGPUs = 1;
    const Neon::int32_3d   dim(64, 64, 64);
    const std::vector<int> gpusIds(nGPUs, 0);

    const Neon::domain::mGridDescriptor descriptor({1, 1, 1, 1, 1});


    for (auto runtime : {Neon::Runtime::openmp, Neon::Runtime::stream}) {

        auto bk = Neon::Backend(gpusIds, runtime);

        Neon::domain::mGrid grid(
            bk,
            dim,
            {[&](const Neon::index_3d id) -> bool {
                 return true;
             },
             [&](const Neon::index_3d& id) -> bool {
                 return true;
             },
             [&](const Neon::index_3d& id) -> bool {
                 return true;
             }},
            Neon::domain::Stencil::s7_Laplace_t(),
            descriptor);
        //grid.topologyToVTK("grid111.vtk", false);

        auto field = grid.newField<Type>("field", 3, -1);


        //Init fields
        for (int l = 0; l < descriptor.getDepth(); ++l) {
            field.forEachActiveCell(
                l,
                [&](const Neon::int32_3d, const int, Type& val) {
                    val = -1;
                });
        }

        if (bk.runtime() == Neon::Runtime::stream) {
            field.updateCompute();
        }
        //field.ioToVtk("f", "f");

        std::vector<Neon::set::Container> containers;

        //map operation at the top level
        {
            int level = descriptor.getDepth() - 1;
            containers.push_back(grid.getContainer(
                "map" + std::to_string(level),
                level,
                [&, level](Neon::set::Loader& loader) {
                    auto& local = field.load(loader, level, Neon::MultiResCompute::MAP);
                    return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::bGrid::Cell& cell) mutable {
                        Neon::index_3d global = local.mapToGlobal(cell);
                        local(cell, 0) = global.v[0];
                        local(cell, 1) = global.v[1];
                        local(cell, 2) = global.v[2];
                    };
                }));
        }

        //all other levels
        for (int level = descriptor.getDepth() - 2; level >= 0; --level) {
            containers.push_back(grid.getContainer(
                "ReadParent" + std::to_string(level),
                level,
                [&, level](Neon::set::Loader& loader) {
                    auto& local = field.load(loader, level, Neon::MultiResCompute::STENCIL_UP);
                    return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::bGrid::Cell& cell) mutable {
                        assert(local.hasParent(cell));

                        local(cell, 0) = local.parent(cell, 0);
                        local(cell, 1) = local.parent(cell, 1);
                        local(cell, 2) = local.parent(cell, 2);
                    };
                }));
        }


        Neon::skeleton::Skeleton skl(grid.getBackend());
        skl.sequence(containers, "MultiResSkeleton");
        skl.ioToDot("MultiRes");
        skl.run();

        grid.getBackend().syncAll();
        if (bk.runtime() == Neon::Runtime::stream) {
            field.updateIO();
            grid.getBackend().syncAll();
        }

        //verify
        for (int l = descriptor.getDepth() - 1; l >= 0; --l) {

            int parent_level = descriptor.getDepth() - 1;
            field.forEachActiveCell(
                l,
                [&](const Neon::int32_3d id, const int card, Type& val) {
                    if (l == descriptor.getDepth() - 1) {
                        EXPECT_EQ(val, id.v[card]);
                    } else {
                        Neon::index_3d parent = descriptor.toBaseIndexSpace(descriptor.childToParent(id, parent_level - 1), parent_level);
                        EXPECT_EQ(val, parent.v[card]) << "Level= " << l;
                    }
                });
        }
    }
}
TEST(MultiRes, Skeleton)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        MultiResSkeleton();
    }
}