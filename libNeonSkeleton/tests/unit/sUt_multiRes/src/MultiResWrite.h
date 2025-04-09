#pragma once

void MultiResParent()
{
    using Type = int32_t;
    const int              nGPUs = 1;
    const Neon::int32_3d   dim(64, 64, 64);
    const std::vector<int> gpusIds(nGPUs, 0);

    Neon::mGridDescriptor<1> descriptor(3);

    for (auto runtime : {Neon::Runtime::openmp /*, Neon::Runtime::stream*/}) {
        auto bk = Neon::Backend(gpusIds, runtime);
        auto peel = [dim](const Neon::index_3d id, int peel_level, bool outwards = false) -> bool {
            if (outwards) {
                bool isInX = (id.x <= peel_level || id.x >= dim.x - 1 - peel_level);
                bool isInY = (id.y <= peel_level || id.y >= dim.y - 1 - peel_level);
                bool isInZ = (id.z <= peel_level || id.z >= dim.z - 1 - peel_level);
                return isInX || isInY || isInZ;
            }
            bool isInX = id.x >= peel_level && id.x <= dim.x - 1 - peel_level;
            bool isInY = id.y >= peel_level && id.y <= dim.y - 1 - peel_level;
            bool isInZ = id.z >= peel_level && id.z <= dim.z - 1 - peel_level;
            return isInX && isInY && isInZ;
        };


        Neon::domain::mGrid grid(
            bk,
            dim,
            {[peel](const Neon::index_3d id) -> bool {
                 //   return peel(id, 3, false);  //&& peel(id, 5, false);
                 return peel(id, 3, true);
             },
             [peel](const Neon::index_3d& id) -> bool {
                 return peel(id, 6, false) && peel(id, 11, true);
             },
             [](const Neon::index_3d& id) -> bool {
                 return true;
                 //                 return peel(id, 7, true);
             }},
            Neon::domain::Stencil::s7_Laplace_t(),
            descriptor,
            true,
            true);

        auto MultiRes_write_test = grid.newField<Type>("MultiRes_write_test", 1, -1);
        MultiRes_write_test.ioToVtk("MultiRes_write_test", true, true, true, false);
        for (int level = 0; level < descriptor.getDepth(); ++level) {

            auto container = grid.newContainer(
                "SameLevelStencil", level, [&, level](Neon::set::Loader& loader) {
                    auto&       y = MultiRes_write_test.load(loader, level, Neon::MultiResCompute::MAP);


                    return [=] NEON_CUDA_HOST_DEVICE(const Neon::domain::mGrid::Idx& cell) mutable {
                            y(cell, 0) = level;
                    };
                });

            container.run(0);
            grid.getBackend().syncAll();
        }
        MultiRes_write_test.ioToVtk("MultiRes_write_after_kernel", true, true, true, false);
    }
}

TEST(MultiRes, Write)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        MultiResParent();
    }
}