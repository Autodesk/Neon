#pragma once
#include "MultiResXlbLattice.h"

void hasChildren_with_direction()
{
    const int edge = 14;


    using Type = int32_t;
    const int              nGPUs = 1;
    const Neon::int32_3d   dim(edge, edge, edge);
    const std::vector<int> gpusIds(nGPUs, 0);

    Neon::mGridDescriptor<1> descriptor(2);

    for (auto runtime : {Neon::Runtime::stream}) {
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
            {
                [peel](const Neon::index_3d id) -> bool {
                    //   return peel(id, 3, false);  //&& peel(id, 5, false);
                    return peel(id, 4, true);  //&& peel(id, 2, false);
                    // return id.x < 10;
                    // if (id.x <= 10 && id.x >= 5) {
                    //
                    //
                    //
                    //     return true;
                    //
                    // }
                    //
                    // return false;
                },
                [peel](const Neon::index_3d& id) -> bool {
                    // return peel(id, 6, false) && peel(id, 11, true);
                    // return id.x > 20;
                    return true;
                },
                // [](const Neon::index_3d& id) -> bool {
                //     return true;
                //     //                 return peel(id, 7, true);
                // }

            },
            xlb::getD3Q19(true),
            descriptor,
            true,
            true);


        auto L1 = grid.operator()(1);
        std::cout << "L1" << L1.toString() << std::endl;
        L1.ioDomainToVtk("L1");
        auto L0 = grid.operator()(0);
        std::cout << "L0" << L0.toString() << std::endl;
        L0.ioDomainToVtk("L0");


        auto f = grid.newField<Type>("f", 1, 0);

        {  // HAS CHILDREN
            auto lattice = xlb::getD3Q19(false);
            auto push_direction = lattice.points()[10];
            auto pull_direction = push_direction * -1;

            std::cout << "push_direction " << push_direction << std::endl;
            std::cout << "pull_direction " << pull_direction << std::endl;

            int level = 0;
                auto container = grid.newContainer<Neon::Execution::host>(
                    "FlagTarget", level, [&, level](Neon::set::Loader& loader) {
                        auto& y = f.load(loader, level, Neon::MultiResCompute::MAP);
                        return [=] NEON_CUDA_HOST_DEVICE(const Neon::domain::mGrid::Idx& cell) mutable {
                            printf("I AM HERE\n");
                            int q = 10;
                            if (y.hasChildren(cell)) {
                                y(cell, 0) = 55;
                                return;
                            }

                            if (y.hasChildren(cell, pull_direction.newType<int8_t>())) {
                                y(cell, 0) = 33;
                                printf("99999999999999999999999\n");

                            } else {
                                printf("77676765435\n");
                                y(cell, 0) = -33;
                            }
                        };
                    });

                container.run(0);
                grid.getBackend().syncAll();

            f.ioToVtk("f", true, true, true, false);
            f.ioToVtk("f_no_overlap", true, true, true, true);
        }
    }
}

TEST(MultiRes, hasChildren_with_direction)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        hasChildren_with_direction();
    }
}