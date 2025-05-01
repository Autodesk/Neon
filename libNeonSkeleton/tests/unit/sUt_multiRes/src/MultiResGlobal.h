// #pragma once
//
// void MultiResGlobalIdx()
// {
//     using Type = int32_t;
//     const int              nGPUs = 1;
//     const Neon::int32_3d   dim(64, 64, 64);
//     const std::vector<int> gpusIds(nGPUs, 0);
//
//     Neon::mGridDescriptor<1> descriptor(2);
//
//     for (auto runtime : {Neon::Runtime::stream}) {
//         auto bk = Neon::Backend(gpusIds, runtime);
//         auto peel = [dim](const Neon::index_3d id, int peel_level, bool outwards = false) -> bool {
//             if (outwards) {
//                 bool isInX = (id.x <= peel_level || id.x >= dim.x - 1 - peel_level);
//                 bool isInY = (id.y <= peel_level || id.y >= dim.y - 1 - peel_level);
//                 bool isInZ = (id.z <= peel_level || id.z >= dim.z - 1 - peel_level);
//                 return isInX || isInY || isInZ;
//             }
//             bool isInX = id.x >= peel_level && id.x <= dim.x - 1 - peel_level;
//             bool isInY = id.y >= peel_level && id.y <= dim.y - 1 - peel_level;
//             bool isInZ = id.z >= peel_level && id.z <= dim.z - 1 - peel_level;
//             return isInX && isInY && isInZ;
//         };
//
//
//         Neon::domain::mGrid grid(
//             bk,
//             dim,
//             {
//                 [peel](const Neon::index_3d id) -> bool {
//                     //   return peel(id, 3, false);  //&& peel(id, 5, false);
//                     return peel(id, 6, true) ;//&& peel(id, 2, false);
//                     //return id.x < 10;
//                     //if (id.x <= 10 && id.x >= 5) {
//                     //
//                     //
//                     //
//                     //    return true;
//                     //
//                     //}
//                     //
//                     //return false;
//                 },
//                 [peel](const Neon::index_3d& id) -> bool {
//                     // return peel(id, 6, false) && peel(id, 11, true);
//                     //return id.x > 20;
//                     return true;
//                 },
//                 // [](const Neon::index_3d& id) -> bool {
//                 //     return true;
//                 //     //                 return peel(id, 7, true);
//                 // }
//
//             },
//             Neon::domain::Stencil::s7_Laplace_t(),
//             descriptor,
//             true,
//             true);
//
//         auto L1 = grid.operator()(1);
//         std::cout << "L1" << L1.toString() << std::endl;
//         L1.ioDomainToVtk("L1");
//         auto L0 = grid.operator()(0);
//         std::cout << "L0" << L0.toString() << std::endl;
//         L0.ioDomainToVtk("L0");
//
//
//         auto MultiRes_write_test = grid.newField<Type>("MultiRes_write_test", 1, -1);
//         MultiRes_write_test.operator()(1).operator()().ioToVtk("MultiRes_write_testLevel_1", "sdfsfdd", true, Neon::IoFileType::BINARY);
//         MultiRes_write_test.operator()(0).operator()().ioToVtk("MultiRes_write_testLevel_0", "sdfsfdd", true);
//
//         MultiRes_write_test.operator()(0).operator()().getGrid().newContainer<Neon::Execution::host>("NN",
//             [&MultiRes_write_test](Neon::set::Loader& loader) {
//             auto& y = loader.load(MultiRes_write_test.operator()(0).operator()(), Neon::Pattern::MAP);
//             return [=](const Neon::domain::mGrid::Idx& cell) mutable {
//                 y(cell, 0) = 33;
//             };
//         }).run(0);
//
//         // {
//         //     for (int level = 0; level < descriptor.getDepth(); ++level) {
//         //
//         //         auto container = grid.newContainer(
//         //             "SameLevelStencil", level, [&, level](Neon::set::Loader& loader) {
//         //                 auto& y = MultiRes_write_test.load(loader, level, Neon::MultiResCompute::MAP);
//         //
//         //
//         //                 return [=] NEON_CUDA_HOST_DEVICE(const Neon::domain::mGrid::Idx& cell) mutable {
//         //                     y(cell, 0) = level + 3;
//         //                 };
//         //             });
//         //
//         //         container.run(0);
//         //         grid.getBackend().syncAll();
//         //     }
//         //     MultiRes_write_test.ioToVtk("MultiRes_write_after_kernel", true, true, true, false);
//         // }
//         {  // HAS CHILDREN
//             for (int level = 1; level < descriptor.getDepth(); ++level) {
//
//                 auto container = grid.newContainer(
//                     "FlagTarget", level, [&, level](Neon::set::Loader& loader) {
//                         auto& y = MultiRes_write_test.load(loader, level, Neon::MultiResCompute::MAP);
//                         return [=] NEON_CUDA_HOST_DEVICE(const Neon::domain::mGrid::Idx& cell) mutable {
//                             auto value = level + 3;
//                             if (y.hasChildren(cell)) {
//                                 value *= -1;
//                             }
//                             y(cell, 0) = value;
//                             auto global = y.getGlobalIndex(cell);
//                             if (level == 1) {
//                                 if ((global.x == 4 && global.y == 4 && global.z == 4) || (global.x == 6 && global.y == 6 && global.z == 6)) {
//
//                                     printf("YESSSSSSS (%d,%d,%d) level %d vs %d\n",
//                                            global.x,
//                                            global.y,
//                                            global.z,
//                                            level,
//                                            1);
// #ifdef NEON_PLACE_CUDA_DEVICE
//                                     printf("neon_print - CUDA - Grid dim (%d, %d, %d) block dim (%d, %d, %d) block (%d, %d, %d) thread (%d %d %d)\n",
//                                            gridDim.x, gridDim.y, gridDim.z,
//                                            blockDim.x, blockDim.y, blockDim.z,
//                                            blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
// #endif
//
//                                     y(cell, 0) = 99;
//                                 }
//                             }
//                         };
//                     });
//
//                 container.run(0);
//                 grid.getBackend().syncAll();
//             }
//             MultiRes_write_test.ioToVtk("MultiRes_GlobalIdx", true, true, true, false);
//         }
//     }
// }
//
// TEST(MultiRes, GlobalIdx)
// {
//     if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
//         MultiResGlobalIdx();
//     }
// }