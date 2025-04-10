// #pragma once
//
// void MultiResWrite()
// {
//     using Type = int32_t;
//     const int              nGPUs = 1;
//     const Neon::int32_3d   dim(24, 24, 24);
//     const std::vector<int> gpusIds(nGPUs, 0);
//
//     int SectionX[3];
//     SectionX[0] = 8;
//     SectionX[1] = 16;
//     SectionX[2] = 24;
//
//     Neon::mGridDescriptor<1> descriptor(3);
//
//     for (auto runtime : {Neon::Runtime::openmp, Neon::Runtime::stream}) {
//
//         auto bk = Neon::Backend(gpusIds, runtime);
//
//
//         Neon::domain::mGrid grid(
//             bk,
//             dim,
//             {[&](const Neon::index_3d id) -> bool {
//                  return id.x < SectionX[0];
//              },
//              [&](const Neon::index_3d& id) -> bool {
//                  return id.x >= SectionX[0] && id.x < SectionX[1];
//              },
//              [&](const Neon::index_3d& id) -> bool {
//                  return id.x >= SectionX[1] && id.x < SectionX[2];
//              }},
//             Neon::domain::Stencil::s7_Laplace_t(),
//             descriptor, true, false);
//
//         auto XField = grid.newField<Type>("XField", 1, -1);
//         auto hasParentField = grid.newField<Type>("hasParent", 1, -1);
//
//
//         //Init fields
//         for (int l = 0; l < descriptor.getDepth(); ++l) {
//             XField.forEachActiveCell(
//                 l,
//                 [&](const Neon::int32_3d, const int, Type& val) {
//                     val = l;
//                 },
//                 false);
//             hasParentField.forEachActiveCell(
//                 l,
//                 [&](const Neon::int32_3d, const int, Type& val) {
//                     val = -1;
//                 },
//                 false);
//         }
//
//         if (bk.runtime() == Neon::Runtime::stream) {
//             XField.updateDeviceData();
//             hasParentField.updateDeviceData();
//         }
//         XField.ioToVtk("XF", true, true, true, false);
//
//
//         // for (int level = 0; level < descriptor.getDepth(); ++level) {
//         //
//         //     auto container = grid.newContainer(
//         //         "Parent", level, [&, level](Neon::set::Loader& loader) {
//         //             auto& xLocal = XField.load(loader, level, Neon::MultiResCompute::MAP);
//         //             auto& hasParentLocal = hasParentField.load(loader, level, Neon::MultiResCompute::MAP);
//         //
//         //             return [=] NEON_CUDA_HOST_DEVICE(const Neon::domain::mGrid::Idx& cell) mutable {
//         //                 if (xLocal.hasParent(cell)) {
//         //                     hasParentLocal(cell, 0) = 1;
//         //                     xLocal(cell, 0) = xLocal.parentVal(cell, 0);
//         //                 } else {
//         //                     hasParentLocal(cell, 0) = -1;
//         //                 }
//         //             };
//         //         });
//         //
//         //     container.run(0);
//         //     grid.getBackend().syncAll();
//         // }
//         //
//         // if (bk.runtime() == Neon::Runtime::stream) {
//         //     XField.updateHostData();
//         //     hasParentField.updateHostData();
//         // }
//         // hasParentField.ioToVtk("hasParentField___", true, true, true, false);
//         //
//         //
//         // //verify
//         // for (int l = 0; l < descriptor.getDepth(); ++l) {
//         //     hasParentField.forEachActiveCell(
//         //         l,
//         //         [&](const Neon::int32_3d, const int, Type& val) {
//         //             if (l != descriptor.getDepth() - 1) {
//         //                 EXPECT_EQ(val, 1);
//         //             } else {
//         //                 EXPECT_EQ(val, -1);
//         //             }
//         //         },
//         //         false);
//         //
//         //     XField.forEachActiveCell(
//         //         l,
//         //         [&](const Neon::int32_3d id, const int, Type& val) {
//         //             if (l != descriptor.getDepth() - 1) {
//         //                 EXPECT_EQ(val, l + 1) << "l = " << l << " id = " << id;
//         //             } else {
//         //                 EXPECT_EQ(val, l) << "l = " << l << " id = " << id;
//         //             }
//         //         },
//         //         false);
//         // }
//     }
// }
// TEST(MultiRes, Write)
// {
//     if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
//         MultiResParent();
//     }
// }
