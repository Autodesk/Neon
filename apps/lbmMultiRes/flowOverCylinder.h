#pragma once
#include "Neon/Neon.h"
#include "Neon/domain/mGrid.h"
#include "lbmMultiRes.h"

template <typename T, int Q>
void flowOverCylinder(const int           problemID,
                      const Neon::Backend backend,
                      const int           numIter,
                      const int           Re,
                      const bool          fineInitStore,
                      const bool          streamFusedExpl,
                      const bool          streamFusedCoal,
                      const bool          streamFuseAll,
                      const bool          collisionFusedStore,
                      const bool          benchmark)
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

    Neon::index_3d gridDim(136, 96, 136);

    Neon::index_4d cylinder(52, 52, 0, 8);

    int depth = 3;

    const Neon::mGridDescriptor<1> descriptor(depth);

    Neon::domain::mGrid grid(
        backend, gridDim,
        {[&](const Neon::index_3d idx) -> bool {
             return idx.x >= 40 && idx.x < 96 && idx.y >= 40 && idx.y < 64;
         },
         [&](const Neon::index_3d idx) -> bool {
             return idx.x >= 24 && idx.x < 112 && idx.y >= 24 && idx.y < 72;
         },
         [&](const Neon::index_3d idx) -> bool {
             return true;
         }},
        Neon::domain::Stencil::s19_t(false), descriptor);


    //LBM problem
    const T               uin = 0.04;
    const T               clength = T(grid.getDimension(descriptor.getDepth() - 1).x);
    const T               visclb = uin * clength / static_cast<T>(Re);
    const T               omega = 1.0 / (3. * visclb + 0.5);
    const Neon::double_3d inletVelocity(uin, 0., 0.);

    //auto test = grid.newField<T>("test", 1, 0);
    //test.ioToVtk("Test", true, true, true, false);
    //exit(0);

    //allocate fields
    auto fin = grid.newField<T>("fin", Q, 0);
    auto fout = grid.newField<T>("fout", Q, 0);
    auto storeSum = grid.newField<float>("storeSum", Q, 0);
    auto cellType = grid.newField<CellType>("CellType", 1, CellType::bulk);

    auto vel = grid.newField<T>("vel", 3, 0);
    auto rho = grid.newField<T>("rho", 1, 0);

    //init fields
    const uint32_t numActiveVoxels = countActiveVoxels(grid, fin);
    initFlowOverCylinder<T, Q>(grid, storeSum, fin, fout, cellType, vel, rho, inletVelocity, cylinder);

    //cellType.updateHostData();
    //cellType.ioToVtk("cellType", true, true, true, true);

    runNonUniformLBM<T, Q>(grid,
                           numActiveVoxels,
                           numIter,
                           Re,
                           fineInitStore,
                           streamFusedExpl,
                           streamFusedCoal,
                           streamFuseAll,
                           collisionFusedStore,
                           benchmark,
                           problemID,
                           "lid",
                           omega,
                           cellType,
                           storeSum,
                           fin,
                           fout,
                           vel,
                           rho);
}