#pragma once
#include "Neon/Neon.h"
#include "Neon/domain/mGrid.h"
#include "lbmMultiRes.h"

template <typename T, int Q>
void lidDrivenCavity(const int           problemID,
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

    Neon::index_3d gridDim;

    //int                depth = 2;
    //std::vector<float> levelSDF(depth + 1);
    //gridDim = Neon::index_3d(6, 6, 6);
    //levelSDF[0] = 0;
    //levelSDF[1] = -2.0 / 3.0;
    //levelSDF[2] = -1.0;


    int                depth = 3;
    std::vector<float> levelSDF(depth + 1);
    gridDim = Neon::index_3d(192, 192, 192);
    levelSDF[0] = 0;
    levelSDF[1] = -36.0 / 96.0;
    levelSDF[2] = -72.0 / 96.0;
    levelSDF[3] = -1.0;


    if (problemID == 0) {
        depth = 3;
        levelSDF.resize(depth + 1);
        gridDim = Neon::index_3d(48, 48, 48);
        levelSDF[0] = 0;
        levelSDF[1] = -8.0 / 24.0;
        levelSDF[2] = -16.0 / 24.0;
        levelSDF[3] = -1.0;
    } else if (problemID == 1) {
        depth = 3;
        levelSDF.resize(depth + 1);
        gridDim = Neon::index_3d(160, 160, 160);
        levelSDF[0] = 0;
        levelSDF[1] = -16.0 / 80.0;
        levelSDF[2] = -32.0 / 80.0;
        levelSDF[3] = -1.0;
    } else if (problemID == 2) {
        depth = 3;
        levelSDF.resize(depth + 1);
        gridDim = Neon::index_3d(240, 240, 240);
        levelSDF[0] = 0;
        levelSDF[1] = -24.0 / 120.0;
        levelSDF[2] = -80.0 / 120.0;
        levelSDF[3] = -1.0;
    } else if (problemID == 3) {
        depth = 3;
        levelSDF.resize(depth + 1);
        gridDim = Neon::index_3d(320, 320, 320);
        levelSDF[0] = 0;
        levelSDF[1] = -32.0 / 160.0;
        levelSDF[2] = -64.0 / 160.0;
        levelSDF[3] = -1.0;
    } else if (problemID == 4) {
        depth = 3;
        levelSDF.resize(depth + 1);
        gridDim = Neon::index_3d(480, 480, 480);
        levelSDF[0] = 0;
        levelSDF[1] = -48.0 / 240.0;
        levelSDF[2] = -96.0 / 240.0;
        levelSDF[3] = -1.0;
    } else if (problemID == 5) {
        depth = 3;
        levelSDF.resize(depth + 1);
        gridDim = Neon::index_3d(512, 512, 512);
        levelSDF[0] = 0;
        levelSDF[1] = -64.0 / 256.0;
        levelSDF[2] = -112.0 / 256.0;
        levelSDF[3] = -1.0;
    } else if (problemID == 6) {
        depth = 4;
        levelSDF.resize(depth + 1);
        gridDim = Neon::index_3d(160, 160, 160);
        levelSDF[0] = 0;
        levelSDF[1] = -32.0 / 160.0;
        levelSDF[2] = -64.0 / 160.0;
        levelSDF[3] = -128.0 / 160.0;
        levelSDF[4] = -1.0;
    } else if (problemID == 7) {
        depth = 4;
        levelSDF.resize(depth + 1);
        gridDim = Neon::index_3d(240, 240, 240);
        levelSDF[0] = 0;
        levelSDF[1] = -32.0 / 120.0;
        levelSDF[2] = -80.0 / 120.0;
        levelSDF[3] = -112.0 / 120.0;
        levelSDF[4] = -1.0;
    } else if (problemID == 8) {
        depth = 4;
        levelSDF.resize(depth + 1);
        gridDim = Neon::index_3d(320, 320, 320);
        levelSDF[0] = 0;
        levelSDF[1] = -32.0 / 160.0;
        levelSDF[2] = -64.0 / 160.0;
        levelSDF[3] = -112.0 / 160.0;
        levelSDF[4] = -1.0;
    } else if (problemID == 9) {
        depth = 4;
        levelSDF.resize(depth + 1);
        gridDim = Neon::index_3d(480, 480, 480);
        levelSDF[0] = 0;
        levelSDF[1] = -48.0 / 240.0;
        levelSDF[2] = -96.0 / 240.0;
        levelSDF[3] = -160.0 / 240.0;
        levelSDF[4] = -1.0;
    } /*else if (problemID == 10) {
        depth = 4;
        levelSDF.resize(depth + 1);
        gridDim = Neon::index_3d(512, 512, 512);
        levelSDF[0] = 0;
        levelSDF[1] = -103.0 / 512.0;
        levelSDF[2] = -205.0 / 512.0;
        levelSDF[3] = -359.0 / 512.0;
        levelSDF[4] = -1.0;
    }*/


    //generatepalabosDATFile(std::string("lid_" + std::to_string(gridDim.x) + "_" +
    //                                   std::to_string(gridDim.y) + "_" +
    //                                   std::to_string(gridDim.x) + ".dat"),
    //                       gridDim,
    //                       depth,
    //                       levelSDF);

    //define the grid
    const Neon::mGridDescriptor<1> descriptor(depth);

    std::vector<std::function<bool(const Neon::index_3d&)>> activeCellLambda(depth);
    for (size_t i = 0; i < depth; ++i) {
        activeCellLambda[i] = [=](const Neon::index_3d id) -> bool {
            float sdf = sdfCube(id, gridDim - 1);
            return sdf <= levelSDF[i] && sdf > levelSDF[i + 1];
        };
    }

    Neon::domain::mGrid grid(
        backend, gridDim,
        activeCellLambda,
        Neon::domain::Stencil::s19_t(false), descriptor);

    //LBM problem
    const T               ulb = 0.04;
    const T               clength = T(grid.getDimension(descriptor.getDepth() - 1).x);
    const T               visclb = ulb * clength / static_cast<T>(Re);
    const T               omega = 1.0 / (3. * visclb + 0.5);
    const Neon::double_3d ulid(ulb, 0., 0.);

    //auto test = grid.newField<T>("test", 1, 0);
    //test.ioToVtk("Test", true, true, true, false, {1, 1, 0});
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
    initLidDrivenCavity<T, Q>(grid, storeSum, fin, fout, cellType, vel, rho, ulid);

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

    verifyLidDrivenCavity<T>(grid,
                             depth,
                             vel,
                             Re,
                             numIter,
                             ulb);
}