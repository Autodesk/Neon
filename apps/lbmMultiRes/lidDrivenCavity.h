#pragma once
#include "Neon/Neon.h"
#include "Neon/domain/mGrid.h"
#include "lbmMultiRes.h"

#include "init.h"

template <typename T, int Q>
void initLidDrivenCavity(Neon::domain::mGrid&                  grid,
                         Neon::domain::mGrid::Field<float>&    sumStore,
                         Neon::domain::mGrid::Field<T>&        fin,
                         Neon::domain::mGrid::Field<T>&        fout,
                         Neon::domain::mGrid::Field<CellType>& cellType,
                         Neon::domain::mGrid::Field<T>&        vel,
                         Neon::domain::mGrid::Field<T>&        rho,
                         const Neon::double_3d                 ulid)
{
    const Neon::index_3d gridDim = grid.getDimension();

    //init fields
    for (int level = 0; level < grid.getDescriptor().getDepth(); ++level) {

        auto container =
            grid.newContainer(
                "Init_" + std::to_string(level), level,
                [&fin, &fout, &cellType, &vel, &rho, &sumStore, level, gridDim, ulid](Neon::set::Loader& loader) {
                    auto& in = fin.load(loader, level, Neon::MultiResCompute::MAP);
                    auto& out = fout.load(loader, level, Neon::MultiResCompute::MAP);
                    auto& type = cellType.load(loader, level, Neon::MultiResCompute::MAP);
                    auto& u = vel.load(loader, level, Neon::MultiResCompute::MAP);
                    auto& rh = rho.load(loader, level, Neon::MultiResCompute::MAP);
                    auto& ss = sumStore.load(loader, level, Neon::MultiResCompute::MAP);

                    return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::mGrid::Idx& cell) mutable {
                        //velocity and density
                        u(cell, 0) = 0;
                        u(cell, 1) = 0;
                        u(cell, 2) = 0;
                        rh(cell, 0) = 0;
                        type(cell, 0) = CellType::bulk;

                        for (int q = 0; q < Q; ++q) {
                            ss(cell, q) = 0;
                            in(cell, q) = 0;
                            out(cell, q) = 0;
                        }

                        if (!in.hasChildren(cell)) {
                            const Neon::index_3d idx = in.getGlobalIndex(cell);

                            //the cell classification
                            if (level == 0) {
                                if (idx.x == 0 || idx.x == gridDim.x - 1 ||
                                    idx.y == 0 || idx.y == gridDim.y - 1 ||
                                    idx.z == 0 || idx.z == gridDim.z - 1) {
                                    type(cell, 0) = CellType::bounceBack;

                                    if (idx.y == gridDim.y - 1) {
                                        type(cell, 0) = CellType::movingWall;
                                    }
                                }
                            }

                            //population init value
                            for (int q = 0; q < Q; ++q) {
                                T pop_init_val = latticeWeights[q];

                                //bounce back
                                if (type(cell, 0) == CellType::bounceBack) {
                                    pop_init_val = 0;
                                }

                                //moving wall
                                if (type(cell, 0) == CellType::movingWall) {
                                    pop_init_val = 0;
                                    for (int d = 0; d < 3; ++d) {
                                        pop_init_val += latticeVelocity[q][d] * ulid.v[d];
                                    }
                                    pop_init_val *= -6. * latticeWeights[q];
                                }

                                out(cell, q) = pop_init_val;
                                in(cell, q) = pop_init_val;
                            }
                        }
                    };
                });

        container.run(0);
    }


    //init sumStore
    initSumStore<T, Q>(grid, sumStore);
}

template <typename T, int Q>
void lidDrivenCavity(const Neon::Backend backend,
                     Params&             params)
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


    if (params.scale == 0) {
        depth = 3;
        levelSDF.resize(depth + 1);
        gridDim = Neon::index_3d(48, 48, 48);
        levelSDF[0] = 0;
        levelSDF[1] = -8.0 / 24.0;
        levelSDF[2] = -16.0 / 24.0;
        levelSDF[3] = -1.0;
    } else if (params.scale == 1) {
        depth = 3;
        levelSDF.resize(depth + 1);
        gridDim = Neon::index_3d(160, 160, 160);
        levelSDF[0] = 0;
        levelSDF[1] = -16.0 / 80.0;
        levelSDF[2] = -32.0 / 80.0;
        levelSDF[3] = -1.0;
    } else if (params.scale == 2) {
        depth = 3;
        levelSDF.resize(depth + 1);
        gridDim = Neon::index_3d(240, 240, 240);
        levelSDF[0] = 0;
        levelSDF[1] = -24.0 / 120.0;
        levelSDF[2] = -80.0 / 120.0;
        levelSDF[3] = -1.0;
    } else if (params.scale == 3) {
        depth = 3;
        levelSDF.resize(depth + 1);
        gridDim = Neon::index_3d(320, 320, 320);
        levelSDF[0] = 0;
        levelSDF[1] = -32.0 / 160.0;
        levelSDF[2] = -64.0 / 160.0;
        levelSDF[3] = -1.0;
    } else if (params.scale == 4) {
        depth = 3;
        levelSDF.resize(depth + 1);
        gridDim = Neon::index_3d(480, 480, 480);
        levelSDF[0] = 0;
        levelSDF[1] = -48.0 / 240.0;
        levelSDF[2] = -96.0 / 240.0;
        levelSDF[3] = -1.0;
    } else if (params.scale == 5) {
        depth = 3;
        levelSDF.resize(depth + 1);
        gridDim = Neon::index_3d(512, 512, 512);
        levelSDF[0] = 0;
        levelSDF[1] = -64.0 / 256.0;
        levelSDF[2] = -112.0 / 256.0;
        levelSDF[3] = -1.0;
    } else if (params.scale == 6) {
        depth = 4;
        levelSDF.resize(depth + 1);
        gridDim = Neon::index_3d(160, 160, 160);
        levelSDF[0] = 0;
        levelSDF[1] = -32.0 / 160.0;
        levelSDF[2] = -64.0 / 160.0;
        levelSDF[3] = -128.0 / 160.0;
        levelSDF[4] = -1.0;
    } else if (params.scale == 7) {
        depth = 4;
        levelSDF.resize(depth + 1);
        gridDim = Neon::index_3d(240, 240, 240);
        levelSDF[0] = 0;
        levelSDF[1] = -32.0 / 120.0;
        levelSDF[2] = -80.0 / 120.0;
        levelSDF[3] = -112.0 / 120.0;
        levelSDF[4] = -1.0;
    } else if (params.scale == 8) {
        depth = 4;
        levelSDF.resize(depth + 1);
        gridDim = Neon::index_3d(320, 320, 320);
        levelSDF[0] = 0;
        levelSDF[1] = -32.0 / 160.0;
        levelSDF[2] = -64.0 / 160.0;
        levelSDF[3] = -112.0 / 160.0;
        levelSDF[4] = -1.0;
    } else if (params.scale == 9) {
        depth = 4;
        levelSDF.resize(depth + 1);
        gridDim = Neon::index_3d(480, 480, 480);
        levelSDF[0] = 0;
        levelSDF[1] = -48.0 / 240.0;
        levelSDF[2] = -96.0 / 240.0;
        levelSDF[3] = -160.0 / 240.0;
        levelSDF[4] = -1.0;
    } /*else if (params.scale == 10) {
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
    const T               visclb = ulb * clength / static_cast<T>(params.Re);
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
    initLidDrivenCavity<T, Q>(grid, storeSum, fin, fout, cellType, vel, rho, ulid);

    runNonUniformLBM<T, Q>(grid,
                           params,
                           clength,
                           omega,
                           visclb,
                           ulid,
                           cellType,
                           storeSum,
                           fin,
                           fout,
                           vel,
                           rho);

    verifyLidDrivenCavity<T>(grid,
                             depth,
                             vel,
                             params.Re,
                             params.numIter,
                             ulb);
}