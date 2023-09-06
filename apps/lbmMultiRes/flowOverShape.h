#pragma once
#include "Neon/Neon.h"
#include "Neon/domain/mGrid.h"
#include "lbmMultiRes.h"

#include "init.h"

template <typename T, int Q, typename sdfT>
void initFlowOverShape(Neon::domain::mGrid&                  grid,
                       Neon::domain::mGrid::Field<float>&    sumStore,
                       Neon::domain::mGrid::Field<T>&        fin,
                       Neon::domain::mGrid::Field<T>&        fout,
                       Neon::domain::mGrid::Field<CellType>& cellType,
                       Neon::domain::mGrid::Field<T>&        vel,
                       Neon::domain::mGrid::Field<T>&        rho,
                       const Neon::double_3d                 inletVelocity,
                       const sdfT                            shapeSDF)
{

    const Neon::index_3d gridDim = grid.getDimension();

    //init fields
    for (int level = 0; level < grid.getDescriptor().getDepth(); ++level) {

        auto container =
            grid.newContainer(
                "Init_" + std::to_string(level), level,
                [&fin, &fout, &cellType, &vel, &rho, &sumStore, level, gridDim, inletVelocity, shapeSDF](Neon::set::Loader& loader) {
                    auto&   in = fin.load(loader, level, Neon::MultiResCompute::MAP);
                    auto&   out = fout.load(loader, level, Neon::MultiResCompute::MAP);
                    auto&   type = cellType.load(loader, level, Neon::MultiResCompute::MAP);
                    auto&   u = vel.load(loader, level, Neon::MultiResCompute::MAP);
                    auto&   rh = rho.load(loader, level, Neon::MultiResCompute::MAP);
                    auto&   ss = sumStore.load(loader, level, Neon::MultiResCompute::MAP);
                    const T usqr = (3.0 / 2.0) * (inletVelocity.x * inletVelocity.x + inletVelocity.y * inletVelocity.y + inletVelocity.z * inletVelocity.z);

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
                            if (idx.y == 0 || idx.y == gridDim.y - (1 << level) ||
                                idx.z == 0 || idx.z == gridDim.z - (1 << level)) {
                                type(cell, 0) = CellType::bounceBack;
                            }

                            if (idx.x == 0) {
                                type(cell, 0) = CellType::inlet;
                            }

                            if (shapeSDF(idx)) {
                                type(cell, 0) = CellType::bounceBack;
                            }

                            //population init value
                            for (int q = 0; q < Q; ++q) {
                                T pop_init_val = latticeWeights[q];

                                //bounce back
                                if (type(cell, 0) == CellType::bounceBack) {
                                    pop_init_val = 0;
                                }

                                if (type(cell, 0) == CellType::inlet) {
                                    pop_init_val = 0;
                                    for (int d = 0; d < 3; ++d) {
                                        pop_init_val += latticeVelocity[q][d] * inletVelocity.v[d];
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
void flowOverJet(const int           problemID,
                 const Neon::Backend backend,
                 const int           numIter,
                 const int           Re,
                 const bool          fineInitStore,
                 const bool          streamFusedExpl,
                 const bool          streamFusedCoal,
                 const bool          streamFuseAll,
                 const bool          collisionFusedStore,
                 const bool          benchmark,
                 const int           freq)
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

    Neon::index_3d gridDim(256, 256, 256);
    Neon::index_3d jetBoxDim(gridDim.x / 2, gridDim.y / 2, gridDim.z / 2);

    Neon::index_3d diffBox((gridDim.x - jetBoxDim.x) / 2,
                           (gridDim.y - jetBoxDim.y) / 2,
                           (gridDim.z - jetBoxDim.z) / 2);

    int depth = 2;

    const Neon::mGridDescriptor<1> descriptor(depth);

    Neon::domain::mGrid grid(
        backend, gridDim,
        {[&](const Neon::index_3d idx) -> bool {
             return idx.x >= diffBox.x && idx.x < diffBox.x + jetBoxDim.x &&
                    idx.y >= diffBox.y && idx.y < diffBox.y + jetBoxDim.y &&
                    idx.z >= diffBox.z && idx.z < diffBox.z + jetBoxDim.z;
         },
         [&](const Neon::index_3d idx) -> bool {
             return true;
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
    //test.ioToVtk("Test", true, true, true, true);
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
    initFlowOverShape<T, Q>(grid, storeSum, fin, fout, cellType, vel, rho, inletVelocity, [=] NEON_CUDA_HOST_DEVICE(Neon::index_3d idx) {
        idx.x -= diffBox.x;
        idx.y -= diffBox.y;
        idx.z -= diffBox.z;
        if (idx.x < 0 || idx.y < 0 || idx.z < 0) {
            return false;
        }

        idx.x = (jetBoxDim.x / 2) - (idx.x - (jetBoxDim.x / 2));
        return sdfJetfighter(glm::ivec3(idx.z, idx.y, idx.x), glm::ivec3(jetBoxDim.x, jetBoxDim.y, jetBoxDim.z)) <= 0;
    });

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
                           freq,
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
template <typename T, int Q>
void flowOverSphere(const int           problemID,
                    const Neon::Backend backend,
                    const int           numIter,
                    const int           Re,
                    const bool          fineInitStore,
                    const bool          streamFusedExpl,
                    const bool          streamFusedCoal,
                    const bool          streamFuseAll,
                    const bool          collisionFusedStore,
                    const bool          benchmark,
                    const int           freq)
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

    int scale = 2;

    Neon::index_3d gridDim(136 * scale, 96 * scale, 136 * scale);

    Neon::index_4d sphere(52 * scale, 52 * scale, 68 * scale, 8 * scale);

    int depth = 3;

    const Neon::mGridDescriptor<1> descriptor(depth);

    Neon::domain::mGrid grid(
        backend, gridDim,
        {[&](const Neon::index_3d idx) -> bool {
             return idx.x >= 40 * scale && idx.x < 96 * scale && idx.y >= 40 * scale && idx.y < 64 * scale && idx.z >= 40 * scale && idx.z < 96 * scale;
         },
         [&](const Neon::index_3d idx) -> bool {
             return idx.x >= 24 * scale && idx.x < 112 * scale && idx.y >= 24 * scale && idx.y < 72 * scale && idx.z >= 24 * scale && idx.z < 112 * scale;
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
    initFlowOverShape<T, Q>(grid, storeSum, fin, fout, cellType, vel, rho, inletVelocity, [sphere] NEON_CUDA_HOST_DEVICE(const Neon::index_3d idx) {
        const T dx = sphere.x - idx.x;
        const T dy = sphere.y - idx.y;
        const T dz = sphere.z - idx.z;

        if ((dx * dx + dy * dy + dz * dz) < sphere.w * sphere.w) {
            return true;
        } else {
            return false;
        }
    });

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
                           freq,
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