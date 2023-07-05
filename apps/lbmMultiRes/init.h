#pragma once
#include "Neon/Neon.h"
#include "Neon/domain/mGrid.h"

#include "lattice.h"


template <typename T, int Q>
uint32_t init(Neon::domain::mGrid&                  grid,
              Neon::domain::mGrid::Field<int>&      sumStore,
              Neon::domain::mGrid::Field<T>&        fin,
              Neon::domain::mGrid::Field<T>&        fout,
              Neon::domain::mGrid::Field<CellType>& cellType,
              Neon::domain::mGrid::Field<T>&        vel,
              Neon::domain::mGrid::Field<T>&        rho,
              const Neon::double_3d                 ulid)
{
    uint32_t* dNumActiveVoxels = nullptr;

    if (grid(0).getBackend().runtime() == Neon::Runtime::stream) {
        cudaMalloc((void**)&dNumActiveVoxels, sizeof(uint32_t));
        cudaMemset(dNumActiveVoxels, 0, sizeof(uint32_t));
    } else {
        dNumActiveVoxels = (uint32_t*)malloc(sizeof(uint32_t));
    }

    const Neon::index_3d gridDim = grid.getDimension();

    //init fields
    for (int level = 0; level < grid.getDescriptor().getDepth(); ++level) {

        auto container =
            grid.newContainer(
                "Init_" + std::to_string(level), level,
                [&fin, &fout, &cellType, &vel, &rho, &sumStore, level, gridDim, ulid, dNumActiveVoxels](Neon::set::Loader& loader) {
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

#ifdef NEON_PLACE_CUDA_DEVICE
                        atomicAdd(dNumActiveVoxels, 1);
#else
#pragma omp atomic
                        dNumActiveVoxels[0] += 1;
#endif

                        if (!in.hasChildren(cell)) {
                            const Neon::index_3d idx = in.getGlobalIndex(cell);

                            //pop
                            for (int q = 0; q < Q; ++q) {
                                T pop_init_val = latticeWeights[q];

                                if (level == 0) {
                                    if (idx.x == 0 || idx.x == gridDim.x - 1 ||
                                        idx.y == 0 || idx.y == gridDim.y - 1 ||
                                        idx.z == 0 || idx.z == gridDim.z - 1) {
                                        type(cell, 0) = CellType::bounceBack;

                                        if (idx.y == gridDim.y - 1) {
                                            type(cell, 0) = CellType::movingWall;
                                            pop_init_val = 0;
                                            for (int d = 0; d < 3; ++d) {
                                                pop_init_val += latticeVelocity[q][d] * ulid.v[d];
                                            }
                                            pop_init_val *= -6. * latticeWeights[q];
                                        } else {
                                            pop_init_val = 0;
                                        }
                                    }
                                }

                                out(cell, q) = pop_init_val;
                                in(cell, q) = pop_init_val;
                            }
                        } else {
                            in(cell, 0) = 0;
                            out(cell, 0) = 0;
                        }
                    };
                });

        container.run(0);
    }


    //init sumStore
    for (int level = 0; level < grid.getDescriptor().getDepth() - 1; ++level) {

        auto container =
            grid.newContainer(
                "InitSumStore_" + std::to_string(level), level,
                [&sumStore, level, gridDim](Neon::set::Loader& loader) {
                    auto& ss = sumStore.load(loader, level, Neon::MultiResCompute::STENCIL_UP);

                    return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::mGrid::Idx& cell) mutable {
                        if (ss.hasParent(cell)) {

                            for (int8_t q = 0; q < Q; ++q) {
                                const Neon::int8_3d qDir = getDir(q);
                                if (qDir.x == 0 && qDir.y == 0 && qDir.z == 0) {
                                    continue;
                                }

                                const Neon::int8_3d uncleDir = uncleOffset(cell.mInDataBlockIdx, qDir);

                                const auto cn = ss.helpGetNghIdx(cell, uncleDir);

                                if (!cn.isActive()) {

                                    const auto uncle = ss.getUncle(cell, uncleDir);
                                    if (uncle.isActive()) {

                                        //locate the coarse cell where we should store this cell info
                                        const Neon::int8_3d CsDir = uncleDir - qDir;

                                        const auto cs = ss.getUncle(cell, CsDir);

                                        if (cs.isActive()) {

#ifdef NEON_PLACE_CUDA_DEVICE
                                            atomicAdd(&ss.uncleVal(cell, CsDir, q), int(1));
#else
#pragma omp atomic
                                            ss.uncleVal(cell, CsDir, q) += 1;
#endif
                                        }
                                    }
                                }
                            }
                        }
                    };
                });

        container.run(0);
    }


    grid.getBackend().syncAll();

    uint32_t hNumActiveVoxels = 0;
    if (grid(0).getBackend().runtime() == Neon::Runtime::stream) {
        cudaMemcpy(&hNumActiveVoxels, dNumActiveVoxels, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaFree(dNumActiveVoxels);
    } else {
        hNumActiveVoxels = dNumActiveVoxels[0];
    }

    return hNumActiveVoxels;
}