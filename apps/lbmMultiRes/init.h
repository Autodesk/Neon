#pragma once
#include "Neon/Neon.h"
#include "Neon/domain/mGrid.h"

#include "lattice.h"


template <typename T, int Q>
void initSumStore(Neon::domain::mGrid&               grid,
                  Neon::domain::mGrid::Field<float>& sumStore)
{

    //init sumStore
    for (int level = 0; level < grid.getDescriptor().getDepth() - 1; ++level) {

        auto container =
            grid.newContainer(
                "InitSumStore_" + std::to_string(level), level,
                [&sumStore, level](Neon::set::Loader& loader) {
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


                                if (!ss.isActive(cn)) {


                                    const auto uncle = ss.getUncle(cell, uncleDir);
                                    if (uncle.isActive()) {

                                        //locate the coarse cell where we should store this cell info
                                        const Neon::int8_3d CsDir = uncleDir - qDir;

                                        const auto cs = ss.getUncle(cell, CsDir);

                                        if (cs.isActive()) {

#ifdef NEON_PLACE_CUDA_DEVICE
                                            atomicAdd(&ss.uncleVal(cell, CsDir, q), 1.f);
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


    for (int level = 1; level < grid.getDescriptor().getDepth(); ++level) {

        auto container =
            grid.newContainer(
                "ReciprocalSumStore_" + std::to_string(level), level,
                [&sumStore, level](Neon::set::Loader& loader) {
                    auto& ss = sumStore.load(loader, level, Neon::MultiResCompute::MAP);

                    return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::mGrid::Idx& cell) mutable {
                        //const T refFactor = ss.getRefFactor(level);
                        constexpr T refFactor = 2.0;
                        for (int8_t q = 0; q < Q; ++q) {
                            if (ss(cell, q) > 0.001) {
                                ss(cell, q) = 1.f / (refFactor * ss(cell, q));
                            }
                        }
                    };
                });

        container.run(0);
    }

    grid.getBackend().syncAll();
}


template <typename T>
std::vector<uint32_t>
countActiveVoxels(Neon::domain::mGrid&           grid,
                  Neon::domain::mGrid::Field<T>& fin)
{

    uint32_t*      dNumActiveVoxels = nullptr;
    const uint32_t depth = grid.getDescriptor().getDepth();
    const size_t   numBytes = depth * sizeof(uint32_t);

    std::vector<uint32_t> ret(depth);

    if (grid(0).getBackend().runtime() == Neon::Runtime::stream) {
        cudaMalloc((void**)&dNumActiveVoxels, numBytes);
        cudaMemset(dNumActiveVoxels, 0, numBytes);
    } else {
        dNumActiveVoxels = ret.data();
    }

    const Neon::index_3d gridDim = grid.getDimension();

    for (int level = 0; level < depth; ++level) {

        auto container =
            grid.newContainer(
                "CountActiveVoxels" + std::to_string(level), level,
                [&fin, level, gridDim, dNumActiveVoxels](Neon::set::Loader& loader) {
                    auto& in = fin.load(loader, level, Neon::MultiResCompute::MAP);

                    return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::mGrid::Idx& cell) mutable {
                        if (!in.hasChildren(cell)) {
#ifdef NEON_PLACE_CUDA_DEVICE
                            atomicAdd(dNumActiveVoxels + level, 1);
#else
#pragma omp atomic
                            dNumActiveVoxels[level] += 1;
#endif
                        }
                    };
                });

        container.run(0);
    }


    if (grid(0).getBackend().runtime() == Neon::Runtime::stream) {
        cudaMemcpy(ret.data(), dNumActiveVoxels, numBytes, cudaMemcpyDeviceToHost);
        cudaFree(dNumActiveVoxels);
    }

    return ret;
}
