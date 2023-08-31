#pragma once

#include "Neon/domain/mGrid.h"
#include "Neon/skeleton/Skeleton.h"

#include "verify.h"

template <typename T, int Q>
void postProcess(Neon::domain::mGrid&                        grid,
                 const int                                   numLevels,
                 const Neon::domain::mGrid::Field<T>&        fpop,
                 const Neon::domain::mGrid::Field<CellType>& cellType,
                 const int                                   iteration,
                 Neon::domain::mGrid::Field<T>&              vel,
                 Neon::domain::mGrid::Field<T>&              rho,
                 bool                                        outputFile)
{
    grid.getBackend().syncAll();

    for (int level = 0; level < numLevels; ++level) {
        auto container =
            grid.newContainer(
                "postProcess_" + std::to_string(level), level,
                [&, level](Neon::set::Loader& loader) {
                    const auto& pop = fpop.load(loader, level, Neon::MultiResCompute::STENCIL);
                    const auto& type = cellType.load(loader, level, Neon::MultiResCompute::MAP);
                    auto&       u = vel.load(loader, level, Neon::MultiResCompute::MAP);
                    auto&       rh = rho.load(loader, level, Neon::MultiResCompute::MAP);


                    return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::mGrid::Idx& cell) mutable {
                        if (!pop.hasChildren(cell)) {
                            if (type(cell, 0) == CellType::bulk) {

                                //fin
                                T ins[Q];
                                for (int i = 0; i < Q; ++i) {
                                    ins[i] = pop(cell, i);
                                }

                                //density
                                T r = 0;
                                for (int i = 0; i < Q; ++i) {
                                    r += ins[i];
                                }
                                rh(cell, 0) = r;

                                //velocity
                                const Neon::Vec_3d<T> vel = velocity<T, Q>(ins, r);

                                u(cell, 0) = vel.v[0];
                                u(cell, 1) = vel.v[1];
                                u(cell, 2) = vel.v[2];
                            }
                            if (type(cell, 0) == CellType::movingWall || type(cell, 0) == CellType::inlet) {
                                rh(cell, 0) = 1.0;

                                for (int q = 0; q < Q; ++q) {
                                    for (int d = 0; d < 3; ++d) {
                                        int d1 = (d + 1) % 3;
                                        int d2 = (d + 2) % 3;
                                        if (latticeVelocity[q][d] == -1 &&
                                            latticeVelocity[q][d1] == 0 &&
                                            latticeVelocity[q][d2] == 0) {
                                            u(cell, d) = pop(cell, q) / (6. * latticeWeights[q]);
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

    
    vel.updateHostData();
    //rho.updateHostData();

    if (outputFile) {
        int                precision = 4;
        std::ostringstream suffix;
        suffix << std::setw(precision) << std::setfill('0') << iteration;

        vel.ioToVtk("Velocity_" + suffix.str(), true, true, true, true, {1, 1, 1});
        //rho.ioToVtk("Density_" + suffix.str());
    }
}


template <typename T>
void verifyLidDrivenCavity(Neon::domain::mGrid&           grid,
                           const int                      numLevels,
                           Neon::domain::mGrid::Field<T>& vel,
                           const int                      Re,
                           const int                      iteration,
                           const T                        ulb)
{
    int                precision = 4;
    std::ostringstream suffix;
    suffix << std::setw(precision) << std::setfill('0') << iteration;

    std::vector<std::pair<T, T>> xPosVal;
    std::vector<std::pair<T, T>> yPosVal;

    const Neon::index_3d grid_dim = grid.getDimension();

    const T scale = 1.0 / ulb;

    for (int level = 0; level < numLevels; ++level) {
        vel.forEachActiveCell(
            level, [&](const Neon::index_3d& id, const int& card, T& val) {
                if (id.x == grid_dim.x / 2 && id.z == grid_dim.z / 2) {
                    if (card == 0) {
                        yPosVal.push_back({static_cast<double>(id.v[1]) / static_cast<double>(grid_dim.y), val * scale});
                    }
                }

                if (id.y == grid_dim.y / 2 && id.z == grid_dim.z / 2) {
                    if (card == 1) {
                        xPosVal.push_back({static_cast<double>(id.v[0]) / static_cast<double>(grid_dim.x), val * scale});
                    }
                }
            },
            Neon::computeMode_t::seq);
    }
    //sort the position so the linear interpolation works
    std::sort(xPosVal.begin(), xPosVal.end(), [=](std::pair<T, T>& a, std::pair<T, T>& b) {
        return a.first < b.first;
    });

    std::sort(yPosVal.begin(), yPosVal.end(), [=](std::pair<T, T>& a, std::pair<T, T>& b) {
        return a.first < b.first;
    });


    NEON_INFO("Max difference = {0:.8f}", verifyGhia1982(Re, xPosVal, yPosVal));


    auto writeToFile = [](const std::vector<std::pair<T, T>>& posVal, std::string filename) {
        std::ofstream file;
        file.open(filename);
        for (auto v : posVal) {
            file << v.first << " " << v.second << "\n";
        }
        file.close();
    };
    writeToFile(yPosVal, "NeonMultiResLBM_" + suffix.str() + "_Y.dat");
    writeToFile(xPosVal, "NeonMultiResLBM_" + suffix.str() + "_X.dat");
}


inline void generatepalabosDATFile(const std::string         filename,
                                   const Neon::index_3d      gridDim,
                                   const int                 depth,
                                   const std::vector<float>& levelSDF)
{
    std::ofstream file;
    file.open(filename);

    float delta = 0.5;

    Neon::index_3d gridDimFull(gridDim.x / delta, gridDim.y / delta, gridDim.z / delta);

    //a cuboid coordinates specified by it cartesian extents x0, x1, y0, y1, z0, z1
    file << 0 << " " << gridDim.x - 1 << " "
         << 0 << " " << gridDim.y - 1 << " "
         << 0 << " " << gridDim.z - 1 << "\n";

    //dx: the finest voxel size
    file << delta << "\n";

    //nx, ny and nz representing the number of finest voxels along each dimension of the cuboid.
    file << gridDimFull.x << " "
         << gridDimFull.y << " "
         << gridDimFull.z << "\n";

    for (int32_t k = 0; k < gridDimFull.z; ++k) {
        for (int32_t j = 0; j < gridDimFull.y; ++j) {
            for (int32_t i = 0; i < gridDimFull.x; ++i) {

                float sdf = sdfCube({i, j, k}, gridDimFull - 1);

                for (int d = 0; d < depth; ++d) {
                    if (sdf <= levelSDF[d] && sdf > levelSDF[d + 1]) {
                        file << float(depth - d - 1) / float(depth - 1) << " ";
                        break;
                    }
                }
            }
            file << "\n";
        }
        file << "\n";
    }


    file.close();
}