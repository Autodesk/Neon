#pragma once

#include "Neon/domain/mGrid.h"
#include "Neon/skeleton/Skeleton.h"

#include "verify.h"

template <typename T, int Q>
void postProcess(Neon::domain::mGrid&                        grid,
                 const int                                   Re,
                 const int                                   numLevels,
                 const Neon::domain::mGrid::Field<T>&        fpop,
                 const Neon::domain::mGrid::Field<CellType>& cellType,
                 const int                                   iteration,
                 Neon::domain::mGrid::Field<T>&              vel,
                 Neon::domain::mGrid::Field<T>&              rho,
                 T                                           ulb,
                 bool                                        verify,
                 bool                                        generateValidateFile)
{
    grid.getBackend().syncAll();

    for (int level = 0; level < numLevels; ++level) {
        auto container =
            grid.getContainer(
                "postProcess_" + std::to_string(level), level,
                [&, level](Neon::set::Loader& loader) {
                    const auto& pop = fpop.load(loader, level, Neon::MultiResCompute::STENCIL);
                    const auto& type = cellType.load(loader, level, Neon::MultiResCompute::MAP);
                    auto&       u = vel.load(loader, level, Neon::MultiResCompute::MAP);
                    auto&       rh = rho.load(loader, level, Neon::MultiResCompute::MAP);


                    return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::bGrid::Cell& cell) mutable {
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
                            if (type(cell, 0) == CellType::movingWall) {
                                rh(cell, 0) = 1.0;

                                for (int d = 0; d < 3; ++d) {
                                    int i = (d == 0) ? 3 : ((d == 1) ? 1 : 9);
                                    u(cell, d) = pop(cell, i) / (6.0 * 1.0 / 18.0);
                                }
                            }
                        }
                    };
                });

        container.run(0);
    }

    grid.getBackend().syncAll();


    vel.updateIO();
    //rho.updateIO();


    int                precision = 4;
    std::ostringstream suffix;
    suffix << std::setw(precision) << std::setfill('0') << iteration;

    vel.ioToVtk("Velocity_" + suffix.str());
    //rho.ioToVtk("Density_" + suffix.str());

    std::vector<std::pair<T, T>> xPosVal;
    std::vector<std::pair<T, T>> yPosVal;
    if (verify || generateValidateFile) {
        const Neon::index_3d grid_dim = grid.getDimension();
        for (int level = 0; level < numLevels; ++level) {
            vel.forEachActiveCell(
                level, [&](const Neon::index_3d& id, const int& card, T& val) {
                    if (id.x == grid_dim.x / 2 && id.z == grid_dim.z / 2) {
                        if (card == 0) {
                            yPosVal.push_back({static_cast<double>(id.v[1]) / static_cast<double>(grid_dim.y), val});
                        }
                    }

                    if (id.y == grid_dim.y / 2 && id.z == grid_dim.z / 2) {
                        if (card == 0) {
                            xPosVal.push_back({static_cast<double>(id.v[0]) / static_cast<double>(grid_dim.x), val});
                        }
                    }
                },
                Neon::computeMode_t::seq);
        }
    }

    if (verify) {
        NEON_INFO("Max difference = {}", verifyGhia1982(Re, xPosVal, yPosVal, 1.0 / ulb));
    }
    if (generateValidateFile) {
        std::ofstream file;
        file.open("NeonMultiResLBM_" + suffix.str() + ".dat");
        for (auto v : yPosVal) {
            file << v.first << " " << v.second << "\n";
        }
        file.close();
    }
}


inline void generatepalabosDATFile(const std::string    filename,
                                   const Neon::index_3d gridDim,
                                   const int            depth,
                                   const float*         levelSDF)
{
    std::ofstream file;
    file.open(filename);

    //a cuboid coordinates specified by it cartesian extents x0, x1, y0, y1, z0, z1
    file << 0 << " " << gridDim.x << " "
         << 0 << " " << gridDim.y << " "
         << 0 << " " << gridDim.z << "\n";

    //dx: the finest voxel size
    file << "1\n";

    //nx, ny and nz representing the number of finest voxels along each dimension of the cuboid.
    file << gridDim.x << " "
         << gridDim.y << " "
         << gridDim.z << "\n";

    for (int32_t k = 0; k < gridDim.z; ++k) {
        for (int32_t j = 0; j < gridDim.y; ++j) {
            for (int32_t i = 0; i < gridDim.x; ++i) {
                float sdf = sdfCube({i, j, k}, gridDim - 1);

                for (int d = 0; d < depth; ++d) {
                    if (sdf <= levelSDF[d] && sdf > levelSDF[d + 1]) {
                        file << float(depth - d - 1) / float(depth - 1) << " ";
                        break;
                    }
                }
            }
        }
        file << "\n";
    }


    file.close();
}