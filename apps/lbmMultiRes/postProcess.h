#pragma once

#include "Neon/domain/mGrid.h"
#include "Neon/skeleton/Skeleton.h"

#include "verify.h"

#include "polyscope/volume_mesh.h"

template <typename T, int Q>
void postProcess(Neon::domain::mGrid&                        grid,
                 const int                                   numLevels,
                 const Neon::domain::mGrid::Field<T>&        fpop,
                 const Neon::domain::mGrid::Field<CellType>& cellType,
                 Neon::domain::mGrid::Field<T>&              vel,
                 Neon::domain::mGrid::Field<T>&              rho,
                 const Neon::int8_3d                         slice,
                 std::string                                 fileName,
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
        vel.ioToVtk(fileName, true, true, true, true, slice);
        //rho.ioToVtk("Density_" + suffix.str());
    }
}


template <typename T>
void initPolyscope(Neon::domain::mGrid&                                      grid,
                   std::vector<std::pair<Neon::domain::mGrid::Idx, int8_t>>& psDrawable,
                   const Neon::int8_3d                                       slice)
{
    polyscope::init();

    //polyscope register points
    std::vector<std::array<int, 8>> psHex;
    std::vector<Neon::float_3d>     psHexVert;
    psDrawable.clear();

    const bool filterOverlaps = true;

    //collect indices to draw
    Neon::SetIdx devID(0);
    for (int l = 0; l < grid.getDescriptor().getDepth(); ++l) {
        const int refFactor = grid.getDescriptor().getRefFactor(l);
        const int voxelSpacing = grid.getDescriptor().getSpacing(l - 1);

        constexpr double      tiny = 1e-7;
        const Neon::double_3d voxelSize(1.0 / grid.getDimension(l).x, 1.0 / grid.getDimension(l).y, 1.0 / grid.getDimension(l).z);

        grid(l).helpGetPartitioner1D().forEachSeq(devID, [&](const uint32_t blockIdx, const Neon::int32_3d memBlockOrigin, auto /*byPartition*/) {
            Neon::index_3d blockOrigin = memBlockOrigin;
            blockOrigin.x *= Neon::domain::details::mGrid::kMemBlockSizeX * voxelSpacing;
            blockOrigin.y *= Neon::domain::details::mGrid::kMemBlockSizeY * voxelSpacing;
            blockOrigin.z *= Neon::domain::details::mGrid::kMemBlockSizeZ * voxelSpacing;

            for (uint32_t k = 0; k < Neon::domain::details::mGrid::kNumUserBlockPerMemBlockZ; ++k) {
                for (uint32_t j = 0; j < Neon::domain::details::mGrid::kNumUserBlockPerMemBlockY; ++j) {
                    for (uint32_t i = 0; i < Neon::domain::details::mGrid::kNumUserBlockPerMemBlockX; ++i) {

                        const Neon::index_3d userBlockOrigin(i * Neon::domain::details::mGrid::kUserBlockSizeX * voxelSpacing + blockOrigin.x,
                                                             j * Neon::domain::details::mGrid::kUserBlockSizeY * voxelSpacing + blockOrigin.y,
                                                             k * Neon::domain::details::mGrid::kUserBlockSizeZ * voxelSpacing + blockOrigin.z);

                        for (int32_t z = 0; z < refFactor; z++) {
                            for (int32_t y = 0; y < refFactor; y++) {
                                for (int32_t x = 0; x < refFactor; x++) {

                                    const Neon::index_3d voxelGlobalID(x * voxelSpacing + userBlockOrigin.x,
                                                                       y * voxelSpacing + userBlockOrigin.y,
                                                                       z * voxelSpacing + userBlockOrigin.z);

                                    if (grid(l).isInsideDomain(voxelGlobalID)) {

                                        bool draw = true;
                                        if (filterOverlaps && l != 0) {
                                            draw = !(grid(l - 1).isInsideDomain(voxelGlobalID));
                                        }

                                        const Neon::double_3d location(double(voxelGlobalID.x) / double(grid.getDimension(0).x),
                                                                       double(voxelGlobalID.y) / double(grid.getDimension(0).y),
                                                                       double(voxelGlobalID.z) / double(grid.getDimension(0).z));

                                        if (draw && (slice.x == 1 || slice.y == 1 || slice.z == 1)) {
                                            draw = false;
                                            for (int s = 0; s < 3 && !draw; ++s) {
                                                if (slice.v[s] == 1 && location.v[s] - tiny <= 0.5 && location.v[s] + voxelSize.v[s] >= 0.5 - tiny) {
                                                    draw = true;
                                                }
                                            }
                                        }


                                        if (draw) {

                                            Neon::domain::mGrid::Idx idx(blockIdx, int8_t(i * Neon::domain::details::mGrid::kUserBlockSizeX + x),
                                                                         int8_t(j * Neon::domain::details::mGrid::kUserBlockSizeY + y),
                                                                         int8_t(k * Neon::domain::details::mGrid::kUserBlockSizeZ + z));

                                            psDrawable.push_back({idx, int8_t(l)});

                                            std::array<int, 8> hex;

                                            const Neon::float_3d id(voxelGlobalID.x, voxelGlobalID.y, voxelGlobalID.z);

                                            //x,y,z
                                            hex[0] = psHexVert.size();
                                            psHexVert.push_back({id.x, id.y, id.z});

                                            //+x,y,z
                                            hex[1] = psHexVert.size();
                                            psHexVert.push_back({id.x + voxelSpacing, id.y, id.z});


                                            //+x,y,+z
                                            hex[2] = psHexVert.size();
                                            psHexVert.push_back({id.x + voxelSpacing, id.y, id.z + voxelSpacing});


                                            //x,y,+z
                                            hex[3] = psHexVert.size();
                                            psHexVert.push_back({id.x, id.y, id.z + voxelSpacing});


                                            //x,+y,z
                                            hex[4] = psHexVert.size();
                                            psHexVert.push_back({id.x, id.y + voxelSpacing, id.z});


                                            //+x,+y,z
                                            hex[5] = psHexVert.size();
                                            psHexVert.push_back({id.x + voxelSpacing, id.y + voxelSpacing, id.z});


                                            //+x,+y,+z
                                            hex[6] = psHexVert.size();
                                            psHexVert.push_back({id.x + voxelSpacing, id.y + voxelSpacing, id.z + voxelSpacing});


                                            //x,+y,+z
                                            hex[7] = psHexVert.size();
                                            psHexVert.push_back({id.x, id.y + voxelSpacing, id.z + voxelSpacing});


                                            psHex.push_back(hex);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    auto psMesh = polyscope::registerHexMesh("LBM", psHexVert, psHex);
    //psMesh->setEdgeWidth(1.0);
    polyscope::options::screenshotExtension = ".png";
}

template <typename T>
void postProcessPolyscope(const std::vector<std::pair<Neon::domain::mGrid::Idx, int8_t>>& psDrawable,
                          const Neon::domain::mGrid::Field<T>&                            vel,
                          std::vector<T>&                                                 psColor,
                          std::string                                                     screenshotName,
                          bool                                                            show)
{
    if (psColor.empty()) {
        psColor.resize(psDrawable.size());
    }
    Neon::SetIdx devID(0);
    for (uint32_t t = 0; t < psDrawable.size(); ++t) {
        const auto id = psDrawable[t].first;
        int        level = psDrawable[t].second;

        T c = 0;
        for (int d = 0; d < 3; ++d) {
            T v = vel(id, d, level);
            c += v * v;
        }
        psColor[t] = std::sqrt(c);
    }

    auto colorQu = polyscope::getVolumeMesh("LBM")->addCellScalarQuantity("Velocity", psColor);
    colorQu->setEnabled(true);
    colorQu->setColorMap("jet");
    //colorQu->setMapRange({0, 0.04});

    polyscope::screenshot(screenshotName + ".png");

    if (show) {
        polyscope::show();
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