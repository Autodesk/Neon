#pragma once

#include "Neon/domain/mGrid.h"
#include "Neon/skeleton/Skeleton.h"

#include "verify.h"

#ifdef NEON_USE_POLYSCOPE
#include "polyscope/surface_mesh.h"
#include "polyscope/volume_mesh.h"
#endif

#include <Eigen/Core>

template <typename T, int Q>
void postProcess(Neon::domain::mGrid&                                            grid,
                 const int                                                       numLevels,
                 const Neon::domain::mGrid::Field<T>&                            fpop,
                 const Neon::domain::mGrid::Field<CellType>&                     cellType,
                 Neon::domain::mGrid::Field<T>&                                  vel,
                 Neon::domain::mGrid::Field<T>&                                  rho,
                 const Neon::int8_3d                                             slice,
                 std::string                                                     fileName,
                 bool                                                            outputFile,
                 const std::vector<std::pair<Neon::domain::mGrid::Idx, int8_t>>& psDrawable,
                 const std::vector<std::array<int, 8>>&                          psHex,
                 const std::vector<Neon::float_3d>&                              psHexVert)
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
                            if (type(cell, 0) == CellType::bulk ) {

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
        //vel.ioToVtk(fileName, true, true, true, true, slice);
        //rho.ioToVtk("Density_" + suffix.str());

        std::ofstream file(fileName + ".vtk");
        file << "# vtk DataFile Version 2.0\n";
        file << "mGrid\n";
        file << "ASCII\n";
        file << "DATASET UNSTRUCTURED_GRID\n";
        file << "POINTS " << psHexVert.size() << " float \n";

        for (size_t v = 0; v < psHexVert.size(); ++v) {
            file << psHexVert[v].x << " " << psHexVert[v].y << " " << psHexVert[v].z << "\n";
        }

        file << "CELLS " << psHex.size() << " " << psHex.size() * 9 << " \n";

        for (uint64_t i = 0; i < psHex.size(); ++i) {
            file << "8 ";
            for (int j = 0; j < 8; ++j) {
                int d = j;
                if (j == 2) {
                    d = 3;
                }
                if (j == 3) {
                    d = 2;
                }
                if (j == 6) {
                    d = 7;
                }
                if (j == 7) {
                    d = 6;
                }
                file << psHex[i][d] << " ";
            }
            file << "\n";
        }

        file << "CELL_TYPES " << psHex.size() << " \n";
        for (uint64_t i = 0; i < psHex.size(); ++i) {
            file << 11 << "\n";
        }

        file << "CELL_DATA " << psHex.size() << " \n";

        //data
        //file << "SCALARS Velocity float 1 \n";
        //file << "LOOKUP_TABLE default \n";
        file << "VECTORS Velocity float \n";

        for (size_t t = 0; t < psDrawable.size(); ++t) {
            const auto id = psDrawable[t].first;
            int        level = psDrawable[t].second;
            
            for (int d = 0; d < 3; ++d) {
                T v = vel(id, d, level);
                file << v << " ";                
            }
            file << "\n";

            //T c = 0;
            //for (int d = 0; d < 3; ++d) {
            //    T v = vel(id, d, level);
            //    c += v * v;
            //}
            //file << c << "\n";
        }

        file.close();
    }
}


template <typename T>
void initVisualization(Neon::domain::mGrid&                                      grid,
                       const Neon::domain::mGrid::Field<T>&                      vel,
                       std::vector<std::pair<Neon::domain::mGrid::Idx, int8_t>>& psDrawable,
                       std::vector<std::array<int, 8>>&                          psHex,
                       std::vector<Neon::float_3d>&                              psHexVert,
                       const Neon::int8_3d                                       slice)
{
    //polyscope register points
    //std::vector<std::array<int, 8>> psHex;
    //std::vector<Neon::float_3d>     psHexVert;
    psHex.clear();
    psHexVert.clear();
    psDrawable.clear();


    for (int l = 0; l < grid.getDescriptor().getDepth(); ++l) {
        constexpr double      tiny = 1e-7;
        const Neon::double_3d voxelSize(1.0 / grid.getDimension(l).x, 1.0 / grid.getDimension(l).y, 1.0 / grid.getDimension(l).z);
        const int             voxelSpacing = grid.getDescriptor().getSpacing(l - 1);
        const Neon::index_3d  dim0 = grid.getDimension(0);

        grid.newContainer<Neon::Execution::host>("initVisualization", l, [&](Neon::set::Loader& loader) {
                const auto& u = vel.load(loader, l, Neon::MultiResCompute::MAP);

                return [&](const typename Neon::domain::mGrid::Idx& cell) mutable {
                    if (!u.hasChildren(cell)) {
                        bool draw = true;

                        Neon::index_3d voxelGlobalLocation = u.getGlobalIndex(cell);

                        if (slice.x == 1 || slice.y == 1 || slice.z == 1) {
                            draw = false;
                            const Neon::double_3d locationScaled(double(voxelGlobalLocation.x) / double(dim0.x),
                                                                 double(voxelGlobalLocation.y) / double(dim0.y),
                                                                 double(voxelGlobalLocation.z) / double(dim0.z));
                            for (int s = 0; s < 3 && !draw; ++s) {
                                if (slice.v[s] == 1 && locationScaled.v[s] - tiny <= 0.5 && locationScaled.v[s] + voxelSize.v[s] >= 0.5 - tiny) {
                                    draw = true;
                                }
                            }
                        }


                        if (draw) {

#pragma omp critical
                            {

                                psDrawable.push_back({cell, int8_t(l)});
                            }

                            std::array<int, 8> hex;

                            const Neon::float_3d gf(voxelGlobalLocation.x, voxelGlobalLocation.y, voxelGlobalLocation.z);

                            //x,y,z
                            hex[0] = psHexVert.size();
                            psHexVert.push_back({gf.x, gf.y, gf.z});

                            //+x,y,z
                            hex[1] = psHexVert.size();
                            psHexVert.push_back({gf.x + voxelSpacing, gf.y, gf.z});


                            //+x,y,+z
                            hex[2] = psHexVert.size();
                            psHexVert.push_back({gf.x + voxelSpacing, gf.y, gf.z + voxelSpacing});


                            //x,y,+z
                            hex[3] = psHexVert.size();
                            psHexVert.push_back({gf.x, gf.y, gf.z + voxelSpacing});


                            //x,+y,z
                            hex[4] = psHexVert.size();
                            psHexVert.push_back({gf.x, gf.y + voxelSpacing, gf.z});


                            //+x,+y,z
                            hex[5] = psHexVert.size();
                            psHexVert.push_back({gf.x + voxelSpacing, gf.y + voxelSpacing, gf.z});


                            //+x,+y,+z
                            hex[6] = psHexVert.size();
                            psHexVert.push_back({gf.x + voxelSpacing, gf.y + voxelSpacing, gf.z + voxelSpacing});


                            //x,+y,+z
                            hex[7] = psHexVert.size();
                            psHexVert.push_back({gf.x, gf.y + voxelSpacing, gf.z + voxelSpacing});

#pragma omp critical
                            {

                                psHex.push_back(hex);
                            }
                        }
                    }
                };
            })
            .run(0);
    }

#ifdef NEON_USE_POLYSCOPE
    if (!polyscope::isInitialized()) {
        polyscope::init();
    }
    polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;
    polyscope::view::projectionMode = polyscope::ProjectionMode::Orthographic;
    //Neon::index_3d dim0 = grid.getDimension(0);
    //polyscope::view::lookAt(glm::vec3{0, 0, 0}, glm::vec3{0., 0., 1.});
    auto psMesh = polyscope::registerHexMesh("LBM", psHexVert, psHex);
    polyscope::options::screenshotExtension = ".png";
#endif
}

#ifdef NEON_USE_POLYSCOPE
template <typename T>
void postProcessPolyscope(const std::vector<std::pair<Neon::domain::mGrid::Idx, int8_t>>& psDrawable,
                          const Neon::domain::mGrid::Field<T>&                            vel,
                          std::vector<T>&                                                 psColor,
                          std::string                                                     screenshotName,
                          bool                                                            show,
                          bool                                                            showEdges)
{
    if (psColor.empty()) {
        psColor.resize(psDrawable.size());
    }
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
    if (showEdges) {
        polyscope::getVolumeMesh("LBM")->setEdgeWidth(1.0);
    } else {
        polyscope::getVolumeMesh("LBM")->setEdgeWidth(0.0);
    }
    //colorQu->setMapRange({0, 0.04});

    polyscope::screenshot(screenshotName + ".png");

    if (show) {
        polyscope::show();
    }
}

void polyscopeAddMesh(
    const std::string      name,
    const Eigen::MatrixXi& faces,
    const Eigen::MatrixXd& vertices)
{
    if (!polyscope::isInitialized()) {
        polyscope::init();
    }

    polyscope::registerSurfaceMesh(polyscope::guessNiceNameFromPath(name), vertices, faces);
}
#endif

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