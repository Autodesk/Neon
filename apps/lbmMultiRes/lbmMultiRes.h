#pragma once
#include "Neon/Neon.h"
#include "Neon/domain/mGrid.h"
#include "Neon/skeleton/Skeleton.h"

#include "collide.h"
#include "fusedFinest.h"
#include "init.h"
#include "lattice.h"
#include "postProcess.h"
#include "store.h"
#include "stream.h"
#include "util.h"

template <typename T, int Q>
void collideStep(Neon::domain::mGrid&                        grid,
                 const bool                                  fineInitStore,
                 const bool                                  collisionFusedStore,
                 const T                                     omega0,
                 const int                                   level,
                 const int                                   numLevels,
                 const Neon::domain::mGrid::Field<CellType>& cellType,
                 Neon::domain::mGrid::Field<T>&              fin,
                 Neon::domain::mGrid::Field<T>&              fout,
                 std::vector<Neon::set::Container>&          containers)
{
    if (!collisionFusedStore) {
        // collision for all voxels at level L=level
#ifdef KBC
        containers.push_back(collideKBC<T, Q>(grid, omega0, level, numLevels, cellType, fin, fout));
#endif

#ifdef BGK
        containers.push_back(collideBGKUnrolled<T, Q>(grid, omega0, level, numLevels, cellType, fin, fout));
#endif

        // Storing fine (level - 1) data for later "coalescence" pulled by the coarse (level)
        if (level != numLevels - 1) {
            if (fineInitStore) {
                containers.push_back(storeFine<T, Q>(grid, level, fout));
            } else {
                containers.push_back(storeCoarse<T, Q>(grid, level + 1, fout));
            }
        }
    } else {
        //  collision for all voxels at level L = level fused with
        //  Storing fine (level) data for later "coalescence" pulled by the coarse(level)
#ifdef KBC
        containers.push_back(collideKBCFusedStore<T, Q>(grid, omega0, level, numLevels, cellType, fin, fout));
#endif

#ifdef BGK
        containers.push_back(collideBGKUnrolledFusedStore<T, Q>(grid, omega0, level, numLevels, cellType, fin, fout));
#endif
    }
}


template <typename T, int Q>
void streamingStep(Neon::domain::mGrid&                        grid,
                   const bool                                  fineInitStore,
                   const bool                                  streamFusedExpl,
                   const bool                                  streamFusedCoal,
                   const bool                                  streamFuseAll,
                   const bool                                  collisionFusedStore,
                   const int                                   level,
                   const int                                   numLevels,
                   const Neon::domain::mGrid::Field<CellType>& cellType,
                   const Neon::domain::mGrid::Field<float>&    sumStore,
                   Neon::domain::mGrid::Field<T>&              fin,
                   Neon::domain::mGrid::Field<T>&              fout,
                   std::vector<Neon::set::Container>&          containers)
{
    // Streaming step that also performs the necessary "explosion" and "coalescence" steps.
    if (streamFusedExpl) {
        streamFusedExplosion<T, Q>(grid, fineInitStore || collisionFusedStore, level, numLevels, cellType, sumStore, fout, fin, containers);
    } else if (streamFusedCoal) {
        streamFusedCoalescence<T, Q>(grid, fineInitStore || collisionFusedStore, level, numLevels, cellType, sumStore, fout, fin, containers);
    } else if (streamFuseAll) {
        streamFusedCoalescenceExplosion<T, Q>(grid, fineInitStore || collisionFusedStore, level, numLevels, cellType, sumStore, fout, fin, containers);
    } else {
        stream<T, Q>(grid, fineInitStore || collisionFusedStore, level, numLevels, cellType, sumStore, fout, fin, containers);
    }
}

template <typename T, int Q>
void collideFusedStreaming(Neon::domain::mGrid&                        grid,
                           const T                                     omega0,
                           const int                                   level,
                           const int                                   numLevels,
                           const Neon::domain::mGrid::Field<CellType>& cellType,
                           Neon::domain::mGrid::Field<T>&              fin,
                           Neon::domain::mGrid::Field<T>&              fout,
                           std::vector<Neon::set::Container>&          containers)
{

#ifdef KBC
    containers.push_back(collideKBCUnrolledFusedAll<T, Q>(grid,
                                                          omega0,
                                                          level,
                                                          numLevels,
                                                          cellType,
                                                          fin,
                                                          fout,
                                                          true));

    containers.push_back(collideKBCUnrolledFusedAll<T, Q>(grid,
                                                          omega0,
                                                          level,
                                                          numLevels,
                                                          cellType,
                                                          fout,
                                                          fin,
                                                          false));
#endif

#ifdef BGK
    containers.push_back(collideBGKUnrolledFusedAll<T, Q>(grid,
                                                          omega0,
                                                          level,
                                                          numLevels,
                                                          cellType,
                                                          fin,
                                                          fout,
                                                          true));

    containers.push_back(collideBGKUnrolledFusedAll<T, Q>(grid,
                                                          omega0,
                                                          level,
                                                          numLevels,
                                                          cellType,
                                                          fout,
                                                          fin,
                                                          false));
#endif
}

template <typename T, int Q>
void nonUniformTimestepRecursive(Neon::domain::mGrid&                        grid,
                                 const bool                                  fineInitStore,
                                 const bool                                  streamFusedExpl,
                                 const bool                                  streamFusedCoal,
                                 const bool                                  streamFuseAll,
                                 const bool                                  collisionFusedStore,
                                 const T                                     omega0,
                                 const int                                   level,
                                 const int                                   numLevels,
                                 const Neon::domain::mGrid::Field<CellType>& cellType,
                                 const Neon::domain::mGrid::Field<float>&    sumStore,
                                 Neon::domain::mGrid::Field<T>&              fin,
                                 Neon::domain::mGrid::Field<T>&              fout,
                                 std::vector<Neon::set::Container>&          containers,
                                 bool                                        fusedFinest)
{
    if (fusedFinest && level == 0) {
        collideFusedStreaming<T, Q>(grid,
                                    omega0,
                                    level,
                                    numLevels,
                                    cellType,
                                    fin,
                                    fout,
                                    containers);
        return;

    } else {
        //Collide
        collideStep<T, Q>(grid,
                          fineInitStore,
                          collisionFusedStore,
                          omega0,
                          level,
                          numLevels,
                          cellType,
                          fin,
                          fout,
                          containers);


        // Recurse down
        if (level != 0) {
            nonUniformTimestepRecursive<T, Q>(grid,
                                              fineInitStore,
                                              streamFusedExpl,
                                              streamFusedCoal,
                                              streamFuseAll,
                                              collisionFusedStore,
                                              omega0,
                                              level - 1,
                                              numLevels,
                                              cellType,
                                              sumStore,
                                              fin,
                                              fout,
                                              containers,
                                              fusedFinest);
        }

        //Streaming
        streamingStep<T, Q>(grid,
                            fineInitStore,
                            streamFusedExpl,
                            streamFusedCoal,
                            streamFuseAll,
                            collisionFusedStore,
                            level,
                            numLevels,
                            cellType,
                            sumStore,
                            fin,
                            fout,
                            containers);

        // Stop
        if (level == numLevels - 1) {
            return;
        }

        //Collide
        collideStep<T, Q>(grid,
                          fineInitStore,
                          collisionFusedStore,
                          omega0,
                          level,
                          numLevels,
                          cellType,
                          fin,
                          fout,
                          containers);

        // Recurse down
        if (level != 0) {
            nonUniformTimestepRecursive<T, Q>(grid,
                                              fineInitStore,
                                              streamFusedExpl,
                                              streamFusedCoal,
                                              streamFuseAll,
                                              collisionFusedStore,
                                              omega0,
                                              level - 1,
                                              numLevels,
                                              cellType,
                                              sumStore,
                                              fin,
                                              fout,
                                              containers,
                                              fusedFinest);
        }

        //Streaming
        streamingStep<T, Q>(grid,
                            fineInitStore,
                            streamFusedExpl,
                            streamFusedCoal,
                            streamFuseAll,
                            collisionFusedStore,
                            level,
                            numLevels,
                            cellType,
                            sumStore,
                            fin,
                            fout,
                            containers);
    }
}


template <typename T, int Q>
void runNonUniformLBM(Neon::domain::mGrid&                        grid,
                      const Params&                               params,
                      const T                                     clength,
                      const T                                     omega,
                      const T                                     visclb,
                      const Neon::double_3d                       velocity,
                      const Neon::domain::mGrid::Field<CellType>& cellType,
                      const Neon::domain::mGrid::Field<float>&    storeSum,
                      Neon::domain::mGrid::Field<T>&              fin,
                      Neon::domain::mGrid::Field<T>&              fout,
                      bool                                        verify = false)
{
    const int  depth = grid.getDescriptor().getDepth();
    const auto gridDim = grid.getDimension();

    std::vector<uint32_t> numActiveVoxels = countActiveVoxels(grid, fin);
    uint32_t              sumActiveVoxels = 0;
    for (auto n : numActiveVoxels) {
        sumActiveVoxels += n;
    }

    Neon::domain::mGrid::Field<T> vel;
    Neon::domain::mGrid::Field<T> rho;
    if (!params.benchmark || verify) {
        vel = grid.newField<T>("vel", 3, 0);
        rho = grid.newField<T>("rho", 1, 0);
    }

    //skeleton
    std::vector<Neon::set::Container> containers;
    nonUniformTimestepRecursive<T, Q>(grid,
                                      params.fineInitStore,
                                      params.streamFusedExpl,
                                      params.streamFusedCoal,
                                      params.streamFuseAll,
                                      params.collisionFusedStore,
                                      omega,
                                      depth - 1,
                                      depth,
                                      cellType,
                                      storeSum,
                                      fin,
                                      fout,
                                      containers,
                                      params.fusedFinest);

    Neon::skeleton::Skeleton skl(grid.getBackend());
    skl.sequence(containers, "MultiResLBM");
    skl.ioToDot("MultiResLBM", "", true);

    const Neon::int8_3d slice(params.sliceX, params.sliceY, params.sliceZ);

    std::vector<std::pair<Neon::domain::mGrid::Idx, int8_t>> psDrawable;
    std::vector<std::array<int, 8>>                          psHex;
    std::vector<Neon::float_3d>                              psHexVert;
    std::vector<T>                                           psColor;

    if (!params.benchmark) {
        initVisualization<T>(grid, vel, psDrawable, psHex, psHexVert, slice);

        if (slice.x == -1 && slice.y == -1 && slice.z == -1) {
            if (sumActiveVoxels != psDrawable.size()) {
                Neon::NeonException exp("runNonUniformLBM");
                exp << "Mismatch between number of active voxels and drawable voxels";
                exp << "psDrawable.size()= " << psDrawable.size() << ", sumActiveVoxels= " << sumActiveVoxels;
                NEON_THROW(exp);
            }
        }
    }

    NEON_INFO("Re: {}", params.Re);
    NEON_INFO("clength: {}", clength);
    NEON_INFO("omega: {}", omega);
    NEON_INFO("visclb: {}", visclb);
#ifdef KBC
    NEON_INFO("Collision: KBC");
#endif
#ifdef BGK
    NEON_INFO("Collision: BGK");
#endif


    NEON_INFO("velocity: {}, {}, {}", velocity.x, velocity.y, velocity.z);


    //execution
    for (uint32_t l = 0; l < depth; ++l) {
        NEON_INFO("numActiveVoxels [{}]: {}", l, numActiveVoxels[l]);
    }
    NEON_INFO("sumActiveVoxels: {}", sumActiveVoxels);
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < params.numIter; ++t) {
        if (t % 100 == 0) {
            NEON_INFO("Non-uniform LBM Iteration: {}", t);
        }
        skl.run();
        if (!params.benchmark && t % params.freq == 0) {
            int                precision = 4;
            std::ostringstream suffix;
            suffix << std::setw(precision) << std::setfill('0') << t;
            std::string fileName = "Velocity_" + suffix.str();

            postProcess<T, Q>(grid, depth, fout, cellType, vel, rho, slice, fileName, params.vtk && t != 0, psDrawable, psHex, psHexVert);
#ifdef NEON_USE_POLYSCOPE
            postProcessPolyscope<T>(psDrawable, vel, psColor, fileName, params.gui, t == 0);
#endif
        }
    }
    grid.getBackend().syncAll();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    //const double MLUPS = static_cast<double>(params.numIter * sumActiveVoxels) / duration.count();
    //NEON_INFO("MLUPS = {0:8.8f}, sumActiveVoxels = {1}", MLUPS, sumActiveVoxels);

    double MLUPS = 0;
    for (uint32_t l = 0; l < depth; ++l) {
        double d = (depth - 1) - l;
        MLUPS += double(params.numIter) * std::pow(2, d) * double(numActiveVoxels[l]);
    }
    MLUPS /= double(duration.count());
    NEON_INFO("MLUPS = {0:8.8f}", MLUPS);
    NEON_INFO("Time = {0:8.8f} (microseconds)", double(duration.count()));

    const double effNumIter = double(params.numIter) * double(1 << (depth - 1));
    const double effMLUPS = (effNumIter * double(gridDim.x) * double(gridDim.y) * double(gridDim.y)) / double(duration.count());
    NEON_INFO("Effective MLUPS = {0:8.8f}, Effective numActiveVoxels = {1}", effMLUPS, gridDim.rMul());

    //Reporting
    auto algoName = [&]() {
        std::string ret;

        if (params.collisionFusedStore) {
            ret = "CH";
        } else if (params.fineInitStore) {
            ret = "C-H";
        } else {
            ret = "C-h";
        }

        if (params.streamFusedExpl) {
            ret += "-SE-O";
        } else if (params.streamFusedCoal) {
            ret += "-SO-E";
        } else if (params.streamFuseAll) {
            ret += "-SEO";
        } else {
            ret += "-S-E-O";
        }

        if (params.fusedFinest) {
            ret += "+";
        }
        return ret;
    };

    auto typeName = [&]() {
        std::string ret;
        if (std::is_same_v<T, float>) {
            ret = "F";
        } else {
            ret = "D";
        }
        return ret;
    };

    auto reportSuffix = [&]() {
        std::string ret = "P" + std::to_string(params.scale) + "_";
        ret += algoName();
        ret += "_" + typeName();

        return ret;
    };

    //system
    report.addMember("DeviceType", Neon::DeviceTypeUtil::toString(grid.getBackend().devType()));

    //grid
    report.addMember("Grid Size X", gridDim.x);
    report.addMember("Grid Size Y", gridDim.y);
    report.addMember("Grid Size Z", gridDim.z);
    report.addMember("Depth", depth);
    report.addMember("DataType", typeName());

    //problem
    report.addMember("ProblemScale", params.scale);
    report.addMember("problemType", params.problemType);
    report.addMember("omega", omega);
    report.addMember("Re", params.Re);
    report.addMember("velocity", velocity.to_string());
    report.addMember("clength", clength);
    report.addMember("visclb", visclb);
#ifdef BGK
    report.addMember("Collision", std::string("BGK"));
#endif
#ifdef KBC
    report.addMember("Collision", std::string("KBC"));
#endif


    //algorithm
    report.addMember("fineInitStore", params.fineInitStore);
    report.addMember("streamFusedExpl", params.streamFusedExpl);
    report.addMember("streamFusedCoal", params.streamFusedCoal);
    report.addMember("streamFuseAll", params.streamFuseAll);
    report.addMember("collisionFusedStore", params.collisionFusedStore);
    report.addMember("fusedFinest", params.fusedFinest);
    report.addMember("Algorithm", algoName());

    //perf
    report.addMember("Time (microsecond)", duration.count());
    report.addMember("MLUPS", MLUPS);
    report.addMember("NumIter", params.numIter);
    report.addMember("NumActiveVoxels", numActiveVoxels);
    report.addMember("EMLUPS", effMLUPS);
    report.addMember("ENumIter", effNumIter);
    report.addMember("ENumVoxels", gridDim.rMul());

    //output
    report.write("MultiResLBM_" + reportSuffix(), true);

    //post process
    if (!params.benchmark) {
        int                precision = 4;
        std::ostringstream suffix;
        suffix << std::setw(precision) << std::setfill('0') << params.numIter;
        std::string fileName = "Velocity_" + suffix.str();
        postProcess<T, Q>(grid, depth, fout, cellType, vel, rho, slice, fileName, params.vtk, psDrawable, psHex, psHexVert);
#ifdef NEON_USE_POLYSCOPE
        postProcessPolyscope<T>(psDrawable, vel, psColor, fileName, params.gui, false);
#endif
    } else if (verify) {
        postProcess<T, Q>(grid, depth, fout, cellType, vel, rho, slice, "", false, psDrawable, psHex, psHexVert);
    }

    if (verify) {
        verifyLidDrivenCavity<T>(grid,
                                 depth,
                                 vel,
                                 params.Re,
                                 params.numIter,
                                 velocity.x);
    }
}
