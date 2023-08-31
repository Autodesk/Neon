#pragma once
#include "Neon/Neon.h"
#include "Neon/domain/mGrid.h"
#include "Neon/skeleton/Skeleton.h"

#include "collide.h"
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
        containers.push_back(collideBGKUnrolledFusedStore<T, Q>(grid, omega0, level, numLevels, cellType, fin, fout));
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
                                 std::vector<Neon::set::Container>&          containers)
{
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
                                          containers);
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
                                          containers);
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


template <typename T, int Q>
void runNonUniformLBM(Neon::domain::mGrid&                        grid,
                      const uint32_t                              numActiveVoxels,
                      const int                                   numIter,
                      const int                                   Re,
                      const bool                                  fineInitStore,
                      const bool                                  streamFusedExpl,
                      const bool                                  streamFusedCoal,
                      const bool                                  streamFuseAll,
                      const bool                                  collisionFusedStore,
                      const bool                                  benchmark,
                      const int                                   freq,
                      const int                                   problemID,
                      const std::string                           problemType,
                      const T                                     omega,
                      const Neon::domain::mGrid::Field<CellType>& cellType,
                      const Neon::domain::mGrid::Field<float>&    storeSum,
                      Neon::domain::mGrid::Field<T>&              fin,
                      Neon::domain::mGrid::Field<T>&              fout,
                      Neon::domain::mGrid::Field<T>&              vel,
                      Neon::domain::mGrid::Field<T>&              rho)
{
    const int  depth = grid.getDescriptor().getDepth();
    const auto gridDim = grid.getDimension();

    //skeleton
    std::vector<Neon::set::Container> containers;
    nonUniformTimestepRecursive<T, Q>(grid,
                                      fineInitStore,
                                      streamFusedExpl,
                                      streamFusedCoal,
                                      streamFuseAll,
                                      collisionFusedStore,
                                      omega,
                                      depth - 1,
                                      depth,
                                      cellType,
                                      storeSum,
                                      fin, fout, containers);

    Neon::skeleton::Skeleton skl(grid.getBackend());
    skl.sequence(containers, "MultiResLBM");
    skl.ioToDot("MultiResLBM", "", true);

    //execution
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < numIter; ++t) {
        if (t % 100 == 0) {
            NEON_INFO("Non-uniform LBM Iteration: {}", t);
        }
        skl.run();
        if (!benchmark && t % freq == 0) {
            postProcess<T, Q>(grid, depth, fout, cellType, t, vel, rho, true);
        }
    }
    grid.getBackend().syncAll();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    const double MLUPS = static_cast<double>(numIter * numActiveVoxels) / duration.count();

    const double effNumIter = double(numIter) * double(1 << (depth - 1));
    const double effMLUPS = (effNumIter * double(gridDim.x) * double(gridDim.y) * double(gridDim.y)) / double(duration.count());


    NEON_INFO("Time = {0:8.8f} (microseconds)", double(duration.count()));
    NEON_INFO("MLUPS = {0:8.8f}, numActiveVoxels = {1}", MLUPS, numActiveVoxels);
    NEON_INFO("Effective MLUPS = {0:8.8f}, Effective numActiveVoxels = {1}", effMLUPS, gridDim.rMul());

    //Reporting
    auto algoName = [&]() {
        std::string ret;
        if (collisionFusedStore) {
            ret = "CH";
        } else {
            ret = "C-H";
        }
        if (streamFusedExpl) {
            ret += "-SE-O";
        } else if (streamFusedCoal) {
            ret += "-SO-E";
        } else if (streamFuseAll) {
            ret += "-SEO";
        } else {
            ret += "-S-E-O";
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
        std::string ret = "P" + std::to_string(problemID) + "_";
        ret += algoName();
        ret += "_" + typeName();

        return ret;
    };

    //system
    report.addMember("DeviceType", Neon::DeviceTypeUtil::toString(grid.getBackend().devType()));

    //grid
    report.addMember("N", gridDim.x);
    report.addMember("Depth", depth);
    report.addMember("DataType", typeName());

    //problem
    report.addMember("ProblemID", problemID);
    report.addMember("problemType", problemType);
    report.addMember("omega", omega);
    report.addMember("Re", Re);
#ifdef BGK
    report.addMember("Collision", std::string("BGK"));
#endif
#ifdef KBC
    report.addMember("Collision", std::string("KBC"));
#endif


    //algorithm
    report.addMember("fineInitStore", fineInitStore);
    report.addMember("streamFusedExpl", streamFusedExpl);
    report.addMember("streamFusedCoal", streamFusedCoal);
    report.addMember("streamFuseAll", streamFuseAll);
    report.addMember("collisionFusedStore", collisionFusedStore);
    report.addMember("Algorithm", algoName());

    //perf
    report.addMember("Time (microsecond)", duration.count());
    report.addMember("MLUPS", MLUPS);
    report.addMember("NumIter", numIter);
    report.addMember("NumVoxels", numActiveVoxels);
    report.addMember("EMLUPS", effMLUPS);
    report.addMember("ENumIter", effNumIter);
    report.addMember("ENumVoxels", gridDim.rMul());

    //output
    report.write("MultiResLBM_" + reportSuffix(), true);

    //post process
    postProcess<T, Q>(grid, depth, fout, cellType, numIter, vel, rho, !benchmark);
}
