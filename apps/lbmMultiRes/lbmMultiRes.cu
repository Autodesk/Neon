#include "Neon/core/tools/clipp.h"

#include "Neon/Neon.h"
#include "Neon/Report.h"
#include "Neon/domain/mGrid.h"
#include "Neon/skeleton/Skeleton.h"


#include "collide.h"
#include "init.h"
#include "lattice.h"
#include "postProcess.h"
#include "store.h"
#include "stream.h"
#include "util.h"

Neon::Report report;

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
                                 const Neon::domain::mGrid::Field<int>&      sumStore,
                                 Neon::domain::mGrid::Field<T>&              fin,
                                 Neon::domain::mGrid::Field<T>&              fout,
                                 std::vector<Neon::set::Container>&          containers)
{
    if (!collisionFusedStore) {
        // 1) collision for all voxels at level L=level
        //containers.push_back(collideKBC<T, Q>(grid, omega0, level, numLevels, cellType, fin, fout));
        containers.push_back(collideBGKUnrolled<T, Q>(grid, omega0, level, numLevels, cellType, fin, fout));

        // 2) Storing fine (level - 1) data for later "coalescence" pulled by the coarse (level)
        if (level != numLevels - 1) {
            if (fineInitStore) {
                containers.push_back(storeFine<T, Q>(grid, level, fout));
            } else {
                containers.push_back(storeCoarse<T, Q>(grid, level + 1, fout));
            }
        }
    } else {
        // 6) collision for all voxels at level L = level fused with
        // 7) Storing fine (level) data for later "coalescence" pulled by the coarse(level)
        containers.push_back(collideBGKUnrolledFusedStore<T, Q>(grid, omega0, level, numLevels, cellType, fin, fout));
    }


    // 3) recurse down
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

    // 4) Streaming step that also performs the necessary "explosion" and "coalescence" steps.
    if (streamFusedExpl) {
        streamFusedExplosion<T, Q>(grid, fineInitStore || collisionFusedStore, level, numLevels, cellType, sumStore, fout, fin, containers);
    } else if (streamFusedCoal) {
        streamFusedCoalescence<T, Q>(grid, fineInitStore || collisionFusedStore, level, numLevels, cellType, sumStore, fout, fin, containers);
    } else if (streamFuseAll) {
        streamFusedCoalescenceExplosion<T, Q>(grid, fineInitStore || collisionFusedStore, level, numLevels, cellType, sumStore, fout, fin, containers);
    } else {
        stream<T, Q>(grid, fineInitStore || collisionFusedStore, level, numLevels, cellType, sumStore, fout, fin, containers);
    }

    // 5) stop
    if (level == numLevels - 1) {
        return;
    }

    if (!collisionFusedStore) {

        // 6) collision for all voxels at level L = level
        //containers.push_back(collideBGK<T, Q>(grid, omega0, level, numLevels, cellType, fin, fout));
        containers.push_back(collideBGKUnrolled<T, Q>(grid, omega0, level, numLevels, cellType, fin, fout));

        // 7) Storing fine(level) data for later "coalescence" pulled by the coarse(level)
        if (level != numLevels - 1) {
            if (fineInitStore) {
                containers.push_back(storeFine<T, Q>(grid, level, fout));
            } else {
                containers.push_back(storeCoarse<T, Q>(grid, level + 1, fout));
            }
        }
    } else {
        // 6) collision for all voxels at level L = level fused with
        // 7) Storing fine (level) data for later "coalescence" pulled by the coarse(level)
        containers.push_back(collideBGKUnrolledFusedStore<T, Q>(grid, omega0, level, numLevels, cellType, fin, fout));
    }

    // 8) recurse down
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

    // 9) Streaming step
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
void runNonUniformLBM(const int           problemID,
                      const Neon::Backend backend,
                      const int           numIter,
                      const bool          fineInitStore,
                      const bool          streamFusedExpl,
                      const bool          streamFusedCoal,
                      const bool          streamFuseAll,
                      const bool          collisionFusedStore,
                      const bool          benchmark)
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

    //constexpr int depth = 2;
    constexpr int depth = 3;

    float levelSDF[depth + 1];

    Neon::index_3d gridDim;

    gridDim = Neon::index_3d(6, 6, 6);
    levelSDF[0] = 0;
    levelSDF[1] = -2.0 / 3.0;
    levelSDF[2] = -1.0;

    if (problemID == 0) {
        gridDim = Neon::index_3d(48, 48, 48);
        levelSDF[0] = 0;
        levelSDF[1] = -8 / 24.0;
        levelSDF[2] = -16 / 24.0;
        levelSDF[3] = -1.0;
    } else if (problemID == 1) {
        gridDim = Neon::index_3d(160, 160, 160);
        levelSDF[0] = 0;
        levelSDF[1] = -31.0 / 160.0;
        levelSDF[2] = -64 / 160.0;
        levelSDF[3] = -1.0;
    }


    generatepalabosDATFile(std::string("lid_" + std::to_string(gridDim.x) + "_" +
                                       std::to_string(gridDim.y) + "_" +
                                       std::to_string(gridDim.x) + ".dat"),
                           gridDim,
                           depth,
                           levelSDF);

    //define the grid
    const Neon::mGridDescriptor<1> descriptor(depth);

    Neon::domain::mGrid grid(
        backend, gridDim,
        {[&](const Neon::index_3d id) -> bool {
             return sdfCube(id, gridDim - 1) <= levelSDF[0] &&
                    sdfCube(id, gridDim - 1) > levelSDF[1];
         },
         [&](const Neon::index_3d& id) -> bool {
             return sdfCube(id, gridDim - 1) <= levelSDF[1] &&
                    sdfCube(id, gridDim - 1) > levelSDF[2];
         },
         [&](const Neon::index_3d& id) -> bool {
             return sdfCube(id, gridDim - 1) <= levelSDF[2] &&
                    sdfCube(id, gridDim - 1) > levelSDF[3];
         }},
        Neon::domain::Stencil::s19_t(false), descriptor);

    //LBM problem
    const T               ulb = 0.04;
    const int             Re = 100;
    const T               clength = T(grid.getDimension(descriptor.getDepth() - 1).x);
    const T               visclb = ulb * clength / static_cast<T>(Re);
    const T               omega = 1.0 / (3. * visclb + 0.5);
    const Neon::double_3d ulid(ulb, 0., 0.);

    //allocate fields
    auto fin = grid.newField<T>("fin", Q, 0);
    auto fout = grid.newField<T>("fout", Q, 0);
    auto storeSum = grid.newField<int>("storeSum", Q, 0);
    auto cellType = grid.newField<CellType>("CellType", 1, CellType::bulk);

    auto vel = grid.newField<T>("vel", 3, 0);
    auto rho = grid.newField<T>("rho", 1, 0);

    //init fields
    uint32_t numActiveVoxels = init<T, Q>(grid, storeSum, fin, fout, cellType, vel, rho, ulid);


    //skeleton
    std::vector<Neon::set::Container> containers;
    nonUniformTimestepRecursive<T, Q>(grid,
                                      fineInitStore,
                                      streamFusedExpl,
                                      streamFusedCoal,
                                      streamFuseAll,
                                      collisionFusedStore,
                                      omega,
                                      descriptor.getDepth() - 1,
                                      descriptor.getDepth(),
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
        if (!benchmark && t % 500 == 0) {
            postProcess<T, Q>(grid, Re, descriptor.getDepth(), fout, cellType, t, vel, rho, ulb, true, false);
        }
    }
    backend.syncAll();
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
    report.addMember("DeviceType", Neon::DeviceTypeUtil::toString(backend.devType()));

    //grid
    report.addMember("N", gridDim.x);
    report.addMember("Depth", depth);
    report.addMember("DataType", typeName());

    //problem
    report.addMember("ProblemID", problemID);
    report.addMember("ulb", ulb);
    report.addMember("Re", Re);
    report.addMember("clength", clength);
    report.addMember("visclb", visclb);
    report.addMember("omega", omega);

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
    postProcess<T, Q>(grid, Re, descriptor.getDepth(), fout, cellType, numIter, vel, rho, ulb, true, true);
}

int main(int argc, char** argv)
{
    Neon::init();

    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        report = Neon::Report("Lid Driven Cavity MultiRes LBM");
        report.commandLine(argc, argv);

        std::string deviceType = "gpu";
        int         deviceId = 99;
        int         numIter = 2;
        bool        benchmark = true;
        bool        fineInitStore = false;
        bool        streamFusedExpl = false;
        bool        streamFusedCoal = false;
        bool        streamFuseAll = false;
        bool        collisionFusedStore = false;
        int         problemId = 0;
        std::string dataType = "float";

        auto cli =
            (clipp::option("--deviceType") & clipp::value("deviceType", deviceType) % "Type of device (gpu, cpu)",
             clipp::option("--deviceId") & clipp::integers("deviceId", deviceId) % "Device id",
             clipp::option("--numIter") & clipp::integer("numIter", numIter) % "LBM number of iterations",
             clipp::option("--problemId") & clipp::integer("problemId", problemId) % "Problem ID (0 or 1)",
             clipp::option("--dataType") & clipp::value("dataType", dataType) % "Data type (float or double)",

             ((clipp::option("--benchmark").set(benchmark, true) % "Run benchmark mode") |
              (clipp::option("--visual").set(benchmark, false) % "Run export partial data")),

             ((clipp::option("--storeFine").set(fineInitStore, true) % "Initiate the Store operation from the fine level") |
              (clipp::option("--storeCoarse").set(fineInitStore, false) % "Initiate the Store operation from the coarse level") |
              (clipp::option("--collisionFusedStore").set(collisionFusedStore, true) % "Fuse Collision with Store operation")),

             ((clipp::option("--streamFusedExpl").set(streamFusedExpl, true) % "Fuse Stream with Explosion") |
              (clipp::option("--streamFusedCoal").set(streamFusedCoal, true) % "Fuse Stream with Coalescence") |
              (clipp::option("--streamFuseAll").set(streamFuseAll, true) % "Fuse Stream with Coalescence and Explosion")));


        if (!clipp::parse(argc, argv, cli)) {
            auto fmt = clipp::doc_formatting{}.doc_column(31);
            std::cout << make_man_page(cli, argv[0], fmt) << '\n';
            return -1;
        }


        //Neon grid
        Neon::Runtime runtime = Neon::Runtime::stream;
        if (deviceType == "cpu") {
            runtime = Neon::Runtime::openmp;
        }

        std::vector<int> gpu_ids{deviceId};
        Neon::Backend    backend(gpu_ids, runtime);

        constexpr int Q = 19;
        if (dataType == "float") {
            runNonUniformLBM<float, Q>(problemId, backend, numIter, fineInitStore, streamFusedExpl, streamFusedCoal, streamFuseAll, collisionFusedStore, benchmark);
        } else if (dataType == "double") {
            runNonUniformLBM<double, Q>(problemId, backend, numIter, fineInitStore, streamFusedExpl, streamFusedCoal, streamFuseAll, collisionFusedStore, benchmark);
        } else {
            Neon::NeonException exp("app-lbmMultiRes");
            exp << "Input data type " << dataType;
            NEON_THROW(exp);
        }
    }
    return 0;
}