#include "Neon/core/tools/clipp.h"

#include "Neon/Neon.h"
#include "Neon/domain/mGrid.h"
#include "Neon/skeleton/Skeleton.h"


#include "collide.h"
#include "init.h"
#include "lattice.h"
#include "postProcess.h"
#include "store.h"
#include "stream.h"
#include "util.h"


template <typename T, int Q>
void nonUniformTimestepRecursive(Neon::domain::mGrid&                        grid,
                                 const T                                     omega0,
                                 const int                                   level,
                                 const int                                   numLevels,
                                 const Neon::domain::mGrid::Field<CellType>& cellType,
                                 Neon::domain::mGrid::Field<T>&              fin,
                                 Neon::domain::mGrid::Field<T>&              fout,
                                 std::vector<Neon::set::Container>&          containers)
{
    // 1) collision for all voxels at level L=level
    containers.push_back(collideBGK<T, Q>(grid, omega0, level, numLevels, cellType, fin, fout));
    //containers.push_back(collideBGKUnrolled<T, Q>(grid, omega0, level, numLevels, cellType, fin, fout));

    // 2) Storing fine (level - 1) data for later "coalescence" pulled by the coarse (level)
    if (level != numLevels - 1) {
        //containers.push_back(storeCoarse<T, Q>(grid, level + 1, fout));
        containers.push_back(storeFine<T, Q>(grid, level, fout));
    }


    // 3) recurse down
    if (level != 0) {
        nonUniformTimestepRecursive<T, Q>(grid, omega0, level - 1, numLevels, cellType, fin, fout, containers);
    }

    // 4) Streaming step that also performs the necessary "explosion" and "coalescence" steps.
    stream<T, Q>(grid, level, numLevels, cellType, fout, fin, containers);

    // 5) stop
    if (level == numLevels - 1) {
        return;
    }

    // 6) collision for all voxels at level L = level
    containers.push_back(collideBGK<T, Q>(grid, omega0, level, numLevels, cellType, fin, fout));
    //containers.push_back(collideBGKUnrolled<T, Q>(grid, omega0, level, numLevels, cellType, fin, fout));


    // 7) Storing fine(level) data for later "coalescence" pulled by the coarse(level)
    if (level != numLevels - 1) {
        //containers.push_back(storeCoarse<T, Q>(grid, level + 1, fout));
        containers.push_back(storeFine<T, Q>(grid, level, fout));
    }

    // 8) recurse down
    if (level != 0) {
        nonUniformTimestepRecursive<T, Q>(grid, omega0, level - 1, numLevels, cellType, fin, fout, containers);
    }

    // 9) Streaming step
    stream<T, Q>(grid, level, numLevels, cellType, fout, fin, containers);
}


template <typename T, int Q>
void runNonUniformLBM(const int           problemID,
                      const Neon::Backend backend,
                      const int           numIter,
                      const bool          benchmark)
{

    //constexpr int depth = 2;
    constexpr int depth = 3;

    float levelSDF[depth + 1];

    Neon::index_3d gridDim;

    //gridDim = Neon::index_3d(6, 6, 6);
    //levelSDF[0] = 0;
    //levelSDF[1] = -2.0 / 3.0;
    //levelSDF[2] = -1.0;

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

    //define the grid
    const Neon::domain::mGridDescriptor descriptor(depth);

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
    const T               Re = 100;
    const T               clength = T(grid.getDimension(descriptor.getDepth() - 1).x);
    const T               visclb = ulb * clength / Re;
    const T               omega = 1.0 / (3. * visclb + 0.5);
    const Neon::double_3d ulid(ulb, 0., 0.);

    //allocate fields
    auto fin = grid.newField<T>("fin", Q, 0);
    auto fout = grid.newField<T>("fout", Q, 0);
    auto cellType = grid.newField<CellType>("CellType", 1, CellType::bulk);

    auto vel = grid.newField<T>("vel", 3, 0);
    auto rho = grid.newField<T>("rho", 1, 0);

    //init fields
    uint32_t numActiveVoxels = init<T, Q>(grid, fin, fout, cellType, vel, rho, ulid);


    //skeleton
    std::vector<Neon::set::Container> containers;
    nonUniformTimestepRecursive<T, Q>(grid,
                                      omega,
                                      descriptor.getDepth() - 1,
                                      descriptor.getDepth(),
                                      cellType,
                                      fin, fout, containers);

    Neon::skeleton::Skeleton skl(grid.getBackend());
    skl.sequence(containers, "MultiResLBM");
    skl.ioToDot("MultiResLBM", "", true);

    //execution
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < numIter; ++t) {
        NEON_INFO("Non-uniform LBM Iteration: {}", t);
        skl.run();
        if (!benchmark && t % 100 == 0) {
            postProcess<T, Q>(grid, descriptor.getDepth(), fout, cellType, t, vel, rho, false);
        }
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    const double mlups = static_cast<double>(numIter * numActiveVoxels) / duration.count();
    const double eff_mlups = static_cast<double>(numIter * gridDim.rMul()) / duration.count();

    NEON_INFO("MLUPS = {}, numActiveVoxels = {}", mlups, numActiveVoxels);
    NEON_INFO("Effective MLUPS = {}, Effective numActiveVoxels = {}", eff_mlups, gridDim.rMul());

    postProcess<T, Q>(grid, descriptor.getDepth(), fout, cellType, numIter, vel, rho, true);
}

int main(int argc, char** argv)
{
    Neon::init();

    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        std::string deviceType = "gpu";
        int         deviceId = 0;
        int         numIter = 2;
        bool        benchmark = true;
        int         problemId = 0;

        auto cli =
            (clipp::option("--deviceType") & clipp::value("deviceType", deviceType) % "Type of device (gpu, cpu)",
             clipp::option("--deviceId") & clipp::integers("deviceId", deviceId) % "Device id",
             clipp::option("--numIter") & clipp::integer("numIter", numIter) % "LBM number of iterations",
             clipp::option("--problemId") & clipp::integer("problemId", problemId) % "Problem ID (0 or 1)",

             ((clipp::option("--benchmark").set(benchmark, true) % "Run benchmark mode") |
              (clipp::option("--visual").set(benchmark, false) % "Run export partial data")));


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

        using T = double;
        constexpr int Q = 19;

        runNonUniformLBM<T, Q>(problemId, backend, numIter, benchmark);
    }
    return 0;
}