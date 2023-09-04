#include "./Config.h"
#include "./Methods.h"
#include "./Metrics.h"
#include "./Repoert.h"
#include "CellType.h"
#include "ContainerFactory.h"
#include "ContainersD3Q19.h"
#include "ContainersD3QXX.h"
#include "D3Q19.h"
#include "Methods.h"
#include "Neon/Neon.h"
#include "Neon/set/Backend.h"
#include "Neon/set/Containter.h"
#include "Neon/skeleton/Skeleton.h"

int backendWasReported = false;

template <typename Grid_,
          lbm::Method method,
          Collision   CollisionId,
          typename Precision_,
          typename Lattice_>
struct Lbm
{
    using Grid = Grid_;
    using Lattice = Lattice_;
    using Precision = Precision_;

    using PField = typename Grid::template Field<typename Precision::Storage, Lattice::Q>;
    using CField = typename Grid::template Field<CellType, 1>;
    using RhoField = typename Grid::template Field<typename Precision::Storage, 1>;
    using UField = typename Grid::template Field<typename Precision::Storage, 3>;

    // using CommonContainerFactory = common::ContainerFactory<Precision, Lattice, Grid>;
    using ContainerFactory = ContainerFactoryD3QXX<Precision, Grid, Lattice, CollisionId>;

    template <typename Lambda>
    Lbm(Config& config,
        Report& report,
        Lambda  activeMask)
    {
        configurations = config;
        reportPtr = &report;

        // Setting the backend
        Neon::Backend bk = [&] {
            if (config.deviceType == "cpu") {
                Neon::Backend bk(config.devices, Neon::Runtime::openmp);
                return bk;
            }
            if (config.deviceType == "gpu") {
                Neon::Backend bk(config.devices, Neon::Runtime::stream);
                return bk;
            }
            Neon::NeonException exce("run");
            exce << config.deviceType << " is not a supported option as device type";
            NEON_THROW(exce);
        }();

        // Setting the grid
        grid = Grid(
            bk, {config.N, config.N, config.N},
            [&](const Neon::index_3d& p) { return activeMask(p); },
            Lattice::template getDirectionAsVector<Lattice::MemoryMapping>(),
            1.0, 0.0,
            config.spaceCurveCli.getOption());

        // Allocating Populations
        for (int i = 0; i < lbm::MethodUtils::getNumberOfPFields<method>(); i++) {
            std::stringstream name;
            name << "PopField_0" << i;
            using Storage = typename Precision::Storage;
            auto field = grid.template newField<Storage,
                                                Lattice::Q>(name.str(),
                                                            Lattice::Q,
                                                            Storage(0.0));
            pFieldList.push_back(field);
        }

        // Allocating cell type field
        CellType defaultCelltype;
        cellFlagField = grid.template newField<CellType, 1>("cellFlags", 1, defaultCelltype);

        // Allocating rho and u
        if (config.vti != 0) {
            std::cout << "Allocating rho and u" << std::endl;
            using Storage = typename Precision::Storage;
            rho = grid.template newField<Storage, 1>("rho", 1, Storage(0.0));
            u = grid.template newField<Storage, 3>("u", 3, Storage(0.0));
        }

        {  // Setting Equilibrium all population field
            for (auto& pField : pFieldList) {
                // Set all to eq
                ContainerFactory::Common::setToEquilibrium(pField, cellFlagField).run(Neon::Backend::mainStreamIdx);
            }
        }
    }

    // Lambda = void(*)(Neon::Index3d) -> std::tuple<BcType, Array<Storage, Lattice::Q>>
    template <typename Lambda>
    auto setBC(Lambda bcSetFunction) -> void
    {
        grid.getBackend().sync(Neon::Backend::mainStreamIdx);
        // Compute ngh mask
        ContainerFactory::Common::userSettingBc(bcSetFunction,
                                                pFieldList[0],
                                                cellFlagField)
            .run(Neon::Backend::mainStreamIdx);

        for (int i = 1; i < int(pFieldList.size()); i++) {
            ContainerFactory::Common::copyPopulation(pFieldList[0],
                                                     pFieldList[i])
                .run(Neon::Backend::mainStreamIdx);
        }
        cellFlagField.newHaloUpdate(Neon::set::StencilSemantic::standard,
                                    Neon::set::TransferMode::get,
                                    Neon::Execution::device)
            .run(Neon::Backend::mainStreamIdx);
        grid.getBackend().sync(Neon::Backend::mainStreamIdx);
        ContainerFactory::Common::computeWallNghMask(cellFlagField,
                                                     cellFlagField)
            .run(Neon::Backend::mainStreamIdx);
    }

    auto helpPrep() -> void
    {
        grid.getBackend().sync(Neon::Backend::mainStreamIdx);
        // One collide if 2Pop - pull
        // One iteration if 2Pop = push
        if constexpr (lbm::Method::pull == method) {
            // For pull we set up the system in a way that it does one single collide as first operation
            using Compute = typename Precision::Compute;
            auto lbmParameters = configurations.template getLbmParameters<Compute>();
            {
                skeleton = std::vector<Neon::skeleton::Skeleton>(2);
                for (int itr_ : {0, 1}) {
                    iteration = itr_;
                    int  skIdx = helpGetSkeletonIdx();
                    auto even = ContainerFactory::Pull::iteration(
                        configurations.stencilSemanticCli.getOption(),
                        pFieldList.at(helpGetInputIdx()),
                        cellFlagField,
                        lbmParameters.omega,
                        pFieldList.at(helpGetOutputIdx()));

                    std::vector<Neon::set::Container> ops;
                    skeleton.at(skIdx) = Neon::skeleton::Skeleton(pFieldList[0].getBackend());
                    Neon::skeleton::Options opt(configurations.occCli.getOption(), configurations.transferModeCli.getOption());
                    ops.push_back(even);
                    std::stringstream appName;

                    if (iteration % 2 == 0)
                        appName << "LBM_push_even";
                    else
                        appName << "LBM_pull_even";

                    skeleton.at(skIdx).sequence(ops, appName.str(), opt);
                }
            }
            {
                // Let's compute 1 collide operation to prepare the input of the first iteration
                iteration = 1;
                ContainerFactory::Pull::collideForStep0(pFieldList.at(helpGetInputIdx()),
                                                        cellFlagField,
                                                        lbmParameters.omega,
                                                        pFieldList.at(helpGetOutputIdx()))
                    .run(Neon::Backend::mainStreamIdx);
                pFieldList[0].getBackend().syncAll();
                iteration = 0;
            }
            return;
        }
        if constexpr (lbm::Method::push == method) {
            using Compute = typename Precision::Compute;
            auto lbmParameters = configurations.template getLbmParameters<Compute>();
            skeleton = std::vector<Neon::skeleton::Skeleton>(2);
            {
                iteration = 0;
                int  skIdx = helpGetSkeletonIdx();
                auto even = ContainerFactory::Push::iteration(
                    configurations.stencilSemanticCli.getOption(),
                    pFieldList.at(helpGetInputIdx()),
                    cellFlagField,
                    lbmParameters.omega,
                    pFieldList.at(helpGetOutputIdx()));

                std::vector<Neon::set::Container> ops;
                skeleton.at(skIdx) = Neon::skeleton::Skeleton(pFieldList[0].getBackend());
                Neon::skeleton::Options opt(configurations.occCli.getOption(), configurations.transferModeCli.getOption());
                ops.push_back(even);
                std::stringstream appName;
                appName << "LBM_push_even";
                skeleton.at(skIdx).sequence(ops, appName.str(), opt);
            }
            {
                iteration = 1;
                int  skIdx = helpGetSkeletonIdx();
                auto odd = ContainerFactory::Push::iteration(
                    configurations.stencilSemanticCli.getOption(),
                    pFieldList.at(helpGetInputIdx()),
                    cellFlagField,
                    lbmParameters.omega,
                    pFieldList.at(helpGetOutputIdx()));

                std::vector<Neon::set::Container> ops;
                skeleton.at(skIdx) = Neon::skeleton::Skeleton(pFieldList[0].getBackend());
                Neon::skeleton::Options opt(configurations.occCli.getOption(), configurations.transferModeCli.getOption());
                ops.push_back(odd);
                std::stringstream appName;
                appName << "LBM_push_odd";
                skeleton.at(skIdx).sequence(ops, appName.str(), opt);
            }

            {
                iteration = 1;
                int skIdx = helpGetSkeletonIdx();
                skeleton.at(skIdx).run();
                iteration = 0;
            }
            return;
        }
        if constexpr (lbm::Method::aa == method) {
            NEON_DEV_UNDER_CONSTRUCTION("");
            return;
        }
    }

    auto iterate() -> void
    {
        helpPrep();
        // Iteration keep track of all iterations
        // clock_iter keeps tracks of the iteration done after the last clock reset

        auto& bk = grid.getBackend();
        auto [start, clock_iter] = metrics::restartClock(bk, true);
        int time_iter = 0;
        // Reset the clock, to be used when a benchmark simulation is executed.
        tie(start, clock_iter) = metrics::restartClock(bk, true);

        for (time_iter = 0; time_iter < configurations.benchMaxIter; ++time_iter) {
            if ((time_iter % configurations.vti) == 0) {
                bk.syncAll();
                helpExportVti();
            }

            if (configurations.benchmark && time_iter == configurations.benchIniIter) {
                std::cout << "Warm up completed (" << time_iter << " iterations ).\n"
                          << "Starting benchmark step ("
                          << configurations.benchMaxIter - configurations.benchIniIter << " iterations)."
                          << std::endl;
                tie(start, clock_iter) = metrics::restartClock(bk, false);
            }

            skeleton[helpGetSkeletonIdx()].run();

            ++clock_iter;
            ++iteration;
        }
        std::cout << "Iterations completed" << std::endl;
        metrics::recordMetrics(bk, configurations, *reportPtr, start, clock_iter);
    }

    auto helpIterateOnce() -> void
    {
        if (lbm::Method::pull == method) {
            NEON_DEV_UNDER_CONSTRUCTION("");
            return;
        }
        if (lbm::Method::push == method) {
            skeleton.at(helpGetSkeletonIdx()).run(Neon::Backend::mainStreamIdx);
            return;
        }
        if (lbm::Method::aa == method) {
            NEON_DEV_UNDER_CONSTRUCTION("");
            return;
        }
    }

    auto helpExportVti() -> void
    {
        grid.getBackend().syncAll();
        auto& pop = pFieldList.at(helpGetOutputIdx());
        bool done = false;
        if constexpr (method == lbm::Method::push) {
            auto computeRhoAndU = ContainerFactory::Push::computeRhoAndU(pop, cellFlagField, rho, u);
            computeRhoAndU.run(Neon::Backend::mainStreamIdx);
            done= true;
        }
        if constexpr (method == lbm::Method::pull) {
            auto computeRhoAndU = ContainerFactory::Pull::computeRhoAndU(pop, cellFlagField, rho, u);
            computeRhoAndU.run(Neon::Backend::mainStreamIdx);
            done= true;
        }
        if(!done){
            NEON_DEV_UNDER_CONSTRUCTION("helpExportVti");
        }
        u.updateHostData(Neon::Backend::mainStreamIdx);
        rho.updateHostData(Neon::Backend::mainStreamIdx);
        // pop.updateHostData(Neon::Backend::mainStreamIdx);
        grid.getBackend().sync(Neon::Backend::mainStreamIdx);

        size_t      numDigits = 5;
        std::string iterIdStr = std::to_string(iteration);
        iterIdStr = std::string(numDigits - std::min(numDigits, iterIdStr.length()), '0') + iterIdStr;

        // pop.ioToVtk("pop_" + iterIdStr, "pop", false);
        u.ioToVtk("u_" + iterIdStr, "u", false);
        rho.ioToVtk("rho_" + iterIdStr, "rho", false);
        cellFlagField.template ioToVtk<int>("cellFlagField_" + iterIdStr, "flag", false);

#if 0
        std::vector<std::pair<double, double>> xPosVal;
        std::vector<std::pair<double, double>> yPosVal;
        const double scale = 1.0 / ulid.v[0];

        const Neon::index_3d grid_dim = grid.getDimension();
        u.forEachActiveCell([&](const Neon::index_3d& id, const int& card, auto& val) {
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

        // sort the position so the linear interpolation works
        std::sort(xPosVal.begin(), xPosVal.end(), [=](std::pair<double, double>& a, std::pair<double, double>& b) {
            return a.first < b.first;
        });

        std::sort(yPosVal.begin(), yPosVal.end(), [=](std::pair<double, double>& a, std::pair<double, double>& b) {
            return a.first < b.first;
        });

        auto writeToFile = [](const std::vector<std::pair<double, double>>& posVal, std::string filename) {
            std::ofstream file;
            file.open(filename);
            for (auto v : posVal) {
                file << v.first << " " << v.second << "\n";
            }
            file.close();
        };
        writeToFile(yPosVal, "NeonUniformLBM_" + iterIdStr + "_Y.dat");
        writeToFile(xPosVal, "NeonUniformLBM_" + iterIdStr + "_X.dat");
#endif
    }

    auto helpUpdateIterationCount() -> void
    {
        iteration++;
    }

    auto helpGetInputIdx() -> int
    {
        return iteration % 2;
    }
    auto helpGetOutputIdx() -> int
    {
        return (iteration + 1) % 2;
    }
    auto helpGetSkeletonIdx() -> int
    {
        return iteration % 2;
    }

    Config                                configurations;
    int                                   iteration = 0;
    bool                                  prepDone = false;
    Grid                                  grid;
    std::vector<PField>                   pFieldList;
    CField                                cellFlagField;
    RhoField                              rho;
    UField                                u;
    std::vector<Neon::skeleton::Skeleton> skeleton;
    Report*                               reportPtr;
};
