#include "Config.h"
#include "D3Q19.h"
#include "Neon/domain/bGrid.h"
#include "Neon/domain/dGrid.h"
#include "Neon/domain/details/dGridSoA/dGridSoA.h"
#include "Neon/domain/eGrid.h"

#include "CellType.h"
#include "LbmSkeleton.h"
#include "Metrics.h"
#include "Repoert.h"

namespace CavityTwoPop {

int backendWasReported = false;

namespace details {
template <typename Grid,
          typename Storage_,
          typename Compute_>
auto run(Config& config,
         Report& report) -> void
{
    using Storage = Storage_;
    using Compute = Compute_;
    using Precision = Precision<Storage, Compute>;
    using Lattice = D3Q19<Precision>;
    using PopulationField = typename Grid::template Field<Storage, Lattice::Q>;

    using PopField = typename Grid::template Field<typename Precision::Storage, Lattice::Q>;
    using CellTypeField = typename Grid::template Field<CellType, 1>;

    using Idx = typename PopField::Idx;
    using RhoField = typename Grid::template Field<typename Precision::Storage, 1>;
    using UField = typename Grid::template Field<typename Precision::Storage, 3>;

    using Skeleton = LbmSkeleton<Precision, Lattice, Grid>;
    using ContainerFactory = ContainerFactory<Precision, Lattice, Grid>;

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

    if (!backendWasReported) {
        metrics::recordBackend(bk, report);
        backendWasReported = true;
    }

    Neon::double_3d ulid(1., 0., 0.);
    // Neon Grid and Fields initialization
    auto [start, clock_iter] = metrics::restartClock(bk, true);
    Grid grid(
        bk, {config.N, config.N, config.N},
        [](const Neon::index_3d&) { return true; },
        Lattice::template getDirectionAsVector<Lattice::MemoryMapping>(),
        0.0, 1.0,
        config.spaceCurve);

    PopulationField pop0 = grid.template newField<Storage, Lattice::Q>("Population", Lattice::Q, Storage(0.0));
    PopulationField pop1 = grid.template newField<Storage, Lattice::Q>("Population", Lattice::Q, Storage(0.0));

    typename Grid::template Field<Storage, 1> rho;
    typename Grid::template Field<Storage, 3> u;

    if (!config.benchmark) {
        std::cout << "Allocating rho and u" << std::endl;
        rho = grid.template newField<Storage, 1>("rho", 1, Storage(0.0));
        u = grid.template newField<Storage, 3>("u", 3, Storage(0.0));
    }


    CellType defaultCelltype;
    auto     flag = grid.template newField<CellType, 1>("Material", 1, defaultCelltype);
    auto     lbmParameters = config.getLbmParameters<Compute>();

    Skeleton iteration(config.stencilSemantic,
                       config.occ,
                       config.transferMode,
                       pop0,
                       pop1,
                       flag,
                       lbmParameters.omega);

    auto exportRhoAndU = [&bk, &rho, &u, &iteration, &flag, &grid, &ulid](int iterationId) {
        if ((iterationId) % 100 == 0) {
            auto& f = iteration.getInput();
            {
                bk.syncAll();
                f.newHaloUpdate(Neon::set::StencilSemantic::standard,
                                Neon::set::TransferMode::get,
                                Neon::Execution::device)
                    .run(Neon::Backend::mainStreamIdx);
                bk.syncAll();
            }

            auto container = ContainerFactory::computeRhoAndU(f, flag, rho, u);
            container.run(Neon::Backend::mainStreamIdx);
            u.updateHostData(Neon::Backend::mainStreamIdx);
            rho.updateHostData(Neon::Backend::mainStreamIdx);
            // iteration.getInput().updateHostData(Neon::Backend::mainStreamIdx);

            bk.syncAll();
            size_t      numDigits = 5;
            std::string iterIdStr = std::to_string(iterationId);
            iterIdStr = std::string(numDigits - std::min(numDigits, iterIdStr.length()), '0') + iterIdStr;

            u.ioToVtk("u_" + iterIdStr, "u", false);
            rho.ioToVtk("rho_" + iterIdStr, "rho", false);
            // iteration.getInput().ioToVtk("pop_" + iterIdStr, "u", false);
            // flag.ioToVtk("flag_" + iterIdStr, "u", false);

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
        }
    };


    metrics::recordGridInitMetrics(bk, report, start);
    tie(start, clock_iter) = metrics::restartClock(bk, true);

    // Problem Setup
    // 1. init all lattice to equilibrium
    {
        auto& inPop = iteration.getInput();
        auto& outPop = iteration.getOutput();

        Neon::index_3d dim(config.N, config.N, config.N);

        //        const auto& t = Lattice::Memory::t;
        //        const auto& c = Lattice::Memory::stencil;

        ContainerFactory::problemSetup(inPop,
                                       outPop,
                                       flag,
                                       ulid,
                                       config.ulb)
            .run(Neon::Backend::mainStreamIdx);


        inPop.updateHostData(Neon::Backend::mainStreamIdx);
        outPop.updateHostData(Neon::Backend::mainStreamIdx);
        flag.updateHostData(Neon::Backend::mainStreamIdx);
        {
            bk.syncAll();
            flag.newHaloUpdate(Neon::set::StencilSemantic::standard /*semantic*/,
                               Neon::set::TransferMode::get /*transferMode*/,
                               Neon::Execution::device /*execution*/)
                .run(Neon::Backend::mainStreamIdx);
            bk.syncAll();
        }

        auto container = ContainerFactory::computeWallNghMask(flag, flag);
        container.run(Neon::Backend::mainStreamIdx);
        bk.syncAll();
    }

    metrics::recordProblemSetupMetrics(bk, report, start);

    // Reset the clock, to be used when a benchmark simulation is executed.
    tie(start, clock_iter) = metrics::restartClock(bk, true);

    int time_iter = 0;

    // The average energy, dependent on time, can be used to monitor convergence, or statistical
    // convergence, of the simulation.
    // Maximum number of time iterations depending on whether the simulation is in benchmark mode or production mode.
    // int max_time_iter = config.benchmark ? config.benchMaxIter : static_cast<int>(config.max_t / config.mLbmParameters.dt);
    int max_time_iter = config.benchMaxIter;

    for (time_iter = 0; time_iter < max_time_iter; ++time_iter) {
        if (!config.benchmark) {
            exportRhoAndU(time_iter);
        }

        if (config.benchmark && time_iter == config.benchIniIter) {
            std::cout << "Warm up completed (" << time_iter << " iterations ).\n"
                      << "Starting benchmark step ("
                      << config.benchMaxIter - config.benchIniIter << " iterations)."
                      << std::endl;
            tie(start, clock_iter) = metrics::restartClock(bk, false);
        }

        iteration.run();

        ++clock_iter;
    }
    std::cout << "Iterations completed" << std::endl;
    metrics::recordMetrics(bk, config, report, start, clock_iter);
}

template <typename Grid, typename Storage>
auto runFilterComputeType(Config& config, Report& report) -> void
{
    if (config.computeType == "double") {
        return run<Grid, Storage, double>(config, report);
    }
    if (config.computeType == "float") {
        return run<Grid, Storage, float>(config, report);
    }
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename Grid>
auto runFilterStoreType(Config& config,
                        Report& report)
    -> void
{
    if (config.storeType == "double") {
        return runFilterComputeType<Grid, double>(config, report);
    }
    if (config.storeType == "float") {
        return runFilterComputeType<Grid, float>(config, report);
    }
}
}  // namespace details

#ifdef NEON_BENCHMARK_DESIGN_OF_EXPERIMENTS
constexpr bool skipTest = false;
#else
constexpr bool skipTest = false;
#endif

auto run(Config& config,
         Report& report) -> void
{
    if (config.gridType == "dGrid") {
        return details::runFilterStoreType<Neon::dGrid>(config, report);
    }
    if (config.gridType == "eGrid") {
        if constexpr (!skipTest) {
            return details::runFilterStoreType<Neon::eGrid>(config, report);
        } else {
            NEON_THROW_UNSUPPORTED_OPERATION("This option was disables. PLease define NEON_BENCHMARK_DESIGN_OF_EXPERIMENTS to enable it.")
        }
    }
    if (config.gridType == "bGrid") {
        return details::runFilterStoreType<Neon::bGrid>(config, report);
    }
    if (config.gridType == "bGrid_4_4_4") {
        if constexpr (!skipTest) {
            using Sblock = Neon::domain::details::bGrid::StaticBlock<4, 4, 4>;
            using Grid = Neon::domain::details::bGrid::bGrid<Sblock>;
            return details::runFilterStoreType<Grid>(config, report);
        } else {
            NEON_THROW_UNSUPPORTED_OPERATION("This option was disables. PLease define NEON_BENCHMARK_DESIGN_OF_EXPERIMENTS to enable it.")
        }
    }
    if (config.gridType == "bGrid_2_2_2") {
        if constexpr (!skipTest) {
            using Sblock = Neon::domain::details::bGrid::StaticBlock<2, 2, 2>;
            using Grid = Neon::domain::details::bGrid::bGrid<Sblock>;
            return details::runFilterStoreType<Grid>(config, report);
        } else {
            NEON_THROW_UNSUPPORTED_OPERATION("This option was disables. PLease define NEON_BENCHMARK_DESIGN_OF_EXPERIMENTS to enable it.")
        }
    }
    if (config.gridType == "bGrid_32_8_4") {
        if constexpr (!skipTest) {
            using Sblock = Neon::domain::details::bGrid::StaticBlock<32, 8, 4>;
            using Grid = Neon::domain::details::bGrid::bGrid<Sblock>;
            return details::runFilterStoreType<Grid>(config, report);
        } else {
            NEON_THROW_UNSUPPORTED_OPERATION("This option was disables. PLease define NEON_BENCHMARK_DESIGN_OF_EXPERIMENTS to enable it.")
        }
    }
    if (config.gridType == "bGrid_32_8_4") {
        if constexpr (!skipTest) {
            using Sblock = Neon::domain::details::bGrid::StaticBlock<32, 8, 4>;
            using Grid = Neon::domain::details::bGrid::bGrid<Sblock>;
            return details::runFilterStoreType<Grid>(config, report);
        } else {
            NEON_THROW_UNSUPPORTED_OPERATION("This option was disables. PLease define NEON_BENCHMARK_DESIGN_OF_EXPERIMENTS to enable it.")
        }
    }
    if (config.gridType == "bGrid_32_2_8") {
        if constexpr (!skipTest) {
            using Sblock = Neon::domain::details::bGrid::StaticBlock<32, 2, 8>;
            using Grid = Neon::domain::details::bGrid::bGrid<Sblock>;
            return details::runFilterStoreType<Grid>(config, report);
        } else {
            NEON_THROW_UNSUPPORTED_OPERATION("This option was disables. PLease define NEON_BENCHMARK_DESIGN_OF_EXPERIMENTS to enable it.")
        }
    }
    if (config.gridType == "bGrid_32_8_2") {
        if constexpr (!skipTest) {
            using Sblock = Neon::domain::details::bGrid::StaticBlock<32, 8, 2>;
            using Grid = Neon::domain::details::bGrid::bGrid<Sblock>;
            return details::runFilterStoreType<Grid>(config, report);
        } else {
            NEON_THROW_UNSUPPORTED_OPERATION("This option was disables. PLease define NEON_BENCHMARK_DESIGN_OF_EXPERIMENTS to enable it.")
        }
    }
    if (config.gridType == "dGridSoA") {
        if constexpr (!skipTest) {
            return details::runFilterStoreType<Neon::domain::details::dGridSoA::dGridSoA>(config, report);
        } else {
            NEON_THROW_UNSUPPORTED_OPERATION("This option was disables. PLease define NEON_BENCHMARK_DESIGN_OF_EXPERIMENTS to enable it.")
        }
    }
}
}  // namespace CavityTwoPop
