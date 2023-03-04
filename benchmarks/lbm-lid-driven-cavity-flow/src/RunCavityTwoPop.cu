#include "Config.h"
#include "D3Q19.h"
#include "Neon/domain/dGrid.h"

#include "CellType.h"
#include "LbmIteration.h"
#include "Metrics.h"
#include "Repoert.h"

namespace CavityTwoPop {

int backendWasReported = false;

namespace details {
template <typename Grid,
          typename StorageFP,
          typename ComputeFP>
auto run(Config& config,
         Report& report) -> void
{
    using Lattice = D3Q19Template<StorageFP, ComputeFP>;
    using PopulationField = typename Grid::template Field<StorageFP, Lattice::Q>;


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

    if(!backendWasReported) {
        metrics::recordBackend(bk, report);
        backendWasReported = true;
    }

    Neon::double_3d ulid(0., 1., 0.);
    Lattice         lattice(bk);

    // Neon Grid and Fields initialization
    auto [start, clock_iter] = metrics::restartClock(bk, true);
    Grid grid(
        bk, {config.N, config.N, config.N},
        [](const Neon::index_3d&) { return true; },
        lattice.c_vect);

    PopulationField pop0 = grid.template newField<StorageFP, Lattice::Q>("Population", Lattice::Q, StorageFP(0.0));
    PopulationField pop1 = grid.template newField<StorageFP, Lattice::Q>("Population", Lattice::Q, StorageFP(0.0));

    typename Grid::template Field<StorageFP, 1> rho;
    typename Grid::template Field<StorageFP, 3> u;

    if (!config.benchmark) {
        std::cout << "Allocating rho and u" << std::endl;
        rho = grid.template newField<StorageFP, 1>("rho", 1, StorageFP(0.0));
        u = grid.template newField<StorageFP, 3>("u", 3, StorageFP(0.0));
    }


    CellType defaultCelltype;
    auto     flag = grid.template newField<CellType, 1>("Material", 1, defaultCelltype);
    auto     lbmParameters = config.getLbmParameters<ComputeFP>();

    LbmIterationD3Q19<PopulationField, ComputeFP>
        iteration(config.stencilSemantic,
                  config.occ,
                  config.transferMode,
                  pop0,
                  pop1,
                  flag,
                  lbmParameters.omega);

    auto exportRhoAndU = [&bk, &rho, &u, &iteration, &flag](int iterationId) {
        if ((iterationId) % 100 == 0) {
            auto& f = iteration.getInput();
            bk.syncAll();
            Neon::set::HuOptions hu(Neon::set::TransferMode::get,
                                    false,
                                    Neon::Backend::mainStreamIdx,
                                    Neon::set::StencilSemantic::standard);

            f.haloUpdate(hu);
            bk.syncAll();
            auto container = LbmContainers<Lattice, PopulationField, ComputeFP>::computeRhoAndU(f, flag, rho, u);

            container.run(Neon::Backend::mainStreamIdx);
            u.updateIO(Neon::Backend::mainStreamIdx);
            rho.updateIO(Neon::Backend::mainStreamIdx);
            // iteration.getInput().updateIO(Neon::Backend::mainStreamIdx);

            bk.syncAll();
            size_t      numDigits = 5;
            std::string iterIdStr = std::to_string(iterationId);
            iterIdStr = std::string(numDigits - std::min(numDigits, iterIdStr.length()), '0') + iterIdStr;

            u.ioToVtk("u_" + iterIdStr, "u", false);
            rho.ioToVtk("rho_" + iterIdStr, "rho", false);
            // iteration.getInput().ioToVtk("pop_" + iterIdStr, "u", false);
            // flag.ioToVtk("flag_" + iterIdStr, "u", false);
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

        const auto& t = lattice.t_vect;
        const auto& c = lattice.c_vect;

        inPop.forEachActiveCell([&c, &t, &dim, &flag, &ulid, &config](const Neon::index_3d& idx,
                                                                      const int&            k,
                                                                      StorageFP&            val) {
            val = t.at(k);

            if (idx.x == 0 || idx.x == dim.x - 1 ||
                idx.y == 0 || idx.y == dim.y - 1 ||
                idx.z == 0 || idx.z == dim.z - 1) {

                if (idx.x == dim.x - 1) {
                    val = -6. * t.at(k) * config.ulb *
                          (c.at(k).v[0] * ulid.v[0] +
                           c.at(k).v[1] * ulid.v[1] +
                           c.at(k).v[2] * ulid.v[2]);
                } else {
                    val = 0;
                }
            }
        });

        outPop.forEachActiveCell([&c, &t, &dim, &flag, &ulid, &config](const Neon::index_3d& idx,
                                                                       const int&            k,
                                                                       StorageFP&            val) {
            val = t.at(k);

            if (idx.x == 0 || idx.x == dim.x - 1 ||
                idx.y == 0 || idx.y == dim.y - 1 ||
                idx.z == 0 || idx.z == dim.z - 1) {

                if (idx.x == dim.x - 1) {
                    val = -6. * t.at(k) * config.ulb *
                          (c.at(k).v[0] * ulid.v[0] +
                           c.at(k).v[1] * ulid.v[1] +
                           c.at(k).v[2] * ulid.v[2]);
                } else {
                    val = 0;
                }
            }
        });

        flag.forEachActiveCell([&dim](const Neon::index_3d& idx,
                                      const int&,
                                      CellType& flagVal) {
            flagVal.classification = CellType::bulk;
            flagVal.wallNghBitflag = 0;

            if (idx.x == 0 || idx.x == dim.x - 1 ||
                idx.y == 0 || idx.y == dim.y - 1 ||
                idx.z == 0 || idx.z == dim.z - 1) {

                flagVal.classification = CellType::bounceBack;

                if (idx.x == dim.x - 1) {
                    flagVal.classification = CellType::movingWall;
                }
            }
        });

        inPop.updateCompute(Neon::Backend::mainStreamIdx);
        outPop.updateCompute(Neon::Backend::mainStreamIdx);

        flag.updateCompute(Neon::Backend::mainStreamIdx);
        bk.syncAll();
        Neon::set::HuOptions hu(Neon::set::TransferMode::get,
                                false,
                                Neon::Backend::mainStreamIdx,
                                Neon::set::StencilSemantic::standard);

        flag.haloUpdate(hu);
        bk.syncAll();
        auto container = LbmContainers<Lattice, PopulationField, ComputeFP>::computeWallNghMask(flag, flag);
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

template <typename Grid, typename StorageFP>
auto runFilterComputeType(Config& config, Report& report) -> void
{
    if (config.computeType == "double") {
        return run<Grid, StorageFP, double>(config, report);
    }
    if (config.computeType == "float") {
        return run<Grid, StorageFP, float>(config, report);
    }
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename Grid>
auto runFilterStoreType(Config& config,
                        Report& report)
    -> void
{
    if (config.storeType == "double") {
        return runFilterComputeType<Neon::dGrid, double>(config, report);
    }
    if (config.storeType == "float") {
        return runFilterComputeType<Neon::dGrid, float>(config, report);
    }
    NEON_DEV_UNDER_CONSTRUCTION("");
}
}  // namespace details

auto run(Config& config,
         Report& report) -> void
{
    if (config.gridType == "dGrid") {
        return details::runFilterStoreType<Neon::dGrid>(config, report);
    }
    if (config.gridType == "eGrid") {
        NEON_DEV_UNDER_CONSTRUCTION("");
    }
    if (config.gridType == "bGrid") {
        NEON_DEV_UNDER_CONSTRUCTION("");
    }
}
}  // namespace CavityTwoPop
