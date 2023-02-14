#include "Config.h"
#include "D3Q19.h"
#include "Neon/domain/dGrid.h"

#include "CellType.h"
#include "LbmSkeleton.h"
#include "Metrics.h"
#include "Repoert.h"
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <fenv.h>
namespace CavityTwoPop {

int backendWasReported = false;

namespace details {
template <typename Grid,
          typename StorageFP,
          typename ComputeFP>
auto runSpecialized(Config& config,
                    Report& report) -> void
{
    using Lattice = D3Q19Template<StorageFP, ComputeFP>;
    using PopulationField = typename Grid::template Field<StorageFP, Lattice::Q>;

    feenableexcept(FE_DIVBYZERO);

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

    Lattice               lattice(bk);
    const double          radiusDomainLenRatio = 1.0 / 7;
    const Neon::double_3d center = {config.N / 2.0, config.N / 2.0, config.N / 2.0};
    const double          radius = config.N * radiusDomainLenRatio;
    const double          rhoPrescribedInlet = 1.0;
    const double          rhoPrescribedOutlet = 1.005;

    auto isFluidDomain =
        [&](const Neon::index_3d& idx)
        -> bool {
        if (idx < 0)
            return false;
        if (idx.x >= config.N ||
            idx.y >= config.N ||
            idx.z >= config.N) {
            return false;
        }
        const auto point = idx.newType<double>();
        const auto offset = std::pow(point.x - center.x, 2) +
                            std::pow(point.y - center.y, 2) +
                            std::pow(point.z - center.z, 2);
        if (offset <= radius * radius) {
            // we are in the sphere
            return false;
        }
        return true;
    };

    auto isInsideSphere =
        [&](const Neon::index_3d& idx) -> bool {
        if (idx.x < 0 ||
            idx.y < 0 ||
            idx.z < 0)
            return false;
        if (idx.x >= config.N ||
            idx.y >= config.N ||
            idx.z >= config.N) {
            return false;
        }
        const auto point = idx.newType<double>();
        const auto offset = std::pow(point.x - center.x, 2) +
                            std::pow(point.y - center.y, 2) +
                            std::pow(point.z - center.z, 2);
        if (offset <= radius * radius) {
            // we are in the sphere
            return true;
        }
        return false;
    };

    auto getBoundaryType =
        [&](const Neon::index_3d& idx) -> CellType::Classification {
        if (idx.z == 0 || idx.z == config.N - 1) {
            return CellType::Classification::bounceBack;
        }
        if (idx.y == 0 || idx.y == config.N - 1) {
            return CellType::Classification::bounceBack;
        }
        if (idx.x == 0 || idx.x == config.N - 1) {
            return CellType::Classification::bounceBack;
        }

        auto idEdge = [idx, config](int d1, int d2) {
            if ((idx.v[d1] == 1 && idx.v[d2] == 1) ||
                (idx.v[d1] == 1 && idx.v[d2] == config.N - 2) ||
                (idx.v[d1] == config.N - 2 && idx.v[d2] == 1) ||
                (idx.v[d1] == config.N - 2 && idx.v[d2] == config.N - 2)) {
                return true;
            }
            return false;
        };

        if (idEdge(0,1)) {
            return CellType::Classification::bulk;
        }
        if (idEdge(0,2)) {
            return CellType::Classification::bulk;
        }
        if (idEdge(1,2)) {
            return CellType::Classification::bulk;
        }

        if (idx.x == 1) {
            return CellType::Classification::pressure;
        }
        if (idx.x == config.N - 2) {
            return CellType::Classification::velocity;
        }
        if (isInsideSphere(idx)) {
            return CellType::Classification::undefined;
        }
        for (int i = -1; i < 2; i++) {
            for (int j = -1; j < 2; j++) {
                for (int k = -1; k < 2; k++) {
                    Neon::index_3d offset(i, j, k);
                    Neon::index_3d neighbour = idx + offset;
                    bool           isIn = isInsideSphere(neighbour);
                    if (isIn) {
                        return CellType::Classification::bounceBack;
                    }
                }
            }
        }
        return CellType::Classification::bulk;
    };

    // Neon Grid and Fields initialization
    auto [start, clock_iter] = metrics::restartClock(bk, true);
    Grid grid(
        bk,
        {config.N, config.N, config.N},
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
    auto     bcTypeForDebugging = grid.template newField<double, 1>("BCtype", 1, 33);

    auto lbmParameters = config.getLbmParameters<ComputeFP>();

    LbmIterationD3Q19<PopulationField, ComputeFP>
        iteration(config.stencilSemantic,
                  config.occ,
                  config.transferMode,
                  pop0,
                  pop1,
                  flag,
                  lbmParameters.omega);

    auto exportRhoAndU = [&bk, &rho, &u, &iteration, &flag](int iterationId) {
        if (true) {
            std::cout << "Exporting\n"
                      << std::endl;
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

        flag.forEachActiveCell([&](const Neon::index_3d& idx,
                                   const int&,
                                   CellType& flagVal) {
            flagVal.classification = CellType::undefined;
            flagVal.wallNghBitflag = 0;
            flagVal.classification = getBoundaryType(idx);

            bcTypeForDebugging.getReference(idx, 0) = static_cast<double>(flagVal.classification);
        });
        bcTypeForDebugging.ioToVtk("bcFlags", "cb", false);

        // Population initialization
        inPop.forEachActiveCell([&](const Neon::index_3d& idx,
                                    const int&            k,
                                    StorageFP&            val) {
            val = t.at(k);
            if (flag(idx, 0).classification == CellType::bounceBack) {
                val = 0;
            }
            if (flag(idx, 0).classification == CellType::pressure) {
                if (k == 0) {
                    flag.getReference(idx, 0).rho = rhoPrescribedOutlet;
                }
            }
            if (flag(idx, 0).classification == CellType::velocity) {
                if (k == 0) {
                    flag.getReference(idx, 0).rho = rhoPrescribedInlet;
                }
            }
        });

        inPop.forEachActiveCell([&](const Neon::index_3d& idx,
                                    const int&            k,
                                    StorageFP&            val) {

            if (flag(idx, 0).classification == CellType::pressure) {
                if (k == 0) {

                }
            }
            if (flag(idx, 0).classification == CellType::velocity) {
                if (k == 0) {
                    flag.getReference(idx, 0).rho = rhoPrescribedInlet;
                }
            }
        });

        outPop.forEachActiveCell([&](const Neon::index_3d& idx,
                                     const int&            k,
                                     StorageFP&            val) {
            val = t.at(k);
            if (flag(idx, 0).classification == CellType::bounceBack) {
                val = 0;
            }
            if (flag(idx, 0).classification == CellType::pressure) {
                if (k == 0) {
                    flag.getReference(idx, 0).rho = rhoPrescribedOutlet;
                }
            }
            if (flag(idx, 0).classification == CellType::velocity) {
                if (k == 0) {
                    flag.getReference(idx, 0).rho = rhoPrescribedInlet;
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
        return runSpecialized<Grid, StorageFP, double>(config, report);
    }
    if (config.computeType == "float") {
        return runSpecialized<Grid, StorageFP, float>(config, report);
    }
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename Grid>
auto runFilterStoreType(Config& config,
                        Report& report)
    -> void
{
    if (config.storeType == "double") {
        return runFilterComputeType<Neon::domain::dGrid, double>(config, report);
    }
    if (config.storeType == "float") {
        return runFilterComputeType<Neon::domain::dGrid, float>(config, report);
    }
    NEON_DEV_UNDER_CONSTRUCTION("");
}
}  // namespace details

auto runTwoPop(Config& config,
               Report& report) -> void
{
    if (config.gridType == "dGrid") {
        return details::runFilterStoreType<Neon::domain::dGrid>(config, report);
    }
    if (config.gridType == "eGrid") {
        NEON_DEV_UNDER_CONSTRUCTION("");
    }
    if (config.gridType == "bGrid") {
        NEON_DEV_UNDER_CONSTRUCTION("");
    }
}
}  // namespace CavityTwoPop
