#include "Config.h"
#include "D3Q19.h"
#include "Neon/domain/dGrid.h"

#include "CellType.h"
#include "Metrics.h"
#include "Repoert.h"

namespace CavityTwoPop {

auto run(Config& config,
         Report& report) -> void
{
    using Grid = Neon::domain::dGrid;
    using StorageFP = double;
    using ComputeFP = double;
    constexpr int latticeRank = 19;

    Neon::Backend bk(config.devices, Neon::Runtime::openmp);

    D3Q19 lattice = D3Q19<StorageFP, ComputeFP>(bk);

    // Neon Grid and Fields initialization
    auto [start, clock_iter] = metrics::restartClock(bk, true);
    Grid grid(
        bk, {config.N, config.N, config.N},
        [](const Neon::index_3d&) { return true; },
        lattice.points);

    auto pop0 = grid.newField<StorageFP, latticeRank>("Population", latticeRank, StorageFP(0.0));
    auto pop1 = grid.newField<StorageFP, latticeRank>("Population", latticeRank, StorageFP(0.0));

    CellType defaultCelltype;
    auto     flag = grid.newField<CellType, 1>("Material", 1, defaultCelltype);
    metrics::recordGridInitMetrics(bk, report, start);

    // Problem Setup


    // Reset the clock, to be used when a benchmark simulation is executed.
    tie(start, clock_iter) = metrics::restartClock(bk, true);

    int time_iter = 0;

    // The average energy, dependent on time, can be used to monitor convergence, or statistical
    // convergence, of the simulation.
    // Maximum number of time iterations depending on whether the simulation is in benchmark mode or production mode.
    int max_time_iter = config.benchmark ? config.benchMaxIter : static_cast<int>(config.max_t / config.mLbmParameters.dt);

    for (time_iter = 0; time_iter < max_time_iter; ++time_iter) {
        if (config.benchmark && time_iter == config.benchIniIter) {
            std::cout << "Warm up completed (" << time_iter << " iterations ).\n"
                      << "Starting benchmark step ("
                      << config.benchMaxIter - config.benchIniIter << " iterations)."
                      << std::endl;
            tie(start, clock_iter) = metrics::restartClock(bk, false);
        }

        usleep(10);

        ++clock_iter;
    }
    std::cout << "Iterations completed" << std::endl;
    metrics::recordMetrics(bk, config, report, start, clock_iter);
}
}  // namespace CavityTwoPop