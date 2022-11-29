#pragma once
#include <iomanip>
#include "Config.h"
#include "Neon/Neon.h"
#include "Neon/set/Backend.h"
#include "Repoert.h"

namespace metrics {
// Return a new clock for the current time, for benchmarking.
namespace {
auto restartClock(Neon::Backend& bk, bool sync = true)
{
    if (sync) {
        bk.syncAll();
    }
    return make_pair(std::chrono::high_resolution_clock::now(), 0);
}
}  // namespace

// Compute the time elapsed since a starting point, and the corresponding
// benchmarks of the code in Mega Lattice site updates per second (MLups).
template <class TimePoint>
void recordMetrics(Neon::Backend& bk,
                   const Config&  config,
                   Report&        report,
                   TimePoint      start,
                   int            clock_iter)
{
    bk.syncAll();
    size_t nElements = config.N * config.N * config.N;
    auto   stop = std::chrono::high_resolution_clock::now();
    auto   duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    double mlups = static_cast<double>(nElements * clock_iter) / duration.count();

    report.recordLoopTime(duration.count(), "microseconds");
    report.recordMLUPS(mlups);

    std::cout << "Metrics: " << std::endl;
    std::cout << "     time: " << std::setprecision(4) << duration.count() << " microseconds" << std::endl;
    std::cout << "    MLUPS: " << std::setprecision(4) << mlups << " MLUPS" << std::endl;
}

template <class TimePoint>
void recordGridInitMetrics(Neon::Backend& bk,
                           Report&        report,
                           TimePoint      start)
{
    bk.syncAll();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    report.recordNeonGridInitTime(duration.count(), "microseconds");

    std::cout << "Metrics: " << std::endl;
    std::cout << "    Grid Init: " << std::setprecision(4) << duration.count() << " microseconds" << std::endl;
}
}  // namespace metrics