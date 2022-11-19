
#include "Config.h"
#include "Neon/core/tools/clipp.h"
#include "Neon/domain/dGrid.h"

Config config;

int main(int argc, char** argv)
{
    config.Re = 100.;         // Reynolds number
    config.ulb = 0.02;        // Velocity in lattice units
    config.N = 60;            // Number of nodes in x-direction
    config.benchmark = true;  // Run in benchmark mode ?
    config.max_t = 10.0;      // Non-benchmark mode: Total time in dim.less units
                              //    config.out_freq = 20000000;     // Non-benchmark mode: Frequency in LU for output of terminal message and profiles (use 0 for no messages)
                              //    config.data_freq = 20000000;    // Non-benchmark mode: Frequency in LU of full data dump (use 0 for no data dump)
                              //    config.bench_ini_iter = 0;      // Benchmark mode: Number of warmup iterations
                              //    config.bench_max_iter = 10000;  // Benchmark mode: Total number of iterations
                              //    config.perKeeperFile = "perf";
                              //    config.devices = {0};
                              //    config.gridType = "dGrid";
                              //    config.occ = Neon::skeleton::Options_t::Occ::none


    if (!clipp::parse(argc, argv, config.getClip())) {
        auto fmt = clipp::doc_formatting{}.doc_column(31);
        std::cout << make_man_page(config.getClip(), argv[0], fmt) << '\n';
        return -1;
    }

    std::cout << "------------ Parameters ------------";
    std::cout << config.toString();
    std::cout << " -------------------------------------";

    return 0;
}
