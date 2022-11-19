#pragma once

#include <string>
#include <vector>
#include "Neon/core/tools/clipp.h"
#include "Neon/skeleton/Skeleton.h"

struct Config
{
    double                      Re = 100.;                                    // Reynolds number
    double                      ulb = 0.02;                                   // Velocity in lattice units
    int                         N = 128;                                      // Number of nodes in x-direction
    bool                        benchmark = false;                            // Run in benchmark mode ?
    double                      max_t = 10.0;                                 // Non-benchmark mode: Total time in dim.less units
    int                         outFrequency = 200;                           // Non-benchmark mode: Frequency in LU for output of terminal message and profiles (use 0 for no messages)
    int                         dataFrequency = 0;                            // Non-benchmark mode: Frequency in LU of full data dump (use 0 for no data dump)
    int                         benchIniIter = 1000;                          // Benchmark mode: Number of warmup iterations
    int                         benchMaxIter = 2000;                          // Benchmark mode: Total number of iterations
    std::vector<int>            devices = std::vector<int>(0);                // Devices for the execution
    std::string                 reportFile = "lbm-lid-driven-cavity-flow";    // Report file name
    std::string                 gridType = "dGrid";                           // Neon grid type
    Neon::skeleton::Occ         occ = Neon::skeleton::Occ::none;              // Neon OCC type
    Neon::set::TransferMode     transferMode = Neon::set::TransferMode::get;  // Neon transfer mode for halo update
    Neon::set::TransferSemantic transferSemantic = Neon::set::TransferSemantic::lattice;
    bool                        vti = false;  // Export vti file

    clipp::group mClip;

    Config();

    auto toString()
        const -> std::string;

    auto getClip() -> clipp::group&;
};


struct LBMparameters
{
    double nu = 0;
    double omega = 0;
    double dx = 0;
    double dt = 0;
};