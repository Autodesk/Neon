#pragma once

#include <string>
#include <vector>
#include "Neon/core/tools/clipp.h"
#include "Neon/skeleton/Skeleton.h"

template <typename ComputeType>
struct LbmParameters
{
    ComputeType nu = 0;
    ComputeType omega = 0;
    ComputeType dx = 0;
    ComputeType dt = 0;
};

struct Config
{
    double                     Re = 100.;            // Reynolds number
    double                     ulb = 0.04;           // Velocity in lattice units
    int                        N = 160;              // Number of nodes in x-direction
    bool                       benchmark = false;    // Run in benchmark mode ?
    double                     max_t = 10.0;         // Non-benchmark mode: Total time in dim.less units
    int                        outFrequency = 200;   // Non-benchmark mode: Frequency in LU for output of terminal message and profiles (use 0 for no messages)
    int                        dataFrequency = 0;    // Non-benchmark mode: Frequency in LU of full data dump (use 0 for no data dump)
    int                        benchIniIter = 1000;  // Benchmark mode: Number of warmup iterations
    int                        benchMaxIter = 2000;  // Benchmark mode: Total number of iterations
    int                        repetitions = 1;      // Benchmark mode: number of time the test is run
    std::string                deviceType = "gpu";
    std::vector<int>           devices = std::vector<int>(0);                // Devices for the execution
    std::string                reportFile = "lbm-lid-driven-cavity-flow";    // Report file name
    std::string                gridType = "dGrid";                           // Neon grid type
    Neon::skeleton::Occ        occ = Neon::skeleton::Occ::none;              // Neon OCC type
    Neon::set::TransferMode    transferMode = Neon::set::TransferMode::get;  // Neon transfer mode for halo update
    Neon::set::StencilSemantic stencilSemantic = Neon::set::StencilSemantic::streaming;
    bool                       vti = false;  // Export vti file
    std::string                computeType = "double";
    std::string                storeType = "double";

    LbmParameters<double> mLbmParameters;

    auto toString()
        const -> std::string;

    auto parseArgs(int argc, char* argv[])
        -> int;

    template <class ComputeType>
    auto getLbmParameters()
        -> LbmParameters<ComputeType>
    {
        LbmParameters<ComputeType> output;
        output.nu = static_cast<ComputeType>(mLbmParameters.nu);
        output.omega = static_cast<ComputeType>(mLbmParameters.omega);
        output.dx = static_cast<ComputeType>(mLbmParameters.dx);
        output.dt = static_cast<ComputeType>(mLbmParameters.dt);

        return output;
    }

   private:
    auto helpSetLbmParameters()
        -> void;
};
