#pragma once

#include <string>
#include <vector>
#include "Collision.h"
#include "Neon/core/tools/clipp.h"
#include "Neon/domain/tools/SpaceCurves.h"
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
    double         Re = 100.;                // Reynolds number
    double         ulb = 0.04;               // Velocity in lattice units
    Neon::index_3d N = Neon::index_3d(160);  // Number of nodes in x-direction
    bool           benchmark = false;        // Run in benchmark mode ?
    double         max_t = 10.0;             // Non-benchmark mode: Total time in dim.less units
    int            benchIniIter = 1000;      // Benchmark mode: Number of warmup iterations
    int            benchMaxIter = 2000;      // Benchmark mode: Total number of iterations
    int            repetitions = 1;          // Benchmark mode: number of time the test is run

    std::string      deviceType = "gpu";
    std::vector<int> devices = std::vector<int>(0);              // Devices for the execution
    std::string      reportFile = "lbm-lid-driven-cavity-flow";  // Report file name

    std::vector<std::string> gridTypeOptions = {"dGrid", "eGrid", "bGrid"};
    std::string              gridType = gridTypeOptions[0];  // Neon grid type

    Neon::skeleton::OccUtils::Cli                         occCli{Neon::skeleton::Occ::none};              // Neon OCC type
    Neon::set::TransferModeUtils::Cli                     transferModeCli{Neon::set::TransferMode::get};  // Neon transfer mode for halo update
    Neon::set::StencilSemanticUtils::Cli                  stencilSemanticCli{Neon::set::StencilSemantic::lattice};
    Neon::domain::tool::spaceCurves::EncoderTypeUtil::Cli spaceCurveCli{Neon::domain::tool::spaceCurves::EncoderType::sweep};
    CollisionUtils::Cli                                   collisionCli{Collision::bgk};
    int                                                   vti = 0;  // Export vti file

    std::vector<std::string> computeTypeOptions = {"double", "float"};
    std::string              computeTypeStr = computeTypeOptions[0];

    std::vector<std::string> storeTypeOptions = {"double", "float"};
    std::string              storeTypeStr = storeTypeOptions[0];


    std::vector<std::string> latticeOptions = {"d3q19", "d3q27"};
    std::string              lattice = latticeOptions[0];

    std::vector<std::string> streamingMethodOption = {"push", "pull"};
    std::string              streamingMethod = "push";

    LbmParameters<double> mLbmParameters;

    std::string mArgv;

    auto getOptionList(std::vector<std::string> list, std::string defaultVal) -> std::string
    {
        std::stringstream s;
        for (int i = 0; i < int(list.size()); i++) {
            s << list[i];
            if (list[i] == defaultVal) {
                s << " (default) ";
            }
        }
        return s.str();
    }

    auto check(std::vector<std::string> list, std::string userValue) -> bool
    {
        for (int i = 0; i < int(list.size()); i++) {
            if (list[i] == userValue) {
                return true;
            }
        }
        return false;
    }

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

    auto fromArgStringToDim(std::string s)
    {
        // tokenize the string base on spaces
        std::vector<std::string> tokens;
        std::string              token;
        std::istringstream       tokenStream(s);
        while (std::getline(tokenStream, token, '_')) {
            tokens.push_back(token);
        }
        // if the number of tokens is one, the set all the component of N to the same number
        if (tokens.size() == 1) {
            N = Neon::index_3d(std::stoi(tokens[0]));
        } else if (tokens.size() == 3) {
            N = Neon::index_3d(std::stoi(tokens[0]), std::stoi(tokens[1]), std::stoi(tokens[2]));
        } else {
            throw std::runtime_error("Error parsing the dimension");
        }
    }

   private:
    auto helpSetLbmParameters()
        -> void;
};
