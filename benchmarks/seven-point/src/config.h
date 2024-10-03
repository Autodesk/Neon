#pragma once

#include <string>
#include <vector>
#include "Neon/core/tools/clipp.h"
#include "Neon/domain/tools/SpaceCurves.h"
#include "Neon/skeleton/Skeleton.h"

struct Config
{
    Neon::index_3d N = Neon::index_3d(160);  // Number of nodes in x-direction
    bool           benchmark = false;        // Run in benchmark mode ?
    int            benchIniIter = 1000;      // Benchmark mode: Number of warmup iterations
    int            benchMaxIter = 2000;      // Benchmark mode: Total number of iterations
    int            repetitions = 1;          // Benchmark mode: number of time the test is run

    std::string      deviceType = "gpu";
    std::vector<int> devices = std::vector<int>(0);              // Devices for the execution
    std::string      reportFile = "seven-point";  // Report file name

    std::vector<std::string> gridTypeOptions = {"dGrid", "eGrid", "bGrid"};
    std::string              gridType = gridTypeOptions[0];  // Neon grid type

    Neon::skeleton::OccUtils::Cli                         occCli{Neon::skeleton::Occ::none};              // Neon OCC type
    Neon::set::TransferModeUtils::Cli                     transferModeCli{Neon::set::TransferMode::get};  // Neon transfer mode for halo update
    Neon::set::StencilSemanticUtils::Cli                  stencilSemanticCli{Neon::set::StencilSemantic::standard};
    Neon::domain::tool::spaceCurves::EncoderTypeUtil::Cli spaceCurveCli{Neon::domain::tool::spaceCurves::EncoderType::sweep};
    int                                                   vti = 0;  // Export vti file

    std::vector<std::string> computeTypeOptions = {"double", "float"};
    std::string              computeTypeStr = computeTypeOptions[0];

    std::vector<std::string> dimSpaceOptions = {"2D", "3D"};
    std::string              dimSpaceStr = computeTypeOptions[0];

    std::string mArgv;

    auto getOptionList(std::vector<std::string> list, const std::string& defaultVal) -> std::string
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

    auto check(std::vector<std::string> list, const std::string& userValue) -> bool
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

};
