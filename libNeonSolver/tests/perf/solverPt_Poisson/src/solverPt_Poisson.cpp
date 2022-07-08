
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>

#include "Neon/Neon.h"

#include "Neon/Report.h"

#include "Neon/core/core.h"
#include "Neon/core/tools/clipp.h"
#include "Neon/set/DevSet.h"
#include "Neon/skeleton/Skeleton.h"
#include "Poisson.h"
#include "gtest/gtest.h"

using namespace Neon::set;
using namespace Neon::solver;
using namespace Neon::domain;

std::vector<int>             DEVICES;            // GPU device IDs
int                          DOMAIN_SIZE = 256;  // Number of voxels along each axis
size_t                       MAX_ITER = 10;      // Maximum iterations for the solver
double                       TOL = 1e-10;        // Absolute tolerance for use in converge check
int                          CARDINALITY = 1;
std::string                  GRID_TYPE = "dGrid";
std::string                  DATA_TYPE = "double";
std::string                  REPORT_FILENAME = "Poisson";
int                          TIMES = 1;
Neon::skeleton::Occ occE = Neon::skeleton::Occ::none;
Neon::set::TransferMode transferE = Neon::set::TransferMode::get;
int                          ARGC;
char**                       ARGV;

template <typename T>
int poissonPerfTestScalability()
{
    assert(GRID_TYPE == "dGrid" || GRID_TYPE == "eGrid" || GRID_TYPE == "bGrid");
    assert(DATA_TYPE == "double" || DATA_TYPE == "single");

    T ZMIN = -20.0;
    T ZMAX = +20.0;

    if (DEVICES.empty()) {
        DEVICES.push_back(0);
    }
    DevSet        deviceSet(Neon::DeviceType::CUDA, DEVICES);
    Neon::Backend backend(deviceSet, Neon::Runtime::stream);
    backend.setAvailableStreamSet(2);

    // Create a report
    Neon::Report report("Poisson_" + std::string(GRID_TYPE) + "_" + std::to_string(CARDINALITY) + "D_" + std::to_string(DEVICES.size()) + "GPUs");

    //report.setToken("Token404");

    report.commandLine(ARGC, ARGV);

    Neon::index_3d dom(DOMAIN_SIZE, DOMAIN_SIZE, DOMAIN_SIZE);

    report.addMember("maxIters", MAX_ITER);
    report.addMember("voxelDomain", dom.to_stringForComposedNames());
    report.addMember("cardinality", CARDINALITY);
    report.addMember("numGPUs", DEVICES.size());
    report.addMember("absTol", TOL);
    report.addMember("gridType", GRID_TYPE);
    report.addMember("dataType", DATA_TYPE);
    report.addMember("skeletonOCC", Neon::skeleton::OccUtils::toString(occE));
    report.addMember("skeletonTransferMode", Neon::set::TransferModeUtils::toString(transferE));

    // Run on different domain sizes
    std::vector<double> solveTime(TIMES);
    std::vector<double> totalTime(TIMES);
    std::vector<double> residualStart(TIMES);
    std::vector<double> residualEnd(TIMES);
    std::vector<size_t> numIterations(TIMES);
    for (int t = 0; t < TIMES; ++t) {
        SolverResultInfo result;
        SolverStatus     status = SolverStatus::Error;

        if (CARDINALITY == 1) {
            std::array<T, 1> bdZMin{ZMIN};
            std::array<T, 1> bdZMax{ZMAX};
            if (GRID_TYPE == "eGrid") {
                std::tie(result, status) = testPoissonContainers<eGrid, T, 1>(backend, "CG", DOMAIN_SIZE, bdZMin, bdZMax, MAX_ITER, static_cast<T>(TOL), occE, transferE);
            } else if (GRID_TYPE == "dGrid") {
                std::tie(result, status) = testPoissonContainers<dGrid, T, 1>(backend, "CG", DOMAIN_SIZE, bdZMin, bdZMax, MAX_ITER, static_cast<T>(TOL), occE, transferE);
            } else if (GRID_TYPE == "bGrid") {
                std::tie(result, status) = testPoissonContainers<bGrid, T, 1>(backend, "CG", DOMAIN_SIZE, bdZMin, bdZMax, MAX_ITER, static_cast<T>(TOL), occE, transferE);
            }
        } else if (CARDINALITY == 3) {
            std::array<T, 3> bdZMin{0, ZMIN, 0};
            std::array<T, 3> bdZMax{0, 0, ZMAX};
            if (GRID_TYPE == "eGrid") {
                std::tie(result, status) = testPoissonContainers<eGrid, T, 3>(backend, "CG", DOMAIN_SIZE, bdZMin, bdZMax, MAX_ITER, static_cast<T>(TOL), occE, transferE);
            } else if (GRID_TYPE == "dGrid") {
                std::tie(result, status) = testPoissonContainers<dGrid, T, 3>(backend, "CG", DOMAIN_SIZE, bdZMin, bdZMax, MAX_ITER, static_cast<T>(TOL), occE, transferE);
            } else if (GRID_TYPE == "bGrid") {
                std::tie(result, status) = testPoissonContainers<bGrid, T, 3>(backend, "CG", DOMAIN_SIZE, bdZMin, bdZMax, MAX_ITER, static_cast<T>(TOL), occE, transferE);
            }
        }

        // Store results
        solveTime[t] = result.solveTime;
        totalTime[t] = result.totalTime;
        residualStart[t] = result.residualStart;
        residualEnd[t] = result.residualEnd;
        numIterations[t] = result.numIterations;
    }
    report.addMember("TimeToSolution_ms", solveTime);
    report.addMember("TimeTotal_ms", totalTime);
    report.addMember("ResidualStart", residualStart);
    report.addMember("ResidualFinal", residualEnd);
    report.addMember("IterationsTaken", numIterations);

    std::stringstream stringstream;
    stringstream << "Saving report file here: " << REPORT_FILENAME << std::endl;
    NEON_INFO(stringstream.str());

    report.write(REPORT_FILENAME);
    return 0;
}

int main(int argc, char** argv)
{
    ARGC = argc;
    ARGV = argv;

    Neon::init();

    // CLI for performance test
    auto cli =
        (clipp::option("--gpus") & clipp::integers("gpus", DEVICES) % "GPU ids to use",
         clipp::option("--grid") & clipp::value("grid", GRID_TYPE) % "Could be eGrid, dGrid, or bGrid",
         clipp::option("--data_type") & clipp::value("data_type", DATA_TYPE) % "Could be single or double",
         clipp::option("--cardinality") & clipp::value("cardinality", CARDINALITY) % "Must be 1 or 3",
         clipp::option("--domain_size") & clipp::integer("domain_size", DOMAIN_SIZE) % "Voxels along each dimension of the cube domain",
         clipp::option("--max_iter") & clipp::integer("max_iter", MAX_ITER) % "Maximum solver iterations",
         clipp::option("--tol") & clipp::number("tol", TOL) % "Absolute tolerance for convergence",
         clipp::option("--report_filename ") & clipp::value("report_filename", REPORT_FILENAME) % "Output report filename",
         clipp::option("--times ") & clipp::integer("times", TIMES) % "Times to run the experiment",
         ((clipp::option("--sOCC ").set(occE, Neon::skeleton::Occ::standard) % "Standard OCC") |
          (clipp::option("--nOCC ").set(occE, Neon::skeleton::Occ::none) % "No OCC (on by default)") |
          (clipp::option("--eOCC ").set(occE, Neon::skeleton::Occ::extended) % "Extended OCC") |
          (clipp::option("--e2OCC ").set(occE, Neon::skeleton::Occ::twoWayExtended) % "Two-way Extended OCC")),
         ((clipp::option("--put ").set(transferE, Neon::set::TransferMode::put) % "Set transfer mode to GET") |
          (clipp::option("--get ").set(transferE, Neon::set::TransferMode::get) % "Set transfer mode to PUT (on by default)")));


    if (!clipp::parse(argc, argv, cli)) {
        auto fmt = clipp::doc_formatting{}.doc_column(31);
        std::cout << make_man_page(cli, argv[0], fmt) << '\n';
        return -1;
    }
    std::cout << " #gpus= " << (DEVICES.empty() ? 1 : DEVICES.size()) << "\n";
    std::cout << " grid= " << GRID_TYPE << "\n";
    std::cout << " data_type= " << DATA_TYPE << "\n";
    std::cout << " cardinality= " << CARDINALITY << "\n";
    std::cout << " domain_size= " << DOMAIN_SIZE << "\n";
    std::cout << " max_iter= " << MAX_ITER << "\n";
    std::cout << " tol= " << TOL << "\n";
    std::cout << " times= " << TIMES << "\n";
    std::cout << " OCC= " << Neon::skeleton::OccUtils::toString(occE) << "\n";
    std::cout << " transfer= " << Neon::set::TransferModeUtils::toString(transferE) << "\n";

    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        if (DATA_TYPE == "single") {
            return poissonPerfTestScalability<float>();
        } else if (DATA_TYPE == "double") {
            return poissonPerfTestScalability<double>();
        } else {
            return -1;
        }
    } else {
        return 0;
    }
}
