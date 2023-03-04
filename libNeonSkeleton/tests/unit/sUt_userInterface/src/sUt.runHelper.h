#pragma once

#include <functional>
#include "Neon/domain/aGrid.h"
#include "Neon/domain/eGrid.h"
#include "gtest/gtest.h"
#include "sUt_common.h"

using aGrid_t = Neon::aGrid;
using eGrid_t = Neon::domain::eGrid;
using dGrid_t = Neon::dGrid;
using bGrid_t = Neon::domain::bGrid;

void runAllTestConfiguration(std::function<void(Neon::int64_3d, int, int, const Neon::Runtime&)> f, int maxNumGpu = 3)
{
    std::vector<int>              nGpuTest{};
    std::vector<int>              cardinalityTest{1, 2, 3, 4, 5};
    std::vector<Neon::index64_3d> dimTest{{117, 100, 21}, {33, 17, 47}, {117, 100, 100}, {33, 100, 100}};

    std::vector<Neon::Runtime> backendTest{Neon::Runtime::openmp};
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        backendTest.push_back(Neon::Runtime::stream);
    }

    for (int i = 0; i < maxNumGpu; i++) {
        nGpuTest.push_back(i + 1);
    }

    for (const auto& ngpu : nGpuTest) {
        for (const auto& card : cardinalityTest) {
            for (const auto& dim : dimTest) {
                for (const auto& backend : backendTest) {
                    std::string       backend_name = (backend == Neon::Runtime::openmp) ? "openmp" : "stream";
                    std::stringstream stringstream;
                    stringstream << "ngpu " << ngpu << " cardinality " << card << " dim " << dim << " backend " << backend_name << std::endl;
                    NEON_INFO(std::string("Test Configuration: ") + stringstream.str());
                    f(dim, ngpu, card, backend);
                }
            }
        }
    }
}