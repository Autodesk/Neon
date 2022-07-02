#pragma once

#include <functional>

#include "Neon/domain/aGrid.h"
#include "Neon/domain/eGrid.h"

#include "gtest/gtest.h"
#include "sUt_common.h"

using aGrid_t = Neon::domain::aGrid;
using eGrid_t = Neon::domain::eGrid;
using dGrid_t = Neon::domain::dGrid;

namespace {
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
                    std::string backend_name = (backend == Neon::Runtime::openmp) ? "openmp" : "stream";
                    std::cout << "ngpu " << ngpu << " cardinality " << card << " dim " << dim << " backend " << backend_name << std::endl;
                    f(dim, ngpu, card, backend);
                }
            }
        }
    }
}

void runOneTestConfiguration(std::function<void(Neon::int64_3d, int, int, const Neon::Runtime&)> f, int)
{
    std::vector<int>              nGpuTest{2};
    std::vector<int>              cardinalityTest{1};
    std::vector<Neon::index64_3d> dimTest{{7, 7, 7}};

    std::vector<Neon::Runtime> backendTest{Neon::Runtime::openmp};
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        backendTest.push_back(Neon::Runtime::stream);
    }


    for (const auto& ngpu : nGpuTest) {
        for (const auto& card : cardinalityTest) {
            for (const auto& dim : dimTest) {
                for (const auto& backend : backendTest) {
                    std::string backend_name = (backend == Neon::Runtime::openmp) ? "openmp" : "stream";
                    std::cout << "ngpu " << ngpu << " cardinality " << card << " dim " << dim << " backend " << backend_name << std::endl;
                    f(dim, ngpu, card, backend);
                }
            }
        }
    }
}
}  // namespace