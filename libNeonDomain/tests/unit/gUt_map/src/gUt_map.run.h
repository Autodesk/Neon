#pragma once

#include <functional>
#include "Neon/domain/aGrid.h"
#include "Neon/domain/eGrid.h"
#include "gUt_map.storage.h"
#include "gtest/gtest.h"

using aGrid_t = Neon::domain::aGrid;
using eGrid_t = Neon::domain::eGrid;

namespace {


void runAllTestConfiguration(std::string                                       gridName,
                             std::function<void(Neon::int64_3d,
                                                int,
                                                int,
                                                const Neon::Runtime&,
                                                Neon::Timer_ms&,
                                                const Neon::MemSetOptions_t&)> f,
                             int                                               maxNumGpu = 3,
                             const Neon::MemSetOptions_t&                      memSetOptions = Neon::MemSetOptions_t())
{
    std::vector<int>              nGpuTest{};
    std::vector<int>              cardinalityTest{1};
    std::vector<Neon::index64_3d> dimTest{{1, 1, 3}};
    std::vector<Neon::Runtime>    backendTest{
        Neon::Runtime::openmp};
    if (maxNumGpu > 1) {
        maxNumGpu = 1;
    }
    for (int i = 0; i < maxNumGpu; i++) {
        nGpuTest.push_back(i + 1);
    }

    for (const auto& ngpu : nGpuTest) {
        for (const auto& card : cardinalityTest) {
            for (const auto& dim : dimTest) {
                for (const auto& backendType : backendTest) {
                    Neon::Timer_ms ms;
                    std::string    backend_name = (backendType == Neon::Runtime::openmp) ? "openmp" : "stream";

                    std::stringstream s;
                    s << "Grid " << gridName << " ngpu " << ngpu << " cardinality " << card << " dim " << dim << " backend " << backend_name;
                    NEON_INFO(s.str());
                    f(dim, ngpu, card, backendType, ms, memSetOptions);
                    NEON_INFO("Time: {}", ms.timeStr());
                }
            }
        }
    }
}

void runOneTestConfiguration(std::function<void(Neon::int64_3d,
                                                int,
                                                int,
                                                const Neon::Runtime&,
                                                Neon::Timer_ms&,
                                                const Neon::MemSetOptions_t&)> f,
                             int                                               maxNumGpu = 3,
                             const Neon::MemSetOptions_t&                      memSetOptions = Neon::MemSetOptions_t())
{
    std::vector<int> nGpuTest{};
    std::vector<int> cardinalityTest{
        2,
    };
    std::vector<Neon::index64_3d> dimTest{{1, 1, 4}};
    std::vector<Neon::Runtime>    backendTest{
        Neon::Runtime::openmp /*, Neon::Backend_t::runtime_et::e::stream*/};

    for (int i = 0; i < maxNumGpu; i++) {
        nGpuTest.push_back(i + 1);
    }
    nGpuTest = std::vector<int>{2};
    for (const auto& ngpu : nGpuTest) {
        for (const auto& card : cardinalityTest) {
            for (const auto& dim : dimTest) {
                for (const auto& backendType : backendTest) {
                    Neon::Timer_ms    ms;
                    std::string       backend_name = (backendType == Neon::Runtime::openmp) ? "openmp" : "stream";
                    std::stringstream s;
                    s << "ngpu " << ngpu << " cardinality " << card << " dim " << dim << " backend " << backend_name;
                    NEON_INFO(s.str());
                    f(dim, ngpu, card, backendType, ms, memSetOptions);
                    NEON_INFO("TIME {}", ms.timeStr());
                }
            }
        }
    }
}

}  // namespace