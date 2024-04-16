#pragma once
#include <map>
#include "gtest/gtest.h"

#include "Neon/core/core.h"
#include "Neon/core/tools/io/ioToVti.h"
#include "Neon/core/types/DataUse.h"
#include "Neon/core/types/DeviceType.h"

#include "Neon/domain/dGrid.h"
#include "Neon/domain/tools/Geometries.h"
#include "Neon/domain/tools/TestData.h"

#include "Neon/domain/bGrid.h"
#include "gtest/gtest.h"

using namespace Neon;
using namespace Neon::domain;

using namespace Neon::domain::tool::testing;
using namespace Neon::domain::tool;

template <typename G, typename T, int C>
void runAllTestConfiguration(
    std::function<void(TestData<G, T, C>&)> f,
    [[maybe_unused]] int                    nGpus,
    [[maybe_unused]] int                    minNumGpus)
{
    std::vector<int> nGpuTest;
    for (int i = minNumGpus; i <= nGpus; i++) {
        nGpuTest.push_back(i);
    }
    // std::vector<int> nGpuTest{2,4,6,8};
    std::vector<int> cardinalityTest{1,3,19};

    std::vector<Neon::index_3d> dimTest{{10, 17, 13}, {1, 1, 100}, {17, 1, 77}};
    std::vector<Neon::Runtime>  runtimeE{Neon::Runtime::openmp};
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        runtimeE.push_back(Neon::Runtime::stream);
    }

    std::vector<Geometry> geos;

    if constexpr (std::is_same_v<G, Neon::dGrid>) {
        geos = std::vector<Geometry>{
            Geometry::FullDomain,
        };
    } else {
        geos = std::vector<Geometry>{
            Geometry::FullDomain,
            //            Geometry::Sphere,
            //            Geometry::HollowSphere,
        };
    }

    for (auto& dim : dimTest) {
        for (const auto& card : cardinalityTest) {
            for (auto& geo : geos) {
                for (const auto& ngpu : nGpuTest) {
                    for (const auto& runtime : runtimeE) {
                        int maxnGPUs = [] {
                            if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
                                return Neon::set::DevSet::maxSet().setCardinality();
                            }
                            return 1;
                        }();

                        std::vector<int> ids;
                        for (int i = 0; i < ngpu; i++) {
                            ids.push_back(i % maxnGPUs);
                        }

                        Neon::Backend       backend(ids, runtime);
                        Neon::MemoryOptions memoryOptions = backend.getMemoryOptions();

                        if constexpr (G::executionThreadSpan == Neon::set::details::ExecutionThreadSpan::d1b3) {
                            if (dim.z < 8 * ngpu * 3) {
                                dim.z = ngpu * 3 * 8;
                            }
                        }

                        TestData<G, T, C> testData(backend,
                                                   dim,
                                                   card,
                                                   memoryOptions,
                                                   geo);

                        NEON_INFO(testData.toString());

                        f(testData);
                    }
                }
            }
        }
    }
}

#if 0

template <typename G, typename T, int C>
void runOneTestConfiguration(const std::string&                      gname,
                             std::function<void(TestData<G, T, C>&)> f,
                             int                                     nGpus,
                             int                                     minNumGpus = 1)
{
    std::vector<int> nGpuTest{2};
    std::vector<int> cardinalityTest{1};

    std::vector<Neon::index_3d> dimTest{{1, 1, 10}};
    std::vector<Neon::Runtime>  runtimeE{Neon::Runtime::openmp};
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        runtimeE.push_back(Neon::Runtime::stream);
    }

    std::vector<Geometry> geos = std::vector<Geometry>{
        Geometry::FullDomain};

    for (const auto& dim : dimTest) {
        for (const auto& card : cardinalityTest) {
            for (auto& geo : geos) {
                for (const auto& ngpu : nGpuTest) {
                    for (const auto& runtime : runtimeE) {
                        int maxnGPUs = Neon::set::DevSet::maxSet().setCardinality();

                        std::vector<int> ids;
                        for (int i = 0; i < ngpu; i++) {
                            ids.push_back(i % maxnGPUs);
                        }

                        Neon::Backend       backend(ids, runtime);
                        Neon::MemoryOptions memoryOptions = backend.getMemoryOptions();

                        TestData<G, T, C> testData(backend,
                                                   dim,
                                                   card,
                                                   memoryOptions,
                                                   geo);

                        NEON_INFO(testData.toString());

                        f(testData);
                    }
                }
            }
        }
    }
}
#endif