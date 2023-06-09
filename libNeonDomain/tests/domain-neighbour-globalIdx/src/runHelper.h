#pragma once
#include <map>
#include "gtest/gtest.h"

#include "Neon/core/core.h"
#include "Neon/core/tools/io/ioToVti.h"
#include "Neon/core/types/DataUse.h"
#include "Neon/core/types/DeviceType.h"

#include "Neon/domain/dGrid.h"
#include "Neon/domain/eGrid.h"
#include "Neon/domain/tools/Geometries.h"
#include "Neon/domain/tools/TestData.h"

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
    std::vector<int> cardinalityTest{1};

    std::vector<Neon::index_3d> dimTest{{17, 33, 71}};
    std::vector<Neon::Runtime>  runtimeE{Neon::Runtime::openmp};
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        runtimeE.push_back(Neon::Runtime::stream);
    }

    std::vector<Geometry>           geos;
    std::vector<Neon::MemoryLayout> memoryLayoutOptions{Neon::MemoryLayout::structOfArrays, Neon::MemoryLayout::arrayOfStructs};
    // std::vector<Neon::MemoryLayout> memoryLayoutOptions{Neon::MemoryLayout::structOfArrays};

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

    for (auto dim : dimTest) {
        for (const auto& card : cardinalityTest) {
            for (auto& geo : geos) {
                for (const auto& ngpu : nGpuTest) {
                    for (const auto& runtime : runtimeE) {
                        for (const auto& memoryLayout : memoryLayoutOptions) {

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
                            memoryOptions.setOrder(memoryLayout);

                            if constexpr (std::is_same_v<G, Neon::bGrid>) {
                                if (dim.z < 8 * ngpu * 3) {
                                    dim.z = ngpu * 3 * 8;
                                }
                                if(memoryLayout == Neon::MemoryLayout::arrayOfStructs){
                                    continue ;
                                }
                            }

                            assert(card == 1);
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
}


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
