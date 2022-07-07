#include "gtest/gtest.h"

#include "Neon/Neon.h"

#include "Neon/domain/bGrid.h"
#include "Neon/domain/dGrid.h"

#include "gUt_storage.h"

using dGrid_t = Neon::domain::dGrid;
using bGrid_t = Neon::domain::bGrid;


void runAllTestConfiguration(
    std::function<void(const Neon::int64_3d,
                       const int,
                       const int,
                       const Neon::Runtime&,
                       const Neon::MemoryLayout&,
                       const Neon::sys::patterns::Engine)> f,
    int                                                    maxNumGpu)
{
    std::vector<int> nGpuTest{};
    std::vector<int> cardinalityTest{1, 3, 5};

    std::vector<Neon::index64_3d> dimTest{
        {117, 100, 21},
        {33, 17, 47},
        {117, 100, 100},
        {33, 100, 100}};
    std::vector<Neon::Runtime> backendTest{
        Neon::Runtime::openmp, Neon::Runtime::stream};


    std::vector<Neon::MemoryLayout> layoutTest{
        Neon::MemoryLayout::arrayOfStructs,
        Neon::MemoryLayout::structOfArrays};

    std::vector<Neon::sys::patterns::Engine> engines{
        Neon::sys::patterns::Engine::CUB,
        Neon::sys::patterns::Engine::cuBlas};

    for (int i = 0; i < maxNumGpu; i++) {
        nGpuTest.push_back(i + 1);
    }

    for (const auto& ngpu : nGpuTest) {
        for (const auto& card : cardinalityTest) {
            for (const auto& dim : dimTest) {
                for (const auto& backend : backendTest) {
                    for (const auto& layout : layoutTest) {
                        for (const auto& eng : engines) {

                            std::stringstream s;
                            s << "ngpu " << ngpu
                              << " cardinality " << card
                              << " dim " << dim
                              << " backend " << Neon::Backend::toString(backend)
                              << " layout " << Neon::MemoryLayoutUtils::toString(layout)
                              << " engine " << (eng == Neon::sys::patterns::Engine::CUB ? "CUB" : "cuBlas");
                            NEON_INFO(s.str());
                            f(dim, ngpu, card, backend, layout, eng);

                            auto res = cudaDeviceReset();
                            if (res != cudaSuccess) {
                                Neon::NeonException exp("runAllTestConfiguration");
                                exp << " cudaDeviceReset failed with message!";
                                exp << cudaGetErrorString(res);
                                NEON_THROW(exp);
                            }
                        }
                    }
                }
            }
        }
    }
}

template <typename GridT, typename T>
void patternDotTest(const Neon::index64_3d            dim,
                    const int                         nGPU,
                    const int                         cardinality,
                    const Neon::Runtime&              backendType,
                    const Neon::MemoryLayout&         layout,
                    const Neon::sys::patterns::Engine eng)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        if (std::is_same_v<GridT, Neon::domain::bGrid> && eng == Neon::sys::patterns::Engine::cuBlas) {
            NEON_INFO("Skipped");
            return;
        }

        Storage<GridT, T> storage(dim, nGPU, cardinality, backendType, layout);
        storage.m_grid.setReduceEngine(eng);
        storage.initConst(-1, 1, 1, 1);

        auto scalar = storage.m_grid.template newPatternScalar<T>();

        auto dot_container = storage.m_grid.dot("GridDot", storage.Xf, storage.Yf, scalar);
        dot_container.run(Neon::Backend::mainStreamIdx);

        T ground_truth = storage.dot(storage.Xd, storage.Yd);

        ASSERT_NEAR(scalar(), ground_truth, 0.001) << ground_truth << " versus " << scalar();
    }
}

template <typename GridT, typename T>
void patternNorm2Test(const Neon::index64_3d            dim,
                      const int                         nGPU,
                      const int                         cardinality,
                      const Neon::Runtime&              backendType,
                      const Neon::MemoryLayout&         layout,
                      const Neon::sys::patterns::Engine eng)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        if (std::is_same_v<GridT, Neon::domain::bGrid> && eng == Neon::sys::patterns::Engine::cuBlas) {
            NEON_INFO("Skipped");
            return;
        }

        Storage<GridT, T> storage(dim, nGPU, cardinality, backendType, layout);
        storage.m_grid.setReduceEngine(eng);
        storage.initConst(-1, 1, 1, 1);

        auto scalar = storage.m_grid.template newPatternScalar<T>();

        auto norm2_container = storage.m_grid.norm2("GridNorm2", storage.Xf, scalar);
        norm2_container.run(Neon::Backend::mainStreamIdx);

        T ground_truth = storage.norm2(storage.Xd);

        ASSERT_NEAR(scalar(), ground_truth, 0.001);
    }
}


TEST(PatternContainerDot, bGrid)
{
    NEON_INFO("bGrid");
    int nGpus = 1;
    runAllTestConfiguration(patternDotTest<bGrid_t, double>, nGpus);
}

TEST(PatternContainerDot, dGrid)
{
    NEON_INFO("dGrid");
    int nGpus = 3;
    runAllTestConfiguration(patternDotTest<dGrid_t, double>, nGpus);
}

TEST(PatternContainerNorm2, bGrid)
{
    NEON_INFO("bGrid");
    int nGpus = 1;
    runAllTestConfiguration(patternNorm2Test<bGrid_t, double>, nGpus);
}

TEST(PatternContainerNorm2, dGrid)
{
    NEON_INFO("dGrid");
    int nGpus = 3;
    runAllTestConfiguration(patternNorm2Test<dGrid_t, double>, nGpus);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
