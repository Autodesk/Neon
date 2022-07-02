#include "gtest/gtest.h"

#include "Neon/Neon.h"
#include "Neon/domain/dGrid.h"
#include "gUt_storage.h"

using dGrid_t = Neon::domain::dGrid;


void runAllTestConfiguration(
    std::function<void(const Neon::int64_3d,
                       const int,
                       const int,
                       const Neon::Runtime&,
                       const Neon::MemoryLayout&)> f,
    int                                            maxNumGpu)
{
    std::vector<int> nGpuTest{};
    std::vector<int> cardinalityTest{1, 2, 3, 4, 5};

    std::vector<Neon::index64_3d> dimTest{
        {117, 100, 21},
        {33, 17, 47},
        {117, 100, 100},
        {33, 100, 100}};
    std::vector<Neon::Runtime> backendTest{
        Neon::Runtime::openmp};
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        backendTest.push_back(Neon::Runtime::stream);
    }

    std::vector<Neon::MemoryLayout> layoutTest{
        Neon::MemoryLayout::arrayOfStructs,
        Neon::MemoryLayout::structOfArrays};


    for (int i = 0; i < maxNumGpu; i++) {
        nGpuTest.push_back(i + 1);
    }


    for (const auto& ngpu : nGpuTest) {
        for (const auto& card : cardinalityTest) {
            for (const auto& dim : dimTest) {
                for (const auto& backend : backendTest) {

                    for (const auto& layout : layoutTest) {

                        std::stringstream s;
                        s << "ngpu " << ngpu
                          << " cardinality " << card
                          << " dim " << dim
                          << " backend " << Neon::Backend::toString(backend)
                          << " layout " << Neon::MemoryLayoutUtils::toString(layout) << std::endl;
                        NEON_INFO(s.str());

                        f(dim, ngpu, card, backend, layout);

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

template <typename GridT, typename T>
void patternDotTest(const Neon::index64_3d    dim,
                    const int                 nGPU,
                    const int                 cardinality,
                    const Neon::Runtime&      backendType,
                    const Neon::MemoryLayout& layout)
{
    Storage<GridT, T> storage(dim, nGPU, cardinality, backendType, layout);
    storage.initConst(-1);

    auto output = storage.m_grid.getDevSet().template newMemDevSet<T>(Neon::DeviceType::CPU, Neon::Allocator::MALLOC, 1);

    Neon::set::patterns::BlasSet<T> blasHandle(storage.m_grid.getDevSet());

    T std_output = 0;
    T int_output = 0;
    T bd_output = 0;
    if (storage.m_grid.getBackend().devType() == Neon::DeviceType::CUDA) {
        auto streams = storage.m_grid.getDevSet().newStreamSet();
        blasHandle.setStream(streams);
        std_output = storage.Xf.dot(blasHandle, storage.Yf, output, Neon::DataView::STANDARD);
        int_output = storage.Xf.dot(blasHandle, storage.Yf, output, Neon::DataView::INTERNAL);
        bd_output = storage.Xf.dot(blasHandle, storage.Yf, output, Neon::DataView::BOUNDARY);
    } else {
        std_output = storage.Xf.dot(blasHandle, storage.Yf, output, Neon::DataView::STANDARD);
        int_output = storage.Xf.dot(blasHandle, storage.Yf, output, Neon::DataView::INTERNAL);
        bd_output = storage.Xf.dot(blasHandle, storage.Yf, output, Neon::DataView::BOUNDARY);
    }

    T ground_truth = storage.dot(storage.Xd, storage.Yd);
    ASSERT_NEAR(std_output, ground_truth, 0.001);
    if (nGPU > 1) {
        ASSERT_NEAR(int_output + bd_output, ground_truth, 0.001);
    }
}

template <typename GridT, typename T>
void patternNorm2Test(const Neon::index64_3d    dim,
                      const int                 nGPU,
                      const int                 cardinality,
                      const Neon::Runtime&      backendType,
                      const Neon::MemoryLayout& layout)
{
    Storage<GridT, T> storage(dim, nGPU, cardinality, backendType, layout);
    storage.initConst(-1);

    auto output = storage.m_grid.getDevSet().template newMemDevSet<T>(Neon::DeviceType::CPU, Neon::Allocator::MALLOC, 1);

    Neon::set::patterns::BlasSet<T> blasHandle(storage.m_grid.getDevSet());

    T std_output = 0;
    T int_output = 0;
    T bd_output = 0;
    if (storage.m_grid.getBackend().devType() == Neon::DeviceType::CUDA) {
        auto streams = storage.m_grid.getDevSet().newStreamSet();
        blasHandle.setStream(streams);
        std_output = storage.Xf.norm2(blasHandle, output, Neon::DataView::STANDARD);
        int_output = storage.Xf.norm2(blasHandle, output, Neon::DataView::INTERNAL);
        bd_output = storage.Xf.norm2(blasHandle, output, Neon::DataView::BOUNDARY);
    } else {
        std_output = storage.Xf.norm2(blasHandle, output, Neon::DataView::STANDARD);
        int_output = storage.Xf.norm2(blasHandle, output, Neon::DataView::INTERNAL);
        bd_output = storage.Xf.norm2(blasHandle, output, Neon::DataView::BOUNDARY);
    }

    T ground_truth = storage.norm2(storage.Xd);
    ASSERT_NEAR(std_output, ground_truth, 0.001);
    if (nGPU > 1) {
        ASSERT_NEAR(int_output * int_output + bd_output * bd_output,
                    ground_truth * ground_truth,
                    0.001);
    }
}


TEST(PatternDot, dGrid)
{
    NEON_INFO("dGrid");
    int nGpus = 3;
    runAllTestConfiguration(patternDotTest<dGrid_t, double>, nGpus);
}


TEST(PatternNorm2, dGrid)
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
