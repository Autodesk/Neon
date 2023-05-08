#include "gtest/gtest.h"

#include "Neon/Neon.h"
#include "Neon/core/tools/metaprogramming/debugHelp.h"
#include "Neon/set/DevSet.h"
#include "Neon/set/Replica.h"

struct TestObj
{
    int a;
};

void run(Neon::Runtime runtime)
{
    Neon::init();
    size_t           numDevices = 2;
    std::vector<int> devIds(numDevices, 0);
    Neon::Backend    backend(devIds, runtime);

    Neon::set::Replica<TestObj> multiDeviceObject(backend);

    backend.devSet().forEachSetIdxSeq([&](const Neon::SetIdx& setIdx) {
        multiDeviceObject(setIdx).a = 33;
    });

    multiDeviceObject.updateDeviceData(0);
    backend.syncAll();


    Neon::set::Container c = multiDeviceObject.newContainer(
        "Test",
        [&](Neon::set::Loader& loader) {
            auto m = loader.load(multiDeviceObject);
            // Neon::meta::debug::printType(m);
            return [=] NEON_CUDA_HOST_DEVICE(const Neon::set::Replica<TestObj>::Idx&) mutable {
                m().a += 17;
            };
        },
        Neon::Execution::device);
    c.run(0);
    multiDeviceObject.updateHostData(0);
    backend.syncAll();

    backend.devSet().forEachSetIdxSeq([&](const Neon::SetIdx& setIdx) {
        ASSERT_EQ(multiDeviceObject(setIdx).a, 50);
    });
}
TEST(MultiDeviceObj, openmp)
{
    run(Neon::Runtime::openmp);
}

TEST(MultiDeviceObj, stream)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        run(Neon::Runtime::stream);
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
