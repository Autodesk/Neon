#include "gtest/gtest.h"

#include "Neon/Neon.h"

#include "Neon/sys/memory/MemDevice.h"
#include "Neon/sys/patterns/Blas.h"

TEST(Blas, IncompatibleAllocator)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        using dataT = float;
        size_t                      buffer_size = 1024;
        Neon::sys::DeviceID         dev_id = 0;
        Neon::sys::GpuDevice        gpu_dev(dev_id);
        Neon::sys::MemDevice<dataT> input(Neon::DeviceType::CUDA, 0, Neon::Allocator::CUDA_MEM_DEVICE, buffer_size);
        Neon::sys::MemDevice<dataT> output(Neon::DeviceType::CUDA, 0, Neon::Allocator::CUDA_MEM_DEVICE, 1);

        Neon::sys::patterns::Blas<dataT> pat(gpu_dev);
        EXPECT_ANY_THROW(pat.absoluteSum(input, output));
    }
}

TEST(Blas, HostSum)
{
    using dataT = float;
    int                         buffer_size = 1024;
    Neon::sys::DeviceID         dev_id = 0;
    Neon::sys::GpuDevice        gpu_dev(dev_id);
    Neon::sys::MemDevice<dataT> input(Neon::DeviceType::CPU, 0, Neon::Allocator::MALLOC, buffer_size);
    Neon::sys::MemDevice<dataT> output(Neon::DeviceType::CPU, 0, Neon::Allocator::MALLOC, 1);

    for (int i = 0; i < buffer_size; ++i) {
        input.mem()[i] = -1;
    }
    output.mem()[0] = 0;

    Neon::sys::patterns::Blas<dataT> pat(gpu_dev);
    pat.absoluteSum(input, output, 0, buffer_size);

    dataT results = output.elRef(0);

    EXPECT_EQ(results, static_cast<dataT>(buffer_size));
    EXPECT_NEAR(results, static_cast<dataT>(buffer_size), 0.001);
}

TEST(Blas, DeviceSum)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        using dataT = float;
        int                         buffer_size = 1024;
        Neon::sys::DeviceID         dev_id = 0;
        Neon::sys::GpuDevice        gpu_dev(dev_id);
        Neon::sys::GpuStream        gpu_stream = gpu_dev.tools.stream();
        Neon::sys::MemDevice<dataT> input(Neon::DeviceType::CPU, dev_id, Neon::Allocator::CUDA_MEM_UNIFIED, buffer_size);
        Neon::sys::MemDevice<dataT> output(Neon::DeviceType::CPU, 0, Neon::Allocator::MALLOC, 1);

        for (int i = 0; i < buffer_size; ++i) {
            input.mem()[i] = -1;
        }
        output.mem()[0] = 0;

        Neon::sys::patterns::Blas<dataT> pat(gpu_dev);
        pat.setStream(gpu_stream);
        pat.absoluteSum(input, output, 0, buffer_size);

        gpu_stream.sync();

        dataT results = output.elRef(0);

        EXPECT_EQ(results, static_cast<dataT>(buffer_size));
        EXPECT_NEAR(results, static_cast<dataT>(buffer_size), 0.001);
    }
}

TEST(Blas, HostNorm2)
{
    using dataT = float;
    int                         buffer_size = 1024;
    Neon::sys::DeviceID         dev_id = 0;
    Neon::sys::GpuDevice        gpu_dev(dev_id);
    Neon::sys::MemDevice<dataT> input(Neon::DeviceType::CPU, 0, Neon::Allocator::MALLOC, buffer_size);
    Neon::sys::MemDevice<dataT> output(Neon::DeviceType::CPU, 0, Neon::Allocator::MALLOC, 1);

    for (int i = 0; i < buffer_size; ++i) {
        input.mem()[i] = -2;
    }
    output.mem()[0] = 0;

    Neon::sys::patterns::Blas<dataT> pat(gpu_dev);
    pat.norm2(input, output, 0, buffer_size);

    dataT results = output.elRef(0);

    EXPECT_NEAR(results, static_cast<dataT>(std::sqrt(2.0 * 2.0 * buffer_size)), 0.001);
}

TEST(Blas, DeviceNorm2)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        using dataT = float;
        int                         buffer_size = 1024;
        Neon::sys::DeviceID         dev_id = 0;
        Neon::sys::GpuDevice        gpu_dev(dev_id);
        Neon::sys::GpuStream        gpu_stream = gpu_dev.tools.stream();
        Neon::sys::MemDevice<dataT> input(Neon::DeviceType::CPU, dev_id, Neon::Allocator::CUDA_MEM_UNIFIED, buffer_size);
        Neon::sys::MemDevice<dataT> output(Neon::DeviceType::CPU, 0, Neon::Allocator::MALLOC, 1);

        for (int i = 0; i < buffer_size; ++i) {
            input.mem()[i] = -2;
        }
        output.mem()[0] = 0;

        Neon::sys::patterns::Blas<dataT> pat(gpu_dev);
        pat.setStream(gpu_stream);
        pat.norm2(input, output, 0, buffer_size);

        gpu_stream.sync();

        dataT results = output.elRef(0);

        EXPECT_NEAR(results, static_cast<dataT>(std::sqrt(2.0 * 2.0 * buffer_size)), 0.001);
    }
}

TEST(Blas, HostDot)
{
    using dataT = float;
    int                         buffer_size = 1024;
    Neon::sys::DeviceID         dev_id = 0;
    Neon::sys::GpuDevice        gpu_dev(dev_id);
    Neon::sys::MemDevice<dataT> input1(Neon::DeviceType::CPU, 0, Neon::Allocator::MALLOC, buffer_size);
    Neon::sys::MemDevice<dataT> input2(Neon::DeviceType::CPU, 0, Neon::Allocator::MALLOC, buffer_size);
    Neon::sys::MemDevice<dataT> output(Neon::DeviceType::CPU, 0, Neon::Allocator::MALLOC, 1);

    for (int i = 0; i < buffer_size; ++i) {
        input1.mem()[i] = -2;
        input2.mem()[i] = 2;
    }
    output.mem()[0] = 0;

    Neon::sys::patterns::Blas<dataT> pat(gpu_dev);
    pat.dot(input1, input2, output, 0, buffer_size);

    dataT results = output.elRef(0);

    EXPECT_NEAR(results, static_cast<dataT>(-2.0 * 2.0 * buffer_size), 0.001);
}

TEST(Blas, DeviceDot)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        using dataT = float;
        int                         buffer_size = 1024;
        Neon::sys::DeviceID         dev_id = 0;
        Neon::sys::GpuDevice        gpu_dev(dev_id);
        Neon::sys::GpuStream        gpu_stream = gpu_dev.tools.stream();
        Neon::sys::MemDevice<dataT> input1(Neon::DeviceType::CPU, 0, Neon::Allocator::CUDA_MEM_UNIFIED, buffer_size);
        Neon::sys::MemDevice<dataT> input2(Neon::DeviceType::CPU, 0, Neon::Allocator::CUDA_MEM_UNIFIED, buffer_size);
        Neon::sys::MemDevice<dataT> output(Neon::DeviceType::CPU, 0, Neon::Allocator::MALLOC, 1);

        for (int i = 0; i < buffer_size; ++i) {
            input1.mem()[i] = -2;
            input2.mem()[i] = 2;
        }
        output.mem()[0] = 0;

        Neon::sys::patterns::Blas<dataT> pat(gpu_dev);
        pat.setStream(gpu_stream);
        pat.dot(input1, input2, output, 0, buffer_size);

        gpu_stream.sync();

        dataT results = output.elRef(0);

        EXPECT_NEAR(results, static_cast<dataT>(-2.0 * 2.0 * buffer_size), 0.001);
    }
}
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
