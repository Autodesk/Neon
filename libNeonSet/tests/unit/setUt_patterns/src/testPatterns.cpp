#include "gtest/gtest.h"

#include "Neon/Neon.h"

#include "Neon/set/DevSet.h"
#include "Neon/set/patterns/BlasSet.h"

TEST(BlasSet, SumUnified)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        using dataT = double;
        int               buffer_size = 1024;
        size_t            num_sets = 2;
        std::vector<int>  dev_ids(num_sets, 0);
        Neon::set::DevSet dev_set(Neon::DeviceType::CUDA, dev_ids);
        auto              inputs = dev_set.newMemDevSet<dataT>(Neon::DeviceType::CPU,
                                                  Neon::Allocator::MALLOC,
                                                  buffer_size);
        auto              output = dev_set.newMemDevSet<dataT>(Neon::DeviceType::CPU,
                                                  Neon::Allocator::MALLOC,
                                                  1);
        auto              streams = dev_set.newStreamSet();
        auto              start_id = dev_set.newDataSet<int>(0);
        auto              num_elements = dev_set.newDataSet<int>(int(buffer_size));
        for (int32_t i = 0; i < int32_t(num_sets); ++i) {
            output.mem(i)[0] = 0;
            for (int j = 0; j < buffer_size; ++j) {
                inputs.mem(i)[j] = -1;
            }
        }

        Neon::set::patterns::BlasSet<dataT> pattern(dev_set);
        pattern.setStream(streams);
        dataT result = pattern.absoluteSum(inputs, output, start_id, num_elements);

        dataT expected_result = static_cast<dataT>(num_sets * buffer_size);
        EXPECT_NEAR(result, expected_result, 0.001);
    }
}

TEST(BlasSet, Sum)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        using dataT = double;
        size_t            buffer_size = 1024;
        size_t            num_sets = 2;
        std::vector<int>  dev_ids(num_sets, 0);
        Neon::set::DevSet dev_set(Neon::DeviceType::CUDA, dev_ids);
        auto              start_id = dev_set.newDataSet<int>(0);
        auto              num_elements = dev_set.newDataSet<int>(int(buffer_size));
        auto              d_inputs = dev_set.newMemDevSet<dataT>(Neon::DeviceType::CUDA,
                                                    Neon::Allocator::CUDA_MEM_DEVICE,
                                                    buffer_size);
        auto              h_inputs = dev_set.newMemDevSet<dataT>(Neon::DeviceType::CPU,
                                                    Neon::Allocator::MALLOC,
                                                    buffer_size);


        auto h_output = dev_set.newMemDevSet<dataT>(Neon::DeviceType::CPU,
                                                    Neon::Allocator::MALLOC,
                                                    1);
        auto streams = dev_set.newStreamSet();

        for (int32_t i = 0; i < int32_t(num_sets); ++i) {
            h_output.mem(i)[0] = 0;
            for (size_t j = 0; j < buffer_size; ++j) {
                h_inputs.mem(i)[j] = -1;
            }
        }

        d_inputs.copyFrom(h_inputs);

        Neon::set::patterns::BlasSet<dataT> pattern(dev_set);
        pattern.setStream(streams);
        dataT result = pattern.absoluteSum(d_inputs, h_output, start_id, num_elements);


        dataT expected_result = static_cast<dataT>(num_sets * buffer_size);
        EXPECT_NEAR(result, expected_result, 0.001);
    }
}

TEST(BlasSet, Norm2Unified)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        using dataT = double;
        size_t            buffer_size = 1024;
        size_t            num_sets = 2;
        std::vector<int>  dev_ids(num_sets, 0);
        Neon::set::DevSet dev_set(Neon::DeviceType::CUDA, dev_ids);
        auto              start_id = dev_set.newDataSet<int>(0);
        auto              num_elements = dev_set.newDataSet<int>(int(buffer_size));
        auto              inputs = dev_set.newMemDevSet<dataT>(Neon::DeviceType::CPU,
                                                  Neon::Allocator::CUDA_MEM_UNIFIED,
                                                  buffer_size);
        auto              output = dev_set.newMemDevSet<dataT>(Neon::DeviceType::CPU,
                                                  Neon::Allocator::MALLOC,
                                                  1);
        auto              streams = dev_set.newStreamSet();

        for (int32_t i = 0; i < int32_t(num_sets); ++i) {
            output.mem(i)[0] = 0;
            for (size_t j = 0; j < buffer_size; ++j) {
                inputs.mem(i)[j] = -2;
            }
        }

        Neon::set::patterns::BlasSet<dataT> pattern(dev_set);
        pattern.setStream(streams);
        dataT result = pattern.norm2(inputs, output, start_id, num_elements);

        dataT expected_result = std::sqrt(2.0 * 2.0 * num_sets * buffer_size);
        EXPECT_NEAR(result, expected_result, 0.001);
    }
}

TEST(BlasSet, Norm2)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        using dataT = double;
        size_t            buffer_size = 1024;
        size_t            num_sets = 2;
        std::vector<int>  dev_ids(num_sets, 0);
        Neon::set::DevSet dev_set(Neon::DeviceType::CUDA, dev_ids);
        auto              start_id = dev_set.newDataSet<int>(0);
        auto              num_elements = dev_set.newDataSet<int>(int(buffer_size));
        auto              d_inputs = dev_set.newMemDevSet<dataT>(Neon::DeviceType::CUDA,
                                                    Neon::Allocator::CUDA_MEM_DEVICE,
                                                    buffer_size);
        auto              h_inputs = dev_set.newMemDevSet<dataT>(Neon::DeviceType::CPU,
                                                    Neon::Allocator::MALLOC,
                                                    buffer_size);

        auto h_output = dev_set.newMemDevSet<dataT>(Neon::DeviceType::CPU,
                                                    Neon::Allocator::MALLOC,
                                                    1);
        auto streams = dev_set.newStreamSet();

        for (int32_t i = 0; i < int32_t(num_sets); ++i) {
            h_output.mem(i)[0] = 0;
            for (size_t j = 0; j < buffer_size; ++j) {
                h_inputs.mem(i)[j] = -2;
            }
        }

        d_inputs.copyFrom(h_inputs);


        Neon::set::patterns::BlasSet<dataT> pattern(dev_set);
        pattern.setStream(streams);
        dataT result = pattern.norm2(d_inputs, h_output, start_id, num_elements);


        dataT expected_result = std::sqrt(2.0 * 2.0 * num_sets * buffer_size);
        EXPECT_NEAR(result, expected_result, 0.001);
    }
}

TEST(BlasSet, DotUnified)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        using dataT = double;
        size_t            buffer_size = 1024;
        size_t            num_sets = 2;
        std::vector<int>  dev_ids(num_sets, 0);
        Neon::set::DevSet dev_set(Neon::DeviceType::CUDA, dev_ids);
        auto              start_id = dev_set.newDataSet<int>(0);
        auto              num_elements = dev_set.newDataSet<int>(int(buffer_size));
        auto              inputs1 = dev_set.newMemDevSet<dataT>(Neon::DeviceType::CPU,
                                                   Neon::Allocator::CUDA_MEM_UNIFIED,
                                                   buffer_size);
        auto              inputs2 = dev_set.newMemDevSet<dataT>(Neon::DeviceType::CPU,
                                                   Neon::Allocator::CUDA_MEM_UNIFIED,
                                                   buffer_size);
        auto              output = dev_set.newMemDevSet<dataT>(Neon::DeviceType::CPU,
                                                  Neon::Allocator::MALLOC,
                                                  1);
        auto              streams = dev_set.newStreamSet();

        for (int32_t i = 0; i < int32_t(num_sets); ++i) {
            output.mem(i)[0] = 0;
            for (size_t j = 0; j < buffer_size; ++j) {
                inputs1.mem(i)[j] = -2;
                inputs2.mem(i)[j] = 2;
            }
        }

        Neon::set::patterns::BlasSet<dataT> pattern(dev_set);
        pattern.setStream(streams);
        dataT result = pattern.dot(inputs1, inputs2, output, start_id, num_elements);

        dataT expected_result = -2.0 * 2.0 * num_sets * buffer_size;
        EXPECT_NEAR(result, expected_result, 0.001);
    }
}

TEST(BlasSet, Dot)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        using dataT = double;
        size_t            buffer_size = 1024;
        size_t            num_sets = 2;
        std::vector<int>  dev_ids(num_sets, 0);
        Neon::set::DevSet dev_set(Neon::DeviceType::CUDA, dev_ids);
        auto              start_id = dev_set.newDataSet<int>(0);
        auto              num_elements = dev_set.newDataSet<int>(int(buffer_size));
        auto              d_inputs1 = dev_set.newMemDevSet<dataT>(Neon::DeviceType::CUDA,
                                                     Neon::Allocator::CUDA_MEM_DEVICE,
                                                     buffer_size);
        auto              h_inputs1 = dev_set.newMemDevSet<dataT>(Neon::DeviceType::CPU,
                                                     Neon::Allocator::MALLOC,
                                                     buffer_size);

        auto d_inputs2 = dev_set.newMemDevSet<dataT>(Neon::DeviceType::CUDA,
                                                     Neon::Allocator::CUDA_MEM_DEVICE,
                                                     buffer_size);
        auto h_inputs2 = dev_set.newMemDevSet<dataT>(Neon::DeviceType::CPU,
                                                     Neon::Allocator::MALLOC,
                                                     buffer_size);

        auto h_output = dev_set.newMemDevSet<dataT>(Neon::DeviceType::CPU,
                                                    Neon::Allocator::MALLOC,
                                                    1);
        auto streams = dev_set.newStreamSet();

        for (int32_t i = 0; i < int32_t(num_sets); ++i) {
            h_output.mem(i)[0] = 0;
            for (size_t j = 0; j < buffer_size; ++j) {
                h_inputs1.mem(i)[j] = -2;
                h_inputs2.mem(i)[j] = 2;
            }
        }
        d_inputs1.copyFrom(h_inputs1);
        d_inputs2.copyFrom(h_inputs2);


        Neon::set::patterns::BlasSet<dataT> pattern(dev_set);
        pattern.setStream(streams);
        dataT result = pattern.dot(d_inputs1, d_inputs2, h_output, start_id, num_elements);


        dataT expected_result = -2.0 * 2.0 * num_sets * buffer_size;
        EXPECT_NEAR(result, expected_result, 0.001);
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
