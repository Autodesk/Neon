#pragma once
#include "Neon/sys/devices/gpu/GpuTools.h"
#include "Neon/sys/patterns/Blas.h"

namespace Neon::sys::patterns {

template <typename T>
Blas<T>::Blas(const Neon::sys::GpuDevice& dev,
              const Neon::DeviceType&     devType,
              const Engine                engine)
{
    mDevType = devType;
    mEngine = engine;
    mNumBlocks = 0;
    mDevID = dev.getIdx();
    mDeviceCUBTempMemBytes = 0;

    mHandle = std::make_shared<cublasHandle_t>();
    if (mDevType == Neon::DeviceType::CUDA) {
        *mHandle = dev.tools.cublasHandle(false);
    }
}

template <typename T>
void Blas<T>::setStream(Neon::sys::GpuStream& stream)
{
    if (mDevType == Neon::DeviceType::CUDA) {
        mStream = stream;
        cublasStatus_t status = cublasSetStream(*mHandle, stream.stream());
        if (status != CUBLAS_STATUS_SUCCESS) {
            NeonException exc;
            exc << "cuBLAS error setting stream with error: " << Neon::sys::cublasGetErrorString(status);
            NEON_THROW(exc);
        }
    }
}

template <typename T>
const Neon::sys::GpuStream& Blas<T>::getStream() const
{
    return mStream;
}

template <typename T>
uint32_t Blas<T>::getNumBlocks() const
{
    return mNumBlocks;
}

template <typename T>
void Blas<T>::checkAllocator(const MemDevice<T>& input, const MemDevice<T>& output)
{
    if (input.allocType() == Neon::Allocator::MEM_ERROR ||
        input.allocType() == Neon::Allocator::NULL_MEM ||
        output.allocType() == Neon::Allocator::MEM_ERROR ||
        output.allocType() == Neon::Allocator::NULL_MEM) {

        NeonException exc("Blas::checkAllocator");
        exc << "Input/output allocators are invalid";
        exc << "\n Input allocator is" << Neon::AllocatorUtils::toString(input.allocType());
        exc << "\n Output allocator is" << Neon::AllocatorUtils::toString(output.allocType());
        NEON_THROW(exc);
    }

    if (output.allocType() != Neon::Allocator::MALLOC && output.allocType() != Neon::Allocator::CUDA_MEM_HOST) {
        NeonException exc("Blas::checkAllocator");
        exc << "Output allocator should be on the host";
        exc << "\n Output allocator is " << Neon::AllocatorUtils::toString(output.allocType());
        NEON_THROW(exc);
    }
}

template <typename T>
void Blas<T>::absoluteSum(const MemDevice<T>& input, MemDevice<T>& output, int start_id, int num_elements)
{
    checkAllocator(input, output);

    num_elements = (num_elements == std::numeric_limits<int>::max()) ? static_cast<int>(input.nElements()) : num_elements;


    if (input.allocType() == Neon::Allocator::CUDA_MEM_DEVICE ||
        input.allocType() == Neon::Allocator::CUDA_MEM_UNIFIED) {

        if constexpr (!std::is_same<T, double>::value && !std::is_same<T, float>::value) {
            NeonException exc("Blas::absoluteSum");
            exc << "cuBLAS engine only works with float and double data type";
            NEON_THROW(exc);
        }
        [[maybe_unused]] auto check_error = [&](cublasStatus_t status) {
            if (status != CUBLAS_STATUS_SUCCESS) {
                NeonException exc("Blas::absoluteSum");
                exc << "cuBLAS error during absolute sum operation with error: " << Neon::sys::cublasGetErrorString(status);
                NEON_THROW(exc);
            }
        };
        if constexpr (std::is_same<T, float>::value) {
            cublasStatus_t status = cublasSasum(*mHandle, num_elements,
                                                input.mem() + start_id, 1,
                                                output.mem());
            check_error(status);
            return;
        }

        if constexpr (std::is_same<T, double>::value) {
            cublasStatus_t status = cublasDasum(*mHandle, static_cast<int>(num_elements),
                                                input.mem() + start_id, 1,
                                                output.mem());
            check_error(status);
            return;
        }
    } else if (input.allocType() == Neon::Allocator::CUDA_MEM_HOST ||
               input.allocType() == Neon::Allocator::MALLOC) {
        T ret = 0;
#pragma omp parallel for reduction(+ \
                                   : ret)
        for (int i = start_id; i < start_id + num_elements; ++i) {
            if constexpr (std::is_signed_v<T>) {
                ret += std::abs(input.mem()[i]);
            } else {
                ret += input.mem()[i];
            }
        }
        output.mem()[0] = ret;
    }
}

template <typename T>
void Blas<T>::dot(const MemDevice<T>& input1, const MemDevice<T>& input2, MemDevice<T>& output, int start_id, int num_elements)
{
    checkAllocator(input1, output);
    checkAllocator(input2, output);

    if (input1.nElements() != input2.nElements()) {
        NeonException exc("Blas::dot");
        exc << "Input1 and output2 size is different";
        exc << "\n Input1 size = " << input1.nElements();
        exc << "\n Input2 size = " << input2.nElements();
        NEON_THROW(exc);
    }

    num_elements = (num_elements == std::numeric_limits<int>::max()) ? static_cast<int>(input1.nElements()) : num_elements;

    if (input1.allocType() == Neon::Allocator::CUDA_MEM_DEVICE ||
        input1.allocType() == Neon::Allocator::CUDA_MEM_UNIFIED) {

        if constexpr (!std::is_same<T, double>::value && !std::is_same<T, float>::value) {
            NeonException exc("Blas::dot");
            exc << "cuBLAS engine only works with float and double data type";
            NEON_THROW(exc);
        }

        [[maybe_unused]] auto check_error = [&](cublasStatus_t status) {
            if (status != CUBLAS_STATUS_SUCCESS) {
                NeonException exc("Blas::dot");
                exc << "cuBLAS error during dot product operation with error: " << Neon::sys::cublasGetErrorString(status);
                NEON_THROW(exc);
            }
        };

        if constexpr (std::is_same<T, float>::value) {
            cublasStatus_t status = cublasSdot(*mHandle, num_elements,
                                               input1.mem() + start_id, 1,
                                               input2.mem() + start_id, 1,
                                               output.mem());
            check_error(status);
            return;
        }

        if constexpr (std::is_same<T, double>::value) {
            cublasStatus_t status = cublasDdot(*mHandle, static_cast<int>(num_elements),
                                               input1.mem() + start_id, 1,
                                               input2.mem() + start_id, 1,
                                               output.mem());
            check_error(status);
            return;
        }
    } else if (input1.allocType() == Neon::Allocator::CUDA_MEM_HOST ||
               input1.allocType() == Neon::Allocator::MALLOC) {
        T ret = 0;
#pragma omp parallel for reduction(+ \
                                   : ret)
        for (int i = start_id; i < start_id + num_elements; ++i) {
            ret += input1.mem()[i] * input2.mem()[i];
        }
        output.mem()[0] = ret;
    }
}

template <typename T>
void Blas<T>::norm2(const MemDevice<T>& input, MemDevice<T>& output, int start_id, int num_elements)
{
    checkAllocator(input, output);

    num_elements = (num_elements == std::numeric_limits<int>::max()) ? static_cast<int>(input.nElements()) : num_elements;

    if (input.allocType() == Neon::Allocator::CUDA_MEM_DEVICE ||
        input.allocType() == Neon::Allocator::CUDA_MEM_UNIFIED) {

        if constexpr (!std::is_same<T, double>::value && !std::is_same<T, float>::value) {
            NeonException exc("Blas::norm2");
            exc << "cuBLAS engine only works with float and double data type";
            NEON_THROW(exc);
        }

        [[maybe_unused]] auto check_error = [&](cublasStatus_t status) {
            if (status != CUBLAS_STATUS_SUCCESS) {
                NeonException exc("Blas::norm2");
                exc << "cuBLAS error during norm2 operation with error: " << Neon::sys::cublasGetErrorString(status);
                NEON_THROW(exc);
            }
        };
        if constexpr (std::is_same<T, float>::value) {
            cublasStatus_t status = cublasSnrm2(*mHandle, num_elements,
                                                input.mem() + start_id, 1,
                                                output.mem());
            check_error(status);
            return;
        }

        if constexpr (std::is_same<T, double>::value) {
            cublasStatus_t status = cublasDnrm2(*mHandle, static_cast<int>(num_elements),
                                                input.mem() + start_id, 1,
                                                output.mem());
            check_error(status);
            return;
        }


    } else if (input.allocType() == Neon::Allocator::CUDA_MEM_HOST ||
               input.allocType() == Neon::Allocator::MALLOC) {
        T ret = 0;
#pragma omp parallel for reduction(+ \
                                   : ret)
        for (int i = start_id; i < start_id + num_elements; ++i) {
            ret += input.mem()[i] * input.mem()[i];
        }
        output.mem()[0] = static_cast<T>(std::sqrt(ret));
    }
}

template <typename T>
MemDevice<T>& Blas<T>::getReducePhase1Output()
{
    return mDevice1stPhaseOutput;
}

template <typename T>
Blas<T>::~Blas() noexcept(false)
{
    if (mHandle.use_count() == 1 && mDevType == Neon::DeviceType::CUDA) {
        cublasStatus_t status = cublasDestroy(*mHandle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            NeonException exc;
            exc << "cuBLAS error destroying handle with error:" << Neon::sys::cublasGetErrorString(status);
            NEON_THROW(exc);
        }
    }
}
}  // namespace Neon::sys::patterns