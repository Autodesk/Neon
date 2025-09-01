    #include "Neon/sys/patterns/Blas.h"

#include <cub/cub.cuh>
#include <thrust/functional.h>

namespace Neon::sys::patterns {


template <typename T>
void Blas<T>::setNumBlocks(const uint32_t numBlocks)
{
    mNumBlocks = numBlocks;
    if (mDevType == Neon::DeviceType::CUDA) {

        mDevice1stPhaseOutput = Neon::sys::MemDevice<T>(Neon::DeviceType::CUDA,
                                                        mDevID,
                                                        Neon::Allocator::CUDA_MEM_DEVICE,
                                                        mNumBlocks);
        void* tempStorage = NULL;
        T*    output = NULL;
        cub::DeviceReduce::Sum(tempStorage,
                               mDeviceCUBTempMemBytes,
                               mDevice1stPhaseOutput.mem(),
                               output,
                               mNumBlocks);

        uint32_t tempStorageTyped = NEON_DIVIDE_UP(mDeviceCUBTempMemBytes, sizeof(T));

        mDeviceCUBTempMem = Neon::sys::MemDevice<T>(Neon::DeviceType::CUDA,
                                                    mDevID,
                                                    Neon::Allocator::CUDA_MEM_DEVICE,
                                                    NEON_DIVIDE_UP(mDeviceCUBTempMemBytes, sizeof(T)));
    }
}

template <typename T>
template <typename ReductionOp>
void Blas<T>::reducePhase2(MemDevice<T>& output, ReductionOp reduction_op, T init)
{
    if (output.allocType() == Neon::Allocator::MEM_ERROR ||
        output.allocType() == Neon::Allocator::NULL_MEM) {

        NeonException exc("Blas::reducePhase2");
        exc << "Output allocators are invalid";
        exc << "\n Output allocator is" << Neon::AllocatorUtils::toString(output.allocType());
        NEON_THROW(exc);
    }

    if (output.allocType() != Neon::Allocator::CUDA_MEM_UNIFIED && output.allocType() != Neon::Allocator::CUDA_MEM_DEVICE) {
        NeonException exc("Blas::reducePhase2");
        exc << "Output allocator should be on the device";
        exc << "\n Output allocator is " << Neon::AllocatorUtils::toString(output.allocType());
        NEON_THROW(exc);
    }

    if (mEngine != Engine::CUB) {
        NeonException exc("Blas::reducePhase2");
        exc << "Can not call reducePhase2 is backend engine is not CUB";
        NEON_THROW(exc);
    }

    cub::DeviceReduce::Reduce((void*)mDeviceCUBTempMem.mem(),
                              mDeviceCUBTempMemBytes,
                              mDevice1stPhaseOutput.mem(),
                              output.mem(),
                              mNumBlocks,
                              reduction_op,
                              init,
                              mStream.stream());
}
template class Blas<float>;
template class Blas<double>;
template class Blas<int32_t>;
template class Blas<int64_t>;
template class Blas<uint32_t>;
template class Blas<uint64_t>;

// Note: cub::Sum() was deprecated in CUDA 13.0, using thrust::plus instead
template void Blas<float>::template reducePhase2<thrust::plus<float>>(
    MemDevice<float>&,
    thrust::plus<float>,
    float);

template void Blas<double>::template reducePhase2<thrust::plus<double>>(
    MemDevice<double>&,
    thrust::plus<double>,
    double);
template void Blas<int32_t>::template reducePhase2<thrust::plus<int32_t>>(
    MemDevice<int32_t>&,
    thrust::plus<int32_t>,
    int32_t);
template void Blas<int64_t>::template reducePhase2<thrust::plus<int64_t>>(
    MemDevice<int64_t>&,
    thrust::plus<int64_t>,
    int64_t);
template void Blas<uint32_t>::template reducePhase2<thrust::plus<uint32_t>>(
    MemDevice<uint32_t>&,
    thrust::plus<uint32_t>,
    uint32_t);
template void Blas<uint64_t>::template reducePhase2<thrust::plus<uint64_t>>(
    MemDevice<uint64_t>&,
    thrust::plus<uint64_t>,
    uint64_t);
}  // namespace Neon::sys::patterns