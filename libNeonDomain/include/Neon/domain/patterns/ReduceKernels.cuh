#include "Neon/core/core.h"

#include <cub/cub.cuh>

namespace Neon::domain::internal {

template <typename T, uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ>
NEON_CUDA_DEVICE_ONLY __inline__ void cubBlockSum(const T threadValue,
                                                  T*      output)
{

    typedef cub::BlockReduce<T, blockDimX,
                             cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                             blockDimY,
                             blockDimZ>
                                                 BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    T                                            block_sum = BlockReduce(temp_storage).Sum(threadValue);

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        output[blockIdx.x +
               blockIdx.y * gridDim.x +
               blockIdx.z * gridDim.x * gridDim.y] = block_sum;
    }
}


template <typename T,
          uint32_t blockDimX,
          uint32_t blockDimY,
          uint32_t blockDimZ,
          typename Partition>
NEON_CUDA_KERNEL void dotKernel(const typename Partition::PartitionIndexSpace indexSpace,
                                const Partition                               input1,
                                const Partition                               input2,
                                T*                                            output)
{
    T                                             threadValue = 0;
    typename Partition::PartitionIndexSpace::Cell cell;
    if (indexSpace.setAndValidate(cell,
                                  threadIdx.x + blockIdx.x * blockDim.x,
                                  threadIdx.y + blockIdx.y * blockDim.y,
                                  threadIdx.z + blockIdx.z * blockDim.z)) {
        int card = input1.cardinality();
        for (int i = 0; i < card; ++i) {
            threadValue += input1(cell, i) * input2(cell, i);
        }
    }

    cubBlockSum<T, blockDimX, blockDimY, blockDimZ>(threadValue, output);
}

template <typename T,
          uint32_t blockDimX,
          uint32_t blockDimY,
          uint32_t blockDimZ,
          typename Partition>
NEON_CUDA_KERNEL void norm2Kernel(const typename Partition::PartitionIndexSpace indexSpace,
                                  const Partition                               input,
                                  T*                                            output)
{
    T                                             threadValue = 0;
    typename Partition::PartitionIndexSpace::Cell cell;
    if (indexSpace.setAndValidate(cell,
                                  threadIdx.x + blockIdx.x * blockDim.x,
                                  threadIdx.y + blockIdx.y * blockDim.y,
                                  threadIdx.z + blockIdx.z * blockDim.z)) {
        int card = input.cardinality();
        for (int i = 0; i < card; ++i) {
            T val = input(cell, i);
            threadValue += val * val;
        }
    }

    cubBlockSum<T, blockDimX, blockDimY, blockDimZ>(threadValue, output);
}

template <typename T,
          uint32_t blockDimX,
          uint32_t blockDimY,
          uint32_t blockDimZ,
          typename Grid,
          typename Field>
auto dotCUB(Neon::set::patterns::BlasSet<T>& blasSet,
            Grid&                            grid,
            const Field&                     input1,
            const Field&                     input2,
            Neon::set::MemDevSet<T>&         output,
            const Neon::DataView&            dataView)
{
    int nGpus = grid.getBackend().devSet().setCardinality();

    const Neon::index_3d blockSize(blockDimX, blockDimY, blockDimZ);
    auto                 launchInfoSet = grid.helpGetLaunchParameters(dataView, blockSize, 0);

#pragma omp parallel for num_threads(nGpus) default(shared)
    for (int idx = 0; idx < nGpus; idx++) {
        const Neon::sys::GpuDevice& dev = Neon::sys::globalSpace::gpuSysObj().dev(
            grid.getBackend().devSet().idSet()[idx]);

        if (launchInfoSet[idx].cudaBlock().x != blockDimX ||
            launchInfoSet[idx].cudaBlock().y != blockDimY ||
            launchInfoSet[idx].cudaBlock().z != blockDimZ) {
            NeonException exc;
            exc << "dotCUB CUDA block template parameter used for dotKernel() does not match the runtime CUDA block size";
            NEON_THROW(exc);
        }

        auto indexSpace = grid.getPartitionIndexSpace(Neon::DeviceType::CUDA, idx, dataView);
        auto inputPartition1 = input1.getPartition(Neon::DeviceType::CUDA, idx, dataView);
        auto inputPartition2 = input2.getPartition(Neon::DeviceType::CUDA, idx, dataView);

        const Neon::sys::GpuStream     gpuStream = blasSet.getBlas(size_t(idx)).getStream();
        const Neon::sys::GpuLaunchInfo kernelInfo = launchInfoSet[idx];

        dev.tools.setActiveDevContext();

        dotKernel<T,
                  blockDimX,
                  blockDimY,
                  blockDimZ><<<kernelInfo.cudaGrid(),
                               kernelInfo.cudaBlock(),
                               0,
                               gpuStream.stream()>>>(indexSpace, inputPartition1, inputPartition2,
                                                     blasSet.getBlas(size_t(idx)).getReducePhase1Output().mem());

        Neon::sys::gpuCheckLastError("from dField::dotPhase1 after launching dotKernel()");

        //gpuStream.sync();

        blasSet.getBlas(idx).reducePhase2(output.getMemDev(idx), cub::Sum(), 0);
    }
}

template <typename T,
          uint32_t blockDimX,
          uint32_t blockDimY,
          uint32_t blockDimZ,
          typename Grid,
          typename Field>
auto norm2CUB(Neon::set::patterns::BlasSet<T>& blasSet,
              Grid&                            grid,
              const Field&                     input,
              Neon::set::MemDevSet<T>&         output,
              const Neon::DataView&            dataView)
{
    int nGpus = grid.getBackend().devSet().setCardinality();

    const Neon::index_3d blockSize(blockDimX, blockDimY, blockDimZ);
    auto                 launchInfoSet = grid.helpGetLaunchParameters(dataView, blockSize, 0);

#pragma omp parallel for num_threads(nGpus) default(shared)
    for (int idx = 0; idx < nGpus; idx++) {
        const Neon::sys::GpuDevice& dev = Neon::sys::globalSpace::gpuSysObj().dev(
            grid.getBackend().devSet().idSet()[idx]);

        if (launchInfoSet[idx].cudaBlock().x != blockDimX ||
            launchInfoSet[idx].cudaBlock().y != blockDimY ||
            launchInfoSet[idx].cudaBlock().z != blockDimZ) {
            NeonException exc;
            exc << "dotCUB CUDA block template parameter used for dotKernel() does not match the runtime CUDA block size";
            NEON_THROW(exc);
        }

        auto indexSpace = grid.getPartitionIndexSpace(Neon::DeviceType::CUDA, idx, dataView);
        auto inputPartition = input.getPartition(Neon::DeviceType::CUDA, idx, dataView);

        const Neon::sys::GpuStream     gpuStream = blasSet.getBlas(size_t(idx)).getStream();
        const Neon::sys::GpuLaunchInfo kernelInfo = launchInfoSet[idx];

        dev.tools.setActiveDevContext();

        norm2Kernel<T,
                    blockDimX,
                    blockDimY,
                    blockDimZ><<<kernelInfo.cudaGrid(),
                                 kernelInfo.cudaBlock(),
                                 0,
                                 gpuStream.stream()>>>(indexSpace, inputPartition,
                                                       blasSet.getBlas(size_t(idx)).getReducePhase1Output().mem());

        Neon::sys::gpuCheckLastError("from dField::dotPhase1 after launching dotKernel()");

        //gpuStream.sync();

        blasSet.getBlas(idx).reducePhase2(output.getMemDev(idx), cub::Sum(), 0);
    }
}
}  // namespace Neon::domain::internal