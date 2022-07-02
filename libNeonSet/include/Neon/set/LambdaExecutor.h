#pragma once
#include <functional>

namespace Neon {
namespace set {
namespace internal {

#ifdef NEON_COMPILER_CUDA
template <typename DataSetContainer_ta,
          typename UserLambda_ta>
NEON_CUDA_KERNEL auto execLambdaWithIterator_cuda(typename DataSetContainer_ta::PartitionIndexSpace i,
                                                  UserLambda_ta                                     userLambdaTa)
    -> void
{
    typename DataSetContainer_ta::PartitionIndexSpace::Cell e;
    if constexpr (DataSetContainer_ta::PartitionIndexSpace::SpaceDim == 1) {
        if (i.setAndValidate(e,
                             threadIdx.x + blockIdx.x * blockDim.x,
                             0,
                             0)) {
            userLambdaTa(e);
        }
    }
    if constexpr (DataSetContainer_ta::PartitionIndexSpace::SpaceDim == 2) {
        if (i.setAndValidate(e,
                             threadIdx.x + blockIdx.x * blockDim.x,
                             threadIdx.y + blockIdx.y * blockDim.y,
                             0)) {
            userLambdaTa(e);
        }
    }
    if constexpr (DataSetContainer_ta::PartitionIndexSpace::SpaceDim == 3) {
        if (i.setAndValidate(e,
                             threadIdx.x + blockIdx.x * blockDim.x,
                             threadIdx.y + blockIdx.y * blockDim.y,
                             threadIdx.z + blockIdx.z * blockDim.z)) {
            userLambdaTa(e);
        }
    }
}
#endif


template <typename DataSetContainer_ta, typename UserLambda_ta>
void execLambdaWithIterator_omp(const Neon::int64_3d&                             gridDim,
                                typename DataSetContainer_ta::PartitionIndexSpace partitionIndexSpace,
                                UserLambda_ta                                     userLambdaTa)
{
    if constexpr (DataSetContainer_ta::PartitionIndexSpace::SpaceDim == 1) {
#ifdef NEON_OS_WINDOWS
#pragma omp parallel for default(shared)
#else
#pragma omp parallel for simd default(shared)
#endif
        for (int64_t x = 0; x < gridDim.x; x++) {
            typename DataSetContainer_ta::PartitionIndexSpace::Cell e;
            if (partitionIndexSpace.setAndValidate(e, x, 0, 0)) {
                userLambdaTa(e);
            }
        }
    }


    if constexpr (DataSetContainer_ta::PartitionIndexSpace::SpaceDim == 2) {
#ifdef NEON_OS_WINDOWS
#pragma omp parallel for default(shared)
#else
#pragma omp parallel for simd collapse(2) default(shared)
#endif
        for (int64_t y = 0; y < gridDim.y; y++) {
            for (int64_t x = 0; x < gridDim.x; x++) {
                typename DataSetContainer_ta::PartitionIndexSpace::Cell e;
                if (partitionIndexSpace.setAndValidate(e, x, y, 0)) {
                    userLambdaTa(e);
                }
            }
        }
    }

    if constexpr (DataSetContainer_ta::PartitionIndexSpace::SpaceDim == 3) {
#ifdef NEON_OS_WINDOWS
#pragma omp parallel for default(shared)
#else
#pragma omp parallel for simd collapse(3) default(shared)
#endif
        for (int64_t z = 0; z < gridDim.z; z++) {
            for (int64_t y = 0; y < gridDim.y; y++) {
                for (int64_t x = 0; x < gridDim.x; x++) {
                    typename DataSetContainer_ta::PartitionIndexSpace::Cell e;
                    if (partitionIndexSpace.setAndValidate(e, x, y, z)) {
                        userLambdaTa(e);
                    }
                }
            }
        }
    }
}


}  // namespace internal
}  // namespace set
}  // namespace Neon