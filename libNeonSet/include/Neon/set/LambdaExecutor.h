#pragma once
#include <functional>

namespace Neon {
namespace set {
namespace internal {

#ifdef NEON_COMPILER_CUDA
template <typename DataSetContainer,
          typename UserLambda>
NEON_CUDA_KERNEL auto execLambdaWithIterator_cuda(typename DataSetContainer::Span span,
                                                  UserLambda                      userLambdaTa)
    -> void
{
    typename DataSetContainer::Idx e;
    if constexpr (DataSetContainer::Span::SpaceDim == 1) {
        if (span.setAndValidate(e,
                                threadIdx.x + blockIdx.x * blockDim.x)) {
            userLambdaTa(e);
        }
    }
    if constexpr (DataSetContainer::Span::SpaceDim == 2) {
        if (span.setAndValidate(e,
                                threadIdx.x + blockIdx.x * blockDim.x,
                                threadIdx.y + blockIdx.y * blockDim.y)) {
            userLambdaTa(e);
        }
    }
    if constexpr (DataSetContainer::Span::SpaceDim == 3) {
        if (span.setAndValidate(e,
                                threadIdx.x + blockIdx.x * blockDim.x,
                                threadIdx.y + blockIdx.y * blockDim.y,
                                threadIdx.z + blockIdx.z * blockDim.z)) {
            userLambdaTa(e);
        }
    }
}
#endif


template <typename DataSetContainer_ta, typename UserLambda_ta>
void execLambdaWithIterator_omp(const Neon::int64_3d&              gridDim,
                                typename DataSetContainer_ta::Span span,
                                UserLambda_ta                      userLambdaTa)
{
    if constexpr (DataSetContainer_ta::Span::SpaceDim == 1) {
#ifdef NEON_OS_WINDOWS
#pragma omp parallel for default(shared)
#else
#pragma omp parallel for simd default(shared)
#endif
        for (int64_t x = 0; x < gridDim.x; x++) {
            typename DataSetContainer_ta::Idx e;
            if (span.setAndValidate(e, x)) {
                userLambdaTa(e);
            }
        }
    }


    if constexpr (DataSetContainer_ta::Span::SpaceDim == 2) {
#ifdef NEON_OS_WINDOWS
#pragma omp parallel for default(shared)
#else
#pragma omp parallel for simd collapse(2) default(shared)
#endif
        for (int64_t y = 0; y < gridDim.y; y++) {
            for (int64_t x = 0; x < gridDim.x; x++) {
                typename DataSetContainer_ta::Idx e;
                if (span.setAndValidate(e, x, y)) {
                    userLambdaTa(e);
                }
            }
        }
    }

    if constexpr (DataSetContainer_ta::Span::SpaceDim == 3) {
#ifdef NEON_OS_WINDOWS
#pragma omp parallel for default(shared)
#else
#pragma omp parallel for simd collapse(3) default(shared)
#endif
        for (int64_t z = 0; z < gridDim.z; z++) {
            for (int64_t y = 0; y < gridDim.y; y++) {
                for (int64_t x = 0; x < gridDim.x; x++) {
                    typename DataSetContainer_ta::Idx e;
                    if (span.setAndValidate(e, x, y, z)) {
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