#include <cuda_runtime_api.h>

#include "Neon/core/core.h"
#include "Neon/sys/devices/gpu/GpuTools.h"


namespace Neon {
namespace sys {

void gpuCheckLastError(const std::string& errorMsg)
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        NeonException exc;
        exc << "\n Error: " << cudaGetErrorString(error) << "\n"
            << errorMsg;
        NEON_THROW(exc);
    }
}

void gpuCheckLastError()
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        NeonException exc;
        exc << "\n Error: " << cudaGetErrorString(error);
        NEON_THROW(exc);
    }
}

std::string cublasGetErrorString(cublasStatus_t status)
{
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
        default:
            return "UNKNOWN_ERROR";
    }
};

}  // namespace sys
}  // namespace Neon