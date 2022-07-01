#pragma once
#include <cublas_v2.h>
#include <string>

namespace Neon {
namespace sys {

void gpuCheckLastError(const std::string& errorMsg);
void gpuCheckLastError();

/**
 * @brief Converting cublas status to string 
 * @param status the input status
 * @return a string of the status 
*/
std::string cublasGetErrorString(cublasStatus_t status);


}  // namespace sys
}  // namespace Neon