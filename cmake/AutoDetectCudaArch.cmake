# AutoDetectCudaArch.cmake
# Detects installed GPU SM architectures at configure time and sets CMAKE_CUDA_ARCHITECTURES.
# The -Wno-deprecated-gpu-targets flag is applied ONLY to the nvcc autodetect run below.

# CUDA should be enabled as a language by the main CMakeLists.txt
if (NOT DEFINED CMAKE_CUDA_COMPILER OR NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA compiler not found")
endif ()

# Force auto-detection to run (we want to override any default CMake detection)
# Auto-detect cuda arch. Inspired by:
#   https://wagonhelm.github.io/articles/2018-03/detecting-cuda-capability-with-cmake
# This will define and populate CMAKE_CUDA_ARCHITECTURES
set(cuda_arch_autodetect_file "${CMAKE_BINARY_DIR}/autodetect_cuda_archs.cu")

file(WRITE "${cuda_arch_autodetect_file}" [[
    #include <cuda_runtime.h>
    #include <stdio.h>
    #include <set>

    int main() {
        int count = 0;
        if (cudaSuccess != cudaGetDeviceCount(&count)) {
            return -1;
        }
        if (count == 0) {
            return -1;
        }

        std::set<int> unique_archs;
        for (int device = 0; device < count; ++device) {
            cudaDeviceProp prop;
            if (cudaSuccess == cudaGetDeviceProperties(&prop, device)) {
                int arch = prop.major * 10 + prop.minor;
                unique_archs.insert(arch);
            } else {
                return -1;
            }
        }

        // Output unique architectures separated by semicolon
        bool first = true;
        for (int arch : unique_archs) {
            if (!first) printf(";");
            printf("%d", arch);
            first = false;
        }
        printf("\n");
        return 0;
    }
]])

# Build the nvcc args just for the autodetect step (do not touch global flags)
set(_NVCC_AUTODETECT_ARGS
        -ccbin "${CMAKE_CXX_COMPILER}"
        -Wno-deprecated-gpu-targets
        --run "${cuda_arch_autodetect_file}"
)

# Pretty log line
string(JOIN " " cuda_detect_cmd "${CMAKE_CUDA_COMPILER}" ${_NVCC_AUTODETECT_ARGS})
message(STATUS "Executing: ${cuda_detect_cmd}")

# Run nvcc with the extra warning suppression ONLY here
execute_process(
        COMMAND "${CMAKE_CUDA_COMPILER}" ${_NVCC_AUTODETECT_ARGS}
        RESULT_VARIABLE CUDA_RETURN_CODE
        OUTPUT_VARIABLE detected_archs
        ERROR_VARIABLE  error_output
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
)

if (CUDA_RETURN_CODE EQUAL 0)
    # Clean output should be semicolon-separated architecture numbers (e.g., "86" or "75;86")
    string(STRIP "${detected_archs}" archs_clean)
    if (archs_clean STREQUAL "")
        message(WARNING "CUDA arch auto-detect yielded empty output. Error: '${error_output}'. Falling back to 86.")
        set(CMAKE_CUDA_ARCHITECTURES "86")
    else ()
        # Convert to a proper CMake list (already semicolon-separated)
        set(CMAKE_CUDA_ARCHITECTURES "${archs_clean}")
        message(STATUS "Auto-detected CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif ()
else ()
    message(WARNING "GPU architectures auto-detect failed; return code: '${CUDA_RETURN_CODE}', error: '${error_output}'. Will build for sm_86 only.")
    set(CMAKE_CUDA_ARCHITECTURES "86")
endif ()
