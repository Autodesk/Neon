find_package(CUDA REQUIRED)

if (NOT DEFINED CUDA_FOUND)
    message(FATAL_ERROR "CUDA not found")
endif ()

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    # Autodetect CUDA Arch
    # Auto-detect cuda arch. Inspired by https://wagonhelm.github.io/articles/2018-03/detecting-cuda-capability-with-cmake
    # This will define and populates CMAKE_CUDA_ARCHITECTURES
    set(cuda_arch_autodetect_file ${CMAKE_BINARY_DIR}/autodetect_cuda_archs.cu)
    file(WRITE ${cuda_arch_autodetect_file} [[
			#include <stdio.h>
			int main() {
			int count = 0; 
			if (cudaSuccess != cudaGetDeviceCount(&count)) { return -1; }
			if (count == 0) { return -1; }
			for (int device = 0; device < count; ++device) {
				cudaDeviceProp prop; 
				bool is_unique = true; 
				if (cudaSuccess == cudaGetDeviceProperties(&prop, device)) {
					for (int device_1 = device - 1; device_1 >= 0; --device_1) {
						cudaDeviceProp prop_1; 
						if (cudaSuccess == cudaGetDeviceProperties(&prop_1, device_1)) {
							if (prop.major == prop_1.major && prop.minor == prop_1.minor) {
								is_unique = false; 
								break; 
							}
						}
						else { return -1; }
					}
					if (is_unique) {
						fprintf(stderr, "%d%d", prop.major, prop.minor);
					}
				}
				else { return -1; }
			}
			return 0; 
			}
			]])

    set(cuda_detect_cmd "${CUDA_NVCC_EXECUTABLE} -ccbin ${CMAKE_CXX_COMPILER} --run ${cuda_arch_autodetect_file}")
    message(STATUS "Executing: ${cuda_detect_cmd}")
    execute_process(COMMAND "${CUDA_NVCC_EXECUTABLE}" "-ccbin" "${CMAKE_CXX_COMPILER}" "--run" "${cuda_arch_autodetect_file}"
            #WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/CMakeFiles/"
            RESULT_VARIABLE CUDA_RETURN_CODE
            OUTPUT_VARIABLE dummy
            ERROR_VARIABLE fprintf_output
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (CUDA_RETURN_CODE EQUAL 0)
        # Sanitize stderr output: extract two-digit compute capabilities (e.g., 86, 75)
        set(archs_raw "${fprintf_output}")
        string(REGEX MATCHALL "([0-9][0-9])" arch_list "${archs_raw}")
        list(REMOVE_DUPLICATES arch_list)
        list(JOIN arch_list ";" archs)
        if (archs STREQUAL "")
            message(WARNING "CUDA arch auto-detect yielded no valid numbers from '${archs_raw}'. Falling back to 86.")
            set(archs "86")
        endif ()
        set(CMAKE_CUDA_ARCHITECTURES "${archs}")
        message(STATUS "Detected CMAKE_CUDA_ARCHITECTURES='${CMAKE_CUDA_ARCHITECTURES}'")
    else ()
        message(WARNING "GPU architectures auto-detect failed; return code: '${CUDA_RETURN_CODE}', stderr: '${fprintf_output}'. Will build for sm_86 only.")
        set(CMAKE_CUDA_ARCHITECTURES "86")
    endif ()
endif ()
###################################################################################
