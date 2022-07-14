# Toggle between building for the installed GPU's CC or all CC
set(NEON_BUILD_ONLY_FOR_INSTALLED_GPU "ON" CACHE BOOL "Build only for the current/installed compute capabilities")

if (NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
    if (${NEON_BUILD_ONLY_FOR_INSTALLED_GPU})
        message(STATUS "Building only for installed GPU's compute capabilities")
        include("${PROJECT_SOURCE_DIR}/cmake/AutoDetectCudaArch.cmake")
    else ()
        message(STATUS "Building for all compute capabilities")
        if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            message(STATUS "GPU architectures was not define (or not detected). Will build for sm_86 only.")
            set(CMAKE_CUDA_ARCHITECTURES 86)
        endif ()
    endif ()
else ()
    message(STATUS "Building for user defined compute capabilities:  ${CMAKE_CUDA_ARCHITECTURES}")
endif ()

# Auto-detect GPU architecture, sets ${CUDA_ARCHS}
message(STATUS "CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")
