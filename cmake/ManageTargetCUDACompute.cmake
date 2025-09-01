# Toggle between building for the installed GPU's CC or all CC
set(NEON_BUILD_ONLY_FOR_INSTALLED_GPU "ON" CACHE BOOL "Build only for the current/installed compute capabilities")

# Check if user explicitly set CUDA architectures via command line  
get_property(user_defined CACHE CMAKE_CUDA_ARCHITECTURES PROPERTY VALUE)
# Check if we should force our own auto-detection
# CMake may auto-detect wrong values (like 52 instead of 86 for A4000)
set(force_our_detection FALSE)
if (${NEON_BUILD_ONLY_FOR_INSTALLED_GPU})
    # Known problematic CMake auto-detected values that should be overridden
    set(problematic_values "52;53;60;61")  # Add more as needed
    if (CMAKE_CUDA_ARCHITECTURES IN_LIST problematic_values)
        set(force_our_detection TRUE)
        message(STATUS "Building only for installed GPU's compute capabilities (overriding CMake's incorrect detection of ${CMAKE_CUDA_ARCHITECTURES})")
    elseif (NOT user_defined)
        set(force_our_detection TRUE)
        message(STATUS "Building only for installed GPU's compute capabilities (no user override)")
    endif ()
endif ()

if (force_our_detection)
    # Clear any CMake-detected value and run our detection
    unset(CMAKE_CUDA_ARCHITECTURES)
    include("${PROJECT_SOURCE_DIR}/cmake/AutoDetectCudaArch.cmake")
elseif (DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    message(STATUS "Building for specified compute capabilities: ${CMAKE_CUDA_ARCHITECTURES}")
else ()
    message(STATUS "No CUDA architectures defined. Will build for sm_86 only.")
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif ()

# Auto-detect GPU architecture, sets ${CUDA_ARCHS}
message(STATUS "CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")
