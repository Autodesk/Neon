# Toggle between building for the installed GPU's CC or all CC
set(NEON_BUILD_ONLY_FOR_INSTALLED_GPU "ON" CACHE BOOL "Build only for the current/installed compute capabilities")

# Option to build for all common GPUs (for distributable wheels)
set(NEON_BUILD_FOR_ALL_GPUS "OFF" CACHE BOOL "Build for all common GPU architectures (for wheel distribution)")

# Define common GPU architectures for distribution:
# - 70: Volta (V100)
# - 75: Turing (RTX 20xx, T4)
# - 80: Ampere (A100, A30)
# - 86: Ampere (RTX 30xx, A40, A10, A4000, A5000, A6000)
# - 89: Ada Lovelace (RTX 40xx, L40, L4)
# - 90: Hopper (H100, H200, GH100)
# - 100: Blackwell data-center (B100, B200, GB200) [requires CUDA >= 12.8]
# - 120: Blackwell consumer (RTX 50xx, GB202/203/205/206) [requires CUDA >= 12.8]
set(NEON_ALL_GPU_ARCHITECTURES "70;75;80;86;89;90;100;120" CACHE STRING "GPU architectures to build for when NEON_BUILD_FOR_ALL_GPUS is ON")

if (${NEON_BUILD_FOR_ALL_GPUS})
    # Build for all common architectures (for distributable wheels)
    set(CMAKE_CUDA_ARCHITECTURES ${NEON_ALL_GPU_ARCHITECTURES})
    message(STATUS "Building for all common GPU architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else ()
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
endif ()

# Report final architecture configuration
message(STATUS "CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")
