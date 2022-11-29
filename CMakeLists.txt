cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

#Project name 
project(Neon
        VERSION 0.3.3
        LANGUAGES CXX C CUDA)

#Default build type is Release if no build type is passed by the user
if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
endif ()

# Manage between building Neon as shared or static library
include("${PROJECT_SOURCE_DIR}/cmake/ManageLibraryType.cmake")

# Manage unit testing compilation following directives from BUILD_NEON_TESTING
include("${PROJECT_SOURCE_DIR}/cmake/ManageUnitTests.cmake")

# Manage CUDA compute capabilities by setting CMAKE_CUDA_ARCHITECTURES
include("${PROJECT_SOURCE_DIR}/cmake/ManageTargetCUDACompute.cmake")

# Toggle NVTX ranges. Enabled by default on linux. To enable use "-DNEON_USE_NVTX=ON"
include("${PROJECT_SOURCE_DIR}/cmake/Nvtx.cmake")

# Direct all output to /bin directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

# Enable IDE folder support
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Fetching external git repositories:
# - spdlog, rapidjson, GoogleTest (only if tests are enabled)
include("${PROJECT_SOURCE_DIR}/cmake/LoadingExternalGitRepositories.cmake")


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Definition of the target interface NeonDeveloperLib
add_library(NeonDeveloperLib INTERFACE)
target_compile_features(NeonDeveloperLib INTERFACE cxx_std_17)

include("${PROJECT_SOURCE_DIR}/cmake/ManageCompilationFlags.cmake")

# Get and store git sha1 https://stackoverflow.com/a/4318642/1608232
include("${PROJECT_SOURCE_DIR}/cmake/ManageGitShaInformation.cmake")

target_compile_options(NeonDeveloperLib INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:${NeonCXXFlags}>
        $<$<COMPILE_LANGUAGE:CUDA>:${NeonCUDAFlags}>
        )

target_include_directories(NeonDeveloperLib INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

if (${BUILD_NEON_SHARED_LIBS})
    set_property(TARGET NeonDeveloperLib PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET NeonDeveloperLib PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif ()

if (${NEON_USE_NVTX})
    target_compile_definitions(NeonDeveloperLib INTERFACE NEON_USE_NVTX)
endif ()

#OpenMP
find_package(OpenMP)
if (NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR "Neon could not find OpenMP")
endif ()
target_link_libraries(NeonDeveloperLib INTERFACE OpenMP::OpenMP_CXX)

#target_link_libraries(libNeonXXX INTERFACE $<BUILD_INTERFACE:NeonDeveloperLib>)

add_subdirectory("libNeonCore")
add_subdirectory("libNeonSys")
add_subdirectory("libNeonSet")
add_subdirectory("libNeonDomain")
add_subdirectory("libNeonSkeleton")
add_subdirectory("libNeonSolver")
add_subdirectory("tutorials")
add_subdirectory("apps")
add_subdirectory("benchmarks")


#-------------------------------------------------------------------#
# Configuration summary
#-------------------------------------------------------------------#
message("//===================================================")
message("|| ${PROJECT_NAME} build configuration:")
message("|| ${PROJECT_NAME} version : ${PROJECT_VERSION}")
message("||===================================================")
message("||  CMake version          : ${CMAKE_VERSION}")
if (NOT CMAKE_BUILD_TYPE)
    message("||  CMake build type       : NOT SET !")
else ()
    message("||  CMake build type       : ${CMAKE_BUILD_TYPE}")
endif ()
message("||  C++ Compiler           : ${CMAKE_CXX_COMPILER_ID} "
        "${CMAKE_CXX_COMPILER_VERSION} "
        "${CMAKE_CXX_COMPILER_WRAPPER}")
message("||  C++ Compiler exec      : ${CMAKE_CXX_COMPILER}")
message("||  OpenMP compile flags   : ${OpenMP_CXX_FLAGS}")
message("||  CUDA compiler ID       : ${CMAKE_CUDA_COMPILER_ID}")
message("||  CUDA compiler Version  : ${CMAKE_CUDA_COMPILER_VERSION}")
message("||  CUDA Compiler          : ${CMAKE_CUDA_COMPILER}")
message("||  CUDA Compiler exec     : ${CUDA_NVCC_EXECUTABLE}")
message("||  CUDA Compile flags     : ${CMAKE_CUDA_FLAGS}")
message("||  CUDA ARCHS             : ${CMAKE_CUDA_ARCHITECTURES}")
#message("||  NeonCXXFlags           : ${NeonCXXFlags}")
#message("||  NeonCUDAFlags          : ${NeonCUDAFlags}")
message("\\===================================================")
