cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE libNeonPyFiles include/*.* src/*.*)

add_library(libNeonPy SHARED ${libNeonPyFiles})
message(STATUS "Configuring libNeonPy as a shared library...")
set_target_properties(libNeonSet PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

enable_language(CUDA)
find_package(CUDAToolkit)

target_link_libraries(libNeonPy
        PUBLIC NeonDeveloperLib
        PUBLIC libNeonCore
        PUBLIC libNeonSys
        PUBLIC libNeonSet
        PUBLIC libNeonDomain
        PUBLIC libNeonSkeleton
        PUBLIC CUDA::cudart
        PUBLIC CUDA::cuda_driver)

#
#target_link_libraries(libNeonPy)
#target_link_libraries(libNeonPy CUDA::cuda_driver)

include("${PROJECT_SOURCE_DIR}/cmake/ExportHeader.cmake")
ExportHeader(libNeonSet)

set_target_properties(libNeonPy PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Workaround for CUDA 12.8+ namespace compilation issue
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8")
    message(STATUS "Applying CUDA 12.8+ workaround for libNeonPy")
    target_compile_options(libNeonPy PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress 177>
    )
endif()

target_include_directories(libNeonPy PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "libNeonPy" FILES ${libNeonPyFiles})

# Install all shared libraries for Python packaging
install(TARGETS libNeonPy libNeonCore libNeonSys libNeonSet libNeonDomain libNeonSkeleton
    LIBRARY DESTINATION neon
    RUNTIME DESTINATION neon
)

# Install Warp's pre-built shared libraries
set(WARP_BIN_DIR "${PROJECT_SOURCE_DIR}/extern/warp/warp/bin")
if(EXISTS "${WARP_BIN_DIR}/warp.so")
    install(FILES 
        "${WARP_BIN_DIR}/warp.so"
        "${WARP_BIN_DIR}/warp-clang.so"
        DESTINATION warp/bin
    )
    message(STATUS "Including Warp binaries from ${WARP_BIN_DIR}")
else()
    message(WARNING "Warp binaries not found at ${WARP_BIN_DIR}. Run 'python build_lib.py' in extern/warp first.")
endif()
