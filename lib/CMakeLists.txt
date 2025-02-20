cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE neonFiles neon/*.*)

if (${NEON_BUILD_SHARED_LIBS})
    add_library(neon SHARED ${neonFiles})
    message(STATUS "Configuring neon as a shared library...")
    set_target_properties(neon PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
else ()
    add_library(neon STATIC ${neonFiles})
    message(STATUS "Configuring neon as a static library...")
endif ()

target_link_libraries(neon
        PUBLIC
        NeonDeveloperLib)

include("${PROJECT_SOURCE_DIR}/cmake/ExportHeader.cmake")
ExportHeader(neon)

set_target_properties(
        neon
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(neon PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")
target_include_directories(neon PUBLIC "${spdlog_SOURCE_DIR}/include")
target_include_directories(neon PUBLIC "${rapidjson_SOURCE_DIR}/include")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_link_libraries(neon PRIVATE "-lstdc++fs")
endif ()

target_include_directories(
        neon
        PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}")

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "neon" FILES ${neonFiles})

#if (${BUILD_NEON_TESTING})
#    add_subdirectory("tests")
#endif ()