cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE libNeonSetFiles include/*.* src/*.*)

#set(CUDA_PROPAGATE_HOST_FLAGS ON)

if (${BUILD_NEON_SHARED_LIBS})
    add_library(libNeonSet SHARED ${libNeonSetFiles})
    message(STATUS "Configuring libNeonSet as a shared library...")
    set_target_properties(libNeonSet PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
else ()
    add_library(libNeonSet STATIC ${libNeonSetFiles})
    message(STATUS "Configuring libNeonSet as a static library...")
endif ()

find_package(MPI REQUIRED)
# Find NCCL
find_library(NCCL_LIBRARIES nccl)
find_path(NCCL_INCLUDE_DIRS nccl.h)

target_link_libraries(libNeonSet
        PUBLIC NeonDeveloperLib
        PUBLIC libNeonCore
        PUBLIC libNeonSys
        PUBLIC MPI::MPI_CXX
        PUBLIC ${NCCL_LIBRARIES})

include("${PROJECT_SOURCE_DIR}/cmake/ExportHeader.cmake")
ExportHeader(libNeonSet)

target_include_directories(libNeonSet PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/" ${MPI_INCLUDE_PATH} ${NCCL_INCLUDE_DIRS})

#add_flag(Public All COMPILE_DEFINITIONS NEONSYS_API_DLL_EXPORT)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "libNeonSet" FILES ${libNeonSetFiles})

if (${BUILD_NEON_TESTING})
    add_subdirectory("tests")
endif ()
