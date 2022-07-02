cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(domainUt_sGrid ${SrcFiles})

target_link_libraries(domainUt_sGrid
        PUBLIC libNeonDomain
        PUBLIC libNeonSkeleton
        PUBLIC gtest_main)

set_target_properties(domainUt_sGrid PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set_target_properties(domainUt_sGrid PROPERTIES FOLDER "libNeonDomain")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "gUt_containers" FILES ${SrcFiles})

add_test(NAME gUt_containers COMMAND gUt_containers)