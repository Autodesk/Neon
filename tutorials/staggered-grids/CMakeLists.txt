cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(tutorial-staggered-grid ${SrcFiles})

target_link_libraries(tutorial-staggered-grid
		PUBLIC libNeonDomain
		PUBLIC libNeonSkeleton
		PUBLIC gtest_main)

set_target_properties(tutorial-staggered-grid PROPERTIES
		CUDA_SEPARABLE_COMPILATION ON
		CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set_target_properties(tutorial-staggered-grid PROPERTIES FOLDER "libNeonDomain")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "gUt_containers" FILES ${SrcFiles})

add_test(NAME tutorial-staggered-grid COMMAND tutorial-staggered-grid)