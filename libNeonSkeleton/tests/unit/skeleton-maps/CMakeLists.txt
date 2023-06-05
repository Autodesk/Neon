cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(skeleton-map ${SrcFiles})

target_link_libraries(skeleton-map 
	PUBLIC libNeonSkeleton
	PUBLIC gtest_main)

set_target_properties(skeleton-map PROPERTIES 
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(skeleton-map PROPERTIES FOLDER "libNeonSkeleton")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "skeleton-map" FILES ${SrcFiles})

add_test(NAME skeleton-map COMMAND skeleton-map)