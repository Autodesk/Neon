cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(skeletonSyntheticBenchmarks ${SrcFiles})

target_link_libraries(skeletonSyntheticBenchmarks
	PUBLIC libNeonSkeleton)

set_target_properties(skeletonSyntheticBenchmarks PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set_target_properties(skeletonSyntheticBenchmarks PROPERTIES FOLDER "libNeonSkeletonTests")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "skeletonSyntheticBenchmarks" FILES ${SrcFiles})