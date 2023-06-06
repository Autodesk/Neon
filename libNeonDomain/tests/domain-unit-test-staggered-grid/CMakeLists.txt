cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(domain-unit-test-staggered-grid ${SrcFiles})

target_link_libraries(domain-unit-test-staggered-grid
		PUBLIC libNeonDomain
		PUBLIC libNeonSkeleton
		PUBLIC gtest_main)

set_target_properties(domain-unit-test-staggered-grid PROPERTIES
		CUDA_SEPARABLE_COMPILATION ON
		CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set_target_properties(domain-unit-test-staggered-grid PROPERTIES FOLDER "libNeonDomain")
source_group(TREE . PREFIX "gUt_containers" FILES ${SrcFiles})

add_test(NAME domain-unit-test-staggered-grid COMMAND domain-unit-test-staggered-grid)