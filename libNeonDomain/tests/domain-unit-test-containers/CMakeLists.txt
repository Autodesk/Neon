cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(domain-unit-test-containers ${SrcFiles})

target_link_libraries(domain-unit-test-containers 
	PUBLIC libNeonDomain
	PUBLIC gtest_main)

set_target_properties(domain-unit-test-containers PROPERTIES 
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(domain-unit-test-containers PROPERTIES FOLDER "libNeonDomain")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "domain-unit-test-containers" FILES ${SrcFiles})

add_test(NAME domain-unit-test-containers COMMAND domain-unit-test-containers)