cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(domain-host-containers ${SrcFiles})

target_link_libraries(domain-host-containers 
	PUBLIC libNeonDomain
	PUBLIC gtest_main)

set_target_properties(domain-host-containers PROPERTIES 
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(domain-host-containers PROPERTIES FOLDER "libNeonDomain")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "domain-host-containers" FILES ${SrcFiles})

add_test(NAME domain-host-containers COMMAND domain-host-containers)