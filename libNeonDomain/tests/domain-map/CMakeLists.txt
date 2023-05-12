cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(domain-map ${SrcFiles})

target_link_libraries(domain-map 
	PUBLIC libNeonDomain
	PUBLIC gtest_main)

set_target_properties(domain-map PROPERTIES 
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(domain-map PROPERTIES FOLDER "libNeonDomain")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "domain-map" FILES ${SrcFiles})

add_test(NAME domain-map COMMAND domain-map)