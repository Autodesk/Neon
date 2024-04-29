cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(domain-unit-test-cast ${SrcFiles})

target_link_libraries(domain-unit-test-cast 
	PUBLIC libNeonDomain
	PUBLIC gtest_main)

set_target_properties(domain-unit-test-cast PROPERTIES 
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(domain-unit-test-cast PROPERTIES FOLDER "libNeonDomainTests")
source_group(TREE . PREFIX "domain-unit-test-cast" FILES ${SrcFiles})

add_test(NAME domain-unit-test-cast COMMAND domain-unit-test-cast)