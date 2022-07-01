cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(gUt_containers ${SrcFiles})

target_link_libraries(gUt_containers 
	PUBLIC libNeonDomain
	PUBLIC gtest_main)

set_target_properties(gUt_containers PROPERTIES 
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(gUt_containers PROPERTIES FOLDER "libNeonDomain")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "gUt_containers" FILES ${SrcFiles})

add_test(NAME gUt_containers COMMAND gUt_containers)