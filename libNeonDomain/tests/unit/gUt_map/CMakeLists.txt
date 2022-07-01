cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(gUt_map ${SrcFiles})

target_link_libraries(gUt_map 
	PUBLIC libNeonDomain
	PUBLIC gtest_main)

set_target_properties(gUt_map PROPERTIES 
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(gUt_map PROPERTIES FOLDER "libNeonDomain")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "gUt_map" FILES ${SrcFiles})

add_test(NAME gUt_map COMMAND gUt_map)