cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(sUt_stencil ${SrcFiles})

target_link_libraries(sUt_stencil 
	PUBLIC libNeonSkeleton
	PUBLIC gtest_main)

set_target_properties(sUt_stencil PROPERTIES 
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(sUt_stencil PROPERTIES FOLDER "libNeonSkeleton")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "sUt_stencil" FILES ${SrcFiles})

add_test(NAME sUt_stencil COMMAND sUt_stencil)