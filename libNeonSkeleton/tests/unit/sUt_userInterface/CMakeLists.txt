cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(sUt_userInterface ${SrcFiles})

target_link_libraries(sUt_userInterface 
	PUBLIC libNeonSkeleton
	PUBLIC gtest_main)

set_target_properties(sUt_userInterface PROPERTIES 
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(sUt_userInterface PROPERTIES FOLDER "libNeonSkeleton")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "sUt_userInterface" FILES ${SrcFiles})

add_test(NAME sUt_userInterface COMMAND sUt_userInterface)