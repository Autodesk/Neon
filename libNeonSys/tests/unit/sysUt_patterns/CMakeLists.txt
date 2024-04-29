cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(sysUt_patterns ${SrcFiles})

target_link_libraries(sysUt_patterns 
	PUBLIC libNeonSys
	PUBLIC gtest_main)

set_target_properties(sysUt_patterns PROPERTIES 
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set_target_properties(sysUt_patterns PROPERTIES FOLDER "libNeonSysTests")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "sysUt_patterns" FILES ${SrcFiles})

add_test(NAME sysUt_patterns COMMAND sysUt_patterns)