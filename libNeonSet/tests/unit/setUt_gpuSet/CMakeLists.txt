cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(setUt_gpuSet ${SrcFiles})

target_link_libraries(setUt_gpuSet 
	PUBLIC libNeonSet
	PUBLIC gtest_main)

set_target_properties(setUt_gpuSet PROPERTIES FOLDER "libNeonSet")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "setUt_gpuSet" FILES ${SrcFiles})

add_test(NAME setUt_gpuSet COMMAND setUt_gpuSet)