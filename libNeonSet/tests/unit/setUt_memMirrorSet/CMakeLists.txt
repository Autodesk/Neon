cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(setUt_memMirrorSet ${SrcFiles})

target_link_libraries(setUt_memMirrorSet 
	PUBLIC libNeonSet
	PUBLIC gtest_main)

set_target_properties(setUt_memMirrorSet PROPERTIES FOLDER "libNeonSetTests")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "setUt_memMirrorSet" FILES ${SrcFiles})

add_test(NAME setUt_memMirrorSet COMMAND setUt_memMirrorSet)