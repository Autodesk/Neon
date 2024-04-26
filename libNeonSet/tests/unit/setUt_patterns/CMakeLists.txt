cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(setUt_patterns ${SrcFiles})

target_link_libraries(setUt_patterns 
	PUBLIC libNeonSet
	PUBLIC gtest_main)

set_target_properties(setUt_patterns PROPERTIES FOLDER "libNeonSetTests")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "setUt_patterns" FILES ${SrcFiles})


add_test(NAME setUt_patterns COMMAND setUt_patterns)