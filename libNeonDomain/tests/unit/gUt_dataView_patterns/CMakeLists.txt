cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(gUt_dataView_patterns ${SrcFiles})

target_link_libraries(gUt_dataView_patterns 
	PUBLIC libNeonDomain
	PUBLIC gtest_main)

set_target_properties(gUt_dataView_patterns PROPERTIES FOLDER "libNeonDomain")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "gUt_dataView_patterns" FILES ${SrcFiles})

add_test(NAME gUt_dataView_patterns COMMAND gUt_dataView_patterns)