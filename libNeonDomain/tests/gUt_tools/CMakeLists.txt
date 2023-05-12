cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(gUt_tools ${SrcFiles})

target_link_libraries(gUt_tools 
	PUBLIC libNeonDomain
	PUBLIC gtest_main)

set_target_properties(gUt_tools PROPERTIES FOLDER "libNeonDomain")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "gUt_tools" FILES ${SrcFiles})

add_test(NAME gUt_tools COMMAND gUt_tools)