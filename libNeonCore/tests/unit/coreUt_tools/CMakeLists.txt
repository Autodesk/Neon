cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(coreUt_tools ${SrcFiles})

target_link_libraries(coreUt_tools 
	PUBLIC libNeonCore
	PUBLIC gtest_main)

set_target_properties(coreUt_tools PROPERTIES FOLDER "libNeonCore")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "coreUt_tools" FILES ${SrcFiles})

add_test(NAME coreUt_tools COMMAND coreUt_tools)