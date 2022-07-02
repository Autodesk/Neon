cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(coreUt_io ${SrcFiles})

target_link_libraries(coreUt_io 
	PUBLIC libNeonCore
	PUBLIC gtest_main)

set_target_properties(coreUt_io PROPERTIES FOLDER "libNeonCore")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "coreUt_io" FILES ${SrcFiles})

add_test(NAME coreUt_io COMMAND coreUt_io)