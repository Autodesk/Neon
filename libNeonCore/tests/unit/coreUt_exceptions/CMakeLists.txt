cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(coreUt_exceptions ${SrcFiles})

target_link_libraries(coreUt_exceptions 
	PUBLIC libNeonCore
	PUBLIC gtest_main)

set_target_properties(coreUt_exceptions PROPERTIES FOLDER "libNeonCoreTests")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "coreUt_exceptions" FILES ${SrcFiles})

add_test(NAME coreUt_exceptions COMMAND coreUt_exceptions)