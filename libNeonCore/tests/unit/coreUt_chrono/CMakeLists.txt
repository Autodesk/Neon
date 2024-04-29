cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(coreUt_chrono ${SrcFiles})

target_link_libraries(coreUt_chrono 
	PUBLIC libNeonCore
	PUBLIC gtest_main)

set_target_properties(coreUt_chrono PROPERTIES FOLDER "libNeonCoreTests")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "coreUt_chrono" FILES ${SrcFiles})

add_test(NAME coreUt_chrono COMMAND coreUt_chrono)
