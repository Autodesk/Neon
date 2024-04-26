cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(coreUt_tuple3d ${SrcFiles})

target_link_libraries(coreUt_tuple3d 
	PUBLIC libNeonCore
	PUBLIC gtest_main)

set_target_properties(coreUt_tuple3d PROPERTIES FOLDER "libNeonCoreTests")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "coreUt_tuple3d" FILES ${SrcFiles})

add_test(NAME coreUt_tuple3d COMMAND coreUt_tuple3d)