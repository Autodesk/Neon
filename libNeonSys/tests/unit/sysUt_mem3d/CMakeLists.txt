cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(sysUt_mem3d ${SrcFiles})

target_link_libraries(sysUt_mem3d 
	PUBLIC libNeonSys
	PUBLIC gtest_main)

set_target_properties(sysUt_mem3d PROPERTIES FOLDER "libNeonSys")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "sysUt_mem3d" FILES ${SrcFiles})

add_test(NAME sysUt_mem3d COMMAND sysUt_mem3d)