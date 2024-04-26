cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(sysUt_mem ${SrcFiles})

target_link_libraries(sysUt_mem 
	PUBLIC libNeonSys
	PUBLIC gtest_main)

set_target_properties(sysUt_mem PROPERTIES FOLDER "libNeonSysTests")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "sysUt_mem" FILES ${SrcFiles})

add_test(NAME sysUt_mem COMMAND sysUt_mem)