cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(sysUt_report ${SrcFiles})

target_link_libraries(sysUt_report 
	PUBLIC libNeonSys
	PUBLIC gtest_main)

set_target_properties(sysUt_report PROPERTIES FOLDER "libNeonSysTests")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "sysUt_report" FILES ${SrcFiles})

add_test(NAME sysUt_report COMMAND sysUt_report)