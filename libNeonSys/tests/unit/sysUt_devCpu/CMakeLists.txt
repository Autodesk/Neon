cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(sysUt_devCpu ${SrcFiles})

target_link_libraries(sysUt_devCpu 
	PUBLIC libNeonSys
	PUBLIC gtest_main)

set_target_properties(sysUt_devCpu PROPERTIES FOLDER "libNeonSys")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "sysUt_devCpu" FILES ${SrcFiles})

add_test(NAME sysUt_devCpu COMMAND sysUt_devCpu)