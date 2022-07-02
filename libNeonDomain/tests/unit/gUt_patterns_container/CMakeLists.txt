cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(gUt_patterns_container ${SrcFiles})

target_link_libraries(gUt_patterns_container 
	PUBLIC libNeonDomain
	PUBLIC gtest_main)

set_target_properties(gUt_patterns_container PROPERTIES FOLDER "libNeonDomain")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "gUt_patterns_container" FILES ${SrcFiles})

add_test(NAME gUt_patterns_container COMMAND gUt_patterns_container)