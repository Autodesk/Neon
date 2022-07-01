cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(gUt_bGrid ${SrcFiles})

target_link_libraries(gUt_bGrid 
	PUBLIC libNeonDomain
	PUBLIC gtest_main)

set_target_properties(gUt_bGrid PROPERTIES FOLDER "libNeonDomain")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "gUt_bGrid" FILES ${SrcFiles})

add_test(NAME gUt_bGrid COMMAND gUt_bGrid)