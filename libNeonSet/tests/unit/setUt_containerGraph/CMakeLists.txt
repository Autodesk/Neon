cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
SET(APP "setUt_containerGraph")
file(GLOB_RECURSE SrcFiles src/*.*)

SET(APP, "setUt_containerGraph")
add_executable(${APP} ${SrcFiles})

target_link_libraries(${APP} 
	PUBLIC libNeonSkeleton
	PUBLIC gtest_main)

set_target_properties(${APP} PROPERTIES 
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set_target_properties(${APP} PROPERTIES FOLDER "libNeonSet")

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "${APP}" FILES ${SrcFiles})

add_test(NAME ${APP} COMMAND ${APP})