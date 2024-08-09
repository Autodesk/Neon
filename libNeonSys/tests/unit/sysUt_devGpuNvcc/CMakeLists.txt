cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(sysUt_devGpuNvcc ${SrcFiles})

target_link_libraries(sysUt_devGpuNvcc 
	PUBLIC libNeonSys
	PUBLIC gtest_main)

set_target_properties(sysUt_devGpuNvcc PROPERTIES 
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(sysUt_devGpuNvcc PROPERTIES FOLDER "libNeonSysTests")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "sysUt_devGpuNvcc" FILES ${SrcFiles})

add_test(NAME sysUt_devGpuNvcc COMMAND sysUt_devGpuNvcc)