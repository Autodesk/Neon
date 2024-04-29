cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(setUt_multiDeviceObject ${SrcFiles})

target_link_libraries(setUt_multiDeviceObject
        PUBLIC libNeonSet
        PUBLIC gtest_main)

set_target_properties(setUt_multiDeviceObject PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set_target_properties(setUt_multiDeviceObject PROPERTIES FOLDER "libNeonSetTests")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "setUt_multiDeviceObject" FILES ${SrcFiles})


add_test(NAME setUt_multiDeviceObject COMMAND setUt_multiDeviceObject)