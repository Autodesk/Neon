find_package(CUDAToolkit REQUIRED)
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(gUt_mGrid ${SrcFiles})

target_link_libraries(gUt_mGrid
		PUBLIC libNeonCore
		PUBLIC libNeonSys
#		PUBLIC CUDA::cudart
		PUBLIC libNeonDomain
		PUBLIC gtest)


set_target_properties(gUt_mGrid PROPERTIES FOLDER "libNeonDomain")
#target_include_directories(gUt_mGrid PUBLIC "${gtest_SOURCE_DIR}/include")

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "gUt_mGrid" FILES ${SrcFiles})

add_test(NAME gUt_mGrid COMMAND gUt_mGrid)