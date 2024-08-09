cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE poissonFiles include/*.* src/*.*)

add_library(poisson STATIC ${poissonFiles})

target_link_libraries(poisson PUBLIC libNeonSolver)

target_include_directories(poisson PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")

set_target_properties(poisson PROPERTIES 
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(poisson PROPERTIES FOLDER "libNeonSolverTests")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "poisson" FILES ${SrcFiles})