cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(solverPt_Poisson ${SrcFiles})

target_link_libraries(solverPt_Poisson 
	PUBLIC libNeonSolver
	PUBLIC poisson
	PUBLIC gtest_main)

set_target_properties(solverPt_Poisson PROPERTIES 
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(solverPt_Poisson PROPERTIES FOLDER "libNeonSolverTests")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "solverPt_Poisson" FILES ${SrcFiles})

add_test(NAME solverPt_Poisson COMMAND solverPt_Poisson)