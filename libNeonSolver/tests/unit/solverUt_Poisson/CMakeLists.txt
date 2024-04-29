cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.*)

add_executable(solverUt_Poisson ${SrcFiles})

target_link_libraries(solverUt_Poisson 
	PUBLIC libNeonSolver
	PUBLIC poisson
	PUBLIC gtest_main)

set_target_properties(solverUt_Poisson PROPERTIES 
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(solverUt_Poisson PROPERTIES FOLDER "libNeonSolverTests")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "solverUt_Poisson" FILES ${SrcFiles})

add_test(NAME solverUt_Poisson COMMAND solverUt_Poisson)