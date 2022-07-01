cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE libNeonSolverFiles include/*.* src/*.*)

if (${BUILD_NEON_SHARED_LIBS})	
	add_library(libNeonSolver SHARED ${libNeonSolverFiles})
	message( STATUS "Configuring libNeonSolver as a shared library...")
	set_target_properties(libNeonSet PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)	
else ()
	add_library(libNeonSolver STATIC ${libNeonSolverFiles})
	message( STATUS "Configuring libNeonSolver as a static library...")	
endif()

target_link_libraries(libNeonSolver 
	PUBLIC NeonDeveloperLib
	PUBLIC libNeonCore
	PUBLIC libNeonSys
	PUBLIC libNeonSet
	PUBLIC libNeonDomain
	PUBLIC libNeonSkeleton)
	
include("${PROJECT_SOURCE_DIR}/cmake/ExportHeader.cmake")
ExportHeader(libNeonSet)

set_target_properties(libNeonSolver PROPERTIES 
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(libNeonSolver PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "libNeonSolver" FILES ${libNeonSolverFiles})

if (${BUILD_NEON_TESTING})
	add_subdirectory("tests")
endif()