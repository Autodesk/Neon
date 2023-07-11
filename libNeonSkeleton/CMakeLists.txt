cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE libNeonSkeletonFiles include/*.* src/*.*)

if (${BUILD_NEON_SHARED_LIBS})	
	add_library(libNeonSkeleton SHARED ${libNeonSkeletonFiles})
	message( STATUS "Configuring libNeonSkeleton as a shared library...")
	set_target_properties(libNeonSet PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)	
else ()
	add_library(libNeonSkeleton STATIC ${libNeonSkeletonFiles})
	message( STATUS "Configuring libNeonSkeleton as a static library...")	
endif()

target_link_libraries(libNeonSkeleton 
	PUBLIC NeonDeveloperLib
	PUBLIC libNeonCore
	PUBLIC libNeonSys
	PUBLIC libNeonSet
	PUBLIC libNeonDomain)
	
include("${PROJECT_SOURCE_DIR}/cmake/ExportHeader.cmake")
ExportHeader(libNeonSet)

set_target_properties(libNeonSkeleton PROPERTIES 
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON)
	
target_include_directories(libNeonSkeleton PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "libNeonSkeleton" FILES ${libNeonSkeletonFiles})

if (${BUILD_NEON_TESTING})
	add_subdirectory("tests")
endif()