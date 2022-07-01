cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE libNeonDomainFiles include/*.* src/*.*)

if (${BUILD_NEON_SHARED_LIBS})	
	add_library(libNeonDomain SHARED ${libNeonDomainFiles})
	message( STATUS "Configuring libNeonDomain as a shared library...")
	set_target_properties(libNeonDomain PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)	
else ()
	add_library(libNeonDomain STATIC ${libNeonDomainFiles})
	message( STATUS "Configuring libNeonDomain as a static library...")	
endif()

target_link_libraries(libNeonDomain 
	PUBLIC NeonDeveloperLib
	PUBLIC libNeonCore
	PUBLIC libNeonSys
	PUBLIC libNeonSet)
	
include("${PROJECT_SOURCE_DIR}/cmake/ExportHeader.cmake")
ExportHeader(libNeonSet)

set_target_properties(libNeonDomain PROPERTIES 
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(libNeonDomain PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "libNeonDomain" FILES ${libNeonDomainFiles})

if (${BUILD_NEON_TESTING})
	add_subdirectory("tests")
endif()