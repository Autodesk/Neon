cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE libNeonSysFiles include/*.* src/*.*)

#set(CUDA_PROPAGATE_HOST_FLAGS ON)

if(${BUILD_NEON_SHARED_LIBS})
	add_library(libNeonSys SHARED ${libNeonSysFiles})
	message( STATUS "Configuring libNeonSys as a shared library...")
	set_target_properties(libNeonSys PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)		
else()
	add_library(libNeonSys STATIC ${libNeonSysFiles})
	message( STATUS "Configuring libNeonSys as a static library...")	
endif()

target_link_libraries(libNeonSys 
	PUBLIC NeonDeveloperLib
	PUBLIC libNeonCore
	PUBLIC ${CUDA_cublas_LIBRARY} ${CUDA_cublas_device_LIBRARY})

include("${PROJECT_SOURCE_DIR}/cmake/ExportHeader.cmake")
ExportHeader(libNeonSys)

set_target_properties(libNeonSys PROPERTIES 
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON)
	
target_include_directories(libNeonSys PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "libNeonSys" FILES ${libNeonSysFiles})

if (${BUILD_NEON_TESTING})
	add_subdirectory("tests")
endif()
