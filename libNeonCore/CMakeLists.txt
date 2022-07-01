cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE libNeonCoreFiles include/*.* src/*.*)

if (${BUILD_NEON_SHARED_LIBS})	
	add_library(libNeonCore SHARED ${libNeonCoreFiles})
	message( STATUS "Configuring libNeonCore as a shared library...")
	set_target_properties(libNeonCore PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)	
else ()
	add_library(libNeonCore STATIC ${libNeonCoreFiles})
	message( STATUS "Configuring libNeonCore as a static library...")	
endif()

target_link_libraries(libNeonCore PUBLIC NeonDeveloperLib)

include("${PROJECT_SOURCE_DIR}/cmake/ExportHeader.cmake")
ExportHeader(libNeonCore)

set_target_properties(libNeonCore PROPERTIES 
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(libNeonCore PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")
target_include_directories(libNeonCore PUBLIC "${spdlog_SOURCE_DIR}/include")
target_include_directories(libNeonCore PUBLIC "${rapidjson_SOURCE_DIR}/include")

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") 
	target_link_libraries(libNeonCore PRIVATE "-lstdc++fs")
endif()

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "libNeonCore" FILES ${libNeonCoreFiles})

if (${BUILD_NEON_TESTING})
	add_subdirectory("tests")
endif()
