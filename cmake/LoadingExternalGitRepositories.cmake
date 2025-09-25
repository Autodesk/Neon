
# spdlog
include(FetchContent)
FetchContent_GetProperties(spdlog)
if (NOT spdlog_POPULATED)
	message(STATUS "Fetching spdlog...")
	FetchContent_Declare(spdlog
			GIT_REPOSITORY https://github.com/massimim/spdlog.git
			GIT_TAG neon
	)
	FetchContent_MakeAvailable(spdlog)
endif ()

if (${BUILD_NEON_TESTING})
	message(STATUS "Fetching googletest...")
	FetchContent_GetProperties(googletest)
	if (NOT googletest_POPULATED)
		# GoogleTest
		FetchContent_Declare(
				googletest
				GIT_REPOSITORY https://github.com/google/googletest.git
				GIT_TAG release-1.12.1  # Specify the desired version
		)
		# Rename the targets to avoid conflicts
		#set_target_properties(gtest PROPERTIES OUTPUT_NAME "gtest_unique")
		#set_target_properties(gtest_main PROPERTIES OUTPUT_NAME "gtest_main_unique")

		# Enable testing
		enable_testing()

		# Include GoogleTest CMake functions
		include(GoogleTest)
		FetchContent_MakeAvailable(googletest)

	endif ()
endif ()

# rapidjson
message(STATUS "Fetching rapidjson...")
FetchContent_GetProperties(rapidjson)
if (NOT rapidjson_POPULATED)
	FetchContent_Declare(rapidjson
			GIT_REPOSITORY https://github.com/Tencent/rapidjson.git
			GIT_TAG 24b5e7a8b27f42fa16b96fc70aade9106cf7102f
	)

	# Set options for RapidJSON
	set(RAPIDJSON_BUILD_DOC OFF CACHE BOOL "")
	set(RAPIDJSON_BUILD_EXAMPLES OFF CACHE BOOL "")
	set(RAPIDJSON_BUILD_TESTS OFF CACHE BOOL "")
	set(RAPIDJSON_BUILD_THIRDPARTY_GTEST OFF CACHE BOOL "")
	set(RAPIDJSON_BUILD_CXX20 ON CACHE BOOL "")
	set(RAPIDJSON_BUILD_CXX17 OFF CACHE BOOL "")
	set(RAPIDJSON_BUILD_CXX11 OFF CACHE BOOL "")


	FetchContent_MakeAvailable(rapidjson)
	# Removing the bin directory from rapidjson sources.
	# The directory contains jsonchecker, which is problematic from a licence prospective.
	file(REMOVE_RECURSE ${rapidjson_SOURCE_DIR}/bin/)
endif ()


# glm
FetchContent_GetProperties(glm)
message(STATUS "Fetching glm...")
if (NOT glm_POPULATED)
	FetchContent_Declare(glm
			GIT_REPOSITORY https://github.com/g-truc/glm.git
			GIT_TAG        master
	)
	FetchContent_Populate(glm)
	add_subdirectory(${glm_SOURCE_DIR})
endif()

#libigl
FetchContent_GetProperties(libigl)
if (NOT libigl_POPULATED)
	message(STATUS "Fetching libigl...")
	FetchContent_Declare(
			libigl
			GIT_REPOSITORY https://github.com/Ahdhn/libigl.git
			GIT_TAG        master
	)
	FetchContent_MakeAvailable(libigl)
endif()


# Set CMAKE_INSTALL_INCLUDEDIR to a relative path
set(CMAKE_INSTALL_INCLUDEDIR "include" CACHE PATH "Installation directory for header files")

## VTK
#FetchContent_Declare(
#        vtk
#        GIT_REPOSITORY https://gitlab.kitware.com/vtk/vtk.git
#        GIT_TAG v9.4.1  # Specify the desired version
#)
#
## Disable all VTK modules by default
#set(VTK_MODULE_ENABLE_VTK_CommonCore "YES" CACHE STRING "")
#set(VTK_MODULE_ENABLE_VTK_IOCore "YES" CACHE STRING "Enable VTK IOCore module")
#set(VTK_MODULE_ENABLE_VTK_RenderingCore "NO" CACHE STRING "Enable VTK RenderingCore module")
#set(VTK_MODULE_ENABLE_VTK_* "NO" CACHE STRING "Disable all other VTK modules")
##VTK_BUILD_TESTING
#set(VTK_MODULE_ENABLE_VTK_eigen "NO" CACHE STRING "")
#
#FetchContent_MakeAvailable(vtk)



#include(FetchContent)
#FetchContent_Declare(
#		Kokkos
#		URL      https://github.com/kokkos/kokkos/releases/download/4.5.01/kokkos-4.5.01.tar.gz
#		URL_HASH SHA256=52d003ffbbe05f30c89966e4009c017efb1662b02b2b73190670d3418719564c
#)
#set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Enable the OpenMP backend for Kokkos")
#set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable the CUDA backend for Kokkos")
#
#FetchContent_MakeAvailable(Kokkos)


#if (${BUILD_NEON_TESTING})
#	# GoogleTest
#	FetchContent_GetProperties(googletest)
#	if (NOT googletest_POPULATED)
#		message(STATUS "Fetching GoogleTest...")
#		set(gtest_force_shared_crt ON CACHE INTERNAL "make gtest link the runtimes dynamically" FORCE)
#		FetchContent_Declare(googletest
#				GIT_REPOSITORY https://github.com/google/googletest.git
#				GIT_TAG eaf9a3fd77869cf95befb87455a2e2a2e85044ff
#				)
#		FetchContent_MakeAvailable(googletest)
#		enable_testing()
#		include(GoogleTest)
#	endif ()
#endif ()