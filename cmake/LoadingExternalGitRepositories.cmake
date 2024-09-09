include(FetchContent)

# nanovdb
if(${NEON_USE_NANOVDB})
	FetchContent_GetProperties(nanovdb)
	if (NOT nanovdb_POPULATED)
		message(STATUS "Fetching nanovdb...")
		FetchContent_Declare(nanovdb
			GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/openvdb.git
			GIT_TAG        master
		)
		FetchContent_Populate(nanovdb)

		# Configure and build nanovdb with the desired options
		add_custom_target(build_nanovdb ALL
			COMMAND ${CMAKE_COMMAND} -S ${nanovdb_SOURCE_DIR} -B ${nanovdb_BINARY_DIR}
				-DNANOVDB_USE_OPENVDB=OFF
				-DUSE_NANOVDB=ON
				-DOPENVDB_BUILD_CORE=OFF
				-DOPENVDB_BUILD_BINARIES=OFF
				-DNANOVDB_USE_TBB=OFF
				-DNANOVDB_USE_CUDA=ON
				-DNANOVDB_USE_BLOSC=OFF
				-DNANOVDB_USE_ZLIB=OFF
				-DCMAKE_INSTALL_PREFIX=/usr/local
			COMMAND ${CMAKE_COMMAND} --build ${nanovdb_BINARY_DIR} --target install
			WORKING_DIRECTORY ${nanovdb_SOURCE_DIR}
		)

		include_directories(${nanovdb_SOURCE_DIR}/nanovdb)
	endif ()
endif ()

# hdf5 and HighFive
if(${NEON_USE_HDF5})
    find_package(HDF5 REQUIRED COMPONENTS CXX)
    if(HDF5_FOUND)
        message(STATUS "HDF5 found: ${HDF5_INCLUDE_DIRS}")

        # Include HDF5 directories and libraries globally
        set(HDF5_INCLUDE_DIRS ${HDF5_INCLUDE_DIRS})
        set(HDF5_LIBRARIES ${HDF5_LIBRARIES})

		message(STATUS "Fetching HighFive...")
        FetchContent_Declare(
            HighFive
            GIT_REPOSITORY https://github.com/BlueBrain/HighFive.git
            GIT_TAG v2.3.1  # Specify the version you want to use
        )
        FetchContent_MakeAvailable(HighFive)
        set(HighFive_FOUND TRUE)
        set(HighFive_INCLUDE_DIRS ${HighFive_INCLUDE_DIRS})
        set(HighFive_LIBRARIES HighFive)

        # Add definitions
        add_definitions(-DNEON_USE_HDF5)
    else()
        message(FATAL_ERROR "HDF5 not found")
    endif()
endif()

# spdlog
FetchContent_GetProperties(spdlog)
if (NOT spdlog_POPULATED)
	message(STATUS "Fetching spdlog...")
	FetchContent_Declare(spdlog
			GIT_REPOSITORY https://github.com/gabime/spdlog.git
			GIT_TAG v1.8.5
			)
	FetchContent_Populate(spdlog)
endif ()

# rapidjson
FetchContent_GetProperties(rapidjson)
if (NOT rapidjson_POPULATED)
	message(STATUS "Fetching rapidjson...")
	FetchContent_Declare(rapidjson
			GIT_REPOSITORY https://github.com/Tencent/rapidjson.git
			GIT_TAG 8f4c021fa2f1e001d2376095928fc0532adf2ae6
			)
	FetchContent_Populate(rapidjson)
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

# polyscope
if(${NEON_USE_POLYSCOPE})
	FetchContent_GetProperties(polyscope)
	if (NOT polyscope_POPULATED)
		message(STATUS "Fetching polyscope...")
		FetchContent_Declare(polyscope
			GIT_REPOSITORY https://github.com/Ahdhn/polyscope.git
			GIT_TAG        834b9c6c1a2675ccefd254d526f2dac3e3f831c6
		)
		FetchContent_MakeAvailable(polyscope)
	endif()
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

if (${BUILD_NEON_TESTING})
	# GoogleTest
	FetchContent_GetProperties(googletest)
	if (NOT googletest_POPULATED)
		message(STATUS "Fetching GoogleTest...")
		set(gtest_force_shared_crt ON CACHE INTERNAL "make gtest link the runtimes dynamically" FORCE)
		FetchContent_Declare(googletest
				GIT_REPOSITORY https://github.com/google/googletest.git
				GIT_TAG eaf9a3fd77869cf95befb87455a2e2a2e85044ff
				)
		FetchContent_MakeAvailable(googletest)
		enable_testing()
		include(GoogleTest)
	endif ()
endif ()