
# spdlog
include(FetchContent)
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