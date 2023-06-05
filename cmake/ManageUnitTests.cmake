# Build Neon unit and benchmarks test
set(BUILD_NEON_TESTING "ON" CACHE BOOL "Build Neon tests")
if (${BUILD_NEON_TESTING})
    message(STATUS "Building Neon unit and performance tests")
else ()
    message(STATUS "Building Neon without unit and performance tests")
endif ()