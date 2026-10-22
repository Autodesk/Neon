include(FetchContent)

# FetchContent_Populate is deprecated in newer CMake; we avoid RapidJSON's root CMakeLists (old policy).
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()

# Pin to a post-v1.1.0 commit on master. v1.1.0 (2016) does not compile under modern Clang
# (>= 16) because of a const-qualified data member assignment in GenericStringRef
# (commit 3b2441b removed it). Pinning to the current master HEAD pulls in that fix plus
# subsequent clang 17/18/19 compatibility tweaks. The rapidjson project has not tagged a
# release since 2016, so master is the only path forward; bump in lockstep when needed.
FetchContent_Declare(
  rapidjson
  GIT_REPOSITORY https://github.com/Tencent/rapidjson.git
  GIT_TAG 24b5e7a8b27f42fa16b96fc70aade9106cf7102f
)

FetchContent_GetProperties(rapidjson)
if(NOT rapidjson_POPULATED)
  FetchContent_Populate(rapidjson)
endif()

add_library(rapidjson INTERFACE)
# Build-time: point at the FetchContent source. Install-time: rely on the rapidjson headers
# being copied into the consumer's prefix by carbon's install rule below; the same path is
# already on carbon's INSTALL_INTERFACE include list, so no extra dir is added here.
target_include_directories(rapidjson INTERFACE
  $<BUILD_INTERFACE:${rapidjson_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Stash the source dir so the parent CMakeLists.txt can install the headers under our prefix
# (rapidjson's own root CMakeLists is skipped via FetchContent_Populate, so it has no install
# rules of its own).
set(CARBON_RAPIDJSON_SOURCE_DIR "${rapidjson_SOURCE_DIR}")
