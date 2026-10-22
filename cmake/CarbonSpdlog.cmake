include(FetchContent)

set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_BENCH OFF CACHE BOOL "" FORCE)

# Required so spdlog's CMakeLists generates its own install rules + spdlogConfig.cmake.
# carbon's public logging.h includes <spdlog/spdlog.h>, so spdlog is part of carbon's
# transitive interface; carbonConfig.cmake calls find_dependency(spdlog) and resolves it
# from the same install prefix where carbon was installed.
set(SPDLOG_INSTALL ON CACHE BOOL "" FORCE)

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.15.0
)

FetchContent_MakeAvailable(spdlog)

# Linking the static spdlog library into libcarbon.so requires position-independent code.
if(TARGET spdlog)
  set_target_properties(spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
