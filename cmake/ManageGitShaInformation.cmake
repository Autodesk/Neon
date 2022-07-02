# Get and store git sha1 https://stackoverflow.com/a/4318642/1608232
list(APPEND CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmakeTools)
include("${PROJECT_SOURCE_DIR}/cmake/GetGitRevisionDescription.cmake")
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
git_local_changes(GIT_LOCAL_CHANGES_STATUS)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/git_sha1.cpp.in" "${CMAKE_CURRENT_SOURCE_DIR}/libNeonCore/src/core/git_sha1.cpp" @ONLY)
