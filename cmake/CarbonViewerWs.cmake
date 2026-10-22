# Browser (WebSocket) backend for the carbon example viewer.
#
# Stack: ggerganov/imgui-ws  ->  ggerganov/incppect  ->  uNetworking/uWebSockets
#        ->  uNetworking/uSockets.  Everything is pinned to the exact commits
# imgui-ws / incppect expect (recent uWS releases changed the App API), fetched
# individually (submodules disabled) and compiled by *this* module rather than
# each project's own CMake, so that:
#   * the whole thing builds without OpenSSL (incppect force-instantiates the
#     SSL variant Incppect<true>; that one line is patched out below), and
#   * imgui-ws links against carbon's single vendored Dear ImGui (carbon_imgui)
#     instead of dragging in a second imgui copy (ODR).
# zlib is used for WebSocket permessage-deflate.

include(FetchContent)

enable_language(C)
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)

set(CARBON_USOCKETS_TAG "2ed0584e0b7fac11a80886d36903517e3a5ef9c3"
    CACHE STRING "uSockets commit (pinned to what uWebSockets expects)")
set(CARBON_UWEBSOCKETS_TAG "8f7d642f3ff7d9eb6ea7ee565afe5791c580fd73"
    CACHE STRING "uWebSockets commit (pinned to what incppect expects)")
set(CARBON_INCPPECT_TAG "ea0528467cc90ff401e4ba6f926e54fee8cc0ce8"
    CACHE STRING "incppect commit")
set(CARBON_IMGUI_WS_TAG "5e51bbf43a50d5f2c53124ab33f8740fa3f99b70"
    CACHE STRING "imgui-ws commit")

FetchContent_Declare(carbon_usockets
  GIT_REPOSITORY https://github.com/uNetworking/uSockets.git
  GIT_TAG ${CARBON_USOCKETS_TAG}
  GIT_SUBMODULES "")
FetchContent_Declare(carbon_uwebsockets
  GIT_REPOSITORY https://github.com/uNetworking/uWebSockets.git
  GIT_TAG ${CARBON_UWEBSOCKETS_TAG}
  GIT_SUBMODULES "")
FetchContent_Declare(carbon_incppect
  GIT_REPOSITORY https://github.com/ggerganov/incppect.git
  GIT_TAG ${CARBON_INCPPECT_TAG}
  GIT_SUBMODULES "")
FetchContent_Declare(carbon_imgui_ws
  GIT_REPOSITORY https://github.com/ggerganov/imgui-ws.git
  GIT_TAG ${CARBON_IMGUI_WS_TAG}
  GIT_SUBMODULES "")

# Populate only -- do NOT add_subdirectory (we compile the sources ourselves).
foreach(_dep carbon_usockets carbon_uwebsockets carbon_incppect carbon_imgui_ws)
  FetchContent_GetProperties(${_dep})
  if(NOT ${_dep}_POPULATED)
    FetchContent_Populate(${_dep})
  endif()
endforeach()

set(_us "${carbon_usockets_SOURCE_DIR}")
set(_uws "${carbon_uwebsockets_SOURCE_DIR}")
set(_inc "${carbon_incppect_SOURCE_DIR}")
set(_iws "${carbon_imgui_ws_SOURCE_DIR}")
set(_gen "${CMAKE_BINARY_DIR}/carbon_viewer_ws")

# --- uSockets (C, epoll, no SSL) -----------------------------------------------
add_library(carbon_usockets STATIC
  "${_us}/src/context.c"
  "${_us}/src/loop.c"
  "${_us}/src/socket.c"
  "${_us}/src/eventing/epoll_kqueue.c")
target_include_directories(carbon_usockets SYSTEM PUBLIC "${_us}/src")
target_compile_definitions(carbon_usockets PUBLIC LIBUS_NO_SSL)
target_compile_options(carbon_usockets PRIVATE -w)
set_target_properties(carbon_usockets PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- uWebSockets (header only) -------------------------------------------------
add_library(carbon_uwebsockets INTERFACE)
target_include_directories(carbon_uwebsockets SYSTEM INTERFACE "${_uws}/src" "${_us}/src")
target_link_libraries(carbon_uwebsockets INTERFACE carbon_usockets ZLIB::ZLIB Threads::Threads)

# --- incppect (patched: drop the SSL instantiation, bake incppect.js) ----------
file(READ "${_inc}/src/incppect.js" src_incppect_js)
configure_file("${_inc}/src/common.h.in" "${_gen}/incppect/common.h" @ONLY)
file(READ "${_inc}/src/incppect.cpp" _inc_cpp)
string(REPLACE
  "template class Incppect<true>;"
  "// Incppect<true> (SSL) intentionally not instantiated: carbon builds no-SSL."
  _inc_cpp "${_inc_cpp}")
file(WRITE "${_gen}/incppect/incppect.cpp" "${_inc_cpp}")

add_library(carbon_incppect STATIC "${_gen}/incppect/incppect.cpp")
target_include_directories(carbon_incppect SYSTEM PUBLIC "${_inc}/include")
target_include_directories(carbon_incppect SYSTEM PRIVATE "${_gen}/incppect")
target_link_libraries(carbon_incppect
  PUBLIC carbon_uwebsockets
  PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)
target_compile_options(carbon_incppect PRIVATE -w)
set_target_properties(carbon_incppect PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- imgui-ws (bake imgui-ws.js, shim the `imgui/imgui.h` include prefix) ------
file(READ "${_iws}/src/imgui-ws.js" src_imgui-ws_js)
configure_file("${_iws}/src/common.h.in" "${_gen}/imgui-ws/common.h" @ONLY)
# The compressors include "imgui/imgui.h"; carbon_imgui exposes it as <imgui.h>.
file(WRITE "${_gen}/imgui_shim/imgui/imgui.h" "#pragma once\n#include <imgui.h>\n")

add_library(carbon_imgui_ws STATIC
  "${_iws}/src/imgui-ws.cpp"
  "${_iws}/src/compressor-xor-rle-per-draw-list.cpp"
  "${_iws}/src/compressor-xor-rle-per-draw-list-with-vtx-offset.cpp")
target_include_directories(carbon_imgui_ws SYSTEM PUBLIC "${_iws}/include")
target_include_directories(carbon_imgui_ws SYSTEM PRIVATE
  "${_gen}/imgui-ws"
  "${_gen}/imgui_shim")
target_link_libraries(carbon_imgui_ws PUBLIC carbon_imgui PRIVATE carbon_incppect)
target_compile_options(carbon_imgui_ws PRIVATE -w)
set_target_properties(carbon_imgui_ws PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(carbon::imgui_ws ALIAS carbon_imgui_ws)
